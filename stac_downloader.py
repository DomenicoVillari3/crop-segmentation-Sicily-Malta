"""
stac_downloader.py
==================
Unica responsabilità: dato un bbox + anno, scarica il cubo 4D
multistagionale Sentinel-2 e ritorna np.ndarray (4, 6, H, W) o None.

Non sa nulla di MinIO, del modello, o dell'API.
Non ha stato: è una funzione con classe per dependency injection del client STAC.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pystac_client
import stackstac

from config import (
    STAC_ENDPOINT,
    STAC_COLLECTION,
    SENTINEL_BANDS,
    SENTINEL_RESOLUTION,
    MAX_CLOUD_COVER,
    SEASONS,
    ORDERED_SEASONS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tipi interni
# ---------------------------------------------------------------------------

class SeasonResult(NamedTuple):
    season:      str
    date_range:  str
    cloud_cover: float
    item_id:     str


# ---------------------------------------------------------------------------
# Helper UTM dinamico
# ---------------------------------------------------------------------------

def _utm_epsg_from_bbox(bbox: list[float]) -> int:
    """
    Calcola il codice EPSG UTM corretto per il centro del bbox.
    Funziona per qualsiasi longitudine mondiale, non solo Sicilia.

    Args:
        bbox: [min_lon, min_lat, max_lon, max_lat]

    Returns:
        EPSG code (es. 32633 per UTM 33N, 32630 per UTM 30N)
    """
    center_lon = (bbox[0] + bbox[2]) / 2
    center_lat = (bbox[1] + bbox[3]) / 2

    zone = int((center_lon + 180) / 6) + 1

    # Nord o Sud
    epsg = 32600 + zone if center_lat >= 0 else 32700 + zone

    logger.debug(f"UTM EPSG calcolato: {epsg} (zona {zone}, centro lon={center_lon:.2f})")
    return epsg


# ---------------------------------------------------------------------------
# Generatore date stagionali
# ---------------------------------------------------------------------------

def _build_date_ranges(year: int) -> dict[str, str]:
    """
    Costruisce i range di date stagionali per un anno dato.

    Le finestre sono leggermente asimmetriche per catturare
    il picco fenologico di ogni stagione mediterranea:
      - Winter:  gen-feb   (cereali emergenti, olive raccolte)
      - Spring:  apr-mag   (massima biomassa cereali, fioritura vigneti)
      - Summer:  lug-ago   (cereali maturi/raccolti, stress idrico)
      - Autumn:  ott-nov   (raccolta uva/olive, risemina)

    Il febbraio degli anni bisestili è gestito automaticamente.
    """
    import calendar
    feb_last = calendar.monthrange(year, 2)[1]  # 28 o 29

    return {
        "winter": f"{year}-01-01/{year}-02-{feb_last}",
        "spring": f"{year}-04-15/{year}-05-30",
        "summer": f"{year}-07-01/{year}-08-15",
        "autumn": f"{year}-10-01/{year}-11-15",
    }


# ---------------------------------------------------------------------------
# STACDownloader
# ---------------------------------------------------------------------------

class STACDownloader:
    """
    Scarica cubi Sentinel-2 multistagionali via STAC API.

    Usage:
        downloader = STACDownloader()
        cube = downloader.download(bbox=[12.4, 36.6, 12.5, 36.7], year=2023)
        # cube: np.ndarray (4, 6, H, W) float32 oppure None
    """

    def __init__(self) -> None:
        self._catalog = pystac_client.Client.open(STAC_ENDPOINT)
        logger.info(f"STACDownloader connesso a: {STAC_ENDPOINT}")

    # -----------------------------------------------------------------------
    # API pubblica
    # -----------------------------------------------------------------------

    def download(
        self,
        bbox: list[float],
        year: int,
        max_cloud_cover: int | None = None,
        fallback_cloud_cover: int = 40,
    ) -> np.ndarray | None:
        """
        Scarica il cubo 4D multistagionale per bbox e anno.

        Args:
            bbox:                 [min_lon, min_lat, max_lon, max_lat]
            year:                 anno solare (es. 2023)
            max_cloud_cover:      soglia nuvole (default da config)
            fallback_cloud_cover: soglia usata se la prima non trova immagini
                                  per qualche stagione (es. zone nuvolose)

        Returns:
            np.ndarray (4, 6, H, W) float32 se successo
            None se almeno una stagione non ha immagini valide
        """
        cloud_limit = max_cloud_cover or MAX_CLOUD_COVER

        logger.info(f"  STAC download | bbox={bbox} | anno={year} | cloud<{cloud_limit}%")

        date_ranges = _build_date_ranges(year)
        epsg = _utm_epsg_from_bbox(bbox)

        # --- Fase 1: selezione best item per stagione ---
        stac_items = []
        season_log: list[SeasonResult] = []

        for season in ORDERED_SEASONS:
            date_range = date_ranges[season]

            item, cloud = self._best_item_for_season(
                bbox, date_range, cloud_limit
            )

            # Fallback automatico con soglia più permissiva
            if item is None and cloud_limit < fallback_cloud_cover:
                logger.warning(
                    f"   ⚠️  {season}: nessuna immagine con cloud<{cloud_limit}%. "
                    f"Riprovo con cloud<{fallback_cloud_cover}%..."
                )
                item, cloud = self._best_item_for_season(
                    bbox, date_range, fallback_cloud_cover
                )

            if item is None:
                logger.error(
                    f"   ❌ {season} ({date_range}): nessuna immagine disponibile. "
                    f"Aumenta il bbox o la tolleranza cloud cover."
                )
                return None  # fast-fail: cubo incompleto → inutilizzabile

            stac_items.append(item)
            season_log.append(SeasonResult(season, date_range, cloud, item.id))
            logger.info(f"   ✅ {season}: {cloud:.1f}% cloud | {item.id}")

        # --- Riepilogo selezione ---
        self._log_season_summary(season_log, year)

        # --- Fase 2: stacking xarray ---
        cube = self._stack_and_compute(stac_items, bbox, epsg)
        if cube is None:
            return None

        # --- Fase 3: validazione dimensioni ---
        if not self._validate_shape(cube, expected_T=4, expected_C=6):
            return None

        logger.info(
            f"✅ Download completato: {cube.shape} | "
            f"range=[{cube.min():.0f}, {cube.max():.0f}]"
        )
        return cube

    def available_years_for_bbox(
        self,
        bbox: list[float],
        years: list[int],
    ) -> list[int]:
        """
        Verifica quali anni hanno almeno un'immagine per stagione nel bbox.
        Utile per popolare il selettore anni nell'UI senza scaricare tutto.

        Nota: fa solo 4 ricerche STAC per anno (solo check esistenza, no download).
        """
        available = []
        for year in years:
            if self._year_has_full_coverage(bbox, year):
                available.append(year)
        return sorted(available, reverse=True)

    # -----------------------------------------------------------------------
    # Metodi privati
    # -----------------------------------------------------------------------

    def _best_item_for_season(
        self,
        bbox: list[float],
        date_range: str,
        max_cloud: int,
    ) -> tuple:
        """
        Cerca il STAC item con meno nuvole in un periodo.

        Returns:
            (item, cloud_cover) oppure (None, None)
        """
        try:
            search = self._catalog.search(
                collections=[STAC_COLLECTION],
                bbox=bbox,
                datetime=date_range,
                query={"eo:cloud_cover": {"lt": max_cloud}},
            )
            items = list(search.item_collection())

            if not items:
                return None, None

            best = min(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))
            return best, best.properties.get("eo:cloud_cover", 0.0)

        except Exception as e:
            logger.error(f"   Errore ricerca STAC [{date_range}]: {e}")
            return None, None

    def _stack_and_compute(
        self,
        items: list,
        bbox: list[float],
        epsg: int,
    ) -> np.ndarray | None:
        """
        Fa lo stack xarray e triggera il compute dask.
        Ritorna np.ndarray (4, 6, H, W) float32 o None se errore.
        """
        try:
            logger.info(f"   📥 Stacking {len(items)} scene | EPSG:{epsg}...")

            data = stackstac.stack(
                items,
                assets=SENTINEL_BANDS,
                bounds_latlon=bbox,
                resolution=SENTINEL_RESOLUTION,
                epsg=epsg,
                fill_value=0,
                rescale=False,
            )

            # Assicura T=4: prendi solo il primo item per timestep
            # (stackstac può aggiungere timestep extra se un item ha più granuli)
            if data.sizes["time"] > 4:
                logger.warning(
                    f"   ⚠️  {data.sizes['time']} timestep trovati, seleziono i primi 4."
                )
                data = data.isel(time=slice(0, 4))

            cube = data.astype("float32").compute().values
            return cube

        except Exception as e:
            logger.error(f"   ❌ Errore stacking: {e}")
            return None

    def _validate_shape(
        self,
        cube: np.ndarray,
        expected_T: int,
        expected_C: int,
    ) -> bool:
        """Valida che il cubo abbia la shape attesa."""
        if cube.ndim != 4:
            logger.error(f"❌ Shape errata: {cube.shape} (atteso 4D)")
            return False
        T, C, H, W = cube.shape
        if T != expected_T:
            logger.error(f"❌ T={T} (atteso {expected_T})")
            return False
        if C != expected_C:
            logger.error(f"❌ C={C} (atteso {expected_C})")
            return False
        if H < 10 or W < 10:
            logger.error(f"❌ Area troppo piccola: {H}x{W} pixel")
            return False
        return True

    def _year_has_full_coverage(self, bbox: list[float], year: int) -> bool:
        """Verifica (senza download) che tutte le stagioni abbiano almeno 1 immagine."""
        date_ranges = _build_date_ranges(year)
        for season in ORDERED_SEASONS:
            item, _ = self._best_item_for_season(
                bbox, date_ranges[season], max_cloud=MAX_CLOUD_COVER
            )
            if item is None:
                return False
        return True

    @staticmethod
    def _log_season_summary(log: list[SeasonResult], year: int) -> None:
        logger.info(f"\n   {'─'*55}")
        logger.info(f"   Selezione scene per {year}:")
        for r in log:
            logger.info(
                f"   {r.season:<8} | cloud={r.cloud_cover:>5.1f}% | {r.id}"
            )
        logger.info(f"   {'─'*55}\n")