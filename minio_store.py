"""
minio_store.py
==============
Wrapper boto3 per il layer di persistenza MinIO.

Espone esattamente 3 operazioni pubbliche:
  - find_tile(lat, lon) / find_tile_by_bbox(bbox)  → (key, bbox) | (None, None)
  - download_cube(key)                              → np.ndarray (4, 6, H, W)
  - upload_cube(cube, bbox, task_id)                → str (npy_key) | None

Logica interna:
  - is_mostly_water_or_empty(cube): scarta tile inutili prima dell'upload
  - _make_rgb_preview(cube):        genera PNG estivo per validazione visiva
  - _paginate_bucket(prefix):       generatore lazy sugli oggetti MinIO
"""

from __future__ import annotations

import io
import json
import logging
from typing import Generator

import datetime
import boto3
import numpy as np
from botocore.exceptions import ClientError
from PIL import Image
from datetime import datetime, timezone

from config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_BUCKET_NAME,
    WATER_NIR_THRESHOLD,
    WATER_RGB_THRESHOLD,
    EMPTY_RATIO_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MinioStore
# ---------------------------------------------------------------------------

class MinioStore:
    """
    Unico punto di accesso a MinIO per tutta la pipeline.
    Thread-safe: boto3 client è thread-safe per operazioni di sola lettura;
    per upload concorrenti è sufficiente un'istanza per worker asyncio.
    """

    # Prefissi bucket
    PREFIX_NPY = "raw_cubes/"
    PREFIX_PNG = "rgb_images/"

    def __init__(self) -> None:
    
        self._client = boto3.client(
            "s3",
            endpoint_url=f"http://{MINIO_ENDPOINT}",
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
        )
        self._bucket = MINIO_BUCKET_NAME
        logger.info(f"MinioStore connesso a {MINIO_ENDPOINT} → bucket '{self._bucket}'")

    # -----------------------------------------------------------------------
    # OPERAZIONE 1 — find_tile  (aggiornata)
    # -----------------------------------------------------------------------

    def find_tile(
        self,
        lat: float,
        lon: float,
        year: int | None = None,   # None → usa l'anno più recente disponibile
    ) -> tuple[str | None, list | None, int | None]:
        """
        Cerca la tile che contiene (lat, lon) per un dato anno.
        Se year=None, ritorna la versione più recente disponibile.

        Returns:
            (key, bbox, year_found) oppure (None, None, None)
        """
        # Determina il prefix di ricerca
        if year is not None:
            prefixes = [f"{self.PREFIX_NPY}year={year}/"]
        else:
            # Cerca tutti gli anni disponibili, ordine discendente
            prefixes = self._list_year_prefixes(descending=True)

        for prefix in prefixes:
            year_from_prefix = int(prefix.split("year=")[1].rstrip("/"))

            for key, meta in self._iter_metadata(prefix):
                bbox = self._parse_bbox(meta)
                if bbox is None:
                    continue

                if self._point_in_bbox(lat, lon, bbox):
                    if self._is_shape_valid(meta):
                        logger.info(f"✅ Tile trovata: {key} (anno {year_from_prefix})")
                        return key, bbox, year_from_prefix
                    else:
                        logger.warning(f"⚠️  Tile corrotta: {key}, continuo...")

        logger.info(f"❌ Nessuna tile trovata per ({lat:.4f}, {lon:.4f}) anno={year}")
        return None, None, None

    def list_available_years(self, lat: float, lon: float) -> list[int]:
        """
        Ritorna tutti gli anni per cui esiste una tile sul punto dato.
        Utile per l'endpoint /api/v1/analysis/available-years.
        """
        found_years = []
        for prefix in self._list_year_prefixes(descending=False):
            year_val = int(prefix.split("year=")[1].rstrip("/"))
            for key, meta in self._iter_metadata(prefix):
                bbox = self._parse_bbox(meta)
                if bbox and self._point_in_bbox(lat, lon, bbox):
                    found_years.append(year_val)
                    break  # basta una tile per anno, passa al prossimo prefix
        return found_years

    def find_tile_by_bbox(self, request_bbox: list[float]) -> tuple[str | None, list | None]:
        """
        Cerca una tile che contenga interamente il bbox richiesto.

        Utile per richieste API che specificano un'area invece di un punto.
        """
        logger.info(f"🔎 Ricerca tile per bbox {request_bbox}...")

        for key, meta in self._iter_metadata(self.PREFIX_NPY):
            tile_bbox = self._parse_bbox(meta)
            if tile_bbox is None:
                continue

            if self._bbox_contains(tile_bbox, request_bbox):
                logger.info(f"✅ Tile trovata: {key}")
                return key, tile_bbox

        logger.info("❌ Nessuna tile trovata per il bbox richiesto.")
        return None, None

    # -----------------------------------------------------------------------
    # OPERAZIONE 2 — download_cube
    # -----------------------------------------------------------------------

    def download_cube(self, key: str) -> np.ndarray:
        """
        Scarica e deserializza un cubo .npy da MinIO.

        Conversione: uint16 → float32 (mantiene i valori originali di riflettanza).

        Returns:
            np.ndarray shape (4, 6, H, W) dtype float32

        Raises:
            ValueError  se la shape non è (4, 6, H, W)
            ClientError se il file non esiste su MinIO
        """
        logger.info(f"⬇️  Download da MinIO: {key}")

        buffer = io.BytesIO()
        self._client.download_fileobj(self._bucket, key, buffer)
        buffer.seek(0)

        cube = np.load(buffer).astype(np.float32)

        # Validazione shape
        if cube.ndim != 4 or cube.shape[0] != 4 or cube.shape[1] != 6:
            raise ValueError(
                f"Shape non valida: {cube.shape}. Atteso (4, 6, H, W). "
                f"File probabilmente corrotto: {key}"
            )

        logger.info(f"   📦 Cubo: {cube.shape}  range=[{cube.min():.0f}, {cube.max():.0f}]")
        return cube

    # -----------------------------------------------------------------------
    # OPERAZIONE 3 — upload_cube  (aggiornata)
    # -----------------------------------------------------------------------

    def upload_cube(
        self,
        cube: np.ndarray,
        bbox: list[float],
        year: int,                          # ora obbligatorio
        task_id: int | str = "onthefly",
        seasons: list[str] | None = None,
    ) -> str | None:

        if self.is_mostly_water_or_empty(cube):
            logger.info("Tile scartata. Upload skippato.")
            return None

        center_lat = (bbox[1] + bbox[3]) / 2
        center_lon = (bbox[0] + bbox[2]) / 2
        lat_s = f"{center_lat:.4f}"
        lon_s = f"{center_lon:.4f}"

        metadata = {
            "bbox":         json.dumps(bbox),
            "center_lat":   str(center_lat),
            "center_lon":   str(center_lon),
            "year":         str(year),
            "seasons":      json.dumps(seasons or ["winter","spring","summer","autumn"]),
            "task_id":      str(task_id),
            "shape":        json.dumps(list(cube.shape)),
            "uploaded_at":   datetime.now(timezone.utc).isoformat(),
        }

        # Chiave strutturata per anno
        npy_key = f"{self.PREFIX_NPY}year={year}/lat_{lat_s}_lon_{lon_s}.npy"
        png_key = f"{self.PREFIX_PNG}year={year}/lat_{lat_s}_lon_{lon_s}.png"

        # Upload NPY
        try:
            npy_buf = io.BytesIO()
            np.save(npy_buf, cube.astype(np.uint16))
            npy_buf.seek(0)
            self._client.upload_fileobj(
                npy_buf, self._bucket, npy_key,
                ExtraArgs={"Metadata": metadata},
            )
            logger.info(f"   📦 NPY: {npy_key}")
        except ClientError as e:
            logger.error(f"❌ Upload NPY fallito: {e}")
            return None

        # Upload PNG (best-effort)
        try:
            preview = self._make_rgb_preview(cube)
            png_buf = io.BytesIO()
            preview.save(png_buf, format="PNG", optimize=True)
            png_buf.seek(0)
            png_meta = {**metadata, "image_type": "rgb_summer",
                        "image_size": f"{preview.width}x{preview.height}",
                        "source_npy": npy_key}
            self._client.upload_fileobj(
                png_buf, self._bucket, png_key,
                ExtraArgs={"Metadata": png_meta, "ContentType": "image/png"},
            )
            logger.info(f"   🖼️  PNG: {png_key}")
        except Exception as e:
            logger.warning(f"   ⚠️  PNG non creato: {e}")

        return npy_key

    # -----------------------------------------------------------------------
    # LOGICA INTERNA — water/empty check
    # -----------------------------------------------------------------------

    def is_mostly_water_or_empty(self, cube: np.ndarray) -> bool:
        """
        Determina se una tile è prevalentemente acqua o dati nulli.

        Criteri (tutti valutati sull'immagine estiva, indice 2):
          1. empty_ratio  > EMPTY_RATIO_THRESHOLD  (es. 0.5 → >50% pixel a zero)
          2. nir_mean     < WATER_NIR_THRESHOLD     (es. 600 → NIR estivo basso = mare)
          3. rgb_mean     < WATER_RGB_THRESHOLD     (es. 300 → visibile scuro = mare scuro)

        Il criterio 1 cattura dati mancanti/corrotti.
        I criteri 2+3 insieme catturano il mare (NIR molto basso E visibile scuro).
        Un campo irrigato ha NIR alto anche se il visibile è scuro.
        """
        summer = cube[2]  # (6, H, W) — stagione estiva

        # 1. Check dati vuoti
        non_zero = np.count_nonzero(cube)
        empty_ratio = 1.0 - (non_zero / cube.size)
        if empty_ratio > EMPTY_RATIO_THRESHOLD:
            logger.info(f"  🚫 Dati vuoti: {empty_ratio:.1%} pixel a zero")
            return True

        # 2. Check NIR estivo (banda 3 = NIR nell'ordine B,G,R,NIR,SWIR1,SWIR2)
        nir = summer[3]
        valid_nir = nir[nir > 0]
        nir_mean = float(np.mean(valid_nir)) if len(valid_nir) > 0 else 0.0

        # 3. Check RGB visibile estivo
        rgb_bands = summer[[0, 1, 2]]  # Blue, Green, Red
        valid_rgb = rgb_bands[rgb_bands > 0]
        rgb_mean = float(np.mean(valid_rgb)) if len(valid_rgb) > 0 else 0.0

        if nir_mean < WATER_NIR_THRESHOLD and rgb_mean < WATER_RGB_THRESHOLD:
            logger.info(
                f"  🌊 Acqua rilevata: NIR_mean={nir_mean:.0f} < {WATER_NIR_THRESHOLD}, "
                f"RGB_mean={rgb_mean:.0f} < {WATER_RGB_THRESHOLD}"
            )
            return True

        return False

    # -----------------------------------------------------------------------
    # HELPERS PRIVATI
    # -----------------------------------------------------------------------

    def _iter_metadata(self, prefix: str) -> Generator[tuple[str, dict], None, None]:
        """
        Generatore lazy: itera gli oggetti in un prefix e ritorna (key, metadata).
        Usa paginazione boto3 → nessun limite sul numero di tile.
        Legge i metadata con head_object (nessun download del contenuto).
        """
        paginator = self._client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self._bucket, Prefix=prefix)

        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".npy"):
                    continue
                try:
                    head = self._client.head_object(Bucket=self._bucket, Key=key)
                    yield key, head.get("Metadata", {})
                except ClientError as e:
                    logger.warning(f"head_object fallito per {key}: {e}")
                    continue

    @staticmethod
    def _parse_bbox(meta: dict) -> list[float] | None:
        """Deserializza il campo 'bbox' dai metadata MinIO."""
        raw = meta.get("bbox")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def _is_shape_valid(meta: dict) -> bool:
        """
        Verifica dai metadata che la shape sia (4, 6, H, W) senza scaricare il file.
        Se il campo 'shape' non è presente (tile vecchie), assume valida.
        """
        raw = meta.get("shape")
        if not raw:
            return True  # campo assente = tile legacy, non scartiamo
        try:
            shape = json.loads(raw)
            return len(shape) == 4 and shape[0] == 4 and shape[1] == 6
        except (json.JSONDecodeError, TypeError, IndexError):
            return True  # in caso di dubbio, tentiamo il download

    @staticmethod
    def _point_in_bbox(lat: float, lon: float, bbox: list[float]) -> bool:
        """[min_lon, min_lat, max_lon, max_lat]"""
        return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]

    @staticmethod
    def _bbox_contains(tile_bbox: list[float], request_bbox: list[float]) -> bool:
        """True se tile_bbox contiene interamente request_bbox."""
        return (
            tile_bbox[0] <= request_bbox[0]
            and tile_bbox[1] <= request_bbox[1]
            and tile_bbox[2] >= request_bbox[2]
            and tile_bbox[3] >= request_bbox[3]
        )

    @staticmethod
    def _make_rgb_preview(cube: np.ndarray) -> Image.Image:
        """
        Genera un'immagine RGB dalla stagione estiva (indice 2).
        Ordine bande Sentinel-2: [B, G, R, NIR, SWIR1, SWIR2] → indici [2,1,0]
        Normalizzazione percentile 2-98% per visualizzazione ottimale.
        """
        summer = cube[2]  # (6, H, W)
        rgb = np.stack([summer[2], summer[1], summer[0]], axis=-1)  # (H, W, 3) → R,G,B

        p2, p98 = np.percentile(rgb, (2, 98))
        rgb_norm = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
        rgb_uint8 = (rgb_norm * 255).astype(np.uint8)

        return Image.fromarray(rgb_uint8, mode="RGB")
    
    def _list_year_prefixes(self, descending: bool = True) -> list[str]:
        """
        Elenca i prefix year=XXXX/ presenti nel bucket.
        Non scarica nessun file: usa list_objects_v2 con delimiter='/'.
        """
        response = self._client.list_objects_v2(
            Bucket=self._bucket,
            Prefix=self.PREFIX_NPY,
            Delimiter="/",   # ritorna solo i "folder" virtuali
        )

        prefixes = []
        for cp in response.get("CommonPrefixes", []):
            p = cp["Prefix"]           # es. "raw_cubes/year=2023/"
            if "year=" in p:
                prefixes.append(p)

        # Ordina per anno (2024, 2023, 2022, ...)
        prefixes.sort(
            key=lambda p: int(p.split("year=")[1].rstrip("/")),
            reverse=descending,
        )

        logger.debug(f"Year prefixes trovati: {prefixes}")
        return prefixes