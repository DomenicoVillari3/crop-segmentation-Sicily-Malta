"""
data_resolver.py
================
Orchestratore della sorgente dati. Unica responsabilità: data una richiesta
(POI o bbox + anno), ritornare un cubo numpy pronto per l'inferenza.

Flusso decisionale:
  1. MinIO hit  → download + smart_crop centrato sul POI
  2. MinIO miss → STAC download → salvataggio cache → ritorna cubo

Non sa nulla del modello o dell'API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal
from datetime import datetime, timezone

import numpy as np

from config import (
    DEFAULT_YEAR,
    POI_BBOX_SIZE_DEG,      # 0.05° ≈ 5.55km — bbox per download on-the-fly
    DOWNLOAD_MARGIN_PX,     # pixel extra su ogni lato durante download STAC
    SENTINEL_RESOLUTION,    # 10 m/pixel
)
from minio_store import MinioStore
from stac_downloader import STACDownloader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tipo di ritorno
# ---------------------------------------------------------------------------

@dataclass
class ResolvedCube:
    cube:   np.ndarray          # (4, 6, H, W) float32, pronto per inferenza
    bbox:   list[float]         # bbox geografico del cubo ritornato
    year:   int                 # anno effettivo dei dati
    source: Literal["minio", "stac"]


# ---------------------------------------------------------------------------
# DataResolver
# ---------------------------------------------------------------------------

class DataResolver:
    """
    Decide da dove prendere i dati e li prepara per l'inferenza.

    Usage:
        resolver = DataResolver()

        # Da punto di interesse
        result = resolver.resolve_from_point(lat=37.5, lon=14.2, year=2023)

        # Da bbox esplicito
        result = resolver.resolve_from_bbox(bbox=[14.0,37.0,15.0,38.0], year=2023)

        if result:
            cube, bbox, year, source = result.cube, result.bbox, result.year, result.source
    """

    def __init__(self) -> None:
        self._minio = MinioStore()
        self._stac  = STACDownloader()

    # -----------------------------------------------------------------------
    # API pubblica
    # -----------------------------------------------------------------------

    def resolve_from_point(
        self,
        lat: float,
        lon: float,
        year: int | None = None,
        crop_size_px: int = 800,
    ) -> ResolvedCube | None:
        """
        Risolve un punto di interesse (POI) in un cubo pronto per l'inferenza.

        Se il dato è su MinIO: scarica la tile grande e ritaglia crop_size_px
        centrato sul POI (smart_crop), così il punto di interesse è sempre
        al centro dell'area analizzata.

        Se non è su MinIO: scarica da STAC un'area di POI_BBOX_SIZE_DEG gradi,
        senza crop (l'area è già dimensionata correttamente), e salva in cache.

        Args:
            lat, lon:     coordinate del punto
            year:         anno dati (None → più recente disponibile)
            crop_size_px: dimensione del crop in pixel (default 800 ≈ 8km)
        """
        _year = year or DEFAULT_YEAR
        logger.info(f" resolve_from_point ({lat}, {lon}) anno={_year}")

        # --- Tentativo MinIO ---
        key, tile_bbox, found_year = self._minio.find_tile(lat, lon, year=_year)

        if key is not None:
            cube_full = self._minio.download_cube(key)
            cube, crop_bbox = self._smart_crop(cube_full, tile_bbox, lat, lon, crop_size_px)
            logger.info(f"✅ Sorgente: MinIO (anno {found_year})")
            return ResolvedCube(cube=cube, bbox=crop_bbox, year=found_year, source="minio")

        # --- Fallback STAC ---
        logger.info("🔄 MinIO miss → fallback STAC")
        download_bbox = self._get_bbox_from_point(lat, lon, size=POI_BBOX_SIZE_DEG)

        # Margine extra: scarica un'area leggermente più grande per evitare
        # che i chip ai bordi siano a corto di contesto dopo il padding
        padded_bbox = self._add_margin(download_bbox, DOWNLOAD_MARGIN_PX)

        cube = self._stac.download(padded_bbox, year=_year)
        if cube is None:
            logger.error("❌ Download STAC fallito. Nessun dato disponibile.")
            return None

        # Rimuove il margine extra prima di ritornare
        cube, final_bbox = self._trim_margin(cube, padded_bbox, download_bbox)

        # Salva in cache per query future (best-effort)
        self._minio.upload_cube(cube, final_bbox, year=_year, task_id="onthefly")

        logger.info(f"✅ Sorgente: STAC (anno {_year})")
        return ResolvedCube(cube=cube, bbox=final_bbox, year=_year, source="stac")

    def resolve_from_bbox(
        self,
        bbox: list[float],
        year: int | None = None,
    ) -> ResolvedCube | None:
        """
        Risolve un bbox esplicito. Non esegue smart_crop.
        Se la tile MinIO contiene interamente il bbox, la usa;
        altrimenti scarica da STAC esattamente il bbox richiesto.
        """
        _year = year or DEFAULT_YEAR
        logger.info(f"📐 resolve_from_bbox {bbox} anno={_year}")

        key, tile_bbox, found_year = self._minio.find_tile_by_bbox(bbox)

        if key is not None:
            cube_full = self._minio.download_cube(key)
            # Ritaglia esattamente il bbox richiesto dalla tile più grande
            cube, crop_bbox = self._crop_to_bbox(cube_full, tile_bbox, bbox)
            logger.info(f"✅ Sorgente: MinIO (anno {found_year})")
            return ResolvedCube(cube=cube, bbox=crop_bbox, year=found_year, source="minio")

        logger.info("🔄 MinIO miss → fallback STAC")
        cube = self._stac.download(bbox, year=_year)
        if cube is None:
            return None

        self._minio.upload_cube(cube, bbox, year=_year, task_id="bbox_request")
        return ResolvedCube(cube=cube, bbox=bbox, year=_year, source="stac")

    # -----------------------------------------------------------------------
    # Logica spaziale
    # -----------------------------------------------------------------------

    @staticmethod
    def _get_bbox_from_point(lat: float, lon: float, size: float) -> list[float]:
        """BBox quadrato centrato sul punto. size in gradi."""
        h = size / 2
        return [lon - h, lat - h, lon + h, lat + h]

    @staticmethod
    def _add_margin(bbox: list[float], margin_px: int) -> list[float]:
        """
        Espande il bbox di margin_px pixel su ogni lato.
        Converte pixel → gradi usando la risoluzione Sentinel-2 (10m/px).

        Utile per evitare artefatti ai bordi dello sliding window:
        i chip marginali hanno meno contesto, il margine viene poi
        rimosso da _trim_margin prima dell'inferenza.
        """
        # 10m per pixel → 1 pixel ≈ 0.0000898° a latitudine media
        DEG_PER_PX = SENTINEL_RESOLUTION / 111_320  # ~0.0000898°
        delta = margin_px * DEG_PER_PX
        return [
            bbox[0] - delta,  # min_lon
            bbox[1] - delta,  # min_lat
            bbox[2] + delta,  # max_lon
            bbox[3] + delta,  # max_lat
        ]

    @staticmethod
    def _trim_margin(
        cube: np.ndarray,
        padded_bbox: list[float],
        target_bbox: list[float],
    ) -> tuple[np.ndarray, list[float]]:
        """
        Ritaglia il cubo dal bbox paddato al bbox originale.
        Inverso di _add_margin: rimuove i pixel extra prima dell'inferenza.
        """
        _, _, H, W = cube.shape

        # Calcola le proporzioni del bbox interno rispetto al paddato
        lon_range = padded_bbox[2] - padded_bbox[0]
        lat_range = padded_bbox[3] - padded_bbox[1]

        x_start = int((target_bbox[0] - padded_bbox[0]) / lon_range * W)
        x_end   = int((target_bbox[2] - padded_bbox[0]) / lon_range * W)
        y_start = int((padded_bbox[3] - target_bbox[3]) / lat_range * H)  # Y invertita
        y_end   = int((padded_bbox[3] - target_bbox[1]) / lat_range * H)

        # Clamp per sicurezza
        x_start, x_end = max(0, x_start), min(W, x_end)
        y_start, y_end = max(0, y_start), min(H, y_end)

        trimmed = cube[:, :, y_start:y_end, x_start:x_end]
        logger.debug(f"_trim_margin: {cube.shape} → {trimmed.shape}")
        return trimmed, target_bbox

    @staticmethod
    def _smart_crop(
        cube: np.ndarray,
        tile_bbox: list[float],
        target_lat: float,
        target_lon: float,
        crop_size_px: int,
    ) -> tuple[np.ndarray, list[float]]:
        """
        Ritaglia crop_size_px × crop_size_px centrati sul POI (lat, lon).

        La tile MinIO è tipicamente grande (es. 1000×1000 px per 0.1°×0.1°).
        Questo crop centra l'area di analisi sul punto richiesto, riducendo
        il lavoro dell'inference_engine senza perdere contesto.

        Gestisce i bordi: se il POI è vicino al margine della tile,
        il crop viene spostato per rimanere interno (no padding artificiale).
        """
        _, _, H, W = cube.shape
        min_lon, min_lat, max_lon, max_lat = tile_bbox

        # Coordinate geografiche → pixel
        px_x = int((target_lon - min_lon) / (max_lon - min_lon) * W)
        px_y = int((max_lat - target_lat) / (max_lat - min_lat) * H)  # Y invertita

        half = crop_size_px // 2

        # Calcola window con clamp ai bordi
        x_start = max(0, px_x - half)
        x_end   = min(W, x_start + crop_size_px)
        x_start = max(0, x_end - crop_size_px)  # aggiusta se x_end è stato clampato

        y_start = max(0, px_y - half)
        y_end   = min(H, y_start + crop_size_px)
        y_start = max(0, y_end - crop_size_px)

        cropped = cube[:, :, y_start:y_end, x_start:x_end]

        # Calcola il nuovo bbox geografico del crop
        deg_per_px_x = (max_lon - min_lon) / W
        deg_per_px_y = (max_lat - min_lat) / H
        crop_bbox = [
            min_lon + x_start * deg_per_px_x,
            max_lat - y_end   * deg_per_px_y,
            min_lon + x_end   * deg_per_px_x,
            max_lat - y_start * deg_per_px_y,
        ]

        logger.info(f"  smart_crop: {cube.shape} → {cropped.shape} | centro ({target_lat:.4f}, {target_lon:.4f})")
        return cropped, crop_bbox

    @staticmethod
    def _crop_to_bbox(
        cube: np.ndarray,
        tile_bbox: list[float],
        target_bbox: list[float],
    ) -> tuple[np.ndarray, list[float]]:
        """
        Ritaglia la tile MinIO per far corrispondere esattamente target_bbox.
        Usato da resolve_from_bbox quando la tile è più grande del richiesto.
        Stessa logica di _smart_crop ma con bbox esplicito invece di POI.
        """
        _, _, H, W = cube.shape
        min_lon, min_lat, max_lon, max_lat = tile_bbox

        x_start = int((target_bbox[0] - min_lon) / (max_lon - min_lon) * W)
        x_end   = int((target_bbox[2] - min_lon) / (max_lon - min_lon) * W)
        y_start = int((max_lat - target_bbox[3]) / (max_lat - min_lat) * H)
        y_end   = int((max_lat - target_bbox[1]) / (max_lat - min_lat) * H)

        x_start, x_end = max(0, x_start), min(W, x_end)
        y_start, y_end = max(0, y_start), min(H, y_end)

        return cube[:, :, y_start:y_end, x_start:x_end], target_bbox