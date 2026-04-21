"""
postprocess.py
==============
Funzioni pure: nessuno stato, nessuna dipendenza da modello o MinIO.
Input:  maschera (H,W) uint8  +  cubo (4,6,H,W) float32
Output: bytes PNG overlay, dict statistiche, dict NDVI timeseries

L'immagine RGB satellite NON viene generata qui:
è già salvata su MinIO da minio_store.upload_cube() come PNG preview.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from config import (
    CLASS_NAMES,
    CLASS_COLORS_HEX,
    CLASS_COLORS_RGB,
    PIXEL_AREA_M2,
    M2_TO_HECTARES,
    OVERLAY_OPACITY,        # 0.55 — opacità della maschera sull'RGB
    ORDERED_SEASONS,
)


# ---------------------------------------------------------------------------
# Dataclasses di output — usate da schemas.py
# ---------------------------------------------------------------------------

@dataclass
class ClassStat:
    class_id:   int
    name:       str
    color_hex:  str
    hectares:   float
    percentage: float      # % sul totale NON-sfondo


@dataclass
class NDVIPoint:
    season:     str        # "winter" | "spring" | "summer" | "autumn"
    mean:       float
    std:        float
    min:        float
    max:        float


@dataclass
class PostprocessResult:
    overlay_png:   bytes              # immagine overlay serializzata
    class_stats:   list[ClassStat]
    ndvi_series:   list[NDVIPoint]    # 4 punti, uno per stagione
    total_ha:      float              # ettari totali classificati (escluso sfondo)


# ---------------------------------------------------------------------------
# API pubblica — 1 funzione principale
# ---------------------------------------------------------------------------

def process(
    mask: np.ndarray,
    cube: np.ndarray,
    crop_black: bool = True,
) -> PostprocessResult:

    if crop_black:
        H, W    = mask.shape
        rgb_raw = cube[2, [2,1,0], :H, :W].transpose(1,2,0)
        rgb_u8  = _percentile_stretch(rgb_raw)
        _, (y0, y1, x0, x1) = _crop_black_borders(rgb_u8)
        mask_for_stats = mask[y0:y1, x0:x1]
        cube_for_stats = cube[:, :, y0:y1, x0:x1]  # ← stesso crop sul cubo
    else:
        mask_for_stats = mask
        cube_for_stats = cube

    overlay_bytes = render_overlay(mask, cube)                          # overlay usa tutto
    stats         = compute_class_stats(mask_for_stats)                 # stats sul crop
    ndvi          = compute_ndvi_timeseries(mask_for_stats, cube_for_stats)  # ← cubo croppato

    total_ha = sum(s.hectares for s in stats)

    return PostprocessResult(
        overlay_png=overlay_bytes,
        class_stats=stats,
        ndvi_series=ndvi,
        total_ha=total_ha,
    )


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def render_overlay(mask, cube, opacity=OVERLAY_OPACITY) -> bytes:
    H, W = mask.shape

    rgb_raw  = cube[2, [2, 1, 0], :H, :W].transpose(1, 2, 0)
    rgb_norm = _percentile_stretch(rgb_raw)

    # Maschera binaria pixel validi (non neri)
    valid_mask = ~np.all(rgb_norm <= 5, axis=2)  # (H, W) bool

    # Compositing overlay
    mask_rgba  = _colorize_mask_rgba(mask, opacity)
    alpha      = mask_rgba[..., 3:4]
    rgb_f      = rgb_norm.astype(np.float32) / 255.0
    color_f    = mask_rgba[..., :3]
    composited = (alpha * color_f + (1.0 - alpha) * rgb_f)
    composited = (np.clip(composited, 0, 1) * 255).astype(np.uint8)

    # ── Applica trasparenza ai pixel neri ──────────────────────
    # Converte in RGBA e mette alpha=0 sui pixel neri
    # L'utente vede sfondo trasparente invece di nero
    rgba_out        = np.zeros((*composited.shape[:2], 4), dtype=np.uint8)
    rgba_out[..., :3] = composited
    rgba_out[..., 3]  = (valid_mask * 255).astype(np.uint8)  # alpha=0 dove nero
    # ───────────────────────────────────────────────────────────

    # Crop rettangolare per ridurre dimensione file
    _, (y0, y1, x0, x1) = _crop_black_borders(rgb_norm)
    rgba_cropped = rgba_out[y0:y1, x0:x1]

    # Salva come PNG con trasparenza (RGBA)
    buf = io.BytesIO()
    Image.fromarray(rgba_cropped, mode="RGBA").save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Statistiche classi
# ---------------------------------------------------------------------------

def compute_class_stats(mask: np.ndarray) -> list[ClassStat]:
    """
    Calcola ettari e percentuale per ogni classe presente nella maschera.
    La classe 0 (Sfondo) viene esclusa dal totale percentuale.
    """
    unique, counts = np.unique(mask, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))

    # Totale pixel non-sfondo per il calcolo percentuale
    total_px = sum(c for cls_id, c in count_map.items() if cls_id != 0)
    if total_px == 0:
        return []

    stats = []
    for cls_id in range(1, len(CLASS_NAMES)):  # skip sfondo
        px = count_map.get(cls_id, 0)
        if px == 0:
            continue

        ha  = (px * PIXEL_AREA_M2) / M2_TO_HECTARES
        pct = (px / total_px) * 100.0

        stats.append(ClassStat(
            class_id   = cls_id,
            name       = CLASS_NAMES[cls_id],
            color_hex  = CLASS_COLORS_HEX[cls_id],
            hectares   = round(ha,  2),
            percentage = round(pct, 2),
        ))

    # Ordina per ettari decrescenti
    return sorted(stats, key=lambda s: s.hectares, reverse=True)


# ---------------------------------------------------------------------------
# NDVI timeseries
# ---------------------------------------------------------------------------

def compute_ndvi_timeseries(
    mask: np.ndarray,
    cube: np.ndarray,
) -> list[NDVIPoint]:
    """
    Calcola l'NDVI medio per stagione, escludendo sfondo e acqua.

    NDVI = (NIR - Red) / (NIR + Red + ε)
    Bande: Red=2, NIR=3  (ordine B,G,R,NIR,SWIR1,SWIR2)

    Calcola solo sui pixel classificati come vegetazione (classe > 0)
    per evitare che suolo nudo e acqua distorcano la media.

    Returns:
        Lista di 4 NDVIPoint (winter, spring, summer, autumn)
    """
    H, W     = mask.shape
    veg_mask = mask > 0                        # pixel con classe reale

    series = []
    for t, season in enumerate(ORDERED_SEASONS):
        red = cube[t, 2, :H, :W].astype(np.float32)  # banda R
        nir = cube[t, 3, :H, :W].astype(np.float32)  # banda NIR

        ndvi_full = (nir - red) / (nir + red + 1e-6)  # (H, W)

        # Applica maschera vegetazione
        ndvi_veg = ndvi_full[veg_mask]

        if len(ndvi_veg) == 0:
            series.append(NDVIPoint(season, 0.0, 0.0, 0.0, 0.0))
            continue

        series.append(NDVIPoint(
            season = season,
            mean   = round(float(ndvi_veg.mean()), 4),
            std    = round(float(ndvi_veg.std()),  4),
            min    = round(float(ndvi_veg.min()),  4),
            max    = round(float(ndvi_veg.max()),  4),
        ))

    return series


# ---------------------------------------------------------------------------
# Helpers privati
# ---------------------------------------------------------------------------

def _percentile_stretch(
    img: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    """Normalizzazione percentile → uint8. Rimuove outlier radiometrici."""
    lo, hi = np.percentile(img, (p_low, p_high))
    stretched = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
    return (stretched * 255).astype(np.uint8)


def _colorize_mask_rgba(
    mask: np.ndarray,
    opacity: float,
) -> np.ndarray:
    """
    Converte maschera uint8 → RGBA float32.
    Classe 0 (Sfondo) → alpha=0 (completamente trasparente).
    """
    H, W  = mask.shape
    rgba  = np.zeros((H, W, 4), dtype=np.float32)

    for cls_id, color_rgb in enumerate(CLASS_COLORS_RGB):
        if cls_id == 0:
            continue                          # sfondo trasparente
        px = mask == cls_id
        rgba[px, 0] = color_rgb[0] / 255.0
        rgba[px, 1] = color_rgb[1] / 255.0
        rgba[px, 2] = color_rgb[2] / 255.0
        rgba[px, 3] = opacity

    return rgba


# postprocess.py — aggiorna _to_png_bytes per supportare RGBA
def _to_png_bytes(img_array: np.ndarray) -> bytes:
    mode = "RGBA" if img_array.shape[2] == 4 else "RGB"
    buf  = io.BytesIO()
    Image.fromarray(img_array, mode=mode).save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.read()

def _crop_black_borders(img_array: np.ndarray, threshold: int = 5) -> tuple[np.ndarray, tuple]:
    """
    Rimuove le fasce nere usando una maschera binaria per riga/colonna.
    Una riga/colonna è considerata "nera" se il 95% dei suoi pixel
    è sotto la soglia — gestisce bordi obliqui come nelle orbite Sentinel-2.
    """
    # Considera nero se TUTTI e 3 i canali sono sotto soglia
    black_pixel_mask = np.all(img_array <= threshold, axis=2)  # (H, W) bool, True=nero
    valid_pixel_mask = ~black_pixel_mask                        # True=valido

    # Una riga è valida se almeno il 5% dei pixel non è nero
    # Questo gestisce i bordi obliqui dell'orbita Sentinel-2
    row_valid = valid_pixel_mask.mean(axis=1) > 0.05  # (H,)
    col_valid = valid_pixel_mask.mean(axis=0) > 0.05  # (W,)

    if not row_valid.any() or not col_valid.any():
        h, w = img_array.shape[:2]
        return img_array, (0, h, 0, w)

    y_start, y_end = np.where(row_valid)[0][[0, -1]]
    x_start, x_end = np.where(col_valid)[0][[0, -1]]

    # +1 su y_end/x_end perché lo slicing è esclusivo
    cropped = img_array[y_start:y_end+1, x_start:x_end+1]
    return cropped, (y_start, y_end+1, x_start, x_end+1)