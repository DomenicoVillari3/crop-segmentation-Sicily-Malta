"""
main.py

=======
uvicorn main:app --reload --host 0.0.0.0 --port 8400

http://IP_MACCHINA:8400/api/v1/docs
=======

FastAPI app — livello API del sistema.

Endpoint:
  POST /api/v1/analysis/point        → avvia analisi da POI
  POST /api/v1/analysis/bbox         → avvia analisi da bbox
  GET  /api/v1/analysis/{task_id}/status  → polling risultato
  GET  /api/v1/analysis/{task_id}/image   → PNG overlay
  GET  /api/v1/analysis/{task_id}/ndvi    → NDVI timeseries
  GET  /api/v1/analysis/{task_id}/legend  → legenda con ettari
  GET  /api/v1/classes                    → legenda statica (senza ettari)
  GET  /api/v1/health                     → stato sistema
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from config import (
    MAX_CONCURRENT_INFERENCES,
    API_V1_PREFIX,
    CLASS_NAMES,
    CLASS_COLORS_HEX,
)
from data_resolver    import DataResolver
from inference_engine import InferenceEngine
from model_service    import ModelService
from minio_store      import MinioStore
from postprocess      import process as postprocess
from schemas import (
    PointRequest, BBoxRequest,
    AnalysisResponse, NDVIResponse, LegendResponse, LegendItem,
    HealthResponse, ClassStat, NDVIPoint,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Smart Food — Crop Segmentation API",
    description=(
        "Segmentazione semantica di colture agricole da immagini Sentinel-2. "
        "Modello: Prithvi EO v2 (IBM/NASA) fine-tuned su 9 classi mediterranee."
    ),
    version="1.0.0",
    docs_url=f"{API_V1_PREFIX}/docs",
    redoc_url=f"{API_V1_PREFIX}/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Stato globale — singleton e task store
# ---------------------------------------------------------------------------

# Semaforo: max N inferenze GPU parallele
_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)

# In-memory task store: {task_id: AnalysisResponse}
# In produzione sostituire con Redis
_tasks: dict[str, dict[str, Any]] = {}

# Pipeline — inizializzati all'avvio
_resolver: DataResolver | None  = None
_engine:   InferenceEngine | None = None


@app.on_event("startup")
async def startup():
    global _resolver, _engine
    logger.info("Avvio pipeline...")
    _resolver = DataResolver()
    _engine   = InferenceEngine()
    logger.info("✅ Pipeline pronta.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_task(task_id: str) -> dict:
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' non trovato.")
    return _tasks[task_id]


def _require_completed(task: dict) -> dict:
    if task["status"] != "completed":
        raise HTTPException(
            status_code=202,
            detail=f"Task in stato '{task['status']}'. Riprova quando status='completed'."
        )
    return task


async def _run_pipeline(task_id: str, lat: float, lon: float, year: int) -> None:
    """
    Esegue la pipeline completa in background.
    Aggiorna _tasks[task_id] ad ogni step per il polling.
    """
    async with _semaphore:
        try:
            _tasks[task_id]["status"]   = "running"
            _tasks[task_id]["progress"] = 10

            # 1. Risolvi dati
            resolved = await asyncio.get_event_loop().run_in_executor(
                None, lambda: _resolver.resolve_from_point(lat=lat, lon=lon, year=year)
            )

            if resolved is None:
                _tasks[task_id]["status"] = "failed"
                _tasks[task_id]["error"]  = (
                    "Nessun dato disponibile. Verifica coordinate e cloud cover."
                )
                return

            _tasks[task_id]["progress"] = 35
            _tasks[task_id]["source"]   = resolved.source
            _tasks[task_id]["year_used"] = resolved.year
            _tasks[task_id]["bbox"]      = resolved.bbox

            # 2. Inferenza
            mask = await asyncio.get_event_loop().run_in_executor(
                None, lambda: _engine.run(resolved.cube)
            )
            _tasks[task_id]["progress"] = 80

            # 3. Post-processing
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: postprocess(mask, resolved.cube)
            )
            _tasks[task_id]["progress"] = 95

            # 4. Salva risultati
            _tasks[task_id].update({
                "status":      "completed",
                "progress":    100,
                "overlay_png": result.overlay_png,
                "class_stats": result.class_stats,
                "ndvi_series": result.ndvi_series,
                "total_ha":    result.total_ha,
            })
            logger.info(f"✅ Task {task_id} completato.")

        except Exception as e:
            logger.error(f"❌ Task {task_id} fallito: {e}")
            _tasks[task_id]["status"] = "failed"
            _tasks[task_id]["error"]  = str(e)


# ---------------------------------------------------------------------------
# Endpoint — avvio analisi
# ---------------------------------------------------------------------------

@app.post(
    f"{API_V1_PREFIX}/analysis/point",
    response_model=AnalysisResponse,
    status_code=202,
    summary="Avvia analisi da punto di interesse (POI)",
    tags=["Analysis"],
)
async def start_analysis_point(req: PointRequest):
    """
    Avvia l'analisi per un punto geografico (lat/lon).
    Ritorna subito un `task_id` — usa `/status` per il polling.
    """
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"status": "pending", "progress": 0}

    asyncio.create_task(
        _run_pipeline(task_id, req.lat, req.lon, req.year)
    )

    return AnalysisResponse(task_id=task_id, status="pending", progress=0)


@app.post(
    f"{API_V1_PREFIX}/analysis/bbox",
    response_model=AnalysisResponse,
    status_code=202,
    summary="Avvia analisi da bounding box",
    tags=["Analysis"],
)
async def start_analysis_bbox(req: BBoxRequest):
    """
    Avvia l'analisi per un'area definita da bounding box.
    """
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"status": "pending", "progress": 0}

    # Per bbox usiamo resolve_from_bbox
    async def _run_bbox():
        async with _semaphore:
            try:
                _tasks[task_id]["status"]   = "running"
                _tasks[task_id]["progress"] = 10

                resolved = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: _resolver.resolve_from_bbox(
                        bbox=req.to_list(), year=req.year
                    )
                )
                if resolved is None:
                    _tasks[task_id]["status"] = "failed"
                    _tasks[task_id]["error"]  = "Nessun dato disponibile."
                    return

                _tasks[task_id]["progress"]  = 35
                _tasks[task_id]["source"]    = resolved.source
                _tasks[task_id]["year_used"] = resolved.year
                _tasks[task_id]["bbox"]      = resolved.bbox

                mask = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: _engine.run(resolved.cube)
                )
                _tasks[task_id]["progress"] = 80

                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: postprocess(mask, resolved.cube)
                )

                _tasks[task_id].update({
                    "status":      "completed",
                    "progress":    100,
                    "overlay_png": result.overlay_png,
                    "class_stats": result.class_stats,
                    "ndvi_series": result.ndvi_series,
                    "total_ha":    result.total_ha,
                })
            except Exception as e:
                _tasks[task_id]["status"] = "failed"
                _tasks[task_id]["error"]  = str(e)

    asyncio.create_task(_run_bbox())
    return AnalysisResponse(task_id=task_id, status="pending", progress=0)


# ---------------------------------------------------------------------------
# Endpoint — polling e risultati
# ---------------------------------------------------------------------------

@app.get(
    f"{API_V1_PREFIX}/analysis/{{task_id}}/status",
    response_model=AnalysisResponse,
    summary="Polling stato analisi",
    tags=["Analysis"],
)
async def get_status(task_id: str):
    """
    Ritorna lo stato corrente del task.

    - `pending`   → in coda
    - `running`   → in esecuzione (progress 0-99)
    - `completed` → risultati disponibili
    - `failed`    → errore (vedi campo `error`)
    """
    task = _get_task(task_id)
    stats = [ClassStat(**vars(s)) for s in task.get("class_stats") or []]
    ndvi  = [NDVIPoint(**vars(n)) for n in task.get("ndvi_series") or []]

    return AnalysisResponse(
        task_id     = task_id,
        status      = task["status"],
        progress    = task.get("progress", 0),
        source      = task.get("source"),
        year_used   = task.get("year_used"),
        bbox        = task.get("bbox"),
        class_stats = stats or None,
        ndvi_series = ndvi or None,
        total_ha    = task.get("total_ha"),
        error       = task.get("error"),
    )


@app.get(
    f"{API_V1_PREFIX}/analysis/{{task_id}}/image",
    response_class=Response,
    summary="Scarica overlay PNG (RGB satellite + segmentazione)",
    tags=["Analysis"],
)
async def get_image(task_id: str):
    """
    Ritorna il PNG overlay direttamente come bytes.
    Il PNG include canale alpha: i bordi dell'orbita Sentinel-2
    sono trasparenti invece che neri.
    """
    task = _get_task(task_id)
    _require_completed(task)

    return Response(
        content=task["overlay_png"],
        media_type="image/png",
        headers={"Content-Disposition": f"inline; filename={task_id}_overlay.png"},
    )


@app.get(
    f"{API_V1_PREFIX}/analysis/{{task_id}}/ndvi",
    response_model=NDVIResponse,
    summary="NDVI timeseries per stagione",
    tags=["Analysis"],
)
async def get_ndvi(task_id: str):
    """
    Ritorna l'NDVI medio per le 4 stagioni (winter/spring/summer/autumn).

    Calcolato solo sui pixel classificati come vegetazione (classe > 0),
    escludendo acqua e sfondo per non distorcere la media.

    Utile per visualizzare la curva fenologica dell'area analizzata.
    """
    task = _get_task(task_id)
    _require_completed(task)

    ndvi = [NDVIPoint(**vars(n)) for n in task["ndvi_series"]]

    return NDVIResponse(
        task_id     = task_id,
        year_used   = task["year_used"],
        bbox        = task["bbox"],
        ndvi_series = ndvi,
    )


@app.get(
    f"{API_V1_PREFIX}/analysis/{{task_id}}/legend",
    response_model=LegendResponse,
    summary="Legenda con ettari dell'analisi corrente",
    tags=["Analysis"],
)
async def get_legend_with_hectares(task_id: str):
    """
    Ritorna la legenda arricchita con ettari e percentuali
    calcolati sull'analisi specifica del task.

    A differenza di `/classes` (legenda statica), questo endpoint
    include i dati quantitativi reali dell'area analizzata.
    """
    task = _get_task(task_id)
    _require_completed(task)

    stats_map = {s.class_id: s for s in task["class_stats"]}

    items = []
    for cls_id in range(1, len(CLASS_NAMES)):  # skip sfondo
        stat = stats_map.get(cls_id)
        items.append(LegendItem(
            class_id   = cls_id,
            name       = CLASS_NAMES[cls_id],
            color_hex  = CLASS_COLORS_HEX[cls_id],
            hectares   = stat.hectares   if stat else 0.0,
            percentage = stat.percentage if stat else 0.0,
        ))

    # Ordina per ettari decrescenti — le classi dominanti in cima
    items.sort(key=lambda x: x.hectares or 0, reverse=True)

    return LegendResponse(
        items    = items,
        total_ha = task.get("total_ha"),
    )


# ---------------------------------------------------------------------------
# Endpoint — statici
# ---------------------------------------------------------------------------

@app.get(
    f"{API_V1_PREFIX}/classes",
    response_model=LegendResponse,
    summary="Legenda statica classi (senza dati analisi)",
    tags=["Info"],
)
async def get_classes():
    """
    Catalogo fisso delle 9 classi con colori.
    Non richiede un task_id — utile per inizializzare la UI
    prima di avviare un'analisi.
    """
    items = [
        LegendItem(
            class_id  = i,
            name      = CLASS_NAMES[i],
            color_hex = CLASS_COLORS_HEX[i],
        )
        for i in range(1, len(CLASS_NAMES))
    ]
    return LegendResponse(items=items)


@app.get(
    f"{API_V1_PREFIX}/health",
    response_model=HealthResponse,
    summary="Stato del sistema",
    tags=["Info"],
)
async def health():
    """
    Verifica che modello, GPU e MinIO siano operativi.
    Esegue una forward pass dummy per confermare che il modello risponde.
    """
    svc   = ModelService()
    model_health = svc.health()

    try:
        MinioStore()
        minio_status = "ok"
    except Exception:
        minio_status = "error"

    overall = "ok" if model_health["status"] == "ok" and minio_status == "ok" else "degraded"

    return HealthResponse(
        status       = overall,
        device       = model_health.get("device", "unknown"),
        vram_used_mb = model_health.get("vram_used_mb", 0.0),
        minio        = minio_status,
        model        = model_health["status"],
        detail       = model_health.get("detail"),
    )