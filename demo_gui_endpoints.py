"""
demo_gui_api.py
===============
Interfaccia Gradio che chiama le API FastAPI.
Prerequisiti: uvicorn main:app --host 0.0.0.0 --port 8400
"""

import gradio as gr
import httpx
import time
import io
import logging
from PIL import Image

from config import CLASS_NAMES, CLASS_COLORS_HEX, SUPPORTED_YEARS, DEFAULT_YEAR, API_V1_PREFIX

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
API_BASE        = f"http://localhost:8400{API_V1_PREFIX}"
POLL_INTERVAL_S = 3
TIMEOUT_S       = 300

POI_PRESETS = {
    "🍊 Agrumeti – Catania":         (37.380, 14.910),
    "🌾 Grano – Enna":               (37.567, 14.279),
    "🌾 Seminativo – Caltanissetta": (37.490, 14.060),
    "🍊 Agrumeti – Lentini":         (37.285, 14.990),
    "🏖️ Palermo Costa":              (38.115, 13.361),
    "🇲🇹 Terreni agricoli – Mosta":  (35.910, 14.425),
}

BBOX_PRESETS = {
    "🍊 Agrumeti – Catania (11km²)": (14.857, 37.327, 14.963, 37.433),
    "🌾 Grano – Enna (10km²)":       (14.229, 37.517, 14.329, 37.617),
    "🇲🇹 Mosta – Malta (8km²)":      (14.395, 35.880, 14.455, 35.940),
}

# ---------------------------------------------------------------------------
# Client API
# ---------------------------------------------------------------------------

def _start_poi(lat: float, lon: float, year: int) -> str | None:
    try:
        r = httpx.post(
            f"{API_BASE}/point",
            json={"lat": lat, "lon": lon, "year": year},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["task_id"]
    except Exception as e:
        logger.error(f"Errore avvio POI: {e}")
        return None


def _start_bbox(min_lon, min_lat, max_lon, max_lat, year) -> str | None:
    try:
        r = httpx.post(
            f"{API_BASE}/bbox",
            json={
                "min_lon": min_lon, "min_lat": min_lat,
                "max_lon": max_lon, "max_lat": max_lat,
                "year": year,
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["task_id"]
    except Exception as e:
        logger.error(f"Errore avvio BBox: {e}")
        return None


def _poll_until_done(task_id: str) -> dict:
    deadline = time.time() + TIMEOUT_S
    while time.time() < deadline:
        try:
            r = httpx.get(f"{API_BASE}/{task_id}/status", timeout=10)
            r.raise_for_status()
            data = r.json()
            status = data["status"]
            logger.info(f"  [{task_id[:8]}] {status} {data.get('progress', 0)}%")

            if status == "completed":
                return data
            if status == "failed":
                raise RuntimeError(data.get("error", "Errore sconosciuto"))

        except RuntimeError:
            raise
        except Exception as e:
            logger.warning(f"Polling error: {e}")

        time.sleep(POLL_INTERVAL_S)

    raise TimeoutError(f"Timeout dopo {TIMEOUT_S}s")


def _get_overlay_image(task_id: str) -> Image.Image | None:
    try:
        r = httpx.get(f"{API_BASE}/{task_id}/image", timeout=30)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content))
    except Exception as e:
        logger.error(f"Errore download immagine: {e}")
        return None


def _check_health() -> dict:
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=5)
        return r.json()
    except Exception:
        return {"status": "error", "detail": "API non raggiungibile"}


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _run_pipeline(task_id: str, year: int, extra_info: str = "") -> tuple:
    """Polling + immagine + formattazione output. Condiviso tra POI e BBox."""
    data        = _poll_until_done(task_id)
    overlay_img = _get_overlay_image(task_id)

    source_info = (
        f"**Sorgente:** `{data.get('source', 'N/A')}` | "
        f"**Anno:** `{data.get('year_used', year)}` | "
        f"{extra_info}"
        f"**Task:** `{task_id[:8]}...`"
    )

    return (
        overlay_img,
        _format_stats(data, source_info),
        _format_ndvi(data),
        _format_diagnostics(data, task_id),
    )


def _error_tuple(msg: str) -> tuple:
    return None, msg, msg, msg


# ---------------------------------------------------------------------------
# Funzioni Gradio
# ---------------------------------------------------------------------------

def run_analysis_poi(preset_name, lat, lon, year):
    try:
        _year = int(year)
        logger.info(f"▶ POI ({lat:.4f}, {lon:.4f}) anno={_year}")

        task_id = _start_poi(lat, lon, _year)
        if task_id is None:
            return _error_tuple("❌ Impossibile avviare l'analisi. Verifica che l'API sia attiva.")

        logger.info(f"Task avviato: {task_id}")
        return _run_pipeline(task_id, _year)

    except TimeoutError:
        return _error_tuple(f"❌ Timeout dopo {TIMEOUT_S}s. Riprova su un'area più piccola.")
    except Exception as e:
        logger.error(str(e))
        return _error_tuple(f"❌ Errore: {str(e)}")


def run_analysis_bbox(min_lon, min_lat, max_lon, max_lat, year):
    try:
        _year = int(year)

        # Validazione client-side
        if max_lon <= min_lon or max_lat <= min_lat:
            return _error_tuple("❌ BBox non valido: max deve essere > min.")

        area_km2 = abs((max_lon - min_lon) * 111) * abs((max_lat - min_lat) * 111)
        if area_km2 > 2500:
            return _error_tuple(
                f"❌ Area troppo grande: {area_km2:.0f} km². Massimo consigliato: 2500 km²."
            )

        logger.info(f"▶ BBox [{min_lon},{min_lat},{max_lon},{max_lat}] anno={_year} ({area_km2:.1f} km²)")

        task_id = _start_bbox(min_lon, min_lat, max_lon, max_lat, _year)
        if task_id is None:
            return _error_tuple("❌ Impossibile avviare l'analisi. Verifica che l'API sia attiva.")

        logger.info(f"Task BBox avviato: {task_id}")
        return _run_pipeline(task_id, _year, extra_info=f"**Area:** `{area_km2:.1f} km²` | ")

    except TimeoutError:
        return _error_tuple(f"❌ Timeout dopo {TIMEOUT_S}s. Riduci l'area.")
    except Exception as e:
        logger.error(str(e))
        return _error_tuple(f"❌ Errore: {str(e)}")


def load_preset(preset_name):
    if preset_name and preset_name in POI_PRESETS:
        return POI_PRESETS[preset_name]
    return 37.5, 14.2


def load_bbox_preset(preset_name):
    if preset_name and preset_name in BBOX_PRESETS:
        return BBOX_PRESETS[preset_name]
    return 14.85, 37.32, 14.97, 37.44


def check_api_status():
    h = _check_health()
    if h.get("status") == "ok":
        return (
            f"✅ **API Online** | "
            f"Device: `{h.get('device', '?')}` | "
            f"VRAM: `{h.get('vram_used_mb', 0):.0f} MB` | "
            f"MinIO: `{h.get('minio', '?')}`"
        )
    return f"❌ **API Offline** — {h.get('detail', 'Controlla uvicorn')}"


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _format_stats(data: dict, source_info: str) -> str:
    stats = data.get("class_stats") or []
    total = data.get("total_ha", 0)
    lines = [
        "### 📊 Distribuzione Superfici",
        source_info,
        f"**Totale classificato:** `{total:.2f} ha`\n",
        "| Classe | Ettari | % |",
        "| :--- | ---: | ---: |",
    ]
    for s in stats:
        lines.append(f"| **{s['name']}** | {s['hectares']:.2f} | {s['percentage']:.1f}% |")
    return "\n".join(lines)


def _format_ndvi(data: dict) -> str:
    series = data.get("ndvi_series") or []
    season_emoji = {"winter": "❄️", "spring": "🌱", "summer": "☀️", "autumn": "🍂"}
    lines = [
        "### 🌿 NDVI per Stagione (solo pixel vegetazione)",
        "| Stagione | Media | Std | Min | Max |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for n in series:
        emoji = season_emoji.get(n["season"], "")
        lines.append(
            f"| {emoji} {n['season'].capitalize()} "
            f"| {n['mean']:.3f} | {n['std']:.3f} | {n['min']:.3f} | {n['max']:.3f} |"
        )
    return "\n".join(lines)


def _format_diagnostics(data: dict, task_id: str) -> str:
    stats = data.get("class_stats") or []
    lines = [
        "### 🔧 Diagnostica",
        f"- **Task ID:** `{task_id}`",
        f"- **Sorgente:** `{data.get('source', 'N/A')}`",
        f"- **Anno usato:** `{data.get('year_used', 'N/A')}`",
        f"- **BBox:** `{data.get('bbox', 'N/A')}`",
        f"- **Totale ha:** `{data.get('total_ha', 0):.2f}`",
        f"- **Classi rilevate:** `{len(stats)}`",
        "",
        "**Link diretti:**",
        f"- [🖼️ Immagine overlay]({API_BASE}/{task_id}/image)",
        f"- [📊 Status JSON]({API_BASE}/{task_id}/status)",
        f"- [🌿 NDVI JSON]({API_BASE}/{task_id}/ndvi)",
        f"- [🎨 Legenda con ettari]({API_BASE}/{task_id}/legend)",
    ]
    return "\n".join(lines)


def _build_legend_md() -> str:
    lines = ["| Colore | Classe |", "| :---: | :--- |"]
    for i in range(1, len(CLASS_NAMES)):
        hex_color = CLASS_COLORS_HEX[i]
        lines.append(
            f"| <span style='background:{hex_color}; padding:4px 14px; "
            f"border-radius:3px;'>&nbsp;</span> | **{CLASS_NAMES[i]}** |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="Smart Food — Segmentazione Agricola") as demo:

        gr.Markdown("""
        # 🌍 Smart Food — Monitoraggio Satellitare
        ### Classificazione automatica colture su Sentinel-2 | Modello Prithvi EO v2
        ---
        """)

        # Barra stato API
        api_status  = gr.Markdown(check_api_status())
        refresh_btn = gr.Button("🔄 Verifica API", size="sm")
        refresh_btn.click(fn=check_api_status, outputs=api_status)

        with gr.Row():

            # ── Colonna input ──────────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📍 Parametri Query")

                with gr.Tabs():

                    # ── Tab POI ──
                    with gr.Tab("📍 Punto (POI)"):
                        preset_dd = gr.Dropdown(
                            choices=["(manuale)"] + list(POI_PRESETS.keys()),
                            label="Punto predefinito",
                            value="(manuale)",
                        )
                        lat_in  = gr.Number(label="Latitudine",  value=37.380, precision=6)
                        lon_in  = gr.Number(label="Longitudine", value=14.910, precision=6)
                        year_poi = gr.Dropdown(
                            choices=[str(y) for y in SUPPORTED_YEARS],
                            value=str(DEFAULT_YEAR),
                            label="Anno",
                            info="⚠️ Anni < 2020: qualità ridotta (domain shift)",
                        )
                        run_poi_btn = gr.Button(
                            "🚀 Avvia Analisi POI", variant="primary", size="lg"
                        )

                    # ── Tab BBox ──
                    with gr.Tab("📐 Area (BBox)"):
                        gr.Markdown(
                            "Inserisci i 4 bordi dell'area. "
                            "Max consigliato: **~50×50 km** (2500 km²)."
                        )
                        bbox_preset_dd = gr.Dropdown(
                            choices=["(manuale)"] + list(BBOX_PRESETS.keys()),
                            label="Area predefinita",
                            value="(manuale)",
                        )
                        with gr.Row():
                            min_lon_in = gr.Number(label="Min Lon (W)", value=14.857, precision=6)
                            max_lon_in = gr.Number(label="Max Lon (E)", value=14.963, precision=6)
                        with gr.Row():
                            min_lat_in = gr.Number(label="Min Lat (S)", value=37.327, precision=6)
                            max_lat_in = gr.Number(label="Max Lat (N)", value=37.433, precision=6)
                        year_bbox = gr.Dropdown(
                            choices=[str(y) for y in SUPPORTED_YEARS],
                            value=str(DEFAULT_YEAR),
                            label="Anno",
                            info="⚠️ Anni < 2020: qualità ridotta (domain shift)",
                        )
                        run_bbox_btn = gr.Button(
                            "🚀 Avvia Analisi BBox", variant="primary", size="lg"
                        )

                gr.Markdown(f"""
                ---
                ℹ️ **Info:**
                - Risoluzione: 10m/pixel
                - 4 stagioni Sentinel-2
                - Cache MinIO attiva
                - API: `{API_BASE}`
                """)

            # ── Colonna output ─────────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Risultati")
                with gr.Tabs():
                    with gr.Tab("🗺️ Overlay"):
                        overlay_out = gr.Image(label="RGB + Predizione", type="pil")
                    with gr.Tab("📈 Statistiche"):
                        stats_out = gr.Markdown("*Avvia un'analisi.*")
                    with gr.Tab("🌿 NDVI"):
                        ndvi_out = gr.Markdown("*Avvia un'analisi.*")
                    with gr.Tab("🔧 Diagnostica"):
                        diag_out = gr.Markdown("*Avvia un'analisi.*")
                    with gr.Tab("🎨 Legenda"):
                        gr.Markdown(_build_legend_md())

        # ── Event handlers ─────────────────────────────────────────────────
        preset_dd.change(
            fn=load_preset,
            inputs=[preset_dd],
            outputs=[lat_in, lon_in],
        )
        bbox_preset_dd.change(
            fn=load_bbox_preset,
            inputs=[bbox_preset_dd],
            outputs=[min_lon_in, min_lat_in, max_lon_in, max_lat_in],
        )
        run_poi_btn.click(
            fn=run_analysis_poi,
            inputs=[preset_dd, lat_in, lon_in, year_poi],
            outputs=[overlay_out, stats_out, ndvi_out, diag_out],
        )
        run_bbox_btn.click(
            fn=run_analysis_bbox,
            inputs=[min_lon_in, min_lat_in, max_lon_in, max_lat_in, year_bbox],
            outputs=[overlay_out, stats_out, ndvi_out, diag_out],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )