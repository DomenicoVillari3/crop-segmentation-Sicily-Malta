"""
demo_gui.py
===========
Test locale dell'intera pipeline con interfaccia Gradio.
Bypassa completamente FastAPI — chiama direttamente i moduli core.

Flusso testato:
    DataResolver → InferenceEngine → postprocess → visualizzazione

Prerequisiti:
    - MinIO attivo (docker-compose up minio)
    - Pesi modello disponibili al path in .env
    - pip install gradio
"""

import gradio as gr
import numpy as np
from PIL import Image
import io
import logging
import traceback

# Pipeline core — la nuova infrastruttura
from data_resolver   import DataResolver
from inference_engine import InferenceEngine
from postprocess     import process as postprocess, PostprocessResult
from config          import CLASS_NAMES, CLASS_COLORS_RGB, CLASS_COLORS_HEX, SUPPORTED_YEARS, DEFAULT_YEAR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inizializzazione singleton — avviene UNA VOLTA all'avvio
# ---------------------------------------------------------------------------
print("🔄 Caricamento modello e connessioni...")
try:
    resolver = DataResolver()
    engine   = InferenceEngine()
    print("✅ Sistema pronto.")
    SYSTEM_OK = True
except Exception as e:
    print(f"❌ Errore inizializzazione: {e}")
    traceback.print_exc()
    SYSTEM_OK = False

# ---------------------------------------------------------------------------
# Punti di interesse predefiniti
# ---------------------------------------------------------------------------
POI_PRESETS = {
    "🍊 Agrumeti – Catania":            (37.380, 14.910),
    "🌾 Grano – Enna":                  (37.567, 14.279),
    "🌾 Seminativo – Caltanissetta":    (37.490, 14.060),
    "🍊 Agrumeti – Lentini":            (37.285, 14.990),
    "🏖️ Palermo Costa":                 (38.115, 13.361),
    "🇲🇹 Terreni agricoli – Mosta":     (35.910, 14.425),
}

# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def run_analysis(preset_name, lat, lon, year):
    """
    Esegue la pipeline completa e ritorna:
      - overlay PIL Image
      - statistiche markdown
      - NDVI markdown
      - diagnostica markdown
    """
    if not SYSTEM_OK:
        err = "❌ Sistema non inizializzato. Controlla i log del terminale."
        return None, err, err, err

    try:
        _year = int(year)
        logger.info(f"▶ Analisi ({lat:.4f}, {lon:.4f}) anno={_year}")

        # 1. Risolvi sorgente dati
        resolved = resolver.resolve_from_point(lat=lat, lon=lon, year=_year)
        if resolved is None:
            msg = (
                "❌ Nessun dato disponibile per queste coordinate.\n\n"
                "Possibili cause:\n"
                "- Area fuori copertura MinIO e STAC senza immagini\n"
                "- Cloud cover >40% per tutte le stagioni\n"
                "- Coordinate fuori range Sentinel-2"
            )
            return None, msg, msg, msg

        source_info = f"**Sorgente:** `{resolved.source}` | **Anno:** `{resolved.year}`"
        logger.info(f"✅ Dati risolti: {resolved.source} | cubo={resolved.cube.shape}")

        # 2. Inferenza
        mask = engine.run(resolved.cube)
        logger.info(f"✅ Inferenza completata: maschera={mask.shape}")

        # 3. Post-processing
        result: PostprocessResult = postprocess(mask, resolved.cube)

        # 4. Prepara output Gradio
        overlay_img  = Image.open(io.BytesIO(result.overlay_png))
        stats_md     = _format_stats(result, source_info)
        ndvi_md      = _format_ndvi(result)
        diag_md      = _format_diagnostics(resolved, mask)

        return overlay_img, stats_md, ndvi_md, diag_md

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Errore pipeline: {e}\n{tb}")
        err = f"❌ Errore durante l'analisi:\n```\n{tb}\n```"
        return None, err, err, err


def load_preset(preset_name):
    """Aggiorna lat/lon quando si seleziona un preset."""
    if preset_name and preset_name in POI_PRESETS:
        lat, lon = POI_PRESETS[preset_name]
        return lat, lon
    return 37.5, 14.2


# ---------------------------------------------------------------------------
# Formatters output
# ---------------------------------------------------------------------------

def _format_stats(result: PostprocessResult, source_info: str) -> str:
    lines = [
        f"### 📊 Distribuzione Superfici",
        source_info,
        f"**Totale classificato:** `{result.total_ha:.2f} ha`\n",
        "| Classe | Ettari | % |",
        "| :--- | ---: | ---: |",
    ]
    for s in result.class_stats:
        lines.append(f"| **{s.name}** | {s.hectares:.2f} | {s.percentage:.1f}% |")
    return "\n".join(lines)


def _format_ndvi(result: PostprocessResult) -> str:
    lines = [
        "### 🌿 NDVI per Stagione (solo pixel vegetazione)",
        "| Stagione | Media | Std | Min | Max |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    season_emoji = {"winter": "❄️", "spring": "🌱", "summer": "☀️", "autumn": "🍂"}
    for n in result.ndvi_series:
        emoji = season_emoji.get(n.season, "")
        lines.append(
            f"| {emoji} {n.season.capitalize()} "
            f"| {n.mean:.3f} | {n.std:.3f} | {n.min:.3f} | {n.max:.3f} |"
        )
    return "\n".join(lines)


def _format_diagnostics(resolved, mask) -> str:
    unique, counts = np.unique(mask, return_counts=True)
    total_px = mask.size
    lines = [
        "### 🔧 Diagnostica Pipeline",
        f"- **Sorgente dati:** `{resolved.source}`",
        f"- **Anno:** `{resolved.year}`",
        f"- **Shape cubo:** `{resolved.cube.shape}`",
        f"- **BBox:** `{[round(x,4) for x in resolved.bbox]}`",
        f"- **Shape maschera:** `{mask.shape}`",
        f"- **Pixel totali:** `{total_px:,}`",
        "",
        "**Distribuzione raw classi:**",
    ]
    for cls_id, count in zip(unique.tolist(), counts.tolist()):
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
        pct  = count / total_px * 100
        lines.append(f"  - `{cls_id}` {name}: {count:,} px ({pct:.1f}%)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Legenda
# ---------------------------------------------------------------------------

def _build_legend_md() -> str:
    lines = ["| Colore | Classe |", "| :---: | :--- |"]
    for i in range(1, len(CLASS_NAMES)):
        hex_color = CLASS_COLORS_HEX[i]
        lines.append(
            f"| <span style='background:{hex_color};"
            f"padding:4px 14px; border-radius:3px;'>&nbsp;</span>"
            f" | **{CLASS_NAMES[i]}** |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interfaccia Gradio
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(
        title="Smart Food — Segmentazione Agricola Sicilia",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            """
            # 🌍 Smart Food — Monitoraggio Satellitare
            ### Classificazione automatica colture su Sentinel-2 | Modello Prithvi EO v2
            ---
            """
        )

        with gr.Row():

            # --- Colonna input ---
            with gr.Column(scale=1):
                gr.Markdown("### 📍 Parametri Query")

                preset_dd = gr.Dropdown(
                    choices=["(manuale)"] + list(POI_PRESETS.keys()),
                    label="Punto predefinito",
                    value="(manuale)",
                )
                lat_in = gr.Number(label="Latitudine",  value=37.380, precision=6)
                lon_in = gr.Number(label="Longitudine", value=14.910, precision=6)
                year_in = gr.Dropdown(
                    choices=[str(y) for y in SUPPORTED_YEARS],
                    value=str(DEFAULT_YEAR),
                    label="Anno",
                    info="⚠️ Anni < 2020 potrebbero avere qualità ridotta (domain shift dal training set)"
                )

                run_btn = gr.Button("🚀 Avvia Analisi", variant="primary", size="lg")

                gr.Markdown(
                    """
                    ---
                    ℹ️ **Info:**
                    - Risoluzione: 10m/pixel
                    - 4 stagioni Sentinel-2
                    - Cache MinIO attiva
                    """
                )

            # --- Colonna output ---
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Risultati")

                with gr.Tabs():

                    with gr.Tab("🗺️ Overlay"):
                        overlay_out = gr.Image(
                            label="RGB Satellite + Predizione",
                            type="pil",
                        )

                    with gr.Tab("📈 Statistiche"):
                        stats_out = gr.Markdown("*Avvia un'analisi per vedere i risultati.*")

                    with gr.Tab("🌿 NDVI"):
                        ndvi_out = gr.Markdown("*Avvia un'analisi per vedere i risultati.*")

                    with gr.Tab("🔧 Diagnostica"):
                        diag_out = gr.Markdown("*Avvia un'analisi per vedere i risultati.*")

                    with gr.Tab("🎨 Legenda"):
                        gr.Markdown(_build_legend_md())

        # --- Event handlers ---
        preset_dd.change(
            fn=load_preset,
            inputs=[preset_dd],
            outputs=[lat_in, lon_in],
        )

        run_btn.click(
            fn=run_analysis,
            inputs=[preset_dd, lat_in, lon_in, year_in],
            outputs=[overlay_out, stats_out, ndvi_out, diag_out],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not SYSTEM_OK:
        print("❌ Sistema non inizializzato. Controlla i log.")
        exit(1)

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )