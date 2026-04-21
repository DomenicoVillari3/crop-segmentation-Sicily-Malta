"""
model_service.py
================
Singleton che carica Prithvi una sola volta all'avvio del processo.
Espone predict_chip(tensor) → (pred_mask, confidence_map).

Thread-safe per inferenza concorrente (read-only dopo il caricamento).
Il lock è necessario solo durante l'inizializzazione, non durante l'inferenza.
"""

from __future__ import annotations

import logging
import threading
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    MODEL_WEIGHTS_PATH,
    BACKBONE_MODEL_NAME,
    NUM_CLASSES,
    NORM_MEANS,
    NORM_STDS,
)

logger = logging.getLogger(__name__)


class ModelService:
    """
    Singleton thread-safe per il modello Prithvi.

    Prima chiamata:  ModelService()  →  carica pesi da disco (~3-5 secondi)
    Chiamate successive: ritorna l'istanza già in memoria istantaneamente.

    Perché Singleton e non variabile globale:
      - Lazy initialization: il modello viene caricato solo quando serve,
        non all'import del modulo (utile per health check e test)
      - Controllabile: si può resettare con ModelService._instance = None
        nei test senza toccare variabili globali
    """

    _instance: ClassVar[ModelService | None] = None
    _lock:     ClassVar[threading.Lock]       = threading.Lock()

    def __new__(cls) -> ModelService:
        # Double-checked locking: evita il lock dopo la prima inizializzazione
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:          # secondo check dentro il lock
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        # __init__ viene chiamato ogni volta che si fa ModelService()
        # ma il caricamento avviene una sola volta grazie al flag
        if self._initialized:
            return
        self._initialized = True
        self._load()

    # -----------------------------------------------------------------------
    # Caricamento
    # -----------------------------------------------------------------------

    def _load(self) -> None:
        """
        Carica backbone + decoder + pesi finetuned su GPU/CPU.

        Passaggi:
          1. Rileva device
          2. Costruisce l'architettura (backbone TerraTorch + decoder)
          3. Carica state_dict pulendo i prefissi _orig_mod. lasciati da
             torch.compile() durante il training
          4. Mette il modello in eval() — disattiva BatchNorm running stats
             update e Dropout, fondamentale per inferenza deterministica
          5. Prepara i tensori means/stds con shape broadcast-compatibile
             (1, 1, C, 1, 1) per normalizzare (B, T, C, H, W) in un'operazione
        """
        # 1. Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ModelService: device={self.device}")

        # 2. Architettura
        try:
            from terratorch.registry import BACKBONE_REGISTRY
            from architecture import PrithviSegmentation4090
        except ImportError as e:
            raise RuntimeError(
                f"Impossibile importare l'architettura del modello: {e}. "
                f"Assicurati che train.py e terratorch siano nel PYTHONPATH."
            ) from e

        logger.info(f"🔄 Costruzione backbone: {BACKBONE_MODEL_NAME}")
        backbone = BACKBONE_REGISTRY.build(BACKBONE_MODEL_NAME, pretrained=False)
        self._model = PrithviSegmentation4090(backbone, NUM_CLASSES)

        # 3. Carica pesi
        logger.info(f"🔄 Caricamento pesi: {MODEL_WEIGHTS_PATH}")
        raw_state = torch.load(MODEL_WEIGHTS_PATH, map_location=self.device)

        # Pulizia prefisso _orig_mod. lasciato da torch.compile()
        # es. "_orig_mod.backbone.patch_embed.weight" → "backbone.patch_embed.weight"
        clean_state = {
            k.replace("_orig_mod.", ""): v
            for k, v in raw_state.items()
        }

        missing, unexpected = self._model.load_state_dict(clean_state, strict=True)
        if missing:
            logger.warning(f"Pesi mancanti ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logger.warning(f"Pesi inattesi ({len(unexpected)}): {unexpected[:5]}...")

        # 4. Eval mode + device
        self._model.to(self.device).eval()

        # 5. Normalizzazione — shape (1, 1, 6, 1, 1) per broadcast su (B, T, C, H, W)
        self._means = (
            torch.tensor(NORM_MEANS, dtype=torch.float32)
            .view(1, 1, len(NORM_MEANS), 1, 1)
            .to(self.device)
        )
        self._stds = (
            torch.tensor(NORM_STDS, dtype=torch.float32)
            .view(1, 1, len(NORM_STDS), 1, 1)
            .to(self.device)
        )

        # Info GPU
        if self.device.type == "cuda":
            vram_mb = torch.cuda.memory_allocated() / 1024**2
            logger.info(f"✅ Modello caricato su GPU | VRAM usata: {vram_mb:.0f} MB")
        else:
            logger.info("✅ Modello caricato su CPU")

    # -----------------------------------------------------------------------
    # Inferenza
    # -----------------------------------------------------------------------

    def predict_chip(
        self,
        chip: torch.Tensor,
        log_prior: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Esegue l'inferenza su un singolo chip.

        Args:
            chip: tensor (1, T, C, H, W) float32, valori grezzi Sentinel-2
                  NON normalizzato — la normalizzazione avviene qui dentro

        Returns:
            pred_mask:      np.ndarray (H, W) uint8   — classe predetta per pixel
            confidence_map: np.ndarray (H, W) float32 — probabilità della classe
                            predetta (0.0–1.0), utile per filtrare pixel incerti

        Note sulla confidence_map:
            È la probabilità softmax della classe vincente, NON l'entropia.
            Valori < 0.5 indicano pixel ambigui (es. bordo tra Grano e Orzo).
            inference_engine.py può usarla per applicare un filtro di confidenza
            minima prima di scrivere la maschera finale.
        """
        # Sposta su device e normalizza
        x = chip.to(self.device, dtype=torch.float32)
        x = (x - self._means) / (self._stds + 1e-6)

        with torch.no_grad():
            logits = self._model(x)          # (1, NUM_CLASSES, H, W)
            if log_prior is not None:           
              logits = logits - log_prior      # Applica correzione bayesiana sottraendo il log-prior (log(p) = log(pred) - log(prior))
            probs  = F.softmax(logits, dim=1) # (1, NUM_CLASSES, H, W)

        # Classe con probabilità massima
        confidence, pred = probs.max(dim=1)  # entrambi (1, H, W)

        pred_mask      = pred.squeeze(0).cpu().numpy().astype(np.uint8)
        confidence_map = confidence.squeeze(0).cpu().numpy().astype(np.float32)

        return pred_mask, confidence_map

    # -----------------------------------------------------------------------
    # Diagnostica
    # -----------------------------------------------------------------------

    @property
    def is_on_gpu(self) -> bool:
        return self.device.type == "cuda"

    @property
    def vram_used_mb(self) -> float:
        if not self.is_on_gpu:
            return 0.0
        return torch.cuda.memory_allocated() / 1024**2

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES

    def health(self) -> dict:
        """
        Usato dall'endpoint GET /api/v1/health per verificare lo stato del modello.
        Esegue una forward pass su un tensor fittizio per confermare che
        il modello risponde correttamente prima di accettare richieste reali.
        """
        try:
            dummy = torch.zeros(1, 4, 6, 224, 224)
            mask, conf = self.predict_chip(dummy)
            return {
                "status":        "ok",
                "device":        str(self.device),
                "vram_used_mb":  round(self.vram_used_mb, 1),
                "num_classes":   self.num_classes,
                "output_shape":  list(mask.shape),
                "conf_range":    [float(conf.min()), float(conf.max())],
            }
        except Exception as e:
            return {"status": "error", "detail": str(e)}