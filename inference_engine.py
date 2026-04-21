"""
inference_engine.py
===================
Prende un cubo numpy (4, 6, H, W), esegue sliding window con overlap,
scrive solo la regione centrale di ogni chip, applica filtro acqua NIR.
Ritorna la maschera finale (H, W) uint8.

Non sa nulla di MinIO, STAC, o API.
Dipende solo da model_service.py e config.py.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from tqdm import tqdm

from config import (
    CHIP_SIZE,
    OVERLAP,
    WATER_NIR_THRESHOLD,
    WATER_NIR_SEASON_IDX,   # indice stagione estiva per il filtro NIR (2 = estate)
    WATER_NIR_BAND_IDX,     # indice banda NIR nel cubo (3 = NIR in B,G,R,NIR,SWIR1,SWIR2)
    MIN_CONFIDENCE,         # soglia sotto cui un pixel è marcato come sfondo (0 = disabilitato)
    APPLY_PRIOR_CORRECTION,
    SICILIA_CLASS_PRIORS
)
from model_service import ModelService

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Esegue l'inferenza su un cubo numpy via sliding window con overlap.

    Il cuore del sistema: trasforma un cubo (4, 6, H, W) in una maschera
    di classi (H, W) pronta per postprocess.py.

    Usage:
        engine = InferenceEngine()
        mask = engine.run(cube)   # np.ndarray (H, W) uint8
    """

    def __init__(self) -> None:
        self._svc  = ModelService()   # Singleton: nessun reload dei pesi
        self._chip = CHIP_SIZE        # 224
        self._overlap = OVERLAP       # es. 0.25

        # Stride = quanto si sposta la finestra ad ogni step
        # Con overlap=0.25: stride = 224 * 0.75 = 168 px
        self._stride = int(self._chip * (1.0 - self._overlap))

        # Margine centrale scritto per ogni chip
        # Con overlap=0.25: margine = 224 * 0.25 / 2 = 28 px su ogni lato
        # Solo la zona [margin:-margin, margin:-margin] di ogni chip viene
        # scritta sulla prediction map finale
        self._margin = int(self._chip * self._overlap / 2)

        # Prior correction — precomputata una volta sola, non ad ogni chip
        if APPLY_PRIOR_CORRECTION:
            priors = torch.tensor(SICILIA_CLASS_PRIORS, dtype=torch.float32)
            self._log_prior = (
                torch.log(priors + 1e-8)
                .view(1, len(SICILIA_CLASS_PRIORS), 1, 1)
                .to(self._svc.device)
            )
            logger.info(f"✅ Prior correction attiva")
        else:
            self._log_prior = None

        logger.info(
            f"InferenceEngine | chip={self._chip}px | "
            f"overlap={self._overlap:.0%} | stride={self._stride}px | "
            f"margin={self._margin}px"
        )

    # -----------------------------------------------------------------------
    # API pubblica
    # -----------------------------------------------------------------------

    def run(self, cube: np.ndarray) -> np.ndarray:
        """
        Pipeline completa: padding → sliding window → unpad → filtro acqua.

        Args:
            cube: np.ndarray (4, 6, H, W) float32, valori grezzi Sentinel-2

        Returns:
            mask: np.ndarray (H, W) uint8 con indici di classe (0 = sfondo/acqua)
        """
        T, C, H_orig, W_orig = cube.shape
        logger.info(f"🧠 Inferenza | cubo={cube.shape} | stride={self._stride}px")

        # 1. Padding reflect per gestire i bordi
        cube_padded, (H_pad, W_pad) = self._pad(cube)

        # 2. Mappe di accumulo
        # pred_accum: somma delle confidence per classe → shape (NUM_CLASSES, H_pad, W_pad)
        # count_map:  quante volte ogni pixel è stato coperto → per normalizzare
        #
        # Usare l'accumulo delle confidence invece del "last write wins" del
        # codice legacy garantisce che pixel sovrapposti da più chip ricevano
        # la classe con la confidence media più alta — molto più robusto ai bordi.
        num_classes = self._svc.num_classes
        conf_accum = np.zeros((num_classes, H_pad, W_pad), dtype=np.float32)
        count_map  = np.zeros((H_pad, W_pad),               dtype=np.uint8)

        # 3. Sliding window
        y_positions = self._positions(H_pad, self._stride, self._chip)
        x_positions = self._positions(W_pad, self._stride, self._chip)
        total_chips = len(y_positions) * len(x_positions)

        with tqdm(total=total_chips, desc="Sliding window", unit="chip") as pbar:
            for y in y_positions:
                for x in x_positions:
                    self._process_chip(
                        cube_padded, conf_accum, count_map,
                        y, x, H_pad, W_pad
                    )
                    pbar.update(1)

        # 4. Predizione finale: argmax sulla mappa di confidence accumulata
        # Dove count_map == 0 (non dovrebbe mai accadere con il padding corretto)
        # forza la classe sfondo
        final_pad = np.where(
            count_map > 0,
            conf_accum.argmax(axis=0).astype(np.uint8),
            0
        )

        # 5. Rimuovi il padding
        final_mask = final_pad[:H_orig, :W_orig]

        # 6. Filtro acqua / mare via NIR estivo
        final_mask = self._apply_water_filter(final_mask, cube)

        # 7. Filtro confidenza minima (opzionale)
        if MIN_CONFIDENCE > 0:
            final_mask = self._apply_confidence_filter(
                final_mask, conf_accum[:, :H_orig, :W_orig], count_map[:H_orig, :W_orig]
            )

        logger.info(f"✅ Inferenza completata | maschera={final_mask.shape}")
        return final_mask

    # -----------------------------------------------------------------------
    # Logica interna
    # -----------------------------------------------------------------------

    def _process_chip(
        self,
        cube_padded: np.ndarray,
        conf_accum:  np.ndarray,
        count_map:   np.ndarray,
        y: int, x: int,
        H_pad: int, W_pad: int,
    ) -> None:
        """
        Estrae un chip, esegue la forward pass, accumula solo la zona centrale.

        La zona centrale è [margin : chip-margin, margin : chip-margin].
        Sui bordi dell'immagine intera il margine viene ridotto a zero per
        non perdere pixel reali: un chip che inizia a y=0 non ha nulla
        sopra di lui, quindi scrive dall'inizio.

        Mappa della zona scritta (con margin=28, chip=224):
        ┌──────────────────────────────┐
        │   28px  margine ignorato     │
        │  ┌────────────────────────┐  │
        │28│  zona scritta 168×168  │28│
        │  └────────────────────────┘  │
        │   28px  margine ignorato     │
        └──────────────────────────────┘
        """
        chip_np = cube_padded[:, :, y:y+self._chip, x:x+self._chip]
        chip_t  = torch.from_numpy(chip_np).unsqueeze(0)  # (1, T, C, H, W)

        pred_mask, confidence = self._svc.predict_chip(chip_t, log_prior=self._log_prior)
        # pred_mask:  (H, W) uint8
        # confidence: (H, W) float32  — probabilità della classe vincente

        # Calcola zona da scrivere (adatta ai bordi dell'immagine)
        m = self._margin
        # Bordo sinistro/superiore: se il chip inizia al bordo, non c'è nulla prima
        wy_start = 0 if y == 0 else m
        wx_start = 0 if x == 0 else m
        # Bordo destro/inferiore
        wy_end = self._chip if (y + self._chip >= H_pad) else self._chip - m
        wx_end = self._chip if (x + self._chip >= W_pad) else self._chip - m

        # Coordinate nella prediction map globale
        py_start = y + wy_start
        py_end   = y + wy_end
        px_start = x + wx_start
        px_end   = x + wx_end

        # Accumula confidence per ogni classe nella zona centrale
        # conf_accum[classe, y, x] += confidence se pred == classe
        for cls_id in range(conf_accum.shape[0]):
            class_conf = np.where(
                pred_mask[wy_start:wy_end, wx_start:wx_end] == cls_id,
                confidence[wy_start:wy_end, wx_start:wx_end],
                0.0
            )
            conf_accum[cls_id, py_start:py_end, px_start:px_end] += class_conf

        count_map[py_start:py_end, px_start:px_end] += 1

    def _pad(
        self,
        cube: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Padding reflect per rendere H e W multipli dello stride.

        Usiamo 'reflect' invece di 'constant' (zero) perché:
        - 'constant' introduce bordi neri artificiali che il modello
          classifica spesso come Incolto o Sfondo
        - 'reflect' specchia i pixel reali, il modello vede dati plausibili
          anche ai bordi → classificazioni più coerenti nelle fasce marginali
        """
        T, C, H, W = cube.shape

        # Serve padding fino al prossimo multiplo di stride che permette
        # di coprire tutta l'immagine
        def pad_to(size: int, stride: int, chip: int) -> int:
            if size <= chip:
                return chip - size
            remainder = (size - chip) % stride
            return 0 if remainder == 0 else stride - remainder

        pad_h = pad_to(H, self._stride, self._chip)
        pad_w = pad_to(W, self._stride, self._chip)

        if pad_h == 0 and pad_w == 0:
            return cube, (H, W)

        cube_padded = np.pad(
            cube,
            ((0, 0), (0, 0), (0, pad_h), (0, pad_w)),
            mode="reflect"
        )
        H_pad, W_pad = cube_padded.shape[2], cube_padded.shape[3]
        logger.debug(f"Padding: ({H},{W}) → ({H_pad},{W_pad})")
        return cube_padded, (H_pad, W_pad)

    @staticmethod
    def _positions(size: int, stride: int, chip: int) -> list[int]:
        """
        Calcola le posizioni di inizio dei chip lungo un asse.
        Garantisce che l'ultimo chip arrivi esattamente a 'size'.
        """
        positions = list(range(0, size - chip + 1, stride))
        # Assicura che l'ultimo chip copra il bordo finale
        if not positions or positions[-1] + chip < size:
            positions.append(size - chip)
        return positions

    def _apply_water_filter(
        self,
        mask: np.ndarray,
        cube: np.ndarray,
    ) -> np.ndarray:
        """
        Azzera i pixel dove il NIR estivo è sotto soglia → acqua/mare.

        Banda NIR = indice 3 (ordine: B, G, R, NIR, SWIR1, SWIR2)
        Stagione estiva = indice 2 (ordine: winter, spring, summer, autumn)

        Perché NIR < 400:
          - Vegetazione sana:  NIR ~2000-4000
          - Suolo nudo:        NIR ~800-1500
          - Acqua/mare:        NIR ~100-400 (assorbe quasi tutto l'infrarosso)
          → Soglia conservativa: falsi positivi rari, falsi negativi accettabili
        """
        H, W = mask.shape
        nir = cube[WATER_NIR_SEASON_IDX, WATER_NIR_BAND_IDX, :H, :W]
        water_mask = nir < WATER_NIR_THRESHOLD

        n_water = water_mask.sum()
        if n_water > 0:
            logger.debug(f"Filtro acqua: {n_water} pixel azzerati ({n_water/(H*W):.1%})")

        result = mask.copy()
        result[water_mask] = 0
        return result

    @staticmethod
    def _apply_confidence_filter(
        mask:       np.ndarray,
        conf_accum: np.ndarray,
        count_map:  np.ndarray,
    ) -> np.ndarray:
        """
        Azzera i pixel dove la confidence media è sotto MIN_CONFIDENCE.
        Utile per marcare come 'incerto' i pixel ai confini tra campi.
        """
        # Confidence media per la classe predetta
        avg_conf = np.where(
            count_map > 0,
            conf_accum.max(axis=0) / np.maximum(count_map, 1),
            0.0
        )
        result = mask.copy()
        result[avg_conf < MIN_CONFIDENCE] = 0
        return result