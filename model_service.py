'''' model_service.py
  Singleton che carica i pesi Prithvi una sola volta all'avvio. Espone
  predict_chip(tensor) → (pred_mask, confidence_map). Gestisce device           
  (CUDA/CPU), normalizzazione con means/stds, e pulizia dello state_dict
  (_orig_mod.). '''