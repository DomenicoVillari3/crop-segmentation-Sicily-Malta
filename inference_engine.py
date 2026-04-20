'''inference_engine.py                                                           
  Prende un cubo numpy, esegue lo sliding window con overlap configurabile
  (logica da minio_inference_v2.py), scrive solo la regione centrale di ogni    
  chip per evitare artefatti. Applica il filtro acqua (NIR estate < 400).
  Ritorna la maschera finale (H, W) uint8. '''