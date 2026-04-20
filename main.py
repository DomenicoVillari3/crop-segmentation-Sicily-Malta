''''main.py         
  FastAPI app. Endpoint:
  - POST /api/v1/analysis/run → accetta POI o bbox, lancia pipeline in
  background, ritorna task_id                                                   
  - GET /api/v1/analysis/{task_id}/status → polling con progress 0-100 e
  risultato finale                                                              
  - GET /api/v1/analysis/{task_id}/image → ritorna il PNG segmentato (o overlay)
  - GET /api/v1/classes → catalogo classi con colori hex                        
  - GET /api/v1/health → stato modello, GPU, MinIO     '''