'''config.py                                                                     
  Centralizza tutte le costanti: endpoint MinIO, bucket name, credenziali, path 
  del modello, parametri STAC (stagioni, cloud cover), chip size, overlap, bande
   Sentinel-2, PIXEL_AREA_M2. Carica anche config.json e lo espone come oggetto 
  unico usato da tutti gli altri moduli. 
  '''
from dotenv import load_dotenv
import os


load_dotenv()  # carica .env se presente


BACKBONE_MODEL_NAME = "terratorch_prithvi_eo_v2_100_tl" 

#MODEL PATH → da env, non hardcoded
MODEL_WEIGHTS_PATH=os.getenv("MODEL_WEIGHTS_PATH")  # Verifica che MODEL_WEIGHTS_PATH sia definito

# Credenziali sensibili → solo da env
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
    raise RuntimeError("Credenziali MinIO mancanti. Controlla il file .env")
MINIO_BUCKET_NAME = "sentinel-2-data-v2"


#Sentinel-2 parameters
SENTINEL_BANDS_CODE = ["B02", "B03", "B04", "B08", "B11", "B12"]  # Blue, Green, Red, NIR, SWIR1, SWIR2
SENTINEL_BANDS= ["blue", "green", "red", "nir08", "swir16", "swir22"]
WATER_NIR_BAND_IDX    = 3   # NIR    (B=0, G=1, R=2, NIR=3, SWIR1=4, SWIR2=5) per filtro acqua
SENTINEL_RESOLUTION = 10          # metri/pixel
MAX_CLOUD_COVER = 20              # percentuale massima accettata
STAC_ENDPOINT = "https://earth-search.aws.element84.com/v1"
STAC_COLLECTION = "sentinel-2-l2a"



# Stagioni: (nome, mese_inizio, mese_fine)
SEASONS = {
    "winter": (12, 2),
    "spring": (3, 5),
    "summer": (6, 8),
    "autumn": (9, 11),
}
ORDERED_SEASONS = ["winter", "spring", "summer", "autumn"]
WATER_NIR_SEASON_IDX  = 2   # estate (winter=0, spring=1, summer=2, autumn=3) per filtro acqua (NIR estivo è più affidabile per distinguere mare da terra)
# Anno di default per nuovi download
DEFAULT_YEAR = 2023

'''anni pre-2020 hanno qualità L2A inferiore e copertura meno uniforme — 
il modello è stato addestrato su dati 2023 quindi le performance su 
anni lontani potrebbero degradare per drift radiometrico. 
Se vuoi essere conservativo per il prodotto commerciale'''
#SUPPORTED_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

SUPPORTED_YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
NUM_TIMESTEPS = 4  # T=4


#Spatial Attributes
CHIP_SIZE = 224           # pixel — dimensione input del modello Prithvi
PIXEL_AREA_M2 = 100.0     # 10m x 10m
M2_TO_HECTARES = 10000.0
# Margine extra durante il download (poi trimmato dopo l'inferenza)
# Serve per evitare che il POI cada sul bordo di un chip
DOWNLOAD_MARGIN_PX = 32   # pixel extra su ogni lato
OVERLAP = 0.25   # 25% — overlap tra chip adiacenti
OVERLAY_OPACITY = 0.55     # bilanciamento visibilità maschera/satellite



NORM_MEANS= [724.812806274414, 1070.410831237793, 1344.193027496338,2834.3394388427732, 2902.6298315429685, 2228.6243228149415]
NORM_STDS= [393.21970404052735, 490.4992893676758, 682.7227749023438, 834.264825805664, 937.5322022094726, 874.680620666504]

NUM_CLASSES = 9
CLASS_NAMES = ["Sfondo", "Olivo", "Vite", "Agrumi", "Frutteto", 
               "Cereali", "Legumi", "Ortaggi", "Incolto"]
# Colori — due formati per usi diversi
CLASS_COLORS_RGB = [
    [0,   0,   0  ],  # 0: Sfondo
    [50,  255, 50 ],  # 1: Olivo
    [255, 0,   255],  # 2: Vite
    [255, 140, 0  ],  # 3: Agrumi
    [0,   102, 255],  # 4: Frutteto
    [255, 255, 0  ],  # 5: Grano
    [0,   255, 255],  # 6: Legumi
    [255, 0,   0  ],  # 7: Ortaggi
    [255, 255, 255],  # 8: Incolto
]
CLASS_COLORS_HEX = [
    "#000000", "#32ff32", "#ff00ff", "#ff8c00",
    "#0066ff", "#ffff00", "#00ffff", "#ff0000", "#ffffff"
]


MAX_CONCURRENT_INFERENCES = 4   # Semaphore per asyncio
TASK_TTL_SECONDS = 3600         # quanto teniamo i risultati in memoria
API_V1_PREFIX = "/api/v1"


# --- Water / Empty detection ---
WATER_NIR_THRESHOLD  = 400    # NIR estivo medio sotto questa soglia → probabile mare
WATER_RGB_THRESHOLD  = 300    # RGB visibile medio sotto questa soglia → scuro (mare/notte)
EMPTY_RATIO_THRESHOLD = 0.50  # Se >50% pixel a zero → tile corrotta o vuota
MIN_CONFIDENCE        = float(os.getenv("MIN_CONFIDENCE", "0.0"))  # 0 = disabilitato


POI_BBOX_SIZE_DEG  = 0.1    # 0.1° ≈ 11km — area download on-the-fly per POI
DOWNLOAD_MARGIN_PX = 32     # pixel extra scaricati e poi rimossi post-inferenza

'''Prior Scaling bayesiano: dividendo la probabilità 
predetta per il prior del training, 
compensando il bias del dataset'''
SICILIA_CLASS_PRIORS = [
    0.5093,  # Sfondo
    0.1022,  # Olivo
    0.0406,  # Vite
    0.0463,  # Agrumi
    0.0584,  # Frutteto
    0.1740,  # Grano   ← alto perché dataset spagnolo è cereal-heavy
    0.0211,  # Legumi
    0.0040,  # Ortaggi
    0.0442,  # Incolto
]
APPLY_PRIOR_CORRECTION = True


