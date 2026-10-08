# Crop Segmentation — Sicily & Malta

Segmentazione semantica delle colture agricole da immagini satellitari **Sentinel-2 L2A**, con una pipeline pensata per l'analisi di aree in Sicilia e a Malta.

Il progetto, presentato nell'interfaccia come **Smart Food**, combina un modello **Prithvi EO v2**, un'API **FastAPI**, una cache **MinIO** e una demo **Gradio**. A partire da un punto geografico o da una bounding box, restituisce un overlay della segmentazione, la distribuzione delle superfici e le statistiche NDVI per quattro stagioni.

> Il repository contiene il codice di inferenza e dei servizi. I pesi del modello devono essere forniti separatamente; non sono inclusi nel repository.

## Funzionalità

- Analisi da punto di interesse (latitudine e longitudine WGS84) o da bounding box.
- Recupero dei cubi satellitari da MinIO e download da STAC quando i dati non sono in cache.
- Input multistagionale: quattro osservazioni e sei bande Sentinel-2.
- Inferenza su chip da **224 × 224 pixel**, con overlap del **25%** e accumulo delle predizioni pesate dalla confidenza nelle zone sovrapposte.
- Filtro acqua basato sul NIR estivo e filtro di confidenza opzionale.
- Overlay PNG con bordi privi di dati resi trasparenti.
- Superfici per classe in ettari e percentuali.
- NDVI stagionale con media, deviazione standard, minimo e massimo.
- Elaborazione in background con polling dello stato.
- Demo web con preset per Sicilia e Malta.

## Pipeline

1. L'API riceve coordinate e anno e restituisce un `task_id`.
2. `DataResolver` cerca una tile in MinIO; in assenza di dati utilizzabili richiede un cubo a `STACDownloader`.
3. Il downloader seleziona una scena per stagione dal catalogo Earth Search, collection `sentinel-2-l2a`, e ricampiona le bande su una griglia UTM a **10 m/pixel**.
4. `ModelService` normalizza l'input e applica il backbone `terratorch_prithvi_eo_v2_100_tl` con il decoder residuo definito in `architecture.py`.
5. `InferenceEngine` ricompone la maschera e applica i filtri.
6. `postprocess.py` genera overlay, superfici e serie NDVI.

Il cubo ha forma `(T, C, H, W) = (4, 6, H, W)`. Le bande, nell'ordine atteso, sono:

| Indice | Banda | Asset STAC | Descrizione |
| ---: | --- | --- | --- |
| 0 | B02 | `blue` | Blu |
| 1 | B03 | `green` | Verde |
| 2 | B04 | `red` | Rosso |
| 3 | B08 | `nir08` | Infrarosso vicino |
| 4 | B11 | `swir16` | SWIR 1 |
| 5 | B12 | `swir22` | SWIR 2 |

Le finestre effettive di selezione sono definite in `stac_downloader.py`:

| Stagione | Finestra nell'anno richiesto |
| --- | --- |
| `winter` | 1 gennaio – ultimo giorno di febbraio |
| `spring` | 15 aprile – 30 maggio |
| `summer` | 1 luglio – 15 agosto |
| `autumn` | 1 ottobre – 15 novembre |

La soglia iniziale di copertura nuvolosa è **< 20%**, con fallback a **< 40%**. L'analisi richiede una scena disponibile per ogni stagione; la selezione usa la copertura nuvolosa della scena e non una maschera nuvole per pixel.

## Classi

| ID | Classe | Colore |
| ---: | --- | --- |
| 0 | Sfondo | `#000000` |
| 1 | Olivo | `#32ff32` |
| 2 | Vite | `#ff00ff` |
| 3 | Agrumi | `#ff8c00` |
| 4 | Frutteto | `#0066ff` |
| 5 | Cereali | `#ffff00` |
| 6 | Legumi | `#00ffff` |
| 7 | Ortaggi | `#ff0000` |
| 8 | Incolto | `#ffffff` |

La classe 0 è esclusa dalle statistiche delle superfici e dall'NDVI. L'endpoint `/satellite/classes` espone le classi da 1 a 8.

## Avvio con Docker

### Prerequisiti

- Git, Docker e Docker Compose.
- Una GPU NVIDIA e il supporto GPU configurato per Docker: il Compose incluso riserva una GPU al servizio API.
- Accesso Internet per scaricare immagini, dipendenze e dati STAC. Il catalogo STAC viene aperto anche all'avvio dell'API.
- Uno `state_dict` PyTorch compatibile con il backbone e il decoder del progetto.

Il Dockerfile API usa `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`; la demo usa `python:3.11-slim`.

### 1. Clona il repository e aggiungi i pesi

```bash
git clone https://github.com/DomenicoVillari3/crop-segmentation-Sicily-Malta.git
cd crop-segmentation-Sicily-Malta
mkdir -p weights
```

Copia il checkpoint compatibile in `weights/model.pth`. Questo nome è un esempio: puoi usare un altro nome aggiornando la configurazione.

Il caricamento usa `load_state_dict(..., strict=True)` e rimuove i prefissi `_orig_mod.`. Il file deve contenere direttamente lo state dictionary compatibile, non un checkpoint con una struttura diversa.

### 2. Crea il file `.env`

Nella radice del progetto:

```dotenv
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=smartfood_local
MINIO_SECRET_KEY=sostituisci_con_una_password_locale
MINIO_BUCKET_NAME=sicily-sentinel-data
MINIO_API_PORT=9000
MINIO_CONSOLE_PORT=9001

MODEL_WEIGHTS_PATH=weights/model.pth
API_BASE_URL=http://localhost:8400
```

Sostituisci la password prima di avviare i servizi.

Il Compose imposta automaticamente `MINIO_ENDPOINT=minio:9000` per l'API e `API_BASE_URL=http://api:8400` per la demo. Per i pesi, antepone `/` al valore di `MODEL_WEIGHTS_PATH`: usa quindi `weights/model.pth`, senza slash iniziale.

Il Dockerfile copia `weights/` in `/weights/`: la directory deve esistere prima della build. Se cambi il checkpoint, ricostruisci l'immagine API.

### 3. Avvia i servizi

```bash
docker compose up -d --build
docker compose logs -f api
```

Il Compose avvia quattro servizi: `minio`, `createbuckets`, `api` e `demo`. Il servizio `createbuckets` crea il bucket; attendi il completamento prima di inviare un'analisi.

| Servizio | Indirizzo locale |
| --- | --- |
| Demo Gradio | http://localhost:7860 |
| Swagger UI | http://localhost:8400/satellite/docs |
| ReDoc | http://localhost:8400/satellite/redoc |
| Stato API | http://localhost:8400/satellite/health |
| Console MinIO | http://localhost:9001 |

Le porte MinIO cambiano se modifichi `MINIO_API_PORT` o `MINIO_CONSOLE_PORT`.

Per fermare lo stack:

```bash
docker compose down
```

I dati MinIO sono conservati nella directory locale `minio_data/`.

## Avvio locale dell'API e della demo

In un ambiente Python con PyTorch e le dipendenze geospaziali compatibili installati:

```bash
python -m pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8400
```

Il file `requirements.txt` non include un requisito attivo per PyTorch: verifica che sia già disponibile nell'ambiente. MinIO deve essere raggiungibile, il bucket deve esistere e `.env` deve indicare il percorso locale del checkpoint.

In un secondo terminale, con lo stesso ambiente:

```bash
python demo_gui_endpoints.py
```

Il modello seleziona CUDA quando disponibile e altrimenti usa la CPU. Per eseguire lo stack Docker senza GPU occorre rimuovere la prenotazione NVIDIA dal servizio `api`; i tempi di inferenza su CPU possono essere maggiori.

## API

Il prefisso configurato è `/satellite`.

| Metodo | Percorso | Risultato |
| --- | --- | --- |
| POST | `/satellite/point` | Avvia un'analisi da punto; HTTP 202 e `task_id` |
| POST | `/satellite/bbox` | Avvia un'analisi da bounding box; HTTP 202 e `task_id` |
| GET | `/satellite/{task_id}/status` | Stato, progresso e risultati disponibili |
| GET | `/satellite/{task_id}/image` | Overlay PNG |
| GET | `/satellite/{task_id}/ndvi` | Statistiche NDVI stagionali |
| GET | `/satellite/{task_id}/legend` | Legenda con ettari e percentuali |
| GET | `/satellite/classes` | Catalogo statico delle classi, escluso lo sfondo |
| GET | `/satellite/health` | Diagnostica modello, device e stato MinIO |

Gli anni attualmente accettati sono **2017–2025**, con default **2023**. Il 2026 non è incluso nella configurazione corrente.

### Esempio: analisi di un punto

```bash
curl -X POST http://localhost:8400/satellite/point \
  -H "Content-Type: application/json" \
  -d '{"lat":37.380,"lon":14.910,"year":2023}'
```

La risposta contiene `task_id`, `status: "pending"` e `progress: 0`; gli altri campi del modello di risposta possono essere `null`.

Copia il valore restituito e interroga lo stato:

```bash
TASK_ID="inserisci_il_task_id_restituito"
curl "http://localhost:8400/satellite/$TASK_ID/status"
```

Gli stati sono `pending`, `running`, `completed` e `failed`. Ripeti il polling fino a `completed`; in caso di `failed`, consulta il campo `error`.

Quando l'analisi è completata:

```bash
curl "http://localhost:8400/satellite/$TASK_ID/image" -o overlay.png
curl "http://localhost:8400/satellite/$TASK_ID/ndvi"
curl "http://localhost:8400/satellite/$TASK_ID/legend"
```

Le richieste ai risultati prima del completamento restituiscono HTTP 202; un task inesistente restituisce HTTP 404.

### Esempio: analisi di una bounding box

L'ordine geografico è `[min_lon, min_lat, max_lon, max_lat]`. Entrambi i valori massimi devono essere maggiori dei rispettivi minimi.

> Il percorso BBox presenta un problema noto nella ricerca della cache, descritto più avanti. Per una prima verifica dello stack usa l'endpoint POI.

```bash
curl -X POST http://localhost:8400/satellite/bbox \
  -H "Content-Type: application/json" \
  -d '{"min_lon":14.395,"min_lat":35.880,"max_lon":14.455,"max_lat":35.940,"year":2023}'
```

La procedura di polling e recupero dei risultati è identica a quella del punto.

## Interpretazione dei risultati

- `source`: `minio` se il cubo proviene dalla cache, `stac` se è stato scaricato.
- `year_used`: anno effettivamente utilizzato.
- `bbox`: area geografica del cubo risolto.
- `class_stats`: superfici per le classi presenti, ordinate per ettari decrescenti.
- `total_ha`: somma delle superfici classificate, **escluso lo sfondo**.
- `percentage`: quota sul totale dei pixel non-sfondo, non sull'intera bounding box.
- `ndvi_series`: quattro punti stagionali, calcolati sui pixel con classe maggiore di 0, inclusa la classe Incolto.

Le superfici sono calcolate assumendo **100 m² per pixel**, cioè **0,01 ha**. L'NDVI usa `(NIR - Red) / (NIR + Red + 1e-6)`. L'overlay combina l'RGB estivo e la segmentazione con opacità predefinita del **55%**.

Per le richieste POI, un hit in cache usa un crop fino a **800 × 800 pixel**; il download STAC usa un bbox centrato sul punto di **0,1° per lato**. L'area restituita può quindi differire tra cache e primo download: consulta sempre `bbox`.

## Configurazione

| Variabile ambiente | Uso |
| --- | --- |
| `MINIO_ENDPOINT` | Host e porta MinIO, senza schema; il client corrente usa HTTP |
| `MINIO_ACCESS_KEY` | Credenziale MinIO, obbligatoria |
| `MINIO_SECRET_KEY` | Credenziale MinIO, obbligatoria |
| `MINIO_BUCKET_NAME` | Bucket per cubi e anteprime |
| `MODEL_WEIGHTS_PATH` | Percorso dello state dictionary PyTorch |
| `API_BASE_URL` | URL base dell'API usato dalla demo, senza `/satellite` |
| `MINIO_API_PORT` | Porta host MinIO nel Compose; default 9000 |
| `MINIO_CONSOLE_PORT` | Porta host della console nel Compose; default 9001 |
| `MIN_CONFIDENCE` | Soglia opzionale; default 0, filtro disabilitato |

Per usare `MIN_CONFIDENCE` nel container API aggiungila al blocco `environment` del servizio: il Compose corrente non la inoltra.

Le altre impostazioni sono costanti in `config.py`: bande, classi, normalizzazione, overlap, soglie acqua, anni supportati e limite di inferenze concorrenti, pari a **4**. La correzione delle predizioni con i prior di classe è attiva tramite `APPLY_PRIOR_CORRECTION=True`.

## Struttura del codice

| File | Responsabilità |
| --- | --- |
| `main.py` | Applicazione FastAPI, task e endpoint |
| `schemas.py` | Validazione e modelli di risposta Pydantic |
| `config.py` | Variabili ambiente e parametri della pipeline |
| `architecture.py` | Decoder residuo e architettura di segmentazione |
| `model_service.py` | Caricamento del modello, normalizzazione e inferenza dei chip |
| `inference_engine.py` | Sliding window, ricomposizione e filtri |
| `data_resolver.py` | Risoluzione dei dati e ritagli geografici |
| `stac_downloader.py` | Selezione stagionale e download Sentinel-2 |
| `minio_store.py` | Ricerca, download e upload nella cache |
| `postprocess.py` | Overlay, statistiche e NDVI |
| `demo_gui_endpoints.py` | Interfaccia Gradio |
| `Dockerfile` / `Dockerfile.demo` | Immagini API e demo |
| `docker-compose.yaml` | Orchestrazione dei servizi |
| `requirements.txt` | Dipendenze Python |

I cubi sono salvati in MinIO sotto `raw_cubes/year=YYYY/`; le anteprime RGB sotto `rgb_images/year=YYYY/`. I metadata includono bbox, anno e forma del cubo.

## Limiti dell'implementazione corrente

- **Percorso BBox:** in `MinioStore.find_tile_by_bbox()` il controllo di contenimento è fuori dal ciclo di ricerca. Con una cache vuota può accedere a variabili non inizializzate; con più tile valuta solo l'ultima. Inoltre `resolve_from_bbox()` non passa l'anno richiesto alla ricerca. Questi punti vanno corretti per rendere affidabili le analisi BBox.
- **Task in memoria:** stato e risultati sono mantenuti nel processo API e si perdono al riavvio. La costante `TASK_TTL_SECONDS` è definita ma non è applicata; non è implementata una pulizia automatica. Usa un singolo worker finché il task store non è condiviso.
- **Dati e validazione:** la disponibilità delle quattro scene dipende da area, anno e nuvole. Il repository non contiene script di training, dataset di valutazione o metriche che consentano di quantificare l'accuratezza in Sicilia e a Malta.
- **Demo:** il polling termina dopo 300 secondi; il timeout della demo non annulla il task API.
- **Deploy:** il Compose rende il bucket pubblico tramite `mc anonymous set public`; l'API non implementa autenticazione e abilita CORS per tutte le origini. Questa configurazione va rivista prima di esporre il servizio fuori dall'ambiente di sviluppo.
- **Diagnostica MinIO:** il controllo `health` istanzia il client ma non verifica con una richiesta la raggiungibilità del bucket.

## Risoluzione dei problemi

| Problema | Verifica |
| --- | --- |
| Credenziali MinIO mancanti | Definisci `MINIO_ACCESS_KEY` e `MINIO_SECRET_KEY` nell'ambiente o in `.env` |
| Build fallita su `COPY weights/` | Crea la directory e inserisci il checkpoint prima della build |
| Pesi non trovati o incompatibili | Controlla percorso e compatibilità dello state dictionary con l'architettura |
| API non parte con Docker | Controlla i log, il supporto GPU e l'accesso al catalogo STAC |
| Nessun dato disponibile | Controlla coordinate, anno e disponibilità di una scena per ogni stagione |
| Analisi BBox fallita | Verifica il problema della cache descritto nei limiti |
| Timeout nella demo | Consulta `/status` e i log API; prova un'area più piccola |

## Licenza

Nel repository non è presente un file `LICENSE`. Le condizioni di riutilizzo del codice devono essere definite dall'autore; verifica separatamente quelle applicabili ai pesi del modello e ai dati satellitari.

## Autore

[Domenico Villari](https://github.com/DomenicoVillari3)

Repository: [crop-segmentation-Sicily-Malta](https://github.com/DomenicoVillari3/crop-segmentation-Sicily-Malta)

