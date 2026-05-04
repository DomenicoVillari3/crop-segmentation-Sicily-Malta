FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py . 
COPY weights/ /weights/
# ← RIMOSSO: COPY .env .  (le credenziali vengono dal docker-compose)

EXPOSE 8400

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8400"]