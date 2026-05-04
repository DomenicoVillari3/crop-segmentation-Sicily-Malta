FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Evita QUALSIASI prompt interattivo durante apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Rome

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdal-dev \
        gdal-bin \
        git \
        curl \
        tzdata \
        libjpeg-dev \
        libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=180 --retries=5 -r requirements.txt

COPY *.py .
COPY weights/ /weights/

EXPOSE 8400
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8400"]