FROM python:3.9-slim

# Installer systemavhengigheter (minimalt sett)
RUN apt-get update && apt-get install -y \
    postgresql-client \
    libpq-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Sett arbeidskatalog
WORKDIR /app

# Kopier requirements først for bedre cache-utnyttelse
COPY requirements.txt .

# Installer Python-avhengigheter med optimalisering
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Kopier resten av applikasjonen
COPY . .

# Opprett nødvendige mapper
RUN mkdir -p static/heightmaps static/textures static/models

# Sett miljøvariabler
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV NVIDIA_VISIBLE_DEVICES=none
ENV PORT=8000
ENV HOST=0.0.0.0
ENV WORKERS=4

# Eksponer porter
EXPOSE 8000

# Helsesjekk
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start applikasjonen
CMD ["python", "run.py", "--reload", "--workers", "4"]