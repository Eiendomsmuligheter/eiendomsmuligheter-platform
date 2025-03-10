# Bruk en lettere base image
FROM python:3.9-slim

# Installer systemavhengigheter (minimalt sett)
RUN apt-get update && apt-get install -y \
    postgresql-client \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Sett arbeidskatalog
WORKDIR /app

# Kopier requirements først for bedre cache-utnyttelse
COPY backend/requirements.txt .

# Installer Python-avhengigheter
RUN pip install --no-cache-dir -r requirements.txt

# Kopier resten av applikasjonen
COPY . .

# Opprett nødvendige kataloger
RUN mkdir -p uploads

# Sett miljøvariabler
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=none

# Eksponer porter
EXPOSE 8000

# Start applikasjonen
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]