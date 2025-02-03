# Installasjonsveiledning - Eiendomsmuligheter Platform

## Systemkrav

### Maskinvare
- CPU: Minimum 8 kjerner
- RAM: Minimum 32GB
- GPU: NVIDIA RTX 3080 eller bedre
- Lagring: 500GB SSD

### Programvare
- Ubuntu 22.04 LTS eller nyere
- Docker 24.0 eller nyere
- NVIDIA Container Toolkit
- Node.js 18.x eller nyere
- Python 3.10 eller nyere
- PostgreSQL 14 eller nyere

## Forberedelser

### 1. Systemoppdateringer
```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Installer nødvendige pakker
```bash
sudo apt install -y \
  docker.io \
  docker-compose \
  nodejs \
  npm \
  python3.10 \
  python3.10-venv \
  postgresql-14 \
  nginx
```

### 3. NVIDIA Setup
```bash
# Installer NVIDIA-drivere
sudo ubuntu-drivers autoinstall

# Installer NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
```

## Installasjon

### 1. Klon repositoriet
```bash
git clone https://github.com/Eiendomsmuligheter/platform.git
cd platform
```

### 2. Sett opp miljøvariabler
```bash
cp .env.example .env

# Rediger .env med dine verdier:
nano .env

# Eksempel på .env innhold:
POSTGRES_USER=eiendom
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=eiendomsmuligheter
NVIDIA_API_KEY=your_key_here
AUTH0_DOMAIN=your_domain
AUTH0_CLIENT_ID=your_client_id
STRIPE_SECRET_KEY=your_stripe_key
```

### 3. Bygg og start tjenestene
```bash
# Bygg alle containere
docker-compose build

# Start tjenestene
docker-compose up -d
```

### 4. Initialiser databasen
```bash
# Kjør migrasjoner
docker-compose exec backend alembic upgrade head

# Last inn standarddata
docker-compose exec backend python scripts/load_initial_data.py
```

## Konfigurasjon

### 1. Nginx Setup
```bash
# Kopier nginx konfigurasjon
sudo cp nginx/eiendomsmuligheter.conf /etc/nginx/sites-available/

# Aktiver siden
sudo ln -s /etc/nginx/sites-available/eiendomsmuligheter.conf /etc/nginx/sites-enabled/

# Test og start nginx
sudo nginx -t
sudo systemctl restart nginx
```

### 2. SSL-sertifikater
```bash
# Installer Certbot
sudo apt install -y certbot python3-certbot-nginx

# Generer sertifikater
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

### 3. Firewall Setup
```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 5432/tcp
sudo ufw enable
```

## Test installasjonen

### 1. Sjekk tjenestestatus
```bash
docker-compose ps
```

### 2. Sjekk logger
```bash
docker-compose logs -f
```

### 3. Verifiser tilgang
```bash
# Test frontend
curl http://localhost:3000

# Test backend
curl http://localhost:8000/api/health
```

## Vedlikehold

### 1. Backup
```bash
# Database backup
docker-compose exec db pg_dump -U eiendom > backup.sql

# Filbackup
tar -czf uploads.tar.gz uploads/
```

### 2. Oppdateringer
```bash
# Pull nyeste kodeendringer
git pull origin master

# Oppdater containers
docker-compose pull
docker-compose up -d
```

### 3. Overvåking
```bash
# Sjekk ressursbruk
docker stats

# Sjekk systemlogger
sudo journalctl -u docker.service
```

## Feilsøking

### 1. Vanlige problemer

#### Docker containers starter ikke
```bash
# Sjekk status
docker-compose ps

# Se logger
docker-compose logs [service_name]

# Restart tjenester
docker-compose restart [service_name]
```

#### Database tilkoblingsproblemer
```bash
# Sjekk PostgreSQL status
sudo systemctl status postgresql

# Sjekk loggfiler
tail -f /var/log/postgresql/postgresql-14-main.log
```

#### NVIDIA problemer
```bash
# Verifiser NVIDIA driver
nvidia-smi

# Sjekk container toolkit
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### 2. Logging
- Frontend logs: `/var/log/nginx/frontend.log`
- Backend logs: `/var/log/eiendomsmuligheter/backend.log`
- Database logs: `/var/log/postgresql/postgresql-14-main.log`

## Sikkerhet

### 1. Backup strategi
- Daglig database backup
- Ukentlig full systembackup
- Kryptering av sensitive data

### 2. Monitorering
- Prometheus metrics
- Grafana dashboards
- ELK stack for logging

### 3. Sikkerhetsoppdateringer
```bash
# Systemoppdateringer
sudo apt update
sudo apt upgrade -y

# Docker images
docker-compose pull
docker-compose up -d
```

## Support
- Teknisk support: support@eiendomsmuligheter.no
- Dokumentasjon: docs.eiendomsmuligheter.no
- Slack: #eiendomsmuligheter-dev

## Lisenser
- Sjekk at alle nødvendige lisenser er på plass:
  - NVIDIA Omniverse Enterprise
  - Auth0 subscription
  - Stripe API keys
  - PostgreSQL enterprise (hvis aktuelt)