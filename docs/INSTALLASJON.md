# Installasjonsveiledning for Eiendomsmuligheter Platform

## Systemkrav

### Maskinvare
- CPU: Minimum 4 kjerner
- RAM: Minimum 16GB
- Lagring: Minimum 100GB SSD
- GPU: NVIDIA GPU med minimum 8GB VRAM for 3D-visualisering

### Programvare
- Ubuntu 22.04 LTS eller nyere
- Docker 24.0 eller nyere
- Docker Compose 2.x
- NVIDIA Container Toolkit
- Python 3.11 eller nyere
- Node.js 20.x eller nyere
- PostgreSQL 15.x eller nyere

## Installasjon

### 1. Klone repositoriet
```bash
git clone https://github.com/Eiendomsmuligheter/eiendomsmuligheter-platform.git
cd eiendomsmuligheter-platform
```

### 2. Installere avhengigheter

#### System-pakker
```bash
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-venv \
    postgresql \
    postgresql-contrib \
    nodejs \
    npm \
    nvidia-container-toolkit
```

#### Python-pakker
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Node.js-pakker
```bash
cd frontend
npm install
cd ..
```

### 3. Konfigurere miljøvariabler
```bash
cp .env.example .env
```

Rediger .env-filen med dine innstillinger:
```ini
# Database
POSTGRES_DB=eiendomsmuligheter
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost

# Auth0
AUTH0_DOMAIN=your-domain.auth0.com
AUTH0_CLIENT_ID=your_client_id
AUTH0_CLIENT_SECRET=your_client_secret

# Stripe
STRIPE_PUBLIC_KEY=your_stripe_public_key
STRIPE_SECRET_KEY=your_stripe_secret_key

# NVIDIA Omniverse
NVIDIA_CLIENT_ID=your_nvidia_client_id
NVIDIA_CLIENT_SECRET=your_nvidia_client_secret
```

### 4. Sette opp databasen
```bash
# Initialisere databasen
sudo -u postgres psql -c "CREATE DATABASE eiendomsmuligheter;"
sudo -u postgres psql -c "CREATE USER your_user WITH PASSWORD 'your_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE eiendomsmuligheter TO your_user;"

# Kjøre migrasjoner
alembic upgrade head
```

### 5. Bygge Docker-containere
```bash
# Bygge alle containere
docker-compose build

# Starte tjenestene
docker-compose up -d
```

### 6. NVIDIA Omniverse Setup
```bash
# Installere NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 7. SSL-sertifikat
```bash
# Installer Certbot
sudo apt-get install certbot python3-certbot-nginx

# Generer sertifikat
sudo certbot --nginx -d your-domain.no
```

## Verifisering av installasjon

### 1. Sjekk tjenestestatus
```bash
docker-compose ps
```

### 2. Kjør tester
```bash
# Backend-tester
pytest tests/

# Frontend-tester
cd frontend
npm test
```

### 3. Sjekk logger
```bash
docker-compose logs
```

## Sikkerhetshensyn

### Brannmur
```bash
# Konfigurere UFW
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 22/tcp
sudo ufw enable
```

### Database-backup
```bash
# Sett opp automatisk backup
sudo crontab -e

# Legg til følgende linje for daglig backup kl 02:00
0 2 * * * pg_dump -U your_user eiendomsmuligheter > /backup/eiendomsmuligheter_$(date +\%Y\%m\%d).sql
```

## Vedlikehold

### Oppdateringer
```bash
# Pull nye endringer
git pull origin master

# Oppdater avhengigheter
pip install -r requirements.txt
cd frontend && npm install && cd ..

# Rebuild og restart containere
docker-compose down
docker-compose build
docker-compose up -d
```

### Overvåkning
```bash
# Installer monitoring verktøy
sudo apt-get install -y prometheus grafana

# Start tjenestene
sudo systemctl enable prometheus grafana-server
sudo systemctl start prometheus grafana-server
```

## Feilsøking

### Vanlige problemer

#### 1. Database-tilkobling feiler
```bash
# Sjekk PostgreSQL status
sudo systemctl status postgresql

# Sjekk logger
sudo tail -f /var/log/postgresql/postgresql-15-main.log
```

#### 2. NVIDIA Omniverse problemer
```bash
# Verifiser GPU-tilgang
nvidia-smi

# Sjekk container GPU-tilgang
docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
```

#### 3. Web-server problemer
```bash
# Sjekk Nginx status og logger
sudo systemctl status nginx
sudo tail -f /var/log/nginx/error.log
```

## Support
For teknisk support, kontakt:
- E-post: tech.support@eiendomsmuligheter.no
- Telefon: 815 22 334
- Slack: #tech-support kanalen