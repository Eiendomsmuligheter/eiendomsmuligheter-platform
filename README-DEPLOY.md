# Eiendomsmuligheter Platform - Installasjons- og kjøreguide

Denne guiden forklarer hvordan du installerer og kjører Eiendomsmuligheter Platform på forskjellige måter. Du kan velge å kjøre hele plattformen med Docker eller kjøre backend og frontend separat uten Docker.

## Innholdsfortegnelse

1. [Forutsetninger](#forutsetninger)
2. [Installasjon](#installasjon)
3. [Kjøre med Docker](#kjøre-med-docker)
4. [Kjøre uten Docker](#kjøre-uten-docker)
5. [Utviklingsmiljø](#utviklingsmiljø)
6. [Feilsøking](#feilsøking)

## Forutsetninger

For å kjøre plattformen trenger du følgende programvare installert:

- Git
- Docker og Docker Compose (for kjøring med Docker)
- Python 3.9+ (for kjøring uten Docker)
- Node.js 16+ og Yarn/npm (for kjøring uten Docker)
- PostgreSQL (for kjøring uten Docker, alternativt kan SQLite brukes)

## Installasjon

1. Klon repositoriet:

```bash
git clone https://github.com/din-organisasjon/eiendomsmuligheter-platform.git
cd eiendomsmuligheter-platform
```

2. Opprett en `.env`-fil med nødvendige miljøvariabler:

```bash
cp .env.example .env
```

Rediger `.env`-filen med riktige verdier for ditt miljø.

## Kjøre med Docker

Den enkleste måten å kjøre hele plattformen på er med Docker Compose:

### Produksjonsmiljø

```bash
# Bygg og start produksjonsmiljø
docker-compose -f docker-compose.prod.yml up -d

# Sjekk status på tjenestene
docker-compose ps

# Stopp tjenestene
docker-compose -f docker-compose.prod.yml down
```

### Utviklingsmiljø

```bash
# Bygg og start utviklingsmiljø (med automatisk opplasting)
docker-compose up -d

# Se logger 
docker-compose logs -f

# Stopp tjenestene
docker-compose down
```

## Kjøre uten Docker

Hvis du foretrekker å kjøre backend og frontend separat uten Docker, følg disse trinnene:

### Backend (Python FastAPI)

1. Opprett og aktiver et virtuelt miljø:

```bash
# Opprett virtuelt miljø
python -m venv venv

# Aktiver virtuelt miljø - Windows
venv\Scripts\activate

# Aktiver virtuelt miljø - macOS/Linux
source venv/bin/activate
```

2. Installer avhengigheter:

```bash
pip install -r backend/requirements.txt
```

3. Kjør backend-tjenesten:

```bash
# Med SQLite database (enklest for utvikling)
python backend/run.py --init-db --use-sqlite --reload

# Med PostgreSQL database
python backend/run.py --init-db --reload

# Se hjelp for flere alternativer
python backend/run.py --help
```

Backend-API-et er nå tilgjengelig på http://localhost:8000/api.
Swagger-dokumentasjon er tilgjengelig på http://localhost:8000/docs.

### Frontend (React/TypeScript)

1. Naviger til frontend-mappen:

```bash
cd frontend
```

2. Installer avhengigheter:

```bash
# Med npm
npm install

# Med yarn
yarn install
```

3. Start utviklingsserveren:

```bash
# Med npm
npm start

# Med yarn
yarn start
```

Frontend-applikasjonen er nå tilgjengelig på http://localhost:3000.

## Utviklingsmiljø

For utvikling anbefales det å bruke følgende verktøy og metoder:

### Backend

- Bruk `--reload`-flagget for automatisk opplasting når koden endres
- Bruk SQLite for utvikling for å unngå avhengighet av PostgreSQL
- Sjekk API-dokumentasjonen på http://localhost:8000/docs

```bash
python backend/run.py --init-db --use-sqlite --reload
```

### Frontend

- Start i utviklingsmodus for hot-reloading:

```bash
cd frontend
yarn start
```

## Feilsøking

### Docker-relaterte problemer

1. **Docker-tjenester starter ikke**
   
   Sjekk logger for detaljer:
   ```bash
   docker-compose logs
   ```

2. **Port-konflikter**
   
   Hvis portene er i bruk, kan du endre dem i `docker-compose.yml`-filen.

### Backend-problemer

1. **Databasetilkoblingsproblemer**
   
   Sjekk at PostgreSQL kjører eller bruk SQLite:
   ```bash
   python backend/run.py --use-sqlite
   ```

2. **Manglende avhengigheter**
   
   Oppdater til nyeste avhengigheter:
   ```bash
   pip install -r backend/requirements.txt --upgrade
   ```

### Frontend-problemer

1. **Node-moduler mangler**
   
   Prøv å slette `node_modules`-mappen og installer på nytt:
   ```bash
   rm -rf node_modules
   yarn install
   ```

2. **API-tilkoblingsproblemer**
   
   Sjekk at API-URL-en er riktig konfigurert i `.env`-filen. 