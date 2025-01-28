# Eiendomsmuligheter Platform

En moderne plattform for analyse av eiendomsutvikling og byggesøknader.

## Funksjonalitet

- Last opp bilder eller skriv inn adresse for analyse
- 3D-visualisering med NVIDIA Omniverse
- Automatisk analyse av utviklingspotensial
- Sjekk av lokale regler og forskrifter
- Energianalyse og Enova-støttemuligheter
- Automatisk generering av byggesøknader og dokumentasjon

## Teknologi

### Frontend
- React med TypeScript
- Material-UI for brukergrensesnitt
- Three.js for 3D-visualisering
- NVIDIA Omniverse integrasjon

### Backend
- FastAPI (Python)
- Computer Vision og ML for bildeanalyse
- Integrasjon med kommunale systemer
- PDF-generering av dokumenter

## Installasjon

1. Klon repositoriet:
```bash
git clone https://github.com/Eiendomsmuligheter/eiendomsmuligheter-platform.git
cd eiendomsmuligheter-platform
```

2. Start med Docker Compose:
```bash
docker-compose up -d
```

Applikasjonen vil være tilgjengelig på:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API dokumentasjon: http://localhost:8000/docs

## Utvikling

### Frontend
```bash
cd frontend
npm install
npm start
```

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # eller .\venv\Scripts\activate på Windows
pip install -r requirements.txt
uvicorn src.main:app --reload
```

## Testing

### Frontend
```bash
cd frontend
npm test
```

### Backend
```bash
cd backend
pytest
```

## Dokumentasjon

- [API Dokumentasjon](docs/api/README.md)
- [Brukermanual](docs/user/README.md)
- [Teknisk Dokumentasjon](docs/technical/README.md)

## Lisens

Dette prosjektet er lisensiert under MIT-lisensen.

## Bidrag

Vi setter pris på bidrag! Se vår [bidragsguide](CONTRIBUTING.md) for mer informasjon.