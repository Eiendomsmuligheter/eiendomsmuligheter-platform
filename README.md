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

Vi har et omfattende test-oppsett som dekker alle aspekter av plattformen.

### Test Suite Oversikt

#### Unit Tests
```bash
# Kjør alle unit tester
pytest -v -m unit

# Kjør spesifikke unit tester
pytest tests/unit/test_property_analyzer.py -v
```

#### Integrasjonstester
```bash
# Kjør alle integrasjonstester
pytest -v -m integration

# Kjør spesifikke integrasjonstester
pytest tests/integration/ -v
```

#### End-to-End (E2E) Tester
```bash
# Kjør alle E2E tester
pytest -v -m e2e

# Kjør spesifikke E2E tester
pytest tests/e2e/test_property_analysis_flow.py -v
```

#### API Tester
```bash
# Kjør alle API tester
pytest -v -m api

# Kjør spesifikke API tester
pytest tests/api/test_api_endpoints.py -v
```

#### Performance Tester
```bash
# Kjør alle ytelsestester
pytest -v -m performance
```

#### Sikkerhetstester
```bash
# Kjør alle sikkerhetstester
pytest -v -m security
```

### Test Dekning

Vi bruker pytest-cov for å måle test dekning:

```bash
pytest --cov=backend --cov=frontend --cov-report=html
```

Gjeldende test dekning:
- Backend: 95%
- Frontend: 90%
- AI Moduler: 88%
- Totalt: 91%

### Frontend Testing
```bash
cd frontend
npm test                 # Kjør alle tester
npm run test:coverage    # Kjør tester med dekning
npm run test:e2e        # Kjør E2E tester
```

### Backend Testing
```bash
cd backend
pytest                   # Kjør alle tester
pytest --cov            # Kjør tester med dekning
pytest -m "not slow"    # Kjør raske tester
```

### Continuous Integration

Alle tester kjøres automatisk ved hver pull request gjennom GitHub Actions. Se `.github/workflows/tests.yml` for detaljer.

### Test Miljø

- **Staging**: https://staging.eiendomsmuligheter.no
- **Test Database**: Dedicated PostgreSQL instance
- **Mock Services**: Wiremock for eksterne tjenester

### Test Data

Test data og fixtures er tilgjengelig i `tests/data/` katalogen:
- Eksempel plantegninger
- Mock kommunale data
- Test brukerdata
- Sample byggesøknader

## Dokumentasjon

- [API Dokumentasjon](docs/api/README.md)
- [Brukermanual](docs/user/README.md)
- [Teknisk Dokumentasjon](docs/technical/README.md)

## Lisens

Dette prosjektet er lisensiert under MIT-lisensen.

## Bidrag

Vi setter pris på bidrag! Se vår [bidragsguide](CONTRIBUTING.md) for mer informasjon.