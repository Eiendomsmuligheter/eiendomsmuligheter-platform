# Utviklerdokumentasjon - Eiendomsmuligheter Platform

## Teknisk Oversikt

### Arkitektur
Eiendomsmuligheter Platform er bygget på en moderne mikroservicearkitektur med følgende hovedkomponenter:

```
[Frontend (React/Next.js)] ←→ [API Gateway (nginx)]
            ↓
[Backend Services (Python/FastAPI)]
            ↓
[Core Modules] ←→ [AI Modules] ←→ [NVIDIA Omniverse]
            ↓
[Database (PostgreSQL/TimescaleDB)]
```

### Teknologistack
- **Frontend**
  - React/Next.js
  - TypeScript
  - TailwindCSS
  - Three.js/React Three Fiber
  - NVIDIA Omniverse Connect

- **Backend**
  - Python 3.11+
  - FastAPI
  - SQLAlchemy
  - Pydantic
  - Celery

- **AI/ML**
  - PyTorch
  - TensorFlow
  - OpenCV
  - Scikit-learn
  - NVIDIA CUDA

- **Database**
  - PostgreSQL
  - TimescaleDB
  - Redis

- **Infrastruktur**
  - Docker
  - Kubernetes
  - Terraform
  - AWS

## Utviklingsmiljø

### Forutsetninger
- Docker og Docker Compose
- Python 3.11+
- Node.js 18+
- NVIDIA GPU med CUDA støtte (for 3D og AI)
- AWS CLI
- kubectl

### Oppsett
1. Klon repositoriet:
```bash
git clone https://github.com/Eiendomsmuligheter/eiendomsmuligheter-platform.git
cd eiendomsmuligheter-platform
```

2. Installer avhengigheter:
```bash
# Backend
python -m venv venv
source venv/bin/activate  # eller .\venv\Scripts\activate på Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Frontend
cd frontend
npm install
```

3. Sett opp miljøvariabler:
```bash
cp .env.example .env
# Rediger .env med dine innstillinger
```

4. Start utviklingsmiljø:
```bash
docker-compose up -d
```

## Kodestruktur

### Backend
```
backend/
├── app/
│   ├── api/            # API endepunkter
│   ├── core/           # Kjernefunksjonalitet
│   ├── db/             # Databasemodeller og migreringer
│   ├── services/       # Forretningslogikk
│   └── utils/          # Hjelpefunksjoner
├── tests/              # Tester
└── alembic/            # Databasemigreringer
```

### Frontend
```
frontend/
├── components/         # React komponenter
├── pages/             # Next.js sider
├── hooks/             # Custom React hooks
├── styles/            # CSS/SCSS filer
└── utils/             # Hjelpefunksjoner
```

### AI Moduler
```
ai_modules/
├── floor_plan_analyzer/    # Plantegningsanalyse
├── image_processor/        # Bildebehandling
├── property_analyzer/      # Eiendomsanalyse
└── model_trainer/          # ML-modelltrening
```

## Testing

### Enhetstester
```bash
# Backend
pytest tests/unit

# Frontend
cd frontend
npm test
```

### Integrasjonstester
```bash
pytest tests/integration
```

### End-to-end Tester
```bash
npm run cypress
```

## Deployment

### Staging
```bash
make deploy-staging
```

### Produksjon
```bash
make deploy-prod
```

## CI/CD Pipeline

GitHub Actions håndterer kontinuerlig integrasjon og deployment:

1. **På Pull Request**:
   - Kjør linting
   - Kjør enhetstester
   - Bygg Docker images
   - Kjør sikkerhetsskanning

2. **På Merge til Main**:
   - Bygg og push Docker images
   - Deploy til staging
   - Kjør integrasjonstester
   - Ved suksess, deploy til produksjon

## Monitoriering

### Metrics
- Prometheus for metrics
- Grafana for visualisering
- Custom dashboards for:
  - API ytelse
  - ML-modell nøyaktighet
  - Brukerengasjement
  - Systemressurser

### Logging
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Strukturert logging med JSON format
- Log levels: DEBUG, INFO, WARNING, ERROR

### Alerting
- PagerDuty integrasjon
- Custom alerting rules
- Automatisk eskalering

## Sikkerhetsrutiner

### Kodegjennomgang
- Mandatory code review
- Automatisk sikkerhetsanalyse
- Dependency scanning
- SAST og DAST

### Data Sikkerhet
- Kryptering i hvile og transit
- Regelmessig backup
- Disaster recovery plan

### Tilgangskontroll
- JWT-basert autentisering
- RBAC (Role-Based Access Control)
- API nøkkelhåndtering

## Bidrag

### Kodestandard
- PEP 8 for Python
- ESLint/Prettier for JavaScript
- Type hints og dokumentasjon påkrevd
- Tester påkrevd for ny funksjonalitet

### PR Prosess
1. Fork repositoriet
2. Opprett feature branch
3. Implementer endringer
4. Skriv/oppdater tester
5. Oppdater dokumentasjon
6. Send pull request

### Versjonering
- Semantic Versioning (MAJOR.MINOR.PATCH)
- Changelog for hver release
- Release notes i GitHub

## API Integrasjon
Se [API Dokumentasjon](../api/README.md)

## SDK-er
- [Python SDK](https://github.com/Eiendomsmuligheter/python-sdk)
- [JavaScript SDK](https://github.com/Eiendomsmuligheter/js-sdk)
- [.NET SDK](https://github.com/Eiendomsmuligheter/dotnet-sdk)

## Ressurser
- [Intern Wiki](https://wiki.eiendomsmuligheter.no)
- [Arkitekturdokumentasjon](./architecture.md)
- [Sikkerhetsdokumentasjon](./security.md)
- [Driftsdokumentasjon](./operations.md)