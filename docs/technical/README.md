# Teknisk Dokumentasjon - Eiendomsmuligheter Platform

## Arkitektur

### Frontend
- **Framework**: React med TypeScript
- **UI Bibliotek**: Material-UI
- **3D Visualisering**: NVIDIA Omniverse SDK + Three.js
- **State Management**: React Query og Context API
- **Form Handling**: Formik med Yup validering

### Backend
- **Framework**: FastAPI (Python 3.9+)
- **Database**: PostgreSQL med SQLAlchemy ORM
- **Cache**: Redis
- **Task Queue**: Celery med RabbitMQ
- **AI/ML**: TensorFlow og OpenCV

### Infrastruktur
- **Container**: Docker
- **Orchestration**: Kubernetes
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana
- **Logging**: ELK Stack

## Systemkrav

### Frontend
```bash
Node.js >= 16.x
npm >= 8.x
```

### Backend
```bash
Python >= 3.9
PostgreSQL >= 13
Redis >= 6.2
RabbitMQ >= 3.9
```

### Infrastruktur
```bash
Docker >= 20.10
Kubernetes >= 1.22
Helm >= 3.8
```

## Installasjon

### Lokal Utvikling

1. Klon repositoriet:
```bash
git clone https://github.com/Eiendomsmuligheter/eiendomsmuligheter-platform.git
cd eiendomsmuligheter-platform
```

2. Frontend oppsett:
```bash
cd frontend
npm install
npm start
```

3. Backend oppsett:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # eller .\venv\Scripts\activate på Windows
pip install -r requirements.txt
uvicorn src.main:app --reload
```

4. Database oppsett:
```bash
docker-compose up -d postgres redis rabbitmq
alembic upgrade head
```

### Produksjonsdeploy

1. Bygg Docker images:
```bash
docker build -t eiendomsmuligheter/frontend:latest frontend/
docker build -t eiendomsmuligheter/backend:latest backend/
```

2. Deploy til Kubernetes:
```bash
kubectl apply -f k8s/
```

## API Dokumentasjon

Se [API Documentation](../api/README.md) for detaljert API-dokumentasjon.

## Sikkerhet

### Autentisering
- JWT-basert autentisering
- Auth0 integrasjon
- Role-based access control (RBAC)

### API Sikkerhet
- Rate limiting
- CORS konfigurasjon
- Input validering
- SQL injection beskyttelse

### Data Sikkerhet
- Kryptert datalagring
- Sikker kommunikasjon (HTTPS)
- Regelmessig backup
- GDPR compliance

## Databasemodeller

### Property
- Eiendomsinformasjon
- Geografisk data
- Tekniske detaljer
- Reguleringsinfo

### Analysis
- Analyseresultater
- AI/ML prediksjoner
- Utviklingspotensial
- Energiberegninger

### Document
- Byggesøknader
- Tekniske tegninger
- Rapporter
- BIM-modeller

## Integrasjoner

### Kommunale Systemer
- Drammen kommune API
- Kartverket API
- Plan- og bygningsetaten API

### Eksterne Tjenester
- NVIDIA Omniverse
- Stripe betalingsløsning
- Enova API
- Auth0

## Feilhåndtering

### Logging
- Strukturert logging med ELK Stack
- Error tracking med Sentry
- Performance monitoring med New Relic

### Error Codes
- 1xxx: Validering/Input feil
- 2xxx: Prosesseringsfeil
- 3xxx: Eksterne API feil
- 4xxx: System/Infrastruktur feil

## Testing

### Enhetstesting
- Jest for frontend
- pytest for backend
- Coverage krav: minimum 80%

### Integrasjonstesting
- Cypress for E2E testing
- Postman/Newman for API testing
- GitHub Actions for CI testing

### Performance Testing
- k6 for load testing
- Lighthouse for frontend performance
- pgBench for database performance

## Vedlikehold

### Backup
- Daglig full backup
- Kontinuerlig WAL archiving
- 30 dagers oppbevaring

### Monitoring
- System metrics med Prometheus
- Application performance med Grafana
- Error tracking med Sentry

### Deploystrategi
- Blue/Green deployment
- Canary releases
- Automatic rollback ved feil

## Skalerbarhet

### Database
- Connection pooling
- Read replicas
- Partitioning strategi

### Applikasjon
- Horizontal pod autoscaling
- Cache strategi
- Task queue distribusjon

### Storage
- S3-kompatibel objektlagring
- CDN for statiske filer
- Database backup strategi