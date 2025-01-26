# AI Assistant Startup Instructions - Eiendomsmuligheter Platform

## VIKTIG: Les dette først!
Dette er dine instruksjoner når en ny samtale starter. Du skal:
1. Generer umiddelbart en statusrapport
2. Informer brukeren om prosjektets status
3. Vær klar til å fortsette arbeidet der det ble avsluttet

## Prosjektstruktur og Filplassering

### Hovedmapper:
- 📂 /home/computeruse/eiendomsmuligheter/ (Hovedmappe)
  - 📂 ai_modules/ (AI-relaterte moduler)
  - 📂 frontend/ (Web-grensesnitt)
  - 📂 backend/ (Server og API)
  - 📂 data/ (Datahåndtering)
  - 📂 docs/ (Dokumentasjon)

### Viktige Filer:
1. 🐍 app.py - Hovedapplikasjon
2. 📊 rental_analyzer.py - Leieinntektsanalyse
3. 📋 regulations_handler.py - Reguleringshåndtering
4. 🌐 index.html - Hovedgrensesnitt
5. 📝 README.md - Prosjektdokumentasjon

### Automatiserte Rapporter:
- 📈 daily_report.md - Daglig statusrapport
- 🔍 project_status.md - Prosjektstatus
- 📊 metrics_report.md - Kodestatistikk
- 🔗 dependencies.md - Avhengigheter

## Prosjektversjonering
- GitHub Repository: https://github.com/Eiendomsmuligheter/eiendomsmuligheter-platform.git
- Branch: main
- Siste commit: [DATO]
- Auto-backup: Hver 5. minutt

## Teknologi Stack
1. Backend:
   - Python 3.9+
   - FastAPI
   - SQLAlchemy
   - NumPy/Pandas

2. Frontend:
   - React
   - Material-UI
   - Chart.js
   - Three.js (3D-visualisering)

3. AI/ML:
   - TensorFlow
   - scikit-learn
   - NVIDIA CUDA
   - OpenAI API

4. Database:
   - PostgreSQL
   - Redis (caching)

## Arbeidsmetodikk for AI-assistent

### Ved oppstart av ny samtale:
1. Les gjennom alle statusfiler
2. Generer umiddelbart en statusrapport
3. Informer om eventuelle ventende oppgaver
4. Sjekk etter nye commits/endringer

### Under arbeid:
1. Oppdater project_log.md ved endringer
2. Følg git-workflow for endringer
3. Kjør tester før commits
4. Dokumenter alle API-endringer

### Ved avslutning:
1. Lagre all progress
2. Oppdater TODO.md
3. Push endringer til GitHub
4. Generer sluttrapport

## Prosjektmål og Tidslinjer

### Hovedmål:
1. AI-drevet eiendomsanalyse
2. 3D-visualisering med NVIDIA RTX
3. Automatisk reguleringssjekk
4. Avansert ROI-kalkulator

### Tidslinjer:
- Q1 2025: Frontend ferdigstilling
- Q2 2025: Backend API komplett
- Q3 2025: AI-modeller integrert
- Q4 2025: Full lansering

## Kommandoer og Verktøy

### Git-kommandoer:
```bash
git pull  # Oppdater lokalt repo
git add .  # Stage endringer
git commit -m "beskrivelse"  # Commit endringer
git push  # Push til GitHub
```

### Docker-kommandoer:
```bash
docker-compose up  # Start miljø
docker-compose down  # Stopp miljø
docker logs -f container_name  # Se logger
```

### Python-miljø:
```bash
source venv/bin/activate  # Aktiver virtuelt miljø
pip install -r requirements.txt  # Installer pakker
python app.py  # Kjør applikasjon
```

## Prosjektstatus og Fremdrift

### Nåværende Status:
- Frontend: 75% fullført
- Backend: 80% fullført
- AI-moduler: 60% fullført
- Testing: 70% dekning
- Dokumentasjon: 85% fullført

### Prioriterte Oppgaver:
1. Implementere sanntids eiendomsanalyse
2. Forbedre 3D-rendering ytelse
3. Optimalisere AI-prediksjoner
4. Utvide test-dekning

## Logging og Rapportering

### Loggstruktur:
1. application.log - Applikasjonslogger
2. error.log - Feilmeldinger
3. ai_operations.log - AI-operasjoner
4. user_actions.log - Brukerhandlinger

### Rapporteringsrutiner:
- Daglig statusrapport genereres automatisk
- Ukentlig fremgangsrapport hver søndag
- Månedlig prosjektgjennomgang
- Kvartalsvis måloppfølging

## AI-assistentens Ansvar

### Hovedoppgaver:
1. Kodegjennomgang og optimalisering
2. Dokumentasjonshåndtering
3. Feilsøking og debugging
4. Implementering av nye funksjoner

### Retningslinjer:
1. Følg etablerte kodestandarter
2. Dokumenter alle endringer
3. Test grundig før implementering
4. Hold oversikt over teknisk gjeld

### Sikkerhetsrutiner:
1. Sjekk for sikkerhetsproblemer
2. Følg OWASP-retningslinjer
3. Implementer sikker koding
4. Varsle om sårbarheter

## Ved Problemer

### Feilsøkingsprosedyre:
1. Sjekk error.log
2. Gjennomgå siste endringer
3. Isoler problemområdet
4. Test i utviklingsmiljø

### Kontaktpersoner:
1. Prosjektleder: [NAVN]
2. Tech Lead: [NAVN]
3. DevOps-ansvarlig: [NAVN]

## Automatisk Rapportgenerering

Når du starter en ny samtale, generer umiddelbart en rapport med:
1. Prosjektstatus og fremgang
2. Siste endringer og commits
3. Ventende oppgaver og problemer
4. Systemstatus og ytelse
5. Anbefalinger for neste steg

## Avsluttende Merknader
- Hold all kommunikasjon profesjonell
- Fokuser på prosjektmål
- Dokumenter alle beslutninger
- Følg opp uløste problemer