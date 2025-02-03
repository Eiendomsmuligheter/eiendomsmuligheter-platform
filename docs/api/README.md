# API Dokumentasjon - Eiendomsmuligheter Platform

## Base URL
```
https://api.eiendomsmuligheter.no/v1
```

## Autentisering

Alle API-kall krever en gyldig JWT token i Authorization header:
```
Authorization: Bearer <your_jwt_token>
```

## Endepunkter

### Eiendomsanalyse

#### Analyser Eiendom
```http
POST /properties/analyze
```

Request Body:
```json
{
  "analysis_type": "address",
  "address": "Eksempelveien 1, 0123 Oslo",
  "image_data": null,
  "link": null
}
```

Response:
```json
{
  "property_info": {
    "address": "Eksempelveien 1, 0123 Oslo",
    "municipality_code": "0301",
    "property_id": "0301-1-1",
    "coordinates": {
      "lat": 59.744225,
      "lon": 10.204458
    }
  },
  "building_analysis": {
    "building_type": "enebolig",
    "stories": 2,
    "has_basement": true,
    "total_area": 150.5,
    "rooms": [...]
  },
  "development_potential": {
    "basement_apartment": {...},
    "extension": {...},
    "lot_division": {...}
  },
  "energy_analysis": {...},
  "recommendations": [...]
}
```

#### Last opp Bilde
```http
POST /properties/analyze/image
```

Request:
- Multipart form data med bilde

Response:
```json
{
  "file_id": "abc123",
  "analysis_results": {...}
}
```

### Kommunale Data

#### Hent Reguleringsinfo
```http
GET /municipalities/{municipality_code}/regulations
```

Response:
```json
{
  "regulations": [
    {
      "title": "Reguleringsbestemmelser",
      "description": "...",
      "type": "regulation",
      "requirements": [...]
    }
  ]
}
```

#### Hent Byggesakshistorikk
```http
GET /properties/{property_id}/history
```

Response:
```json
{
  "cases": [
    {
      "case_number": "123/45",
      "title": "Byggesøknad tilbygg",
      "status": "approved",
      "date": "2024-01-15T12:00:00Z",
      "documents": [...]
    }
  ]
}
```

### ENOVA-støtte

#### Hent Støtteordninger
```http
GET /properties/{property_id}/enova-support
```

Response:
```json
{
  "support_options": [
    {
      "title": "Varmepumpe",
      "description": "...",
      "amount": 25000,
      "requirements": [...],
      "benefits": [...]
    }
  ]
}
```

#### Sjekk Støtteberettigelse
```http
GET /properties/{property_id}/support-eligibility/{support_id}
```

Response:
```json
{
  "eligible": true,
  "requirements_met": [...],
  "missing_requirements": [],
  "next_steps": [...]
}
```

### Dokumentgenerering

#### Generer Dokumenter
```http
POST /properties/{property_id}/documents/generate
```

Request Body:
```json
{
  "document_type": "building_application",
  "options": {
    "include_drawings": true,
    "include_calculations": true
  }
}
```

Response:
```json
{
  "document_id": "doc123",
  "status": "generated",
  "download_url": "https://...",
  "expires_at": "2024-02-10T12:00:00Z"
}
```

## Feilhåndtering

### Feilkoder

- 400: Bad Request - Ugyldig input
- 401: Unauthorized - Manglende eller ugyldig token
- 403: Forbidden - Manglende tilgang
- 404: Not Found - Ressurs ikke funnet
- 422: Unprocessable Entity - Valideringsfeil
- 429: Too Many Requests - Rate limit overskredet
- 500: Internal Server Error - Serverfeil

### Feilrespons Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": [
      {
        "field": "address",
        "message": "Address is required"
      }
    ]
  }
}
```

## Rate Limiting

- Standard: 100 requests per minutt
- Premium: 1000 requests per minutt
- Enterprise: Ubegrenset

## Versjonering

- API er versjonert i URL path
- Støtter multiple versjoner
- Deprecation notices sendes via response headers

## Webhook Support

### Registrer Webhook
```http
POST /webhooks
```

Request Body:
```json
{
  "url": "https://your-domain.com/webhook",
  "events": ["analysis.completed", "document.generated"]
}
```

### Webhook Event Format
```json
{
  "event": "analysis.completed",
  "timestamp": "2024-02-03T12:00:00Z",
  "data": {
    "analysis_id": "abc123",
    "status": "completed",
    "results": {...}
  }
}
```

## SDKs og Klientbiblioteker

- [Python SDK](https://github.com/Eiendomsmuligheter/python-sdk)
- [JavaScript SDK](https://github.com/Eiendomsmuligheter/js-sdk)
- [.NET SDK](https://github.com/Eiendomsmuligheter/dotnet-sdk)

## API Changelog

### v1.1.0 (2025-02-01)
- Lagt til støtte for BIM-modell eksport
- Forbedret ENOVA-integrasjon
- Ny endepunkt for batch-analyse

### v1.0.0 (2025-01-15)
- Initial release