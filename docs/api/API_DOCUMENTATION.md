# API Dokumentasjon - Eiendomsmuligheter Platform

## Oversikt
Denne dokumentasjonen beskriver REST API-et for Eiendomsmuligheter Platform. API-et er bygget med FastAPI og følger OpenAPI (Swagger) spesifikasjonen.

## Base URL
```
https://api.eiendomsmuligheter.no/v1
```

## Autentisering
API-et bruker Auth0 for autentisering. Inkluder JWT-token i Authorization header:
```
Authorization: Bearer {your_jwt_token}
```

## Endepunkter

### Eiendomsanalyse

#### Analyser eiendom
```http
POST /property/analyze
```

##### Request Body
```json
{
  "address": "string",
  "gnr": "integer?",
  "bnr": "integer?",
  "municipality": "string",
  "files": "array<binary>?"
}
```

##### Response
```json
{
  "analysis_id": "string",
  "status": "string",
  "property_data": {
    "address": "string",
    "gnr": "integer",
    "bnr": "integer",
    "municipality": "string",
    "area": "number",
    "existing_buildings": [
      {
        "type": "string",
        "area": "number",
        "floors": "integer",
        "usage": "string"
      }
    ]
  },
  "development_potential": {
    "max_utilization": "number",
    "current_utilization": "number",
    "potential_area": "number",
    "possible_units": "integer",
    "recommendations": [
      {
        "type": "string",
        "description": "string",
        "estimated_cost": "number",
        "estimated_value": "number",
        "roi": "number"
      }
    ]
  }
}
```

#### Hent kommunale regler
```http
GET /municipality/rules/{municipality_id}
```

##### Response
```json
{
  "municipality_id": "string",
  "name": "string",
  "rules": {
    "zoning_plan": {
      "code": "string",
      "description": "string",
      "max_utilization": "number",
      "height_restrictions": {
        "max_height": "number",
        "max_stories": "integer"
      }
    },
    "building_regulations": [
      {
        "type": "string",
        "description": "string",
        "requirements": "object"
      }
    ]
  }
}
```

#### Generer dokumenter
```http
POST /documents/generate
```

##### Request Body
```json
{
  "analysis_id": "string",
  "document_types": [
    "building_application",
    "situation_plan",
    "floor_plan",
    "facade_drawing"
  ]
}
```

##### Response
```json
{
  "documents": [
    {
      "type": "string",
      "url": "string",
      "expiry": "string"
    }
  ]
}
```

#### Energianalyse og Enova-støtte
```http
POST /energy/analyze
```

##### Request Body
```json
{
  "property_id": "string",
  "current_energy_label": "string?",
  "heating_system": "string?",
  "construction_year": "integer?"
}
```

##### Response
```json
{
  "current_status": {
    "energy_label": "string",
    "annual_consumption": "number",
    "co2_emissions": "number"
  },
  "improvement_potential": {
    "possible_label": "string",
    "energy_savings": "number",
    "cost_savings": "number"
  },
  "enova_support": [
    {
      "program": "string",
      "description": "string",
      "max_support": "number",
      "requirements": "array<string>"
    }
  ]
}
```

### 3D-visualisering

#### Generer 3D-modell
```http
POST /3d/generate
```

##### Request Body
```json
{
  "property_id": "string",
  "quality": "string",
  "include_interior": "boolean"
}
```

##### Response
```json
{
  "model_id": "string",
  "viewer_url": "string",
  "download_url": "string"
}
```

## Feilkoder

| Kode | Beskrivelse |
|------|-------------|
| 400  | Ugyldig forespørsel |
| 401  | Ikke autentisert |
| 403  | Ikke autorisert |
| 404  | Ressurs ikke funnet |
| 422  | Valideringsfeil |
| 429  | For mange forespørsler |
| 500  | Serverfeil |

## Rate Limiting
- 100 forespørsler per minutt for Basic-abonnement
- 1000 forespørsler per minutt for Pro-abonnement
- Ubegrenset for Enterprise-abonnement

## Webhook-integrasjon
```http
POST {your_webhook_url}
```

### Webhook Events
```json
{
  "event_type": "string",
  "timestamp": "string",
  "data": {
    "analysis_id": "string",
    "status": "string",
    "result": "object?"
  }
}
```

## SDK-er og Integrasjonsbiblioteker
- [Python SDK](https://github.com/Eiendomsmuligheter/python-sdk)
- [JavaScript SDK](https://github.com/Eiendomsmuligheter/js-sdk)
- [.NET SDK](https://github.com/Eiendomsmuligheter/dotnet-sdk)

## Eksempler

### Python
```python
from eiendomsmuligheter import Client

client = Client('your_api_key')

# Analyser eiendom
analysis = client.analyze_property(
    address="Storgata 1",
    municipality="Drammen"
)

# Vent på resultater
results = analysis.wait_for_completion()
```

### JavaScript
```javascript
import { EiendomsmulighetClient } from '@eiendomsmuligheter/sdk';

const client = new EiendomsmulighetClient('your_api_key');

// Analyser eiendom
const analysis = await client.analyzeProperty({
  address: 'Storgata 1',
  municipality: 'Drammen'
});

// Hent resultater
const results = await analysis.waitForCompletion();
```

## Versjonshistorikk

### v1.0.0 (2025-02-03)
- Initial API release
- Komplett analysestøtte
- NVIDIA Omniverse integrasjon
- Dokumentgenerering
- Enova-støtteberegning

## Support
- E-post: api-support@eiendomsmuligheter.no
- API Status: [status.eiendomsmuligheter.no](https://status.eiendomsmuligheter.no)
- Developer Forum: [forum.eiendomsmuligheter.no](https://forum.eiendomsmuligheter.no)