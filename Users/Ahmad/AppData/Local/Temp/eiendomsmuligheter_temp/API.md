# Eiendomsmuligheter API Dokumentasjon

## Autentisering

### Registrer ny bruker
```
POST /api/auth/register
Content-Type: application/json

{
    "email": "string",
    "password": "string"
}
```

### Logg inn
```
POST /api/auth/login
Content-Type: application/json

{
    "email": "string",
    "password": "string"
}
```

## Analyse

### Start ny analyse
```
POST /api/analysis
Authorization: Bearer <token>
Content-Type: application/json

{
    "address": "string",
    "propertyData": {
        "size": "number",
        "yearBuilt": "number",
        "propertyType": "string",
        "zoning": "string"
    }
}
```

### Hent analyser for bruker
```
GET /api/analysis
Authorization: Bearer <token>
```

### Hent spesifikk analyse
```
GET /api/analysis/:id
Authorization: Bearer <token>
```

## Betaling

### Opprett abonnement
```
POST /api/payment/create-subscription
Authorization: Bearer <token>
Content-Type: application/json

{
    "paymentMethodId": "string",
    "priceId": "string"
}
```

### Kanseller abonnement
```
POST /api/payment/cancel-subscription
Authorization: Bearer <token>
```

### Kjøp analysekreditter
```
POST /api/payment/purchase-credits
Authorization: Bearer <token>
Content-Type: application/json

{
    "amount": "number",
    "paymentMethodId": "string"
}
```

## Feilkoder

- 400: Ugyldig forespørsel
- 401: Ikke autentisert
- 403: Ikke autorisert
- 404: Ressurs ikke funnet
- 500: Serverfeil

## Respons Formater

### Analyse Respons
```json
{
    "id": "string",
    "address": "string",
    "propertyData": {
        "size": "number",
        "yearBuilt": "number",
        "propertyType": "string",
        "zoning": "string"
    },
    "possibilities": [
        {
            "type": "string",
            "feasibility": "string",
            "estimatedCost": "string",
            "requirements": ["string"],
            "description": "string"
        }
    ],
    "aiAnalysis": {
        "recommendation": "string",
        "confidenceScore": "number",
        "potentialValue": "number"
    },
    "status": "string",
    "createdAt": "date"
}
```

### Error Respons
```json
{
    "message": "string"
}
```

## Rate Limiting
API-et har rate limiting på 100 forespørsler per time per IP-adresse.