# Eiendomsmuligheter API Documentation

## Overview
The Eiendomsmuligheter API provides comprehensive property analysis and development potential assessment services. This RESTful API supports property analysis, municipal regulation checks, document generation, and payment processing.

## Base URL
```
https://api.eiendomsmuligheter.no/v1
```

## Authentication
All API requests require authentication using JWT tokens. Obtain a token through the Auth0 authentication flow.

```http
Authorization: Bearer your-jwt-token
```

## Endpoints

### Property Analysis

#### POST /analyze
Analyze a property through image upload or address.

**Request**
```http
POST /analyze
Content-Type: multipart/form-data

{
    "file": (binary),
    "address": "string",
    "analysis_type": "full" | "basic" | "regulation"
}
```

**Response**
```json
{
    "analysis_id": "string",
    "property_info": {
        "address": "string",
        "gnr": "string",
        "bnr": "string",
        "area": "number",
        "building_type": "string"
    },
    "regulation_data": {
        "zoning_plan": "string",
        "building_restrictions": {},
        "development_potential": {}
    },
    "energy_analysis": {
        "current_rating": "string",
        "improvement_potential": {},
        "enova_support": {}
    }
}
```

### Municipality Service

#### GET /municipality/{municipality_id}/regulations
Get municipal regulations for a specific property.

**Parameters**
- municipality_id (string): Kommune ID
- gnr (string): Gårdsnummer
- bnr (string): Bruksnummer

**Response**
```json
{
    "zoning_plan": "string",
    "building_restrictions": {
        "max_height": "number",
        "max_bya": "number",
        "min_distance": "number"
    },
    "historical_cases": [
        {
            "case_id": "string",
            "type": "string",
            "status": "string",
            "date": "string"
        }
    ]
}
```

### Document Generation

#### POST /documents/generate
Generate building application documents.

**Request**
```json
{
    "analysis_id": "string",
    "document_types": ["building_application", "situation_plan", "floor_plan"]
}
```

**Response**
```json
{
    "documents": {
        "building_application": "string (base64)",
        "situation_plan": "string (base64)",
        "floor_plan": "string (base64)"
    }
}
```

### Payment Integration

#### POST /payment/create-session
Create a payment session for service access.

**Request**
```json
{
    "plan_id": "string",
    "currency": "nok",
    "customer_email": "string"
}
```

**Response**
```json
{
    "session_id": "string",
    "public_key": "string",
    "client_secret": "string"
}
```

## Error Codes
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 429: Too Many Requests
- 500: Internal Server Error

## Rate Limits
- 100 requests per minute per IP
- 1000 requests per day per API key

## Webhooks
The API supports webhooks for asynchronous notifications about:
- Analysis completion
- Payment status changes
- Document generation completion

## SDK Support
Official SDKs available for:
- Python
- JavaScript/TypeScript
- C#
- Java