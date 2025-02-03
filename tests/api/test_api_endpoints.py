import pytest
from fastapi.testclient import TestClient
from backend.main import app
import json

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test_token"}

@pytest.mark.api
class TestPropertyAPI:
    def test_analyze_property_endpoint(self, client, auth_headers):
        """Test eiendomsanalyse endepunkt"""
        payload = {
            "address": "Storgata 1",
            "municipality": "Drammen",
            "gnr": 1,
            "bnr": 1
        }
        
        response = client.post(
            "/api/v1/property/analyze",
            json=payload,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "analysis_id" in data
        assert "status" in data
        assert data["status"] == "completed"

    def test_municipality_rules_endpoint(self, client, auth_headers):
        """Test kommune-regler endepunkt"""
        response = client.get(
            "/api/v1/municipality/rules/drammen",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "zoning_plan" in data
        assert "building_regulations" in data

    def test_document_generation_endpoint(self, client, auth_headers):
        """Test dokumentgenerering endepunkt"""
        payload = {
            "analysis_id": "test_analysis",
            "document_types": ["building_application", "situation_plan"]
        }
        
        response = client.post(
            "/api/v1/documents/generate",
            json=payload,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert len(data["documents"]) == 2

    def test_enova_support_endpoint(self, client, auth_headers):
        """Test Enova-støtte endepunkt"""
        payload = {
            "property_id": "test_property",
            "energy_data": {
                "current_rating": "D",
                "building_type": "residential",
                "area": 150
            }
        }
        
        response = client.post(
            "/api/v1/enova/calculate-support",
            json=payload,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "available_support" in data
        assert "recommendations" in data

    def test_3d_model_endpoint(self, client, auth_headers):
        """Test 3D-modell endepunkt"""
        payload = {
            "property_id": "test_property",
            "quality": "high"
        }
        
        response = client.post(
            "/api/v1/3d/generate",
            json=payload,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "model_url" in data
        assert "format" in data

@pytest.mark.api
class TestAuthAPI:
    def test_login_endpoint(self, client):
        """Test innlogging endepunkt"""
        payload = {
            "email": "test@example.com",
            "password": "test123"
        }
        
        response = client.post("/api/v1/auth/login", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data

    def test_protected_endpoint_without_auth(self, client):
        """Test beskyttet endepunkt uten autentisering"""
        response = client.get("/api/v1/protected-resource")
        assert response.status_code == 401

    def test_protected_endpoint_with_auth(self, client, auth_headers):
        """Test beskyttet endepunkt med autentisering"""
        response = client.get(
            "/api/v1/protected-resource",
            headers=auth_headers
        )
        assert response.status_code == 200

@pytest.mark.api
class TestPaymentAPI:
    def test_create_payment_intent(self, client, auth_headers):
        """Test opprettelse av betalingsintensjon"""
        payload = {
            "amount": 1000,
            "currency": "nok",
            "payment_method_types": ["card"]
        }
        
        response = client.post(
            "/api/v1/payment/create-intent",
            json=payload,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "client_secret" in data
        assert "payment_intent_id" in data

    def test_subscription_endpoint(self, client, auth_headers):
        """Test abonnement endepunkt"""
        payload = {
            "plan": "pro",
            "payment_method_id": "pm_test_123"
        }
        
        response = client.post(
            "/api/v1/payment/create-subscription",
            json=payload,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "subscription_id" in data
        assert "status" in data

@pytest.mark.api
class TestErrorHandling:
    def test_invalid_property_data(self, client, auth_headers):
        """Test feilhåndtering ved ugyldig eiendomsdata"""
        payload = {
            "address": "",  # Ugyldig tomt felt
            "municipality": "Drammen"
        }
        
        response = client.post(
            "/api/v1/property/analyze",
            json=payload,
            headers=auth_headers
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_rate_limiting(self, client, auth_headers):
        """Test rate limiting"""
        for _ in range(101):  # Over grensen på 100 forespørsler
            client.get(
                "/api/v1/municipality/rules/drammen",
                headers=auth_headers
            )
        
        response = client.get(
            "/api/v1/municipality/rules/drammen",
            headers=auth_headers
        )
        assert response.status_code == 429

    def test_invalid_token(self, client):
        """Test ugyldig token"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get(
            "/api/v1/protected-resource",
            headers=headers
        )
        assert response.status_code == 401