"""
Ende-til-ende tester for fullstendig analyse
"""
import pytest
from fastapi.testclient import TestClient
from backend.api.main import app
import json
import time

client = TestClient(app)

def test_complete_analysis_flow():
    """Test hele analyseprosessen fra start til slutt"""
    
    # 1. Registrer bruker
    user_data = {
        "email": "test@example.com",
        "password": "TestPassword123!",
        "full_name": "Test User"
    }
    response = client.post("/api/auth/register", json=user_data)
    assert response.status_code == 200
    token = response.json()["access_token"]
    
    # 2. Last opp eiendomsinformasjon
    property_data = {
        "address": "Testveien 1",
        "municipality": "Drammen",
        "gnr": 1,
        "bnr": 1
    }
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/api/properties/", json=property_data, headers=headers)
    assert response.status_code == 200
    property_id = response.json()["id"]
    
    # 3. Last opp plantegning
    with open("tests/test_data/floor_plan.pdf", "rb") as f:
        response = client.post(
            f"/api/properties/{property_id}/floor-plan",
            files={"file": f},
            headers=headers
        )
    assert response.status_code == 200
    
    # 4. Start analyse
    analysis_request = {
        "property_id": property_id,
        "analysis_types": ["development", "rental", "energy"],
        "requirements": {
            "rental_unit": True,
            "min_area": 50,
            "max_investment": 1000000
        }
    }
    response = client.post("/api/analysis/start", json=analysis_request, headers=headers)
    assert response.status_code == 200
    analysis_id = response.json()["id"]
    
    # 5. Vent på analyse resultat
    max_wait = 60  # sekunder
    start_time = time.time()
    while time.time() - start_time < max_wait:
        response = client.get(f"/api/analysis/{analysis_id}", headers=headers)
        assert response.status_code == 200
        if response.json()["status"] == "completed":
            break
        time.sleep(2)
    
    assert response.json()["status"] == "completed"
    result = response.json()["result"]
    
    # 6. Verifiser analyseresultatet
    assert "development_potential" in result
    assert "rental_potential" in result
    assert "energy_analysis" in result
    assert "recommendations" in result
    assert len(result["recommendations"]) > 0
    
    # 7. Generer rapport
    response = client.post(
        f"/api/analysis/{analysis_id}/report",
        headers=headers,
        json={"format": "pdf"}
    )
    assert response.status_code == 200
    assert "report_url" in response.json()
    
    # 8. Test 3D visualisering
    response = client.get(
        f"/api/properties/{property_id}/3d-model",
        headers=headers
    )
    assert response.status_code == 200
    assert "model_url" in response.json()
    assert "scene_data" in response.json()
    
    # 9. Test betalingsintegrering
    payment_request = {
        "analysis_id": analysis_id,
        "plan": "professional"
    }
    response = client.post("/api/payments/create-session", json=payment_request, headers=headers)
    assert response.status_code == 200
    assert "session_id" in response.json()
    
    # 10. Test dokumentgenerering
    doc_request = {
        "analysis_id": analysis_id,
        "document_types": ["building_application", "floor_plan", "situation_plan"]
    }
    response = client.post("/api/documents/generate", json=doc_request, headers=headers)
    assert response.status_code == 200
    documents = response.json()["documents"]
    assert len(documents) == 3
    assert all("url" in doc for doc in documents)

def test_error_handling():
    """Test feilhåndtering i systemet"""
    
    # Test ugyldig eiendomsdata
    invalid_property = {
        "address": "",  # Ugyldig tomt felt
        "municipality": "Drammen"
    }
    response = client.post("/api/properties/", json=invalid_property)
    assert response.status_code == 422
    
    # Test ugyldig analyse
    invalid_analysis = {
        "property_id": 99999,  # Ikke-eksisterende eiendom
        "analysis_types": ["invalid_type"]
    }
    response = client.post("/api/analysis/start", json=invalid_analysis)
    assert response.status_code == 404
    
    # Test ugyldig autentisering
    response = client.get("/api/properties/1", headers={"Authorization": "Bearer invalid_token"})
    assert response.status_code == 401
    
    # Test manglende tillatelser
    # TODO: Implementer test for tilgangskontroll
    
    # Test ugyldig fil-opplasting
    with open("tests/test_data/invalid_file.txt", "rb") as f:
        response = client.post(
            "/api/properties/1/floor-plan",
            files={"file": f}
        )
    assert response.status_code == 400

def test_performance():
    """Test systemets ytelse"""
    
    # Test responstider
    start_time = time.time()
    response = client.get("/api/properties/1")
    assert time.time() - start_time < 0.5  # Maks 500ms responstid
    
    # Test samtidig behandling
    # TODO: Implementer test for samtidig behandling
    
    # Test caching
    # TODO: Implementer test for caching
    
    # Test databaseytelse
    # TODO: Implementer test for databaseytelse