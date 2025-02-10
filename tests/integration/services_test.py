import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from app import app
from core_modules.property_analyzer import PropertyAnalyzer
from core_modules.municipality_service import MunicipalityService
from core_modules.document_generator import DocumentGenerator

client = TestClient(app)

@pytest.fixture
def test_image():
    return Path("tests/test_data/test_property.jpg")

@pytest.fixture
def test_address():
    return "Testveien 1, 3014 Drammen"

def test_property_analysis_integration(test_image, test_address):
    # Test full property analysis flow
    with open(test_image, "rb") as f:
        response = client.post(
            "/api/analyze",
            files={"file": f},
            data={"address": test_address}
        )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify comprehensive analysis results
    assert "property_info" in data
    assert "regulation_data" in data
    assert "development_potential" in data
    assert "energy_analysis" in data
    
    # Verify specific analysis details
    assert data["property_info"]["address"] == test_address
    assert "zoning_plan" in data["regulation_data"]
    assert "potential_units" in data["development_potential"]
    assert "energy_rating" in data["energy_analysis"]

def test_municipality_service_integration():
    # Test municipality service integration
    service = MunicipalityService()
    result = service.get_property_regulations("3014", "1", "1")  # Test GNR/BNR
    
    assert "zoning_plan" in result
    assert "building_restrictions" in result
    assert "historical_cases" in result

def test_document_generation_integration(test_image, test_address):
    # Test document generation flow
    analyzer = PropertyAnalyzer()
    analysis_results = analyzer.analyze_property(test_image, test_address)
    
    generator = DocumentGenerator()
    documents = generator.generate_all_documents(analysis_results)
    
    # Verify all required documents are generated
    assert "building_application" in documents
    assert "situation_plan" in documents
    assert "floor_plan" in documents
    assert "facade_drawing" in documents
    
    # Verify document content
    building_app = documents["building_application"]
    assert "søknad om tillatelse til tiltak" in building_app.lower()
    assert test_address in building_app