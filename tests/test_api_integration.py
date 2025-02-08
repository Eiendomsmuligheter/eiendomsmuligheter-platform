import pytest
from fastapi.testclient import TestClient
from core_modules.auth_service import User, UserRole
import json
import os
from datetime import datetime

def test_auth_endpoints(test_client, mock_jwt_token):
    """Test authentication endpoints"""
    headers = {"Authorization": f"Bearer {mock_jwt_token}"}
    
    # Test login endpoint
    response = test_client.post("/api/auth/login")
    assert response.status_code == 200
    assert "url" in response.json()
    
    # Test callback endpoint
    response = test_client.get("/api/auth/callback?code=test_code&state=test_state")
    assert response.status_code == 200
    assert "access_token" in response.json()
    
    # Test profile endpoint
    response = test_client.get("/api/auth/profile", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "test@example.com"
    assert "professional" in data["roles"]
    
    # Test profile update
    update_data = {"company": "Updated Company"}
    response = test_client.put(
        "/api/auth/profile",
        headers=headers,
        json=update_data
    )
    assert response.status_code == 200
    assert response.json()["company"] == "Updated Company"

def test_payment_endpoints(test_client, mock_jwt_token):
    """Test payment endpoints"""
    headers = {"Authorization": f"Bearer {mock_jwt_token}"}
    
    # Test get plans
    response = test_client.get("/api/payments/plans", headers=headers)
    assert response.status_code == 200
    plans = response.json()
    assert len(plans) == 3
    assert "basic" in plans
    assert "pro" in plans
    assert "enterprise" in plans
    
    # Test create subscription
    subscription_data = {
        "plan_id": "price_pro",
        "payment_method_id": "pm_test123"
    }
    response = test_client.post(
        "/api/payments/create-subscription",
        headers=headers,
        json=subscription_data
    )
    assert response.status_code == 200
    assert "subscriptionId" in response.json()
    
    # Test get subscription status
    sub_id = response.json()["subscriptionId"]
    response = test_client.get(
        f"/api/payments/subscription-status/{sub_id}",
        headers=headers
    )
    assert response.status_code == 200
    assert "status" in response.json()
    
    # Test cancel subscription
    response = test_client.post(
        f"/api/payments/cancel-subscription/{sub_id}",
        headers=headers
    )
    assert response.status_code == 200
    assert response.json()["status"] == "canceled"

def test_property_analysis_endpoints(
    test_client,
    mock_jwt_token,
    mock_upload_file,
    mock_property_data
):
    """Test property analysis endpoints"""
    headers = {"Authorization": f"Bearer {mock_jwt_token}"}
    
    # Test property analysis with file upload
    with open(mock_upload_file, "rb") as f:
        files = {"file": ("test_floorplan.pdf", f, "application/pdf")}
        response = test_client.post(
            "/api/analysis/analyze-property",
            headers=headers,
            files=files,
            data={"property_data": json.dumps(mock_property_data)}
        )
    assert response.status_code == 200
    analysis_result = response.json()
    assert "potential_rental_units" in analysis_result
    assert "estimated_rental_income" in analysis_result
    
    # Test property analysis with Finn.no URL
    url_data = {
        "url": "https://www.finn.no/realestate/homes/ad.html?finnkode=123456789"
    }
    response = test_client.post(
        "/api/analysis/analyze-property-url",
        headers=headers,
        json=url_data
    )
    assert response.status_code == 200
    assert "analysis_results" in response.json()

def test_document_generation_endpoints(
    test_client,
    mock_jwt_token,
    mock_analysis_results
):
    """Test document generation endpoints"""
    headers = {"Authorization": f"Bearer {mock_jwt_token}"}
    
    # Test generate analysis report
    response = test_client.post(
        "/api/documents/generate-report",
        headers=headers,
        json=mock_analysis_results
    )
    assert response.status_code == 200
    assert "report_url" in response.json()
    
    # Test generate building application
    application_data = {
        **mock_analysis_results,
        "municipality": "Drammen",
        "property_id": "1234/56"
    }
    response = test_client.post(
        "/api/documents/generate-application",
        headers=headers,
        json=application_data
    )
    assert response.status_code == 200
    assert "application_url" in response.json()

def test_3d_visualization_endpoints(
    test_client,
    mock_jwt_token,
    mock_property_data
):
    """Test 3D visualization endpoints"""
    headers = {"Authorization": f"Bearer {mock_jwt_token}"}
    
    # Test generate 3D model
    response = test_client.post(
        "/api/visualization/generate-3d",
        headers=headers,
        json=mock_property_data
    )
    assert response.status_code == 200
    assert "model_url" in response.json()
    
    # Test update visualization settings
    settings = {
        "quality": "high",
        "show_measurements": True,
        "view_type": "aerial"
    }
    response = test_client.post(
        "/api/visualization/update-settings",
        headers=headers,
        json=settings
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_municipality_integration_endpoints(
    test_client,
    mock_jwt_token,
    mock_property_data
):
    """Test municipality integration endpoints"""
    headers = {"Authorization": f"Bearer {mock_jwt_token}"}
    
    # Test get zoning regulations
    property_id = mock_property_data["property_id"]
    response = test_client.get(
        f"/api/municipality/zoning/{property_id}",
        headers=headers
    )
    assert response.status_code == 200
    assert "zoning_type" in response.json()
    assert "regulations" in response.json()
    
    # Test get building history
    response = test_client.get(
        f"/api/municipality/building-history/{property_id}",
        headers=headers
    )
    assert response.status_code == 200
    assert "previous_applications" in response.json()

def test_error_handling(test_client):
    """Test API error handling"""
    # Test invalid token
    response = test_client.get(
        "/api/auth/profile",
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 401
    
    # Test missing authorization
    response = test_client.get("/api/auth/profile")
    assert response.status_code == 401
    
    # Test invalid subscription
    response = test_client.get(
        "/api/payments/subscription-status/invalid_sub",
        headers={"Authorization": f"Bearer {mock_jwt_token}"}
    )
    assert response.status_code == 400

def test_rate_limiting(test_client, mock_jwt_token):
    """Test API rate limiting"""
    headers = {"Authorization": f"Bearer {mock_jwt_token}"}
    
    # Make multiple requests in quick succession
    for _ in range(10):
        response = test_client.get("/api/auth/profile", headers=headers)
    
    # The last request should be rate limited
    assert response.status_code == 429
    assert "retry_after" in response.json()

@pytest.mark.parametrize("endpoint,method,expected_status", [
    ("/api/analysis/analyze-property", "POST", 401),
    ("/api/payments/plans", "GET", 401),
    ("/api/documents/generate-report", "POST", 401),
    ("/api/visualization/generate-3d", "POST", 401),
])
def test_authentication_required(test_client, endpoint, method, expected_status):
    """Test authentication requirements"""
    if method == "GET":
        response = test_client.get(endpoint)
    else:
        response = test_client.post(endpoint, json={})
    
    assert response.status_code == expected_status
    assert "detail" in response.json()