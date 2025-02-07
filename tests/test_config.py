"""
Test konfigurasjon for Eiendomsmuligheter Platform
"""
import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.database.models import Base

# Test database URL
TEST_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/eiendomsmuligheter_test"

@pytest.fixture(scope="session")
def engine():
    """Create test database engine"""
    engine = create_engine(TEST_DATABASE_URL)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)

@pytest.fixture(scope="session")
def db_session(engine):
    """Create test database session"""
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

@pytest.fixture
def test_client():
    """Create test client for API endpoints"""
    from backend.api.main import app
    from fastapi.testclient import TestClient
    return TestClient(app)

@pytest.fixture
def test_property_data():
    """Sample property data for testing"""
    return {
        "address": "Testveien 1",
        "municipality": "Drammen",
        "gnr": 1,
        "bnr": 1,
        "property_type": "enebolig",
        "total_area": 150.5,
        "building_area": 120.0
    }

@pytest.fixture
def test_analysis_data():
    """Sample analysis data for testing"""
    return {
        "analysis_type": "development",
        "input_data": {
            "property_id": 1,
            "requirements": {
                "min_area": 50,
                "max_stories": 2,
                "rental_unit": True
            }
        }
    }

@pytest.fixture
def test_user_data():
    """Sample user data for testing"""
    return {
        "email": "test@example.com",
        "password": "TestPassword123!",
        "full_name": "Test User"
    }