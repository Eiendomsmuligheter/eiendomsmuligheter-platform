import pytest
from fastapi.testclient import TestClient
from typing import Generator, Dict
import os
import asyncio
from datetime import datetime, timedelta
from jose import jwt
import json
from core_modules.auth_service import AuthService, User, UserRole
from core_modules.payment_service import PaymentService
from pathlib import Path

# Test configuration
@pytest.fixture(scope="session")
def test_config() -> Dict:
    return {
        "AUTH0_DOMAIN": "test-domain.auth0.com",
        "AUTH0_CLIENT_ID": "test-client-id",
        "AUTH0_CLIENT_SECRET": "test-client-secret",
        "AUTH0_AUDIENCE": "test-audience",
        "STRIPE_SECRET_KEY": "test-stripe-key",
        "STRIPE_PUBLISHABLE_KEY": "test-stripe-pub-key",
        "STRIPE_WEBHOOK_SECRET": "test-webhook-secret",
    }

# Mock JWT token
@pytest.fixture
def mock_jwt_token() -> str:
    payload = {
        "sub": "auth0|123456789",
        "iss": "https://test-domain.auth0.com/",
        "aud": "test-audience",
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=1),
        "scope": "openid profile email",
        "permissions": ["read:basic", "read:professional"]
    }
    
    return jwt.encode(
        payload,
        "test-secret",
        algorithm="HS256"
    )

# Mock user
@pytest.fixture
def mock_user() -> User:
    return User(
        email="test@example.com",
        sub="auth0|123456789",
        roles=[UserRole.PROFESSIONAL],
        name="Test User",
        stripe_customer_id="cus_test123",
        company="Test Company",
        is_active=True
    )

# Mock Auth Service
@pytest.fixture
def mock_auth_service(mock_user: User, mock_jwt_token: str):
    class MockAuthService(AuthService):
        async def verify_token(self, token: str) -> Dict:
            return jwt.decode(
                token,
                "test-secret",
                algorithms=["HS256"],
                audience="test-audience"
            )
            
        async def get_user_profile(self, token: str) -> Dict:
            return {
                "email": mock_user.email,
                "sub": mock_user.sub,
                "name": mock_user.name,
                "company": mock_user.company
            }
            
        async def get_user_roles(self, token: str) -> list:
            return [UserRole.PROFESSIONAL]
    
    return MockAuthService()

# Mock Payment Service
@pytest.fixture
def mock_payment_service():
    class MockPaymentService(PaymentService):
        async def create_customer(self, email: str, name: str) -> Dict:
            return {
                "id": "cus_test123",
                "email": email,
                "name": name
            }
            
        async def create_subscription(self, customer_id: str, plan_id: str) -> Dict:
            return {
                "id": "sub_test123",
                "customer": customer_id,
                "plan": plan_id,
                "status": "active"
            }
    
    return MockPaymentService(api_key="test-key")

# Test client
@pytest.fixture
def test_client(
    mock_auth_service,
    mock_payment_service
) -> Generator:
    from app import app
    
    # Override services with mocks
    app.dependency_overrides = {
        "get_auth_service": lambda: mock_auth_service,
        "get_payment_service": lambda: mock_payment_service
    }
    
    with TestClient(app) as client:
        yield client

# Test database
@pytest.fixture(scope="session")
def test_db():
    """Create test database and tables"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from core_modules.database import Base
    
    SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///./test.db"
    engine = create_engine(SQLALCHEMY_TEST_DATABASE_URL)
    
    # Create test database and tables
    Base.metadata.create_all(bind=engine)
    
    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )
    
    yield TestingSessionLocal
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("./test.db"):
        os.remove("./test.db")

# Async test database session
@pytest.fixture
async def async_test_db_session(test_db):
    async with test_db() as session:
        yield session
        await session.rollback()

# Mock file upload
@pytest.fixture
def mock_upload_file(tmp_path: Path):
    test_file = tmp_path / "test_floorplan.pdf"
    test_file.write_bytes(b"Test PDF content")
    return test_file

# Mock property data
@pytest.fixture
def mock_property_data() -> Dict:
    return {
        "address": "Testveien 1",
        "municipality": "Drammen",
        "property_id": "1234/56",
        "area": 150.5,
        "floors": 2,
        "has_basement": True,
        "has_attic": True,
        "year_built": 1985
    }

# Mock analysis results
@pytest.fixture
def mock_analysis_results() -> Dict:
    return {
        "potential_rental_units": 2,
        "estimated_rental_income": 25000,
        "renovation_cost": 500000,
        "roi": 15.5,
        "payback_period": 8.2,
        "regulations": {
            "zoning": "residential",
            "max_height": 9.0,
            "max_coverage": 30.0,
            "min_distance": 4.0
        },
        "recommendations": [
            "Convert basement to rental unit",
            "Add separate entrance",
            "Upgrade electrical system"
        ]
    }

# Event loop
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()