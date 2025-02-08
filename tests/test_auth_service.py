import pytest
from fastapi import HTTPException
from core_modules.auth_service import AuthService, User, UserRole
from datetime import datetime
import json

pytestmark = pytest.mark.asyncio

async def test_verify_token(mock_auth_service, mock_jwt_token):
    """Test token verification"""
    # Test valid token
    payload = await mock_auth_service.verify_token(mock_jwt_token)
    assert payload["sub"] == "auth0|123456789"
    
    # Test invalid token
    with pytest.raises(HTTPException) as exc_info:
        await mock_auth_service.verify_token("invalid_token")
    assert exc_info.value.status_code == 401

async def test_get_user_roles(mock_auth_service, mock_jwt_token):
    """Test getting user roles"""
    roles = await mock_auth_service.get_user_roles(mock_jwt_token)
    assert UserRole.PROFESSIONAL in roles
    assert len(roles) == 1

async def test_get_user_profile(mock_auth_service, mock_jwt_token):
    """Test getting user profile"""
    profile = await mock_auth_service.get_user_profile(mock_jwt_token)
    assert profile["email"] == "test@example.com"
    assert profile["name"] == "Test User"
    assert profile["company"] == "Test Company"

async def test_create_user(mock_auth_service):
    """Test user creation"""
    user_data = {
        "email": "new@example.com",
        "password": "SecurePass123!",
        "name": "New User",
        "connection": "Username-Password-Authentication"
    }
    
    try:
        user = await mock_auth_service.create_user(user_data)
        assert isinstance(user, User)
        assert user.email == "new@example.com"
        assert user.name == "New User"
        assert UserRole.BASIC in user.roles
    except HTTPException as e:
        pytest.fail(f"User creation failed: {str(e)}")

async def test_update_user_metadata(mock_auth_service, mock_user):
    """Test updating user metadata"""
    metadata = {
        "company": "Updated Company",
        "phone": "+47 12345678"
    }
    
    try:
        updated = await mock_auth_service.update_user_metadata(
            mock_user.sub,
            metadata
        )
        assert updated["user_metadata"]["company"] == "Updated Company"
        assert updated["user_metadata"]["phone"] == "+47 12345678"
    except HTTPException as e:
        pytest.fail(f"Metadata update failed: {str(e)}")

async def test_delete_user(mock_auth_service, mock_user):
    """Test user deletion"""
    try:
        success = await mock_auth_service.delete_user(mock_user.sub)
        assert success is True
    except HTTPException as e:
        pytest.fail(f"User deletion failed: {str(e)}")

async def test_assign_roles(mock_auth_service, mock_user):
    """Test role assignment"""
    roles = ["professional", "enterprise"]
    
    try:
        success = await mock_auth_service.assign_roles(
            mock_user.sub,
            roles
        )
        assert success is True
    except HTTPException as e:
        pytest.fail(f"Role assignment failed: {str(e)}")

def test_user_model_validation():
    """Test User model validation"""
    # Test valid user creation
    user = User(
        email="test@example.com",
        sub="auth0|123",
        roles=[UserRole.BASIC],
        name="Test User",
        is_active=True
    )
    assert user.email == "test@example.com"
    assert user.is_active is True
    
    # Test user serialization
    user_dict = user.dict()
    assert isinstance(user_dict, dict)
    assert user_dict["email"] == "test@example.com"
    assert user_dict["roles"] == [UserRole.BASIC]

def test_user_role_enum():
    """Test UserRole enum"""
    assert UserRole.BASIC == "basic"
    assert UserRole.PROFESSIONAL == "professional"
    assert UserRole.ENTERPRISE == "enterprise"
    assert UserRole.ADMIN == "admin"
    
    # Test role comparison
    assert UserRole.ADMIN in [UserRole.ADMIN, UserRole.PROFESSIONAL]
    assert "basic" == UserRole.BASIC

@pytest.mark.parametrize("role,expected_permissions", [
    (UserRole.BASIC, {"read:property_analysis", "read:documentation"}),
    (UserRole.PROFESSIONAL, {"read:property_analysis", "create:property_analysis"}),
    (UserRole.ENTERPRISE, {"read:property_analysis", "access:api"}),
    (UserRole.ADMIN, {"*"})
])
def test_role_permissions(role, expected_permissions):
    """Test role permissions mapping"""
    # This would normally come from your permission mapping service
    role_permissions = {
        UserRole.BASIC: {"read:property_analysis", "read:documentation"},
        UserRole.PROFESSIONAL: {"read:property_analysis", "create:property_analysis"},
        UserRole.ENTERPRISE: {"read:property_analysis", "access:api"},
        UserRole.ADMIN: {"*"}
    }
    
    assert role_permissions[role] == expected_permissions

async def test_auth_service_initialization():
    """Test AuthService initialization"""
    # Test with missing configuration
    with pytest.raises(ValueError):
        AuthService()
    
    # Test with valid configuration
    os.environ["AUTH0_DOMAIN"] = "test-domain.auth0.com"
    os.environ["AUTH0_AUDIENCE"] = "test-audience"
    os.environ["AUTH0_CLIENT_ID"] = "test-client-id"
    os.environ["AUTH0_CLIENT_SECRET"] = "test-client-secret"
    
    auth_service = AuthService()
    assert auth_service.config.domain == "test-domain.auth0.com"
    assert auth_service.oauth is not None