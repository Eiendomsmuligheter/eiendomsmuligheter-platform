from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2AuthorizationCodeBearer
from core_modules.auth_service import auth_service, get_current_user, User
from typing import Optional, Dict
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["authentication"])

class UserCreate(BaseModel):
    email: str
    password: str
    name: Optional[str] = None
    company: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

@router.post("/login")
async def login(request: Request):
    """
    Initiate Auth0 login process
    Returns Auth0 authorization URL
    """
    try:
        redirect_uri = str(request.url_for('callback'))
        return await auth_service.oauth.auth0.authorize_redirect(
            request, redirect_uri
        )
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login process failed"
        )

@router.get("/callback")
async def callback(request: Request):
    """
    Handle Auth0 callback after successful authentication
    Returns access token and user info
    """
    try:
        token = await auth_service.oauth.auth0.authorize_access_token(request)
        user_info = await auth_service.get_user_profile(token['access_token'])
        
        return {
            "access_token": token['access_token'],
            "token_type": "bearer",
            "user": user_info
        }
    except Exception as e:
        logger.error(f"Callback processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication callback failed"
        )

@router.post("/register", response_model=User)
async def register_user(user_data: UserCreate):
    """
    Register a new user
    Creates user in Auth0 and returns user object
    """
    try:
        auth0_user_data = {
            "email": user_data.email,
            "password": user_data.password,
            "connection": "Username-Password-Authentication",
            "name": user_data.name,
            "user_metadata": {
                "company": user_data.company
            }
        }
        
        user = await auth_service.create_user(auth0_user_data)
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User registration failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user"
        )

@router.get("/profile", response_model=User)
async def get_profile(current_user: User = Depends(get_current_user)):
    """
    Get current user profile
    Requires authentication
    """
    return current_user

@router.put("/profile", response_model=User)
async def update_profile(
    metadata: Dict,
    current_user: User = Depends(get_current_user)
):
    """
    Update user profile metadata
    Requires authentication
    """
    try:
        updated_user = await auth_service.update_user_metadata(
            current_user.sub,
            metadata
        )
        return User(
            email=current_user.email,
            sub=current_user.sub,
            roles=current_user.roles,
            name=updated_user.get("name", current_user.name),
            company=metadata.get("company", current_user.company),
            stripe_customer_id=current_user.stripe_customer_id,
            is_active=current_user.is_active
        )
    except Exception as e:
        logger.error(f"Profile update failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )

@router.delete("/profile")
async def delete_profile(current_user: User = Depends(get_current_user)):
    """
    Delete user account
    Requires authentication
    """
    try:
        success = await auth_service.delete_user(current_user.sub)
        if success:
            return {"message": "User account deleted successfully"}
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user account"
        )
    except Exception as e:
        logger.error(f"Profile deletion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user account"
        )

@router.post("/logout")
async def logout(request: Request, response: Response):
    """
    Logout user and clear session
    """
    try:
        await auth_service.oauth.auth0.revoke_token(request)
        response.delete_cookie("auth_token")
        return {"message": "Logged out successfully"}
    except Exception as e:
        logger.error(f"Logout failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.get("/permissions")
async def get_permissions(current_user: User = Depends(get_current_user)):
    """
    Get user permissions based on roles
    Requires authentication
    """
    role_permissions = {
        "basic": [
            "read:property_analysis",
            "read:documentation",
            "create:support_ticket"
        ],
        "professional": [
            "read:property_analysis",
            "read:documentation",
            "create:support_ticket",
            "create:property_analysis",
            "export:reports",
            "access:3d_visualization"
        ],
        "enterprise": [
            "read:property_analysis",
            "read:documentation",
            "create:support_ticket",
            "create:property_analysis",
            "export:reports",
            "access:3d_visualization",
            "access:api",
            "create:team_member",
            "customize:reports"
        ],
        "admin": [
            "*"  # All permissions
        ]
    }
    
    user_permissions = set()
    for role in current_user.roles:
        user_permissions.update(role_permissions.get(role, []))
    
    return {"permissions": list(user_permissions)}