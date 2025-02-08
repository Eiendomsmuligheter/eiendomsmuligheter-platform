from typing import Optional, Dict, List
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2AuthorizationCodeBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta
import os
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from pydantic import BaseModel
import logging
from enum import Enum
import aiohttp
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserRole(str, Enum):
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"

class User(BaseModel):
    email: str
    sub: str  # Auth0 user ID
    roles: List[UserRole]
    stripe_customer_id: Optional[str] = None
    name: Optional[str] = None
    company: Optional[str] = None
    is_active: bool = True
    created_at: datetime = datetime.now()
    last_login: Optional[datetime] = None

class AuthConfig:
    """Configuration for Auth0 authentication"""
    def __init__(self):
        self.domain = os.getenv("AUTH0_DOMAIN")
        self.api_audience = os.getenv("AUTH0_AUDIENCE")
        self.client_id = os.getenv("AUTH0_CLIENT_ID")
        self.client_secret = os.getenv("AUTH0_CLIENT_SECRET")
        self.algorithms = ["RS256"]
        
        if not all([self.domain, self.api_audience, self.client_id, self.client_secret]):
            raise ValueError("Missing required Auth0 configuration")

class AuthService:
    def __init__(self):
        """Initialize the authentication service"""
        self.config = AuthConfig()
        self.oauth = OAuth()
        
        # Configure Auth0
        self.oauth.register(
            "auth0",
            client_id=self.config.client_id,
            client_secret=self.config.client_secret,
            api_base_url=f"https://{self.config.domain}",
            access_token_url=f"https://{self.config.domain}/oauth/token",
            authorize_url=f"https://{self.config.domain}/authorize",
            client_kwargs={
                "scope": "openid profile email",
            }
        )

    async def verify_token(self, token: str) -> Dict:
        """Verify the JWT token and return the payload"""
        try:
            # Get the public key from Auth0
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://{self.config.domain}/.well-known/jwks.json") as response:
                    jwks = await response.json()

            # Find the key that matches the token
            unverified_header = jwt.get_unverified_header(token)
            rsa_key = {}
            for key in jwks["keys"]:
                if key["kid"] == unverified_header["kid"]:
                    rsa_key = {
                        "kty": key["kty"],
                        "kid": key["kid"],
                        "n": key["n"],
                        "e": key["e"]
                    }
                    break

            if not rsa_key:
                raise JWTError("Unable to find appropriate key")

            # Verify the token
            payload = jwt.decode(
                token,
                rsa_key,
                algorithms=self.config.algorithms,
                audience=self.config.api_audience,
                issuer=f"https://{self.config.domain}/"
            )
            
            return payload

        except JWTError as e:
            logger.error(f"JWT verification failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except Exception as e:
            logger.error(f"Token verification failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def get_user_roles(self, token: str) -> List[UserRole]:
        """Get user roles from the token"""
        payload = await self.verify_token(token)
        permissions = payload.get("permissions", [])
        
        # Map permissions to roles
        roles = []
        if "read:basic" in permissions:
            roles.append(UserRole.BASIC)
        if "read:professional" in permissions:
            roles.append(UserRole.PROFESSIONAL)
        if "read:enterprise" in permissions:
            roles.append(UserRole.ENTERPRISE)
        if "admin" in permissions:
            roles.append(UserRole.ADMIN)
            
        return roles

    async def get_user_profile(self, token: str) -> Dict:
        """Get user profile from Auth0"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {token}"}
                async with session.get(
                    f"https://{self.config.domain}/userinfo",
                    headers=headers
                ) as response:
                    return await response.json()
        except Exception as e:
            logger.error(f"Failed to get user profile: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get user profile"
            )

    async def create_user(self, user_data: Dict) -> User:
        """Create a new user in Auth0"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get management API token
                token_data = {
                    "client_id": self.config.client_id,
                    "client_secret": self.config.client_secret,
                    "audience": f"https://{self.config.domain}/api/v2/",
                    "grant_type": "client_credentials"
                }
                async with session.post(
                    f"https://{self.config.domain}/oauth/token",
                    json=token_data
                ) as response:
                    token_response = await response.json()
                    mgmt_token = token_response["access_token"]

                # Create user
                headers = {
                    "Authorization": f"Bearer {mgmt_token}",
                    "Content-Type": "application/json"
                }
                async with session.post(
                    f"https://{self.config.domain}/api/v2/users",
                    headers=headers,
                    json=user_data
                ) as response:
                    auth0_user = await response.json()
                    
                    if response.status >= 400:
                        raise HTTPException(
                            status_code=response.status,
                            detail=auth0_user.get("message", "Failed to create user")
                        )
                    
                    return User(
                        email=auth0_user["email"],
                        sub=auth0_user["user_id"],
                        roles=[UserRole.BASIC],  # Default role
                        name=auth0_user.get("name"),
                        is_active=True
                    )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to create user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )

    async def update_user_metadata(self, user_id: str, metadata: Dict) -> Dict:
        """Update user metadata in Auth0"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get management API token
                token_data = {
                    "client_id": self.config.client_id,
                    "client_secret": self.config.client_secret,
                    "audience": f"https://{self.config.domain}/api/v2/",
                    "grant_type": "client_credentials"
                }
                async with session.post(
                    f"https://{self.config.domain}/oauth/token",
                    json=token_data
                ) as response:
                    token_response = await response.json()
                    mgmt_token = token_response["access_token"]

                # Update user
                headers = {
                    "Authorization": f"Bearer {mgmt_token}",
                    "Content-Type": "application/json"
                }
                async with session.patch(
                    f"https://{self.config.domain}/api/v2/users/{user_id}",
                    headers=headers,
                    json={"user_metadata": metadata}
                ) as response:
                    return await response.json()

        except Exception as e:
            logger.error(f"Failed to update user metadata: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update user metadata"
            )

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user from Auth0"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get management API token
                token_data = {
                    "client_id": self.config.client_id,
                    "client_secret": self.config.client_secret,
                    "audience": f"https://{self.config.domain}/api/v2/",
                    "grant_type": "client_credentials"
                }
                async with session.post(
                    f"https://{self.config.domain}/oauth/token",
                    json=token_data
                ) as response:
                    token_response = await response.json()
                    mgmt_token = token_response["access_token"]

                # Delete user
                headers = {"Authorization": f"Bearer {mgmt_token}"}
                async with session.delete(
                    f"https://{self.config.domain}/api/v2/users/{user_id}",
                    headers=headers
                ) as response:
                    return response.status == 204

        except Exception as e:
            logger.error(f"Failed to delete user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete user"
            )

    async def assign_roles(self, user_id: str, roles: List[str]) -> bool:
        """Assign roles to a user in Auth0"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get management API token
                token_data = {
                    "client_id": self.config.client_id,
                    "client_secret": self.config.client_secret,
                    "audience": f"https://{self.config.domain}/api/v2/",
                    "grant_type": "client_credentials"
                }
                async with session.post(
                    f"https://{self.config.domain}/oauth/token",
                    json=token_data
                ) as response:
                    token_response = await response.json()
                    mgmt_token = token_response["access_token"]

                # Assign roles
                headers = {
                    "Authorization": f"Bearer {mgmt_token}",
                    "Content-Type": "application/json"
                }
                async with session.post(
                    f"https://{self.config.domain}/api/v2/users/{user_id}/roles",
                    headers=headers,
                    json={"roles": roles}
                ) as response:
                    return response.status == 204

        except Exception as e:
            logger.error(f"Failed to assign roles: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to assign roles"
            )

# Initialize auth service
auth_service = AuthService()

# FastAPI dependency for getting current user
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=f"https://{auth_service.config.domain}/authorize",
    tokenUrl=f"https://{auth_service.config.domain}/oauth/token",
)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Dependency for getting the current authenticated user"""
    try:
        payload = await auth_service.verify_token(token)
        user_profile = await auth_service.get_user_profile(token)
        roles = await auth_service.get_user_roles(token)
        
        return User(
            email=user_profile["email"],
            sub=payload["sub"],
            roles=roles,
            name=user_profile.get("name"),
            stripe_customer_id=user_profile.get("https://api.eiendomsmuligheter.no/stripe_customer_id"),
            company=user_profile.get("https://api.eiendomsmuligheter.no/company"),
            is_active=True,
            last_login=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to get current user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )