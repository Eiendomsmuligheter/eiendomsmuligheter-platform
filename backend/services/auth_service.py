import jwt
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import logging
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2AuthorizationCodeBearer
import aiohttp
import json
from ..models.auth import UserProfile, TokenResponse, AuthResponse
from jose import JWTError, jwt
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

class AuthService:
    """
    Håndterer autentisering og autorisasjon med Auth0.
    Støtter:
    - OAuth2 autentisering
    - JWT validering
    - Rollebasert tilgangskontroll
    - Brukerprofilhåndtering
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.domain = self.config["auth0"]["domain"]
        self.client_id = self.config["auth0"]["client_id"]
        self.client_secret = self.config["auth0"]["client_secret"]
        self.algorithm = self.config["auth0"]["algorithm"]
        self.audience = self.config["auth0"]["audience"]
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Last autentiseringskonfigurasjon"""
        if config_path:
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            "auth0": {
                "domain": "${AUTH0_DOMAIN}",
                "client_id": "${AUTH0_CLIENT_ID}",
                "client_secret": "${AUTH0_CLIENT_SECRET}",
                "algorithm": "RS256",
                "audience": "${AUTH0_AUDIENCE}",
                "scopes": {
                    "read:analysis": "Lese analyseresultater",
                    "write:analysis": "Utføre nye analyser",
                    "read:properties": "Se eiendomsinformasjon",
                    "admin:system": "Administrere systemet"
                }
            },
            "jwt": {
                "access_token_expire_minutes": 30,
                "refresh_token_expire_days": 7
            }
        }
        
    async def authenticate_user(self,
                              code: str,
                              redirect_uri: str) -> AuthResponse:
        """
        Autentiser bruker med Auth0 authorization code
        """
        try:
            # Hent access token fra Auth0
            token_response = await self._get_token(code, redirect_uri)
            
            # Hent brukerprofil
            user_profile = await self._get_user_profile(
                token_response["access_token"]
            )
            
            # Opprett eller oppdater lokal brukerprofil
            await self._sync_user_profile(user_profile)
            
            return AuthResponse(
                access_token=token_response["access_token"],
                refresh_token=token_response["refresh_token"],
                token_type="Bearer",
                expires_in=token_response["expires_in"],
                user=user_profile
            )
            
        except Exception as e:
            logger.error(f"Autentiseringsfeil: {str(e)}")
            raise HTTPException(
                status_code=401,
                detail=f"Autentisering feilet: {str(e)}"
            )
            
    async def refresh_token(self, refresh_token: str) -> TokenResponse:
        """
        Forny access token med refresh token
        """
        try:
            # Valider refresh token
            payload = jwt.decode(
                refresh_token,
                self.client_secret,
                algorithms=[self.algorithm]
            )
            
            # Sjekk om token er utløpt
            if datetime.fromtimestamp(payload["exp"]) < datetime.utcnow():
                raise HTTPException(
                    status_code=401,
                    detail="Refresh token er utløpt"
                )
                
            # Hent ny access token fra Auth0
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"https://{self.domain}/oauth/token",
                    json={
                        "grant_type": "refresh_token",
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "refresh_token": refresh_token
                    }
                ) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=401,
                            detail="Kunne ikke fornye token"
                        )
                        
                    token_data = await response.json()
                    
            return TokenResponse(
                access_token=token_data["access_token"],
                token_type="Bearer",
                expires_in=token_data["expires_in"]
            )
            
        except JWTError:
            raise HTTPException(
                status_code=401,
                detail="Ugyldig refresh token"
            )
            
    async def validate_token(self, token: str) -> Dict:
        """
        Valider JWT token og returner payload
        """
        try:
            # Hent Auth0 public key
            jwks = await self._get_jwks()
            
            # Dekod token header
            unverified_header = jwt.get_unverified_header(token)
            
            # Finn riktig public key
            rsa_key = {}
            for key in jwks["keys"]:
                if key["kid"] == unverified_header["kid"]:
                    rsa_key = {
                        "kty": key["kty"],
                        "kid": key["kid"],
                        "use": key["use"],
                        "n": key["n"],
                        "e": key["e"]
                    }
                    break
                    
            if not rsa_key:
                raise HTTPException(
                    status_code=401,
                    detail="Kunne ikke finne gyldig public key"
                )
                
            # Valider token
            payload = jwt.decode(
                token,
                rsa_key,
                algorithms=[self.algorithm],
                audience=self.audience,
                issuer=f"https://{self.domain}/"
            )
            
            return payload
            
        except JWTError:
            raise HTTPException(
                status_code=401,
                detail="Ugyldig token"
            )
            
    async def get_user_permissions(self, user_id: str) -> List[str]:
        """
        Hent brukerens tillatelser fra Auth0
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://{self.domain}/api/v2/users/{user_id}/permissions",
                    headers={
                        "Authorization": f"Bearer {await self._get_management_token()}"
                    }
                ) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=400,
                            detail="Kunne ikke hente brukertillatelser"
                        )
                        
                    permissions = await response.json()
                    return [p["permission_name"] for p in permissions]
                    
        except Exception as e:
            logger.error(f"Feil ved henting av brukertillatelser: {str(e)}")
            raise
            
    async def assign_role(self,
                         user_id: str,
                         role_name: str) -> Dict:
        """
        Tildel rolle til bruker
        """
        try:
            # Hent rolle-ID fra Auth0
            role_id = await self._get_role_id(role_name)
            
            # Tildel rolle til bruker
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"https://{self.domain}/api/v2/users/{user_id}/roles",
                    headers={
                        "Authorization": f"Bearer {await self._get_management_token()}"
                    },
                    json={
                        "roles": [role_id]
                    }
                ) as response:
                    if response.status != 204:
                        raise HTTPException(
                            status_code=400,
                            detail="Kunne ikke tildele rolle"
                        )
                        
            return {
                "success": True,
                "message": f"Rolle {role_name} tildelt til bruker {user_id}"
            }
            
        except Exception as e:
            logger.error(f"Feil ved tildeling av rolle: {str(e)}")
            raise
            
    async def _get_token(self, code: str, redirect_uri: str) -> Dict:
        """Hent access token fra Auth0"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://{self.domain}/oauth/token",
                json={
                    "grant_type": "authorization_code",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri
                }
            ) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=401,
                        detail="Kunne ikke hente token"
                    )
                    
                return await response.json()
                
    async def _get_user_profile(self, access_token: str) -> UserProfile:
        """Hent brukerprofil fra Auth0"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://{self.domain}/userinfo",
                headers={
                    "Authorization": f"Bearer {access_token}"
                }
            ) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=401,
                        detail="Kunne ikke hente brukerprofil"
                    )
                    
                profile_data = await response.json()
                return UserProfile(**profile_data)
                
    async def _sync_user_profile(self, profile: UserProfile):
        """Synkroniser brukerprofil med lokal database"""
        # TODO: Implementer databasesynkronisering
        pass
        
    async def _get_jwks(self) -> Dict:
        """Hent JSON Web Key Set fra Auth0"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://{self.domain}/.well-known/jwks.json"
            ) as response:
                return await response.json()
                
    async def _get_management_token(self) -> str:
        """Hent management API token fra Auth0"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://{self.domain}/oauth/token",
                json={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "audience": f"https://{self.domain}/api/v2/"
                }
            ) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=500,
                        detail="Kunne ikke hente management token"
                    )
                    
                token_data = await response.json()
                return token_data["access_token"]
                
    async def _get_role_id(self, role_name: str) -> str:
        """Hent rolle-ID fra Auth0"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://{self.domain}/api/v2/roles",
                headers={
                    "Authorization": f"Bearer {await self._get_management_token()}"
                },
                params={
                    "name": role_name
                }
            ) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Kunne ikke finne rolle: {role_name}"
                    )
                    
                roles = await response.json()
                if not roles:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Rolle ikke funnet: {role_name}"
                    )
                    
                return roles[0]["id"]