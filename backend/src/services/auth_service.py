from typing import Dict, Optional
import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
import os

class AuthService:
    def __init__(self):
        self.auth0_domain = os.getenv('AUTH0_DOMAIN')
        self.api_audience = os.getenv('AUTH0_AUDIENCE')
        self.algorithms = ['RS256']
        self.security = HTTPBearer()

    async def get_public_key(self) -> str:
        """Henter Auth0 public key for token validering"""
        # I produksjon bør denne caches
        try:
            jwks_url = f'https://{self.auth0_domain}/.well-known/jwks.json'
            jwks_client = jwt.PyJWKClient(jwks_url)
            return jwks_client
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def verify_token(self, auth: HTTPAuthorizationCredentials = Security(HTTPBearer())) -> Dict:
        """Validerer JWT token og returnerer brukerinfo"""
        try:
            token = auth.credentials
            jwks_client = await self.get_public_key()
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=self.algorithms,
                audience=self.api_audience,
                issuer=f'https://{self.auth0_domain}/'
            )
            
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail='Token er utløpt')
        except jwt.JWTClaimsError:
            raise HTTPException(status_code=401, detail='Ugyldige token claims')
        except Exception as e:
            raise HTTPException(status_code=401, detail=str(e))

    async def get_user_permissions(self, token_payload: Dict) -> list:
        """Henter brukerrettigheter fra token"""
        return token_payload.get('permissions', [])