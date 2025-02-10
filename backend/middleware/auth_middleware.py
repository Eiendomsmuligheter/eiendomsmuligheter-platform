from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt
from jose.exceptions import JWTError
import os
from typing import Optional
import httpx
from functools import lru_cache

class Auth0Validator:
    """
    Validerer Auth0 JWT tokens og håndterer brukerautentisering
    """
    
    def __init__(self):
        self.domain = os.getenv('AUTH0_DOMAIN')
        self.audience = os.getenv('AUTH0_AUDIENCE')
        self.algorithms = ['RS256']
        self._jwks = None
    
    @lru_cache(maxsize=1)
    async def get_jwks(self) -> dict:
        """Hent JWKS (JSON Web Key Set) fra Auth0"""
        if not self._jwks:
            async with httpx.AsyncClient() as client:
                response = await client.get(f'https://{self.domain}/.well-known/jwks.json')
                self._jwks = response.json()
        return self._jwks
    
    def _get_signing_key(self, kid: str) -> Optional[dict]:
        """Hent signeringsnøkkel basert på key ID"""
        jwks = self._jwks['keys']
        for key in jwks:
            if key['kid'] == kid:
                return key
        return None
    
    async def validate_token(self, token: str) -> dict:
        """Valider JWT token og returner claims"""
        try:
            jwks = await self.get_jwks()
            unverified_header = jwt.get_unverified_header(token)
            rsa_key = self._get_signing_key(unverified_header['kid'])
            
            if not rsa_key:
                raise HTTPException(
                    status_code=401,
                    detail='Ugyldig autentiseringstoken'
                )
            
            payload = jwt.decode(
                token,
                rsa_key,
                algorithms=self.algorithms,
                audience=self.audience,
                issuer=f'https://{self.domain}/'
            )
            return payload
            
        except JWTError as e:
            raise HTTPException(
                status_code=401,
                detail=f'Kunne ikke validere token: {str(e)}'
            )

class Auth0Middleware(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
        self.validator = Auth0Validator()
    
    async def __call__(self, request: Request) -> dict:
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail='Ingen autentiseringstoken funnet'
            )
            
        if not credentials.scheme == "Bearer":
            raise HTTPException(
                status_code=401,
                detail='Feil autentiseringstype'
            )
            
        return await self.validator.validate_token(credentials.credentials)

# Opprett en instans av middleware
auth = Auth0Middleware()

# Roller og tilganger
def has_permission(required_permissions: list):
    """Dekoratør for å sjekke brukertilganger"""
    async def decorator(request: Request) -> bool:
        token = await auth(request)
        token_permissions = token.get('permissions', [])
        
        for permission in required_permissions:
            if permission not in token_permissions:
                raise HTTPException(
                    status_code=403,
                    detail='Mangler nødvendige tilganger'
                )
        return True
        
    return decorator

# Brukerroller
ROLES = {
    'basic': ['read:analyses'],
    'pro': ['read:analyses', 'write:analyses', 'read:reports'],
    'enterprise': ['read:analyses', 'write:analyses', 'read:reports', 'write:reports', 'access:api']
}