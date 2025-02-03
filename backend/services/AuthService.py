from typing import Optional, Dict, Any
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from ..models.database import User

security = HTTPBearer()

class AuthService:
    def __init__(self):
        self.domain = "eiendomsmuligheter.eu.auth0.com"
        self.api_audience = "https://api.eiendomsmuligheter.no"
        self.algorithms = ["RS256"]
        self.jwks = self._get_jwks()
        
    def _get_jwks(self):
        import requests
        url = f"https://{self.domain}/.well-known/jwks.json"
        return requests.get(url).json()
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """
        Dekoder og verifiserer JWT token
        """
        try:
            unverified_header = jwt.get_unverified_header(token)
            rsa_key = {}
            
            for key in self.jwks["keys"]:
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
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Unable to find appropriate key"
                )
            
            payload = jwt.decode(
                token,
                rsa_key,
                algorithms=self.algorithms,
                audience=self.api_audience,
                issuer=f"https://{self.domain}/"
            )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
            
        except jwt.JWTClaimsError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid claims"
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e)
            )
    
    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: Session = None
    ) -> User:
        """
        Hent nåværende autentisert bruker
        """
        try:
            payload = self.decode_token(credentials.credentials)
            user_id: str = payload.get("sub")
            
            if user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials"
                )
                
            user = db.query(User).filter(User.auth0_id == user_id).first()
            
            if not user:
                # Opprett ny bruker hvis den ikke eksisterer
                user = User(
                    auth0_id=user_id,
                    email=payload.get("email"),
                    full_name=payload.get("name")
                )
                db.add(user)
                db.commit()
                db.refresh(user)
                
            return user
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e)
            )
    
    def verify_permissions(self, permissions: list) -> bool:
        """
        Verifiser at brukeren har nødvendige tillatelser
        """
        def inner(token: str = Depends(security)) -> bool:
            try:
                payload = self.decode_token(token.credentials)
                token_permissions = payload.get("permissions", [])
                
                for permission in permissions:
                    if permission not in token_permissions:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="Permission denied"
                        )
                        
                return True
                
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=str(e)
                )
        
        return inner