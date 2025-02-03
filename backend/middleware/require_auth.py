from fastapi import Request, HTTPException
from functools import wraps
from ..services.AuthService import AuthService

auth_service = AuthService()

def require_auth(func):
    @wraps(func)
    async def wrapper(*args, request: Request, **kwargs):
        try:
            # Hent token fra Authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                raise HTTPException(status_code=401, detail="No authorization header")
            
            # Fjern "Bearer " fra token
            token = auth_header.split(" ")[1]
            
            # Valider token og hent payload
            payload = auth_service.decode_token(token)
            
            # Legg til bruker i request state
            request.state.user = payload
            
            return await func(*args, request=request, **kwargs)
            
        except Exception as e:
            raise HTTPException(status_code=401, detail=str(e))
    
    return wrapper

def require_permissions(permissions: list):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, request: Request, **kwargs):
            try:
                # Hent token fra Authorization header
                auth_header = request.headers.get("Authorization")
                if not auth_header:
                    raise HTTPException(status_code=401, detail="No authorization header")
                
                # Fjern "Bearer " fra token
                token = auth_header.split(" ")[1]
                
                # Valider token og hent payload
                payload = auth_service.decode_token(token)
                
                # Sjekk tillatelser
                token_permissions = payload.get("permissions", [])
                for permission in permissions:
                    if permission not in token_permissions:
                        raise HTTPException(
                            status_code=403,
                            detail=f"Missing required permission: {permission}"
                        )
                
                # Legg til bruker i request state
                request.state.user = payload
                
                return await func(*args, request=request, **kwargs)
                
            except Exception as e:
                raise HTTPException(status_code=401, detail=str(e))
        
        return wrapper
    return decorator