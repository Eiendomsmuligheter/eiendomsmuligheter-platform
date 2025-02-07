from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import OAuth2AuthorizationCodeBearer
from typing import List
from ..services.auth_service import AuthService
from ..models.auth import (
    AuthResponse,
    TokenRequest,
    TokenResponse,
    UserProfile,
    UserPermissions,
    RoleAssignment,
    LoginRequest,
    SignupRequest,
    PasswordResetRequest,
    PasswordUpdateRequest,
    ProfileUpdateRequest,
    SecuritySettings,
    SessionInfo
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["authentication"])
auth_service = AuthService()

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=f"https://{auth_service.domain}/authorize",
    tokenUrl=f"https://{auth_service.domain}/oauth/token"
)

@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest) -> AuthResponse:
    """
    Autentiser bruker med e-post og passord
    """
    try:
        return await auth_service.authenticate_user(
            request.email,
            request.password
        )
    except Exception as e:
        logger.error(f"Login feilet: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail=f"Login feilet: {str(e)}"
        )

@router.post("/signup", response_model=AuthResponse)
async def signup(request: SignupRequest) -> AuthResponse:
    """
    Registrer ny bruker
    """
    try:
        return await auth_service.create_user(
            email=request.email,
            password=request.password,
            name=request.name,
            nickname=request.nickname
        )
    except Exception as e:
        logger.error(f"Registrering feilet: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Registrering feilet: {str(e)}"
        )

@router.post("/token", response_model=TokenResponse)
async def get_token(request: TokenRequest) -> TokenResponse:
    """
    Hent token med autorisasjonskode
    """
    try:
        return await auth_service.get_token(
            code=request.code,
            redirect_uri=str(request.redirect_uri)
        )
    except Exception as e:
        logger.error(f"Token-henting feilet: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Kunne ikke hente token: {str(e)}"
        )

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(token: str = Depends(oauth2_scheme)) -> TokenResponse:
    """
    Forny access token med refresh token
    """
    try:
        return await auth_service.refresh_token(token)
    except Exception as e:
        logger.error(f"Token-fornyelse feilet: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail=f"Kunne ikke fornye token: {str(e)}"
        )

@router.get("/profile", response_model=UserProfile)
async def get_profile(token: str = Depends(oauth2_scheme)) -> UserProfile:
    """
    Hent brukerprofil
    """
    try:
        return await auth_service.get_user_profile(token)
    except Exception as e:
        logger.error(f"Profilhenting feilet: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Kunne ikke hente profil: {str(e)}"
        )

@router.put("/profile", response_model=UserProfile)
async def update_profile(
    request: ProfileUpdateRequest,
    token: str = Depends(oauth2_scheme)
) -> UserProfile:
    """
    Oppdater brukerprofil
    """
    try:
        return await auth_service.update_user_profile(
            token,
            request.dict(exclude_unset=True)
        )
    except Exception as e:
        logger.error(f"Profiloppdatering feilet: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Kunne ikke oppdatere profil: {str(e)}"
        )

@router.post("/password/reset", response_model=dict)
async def reset_password(request: PasswordResetRequest) -> dict:
    """
    Send e-post for passordtilbakestilling
    """
    try:
        await auth_service.send_password_reset_email(request.email)
        return {"message": "E-post for passordtilbakestilling er sendt"}
    except Exception as e:
        logger.error(f"Passordtilbakestilling feilet: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Kunne ikke tilbakestille passord: {str(e)}"
        )

@router.put("/password", response_model=dict)
async def update_password(
    request: PasswordUpdateRequest,
    token: str = Depends(oauth2_scheme)
) -> dict:
    """
    Oppdater passord
    """
    try:
        await auth_service.update_password(
            token,
            request.old_password,
            request.new_password
        )
        return {"message": "Passord oppdatert"}
    except Exception as e:
        logger.error(f"Passordoppdatering feilet: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Kunne ikke oppdatere passord: {str(e)}"
        )

@router.get("/permissions", response_model=UserPermissions)
async def get_permissions(token: str = Depends(oauth2_scheme)) -> UserPermissions:
    """
    Hent brukerens tillatelser
    """
    try:
        return await auth_service.get_user_permissions(token)
    except Exception as e:
        logger.error(f"Tillatelseshenting feilet: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Kunne ikke hente tillatelser: {str(e)}"
        )

@router.post("/roles", response_model=dict)
async def assign_role(
    request: RoleAssignment,
    token: str = Depends(oauth2_scheme)
) -> dict:
    """
    Tildel rolle til bruker
    """
    try:
        await auth_service.assign_role(
            request.user_id,
            request.role_name
        )
        return {"message": f"Rolle {request.role_name} tildelt"}
    except Exception as e:
        logger.error(f"Rolletildeling feilet: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Kunne ikke tildele rolle: {str(e)}"
        )

@router.get("/sessions", response_model=List[SessionInfo])
async def get_sessions(token: str = Depends(oauth2_scheme)) -> List[SessionInfo]:
    """
    Hent aktive økter for brukeren
    """
    try:
        return await auth_service.get_user_sessions(token)
    except Exception as e:
        logger.error(f"Øktlistehenting feilet: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Kunne ikke hente økter: {str(e)}"
        )

@router.delete("/sessions/{session_id}", response_model=dict)
async def revoke_session(
    session_id: str,
    token: str = Depends(oauth2_scheme)
) -> dict:
    """
    Avslutt en spesifikk økt
    """
    try:
        await auth_service.revoke_session(token, session_id)
        return {"message": "Økt avsluttet"}
    except Exception as e:
        logger.error(f"Øktavslutning feilet: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Kunne ikke avslutte økt: {str(e)}"
        )

@router.put("/security", response_model=SecuritySettings)
async def update_security_settings(
    settings: SecuritySettings,
    token: str = Depends(oauth2_scheme)
) -> SecuritySettings:
    """
    Oppdater sikkerhetsinnstillinger
    """
    try:
        return await auth_service.update_security_settings(token, settings)
    except Exception as e:
        logger.error(f"Sikkerhetsoppdatering feilet: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Kunne ikke oppdatere sikkerhetsinnstillinger: {str(e)}"
        )