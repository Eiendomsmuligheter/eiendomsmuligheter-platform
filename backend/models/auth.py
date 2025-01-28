from pydantic import BaseModel, EmailStr, HttpUrl
from typing import List, Optional, Dict
from datetime import datetime

class UserProfile(BaseModel):
    """Brukerprofilmodell"""
    sub: str  # Auth0 bruker-ID
    email: EmailStr
    email_verified: bool
    name: Optional[str] = None
    nickname: Optional[str] = None
    picture: Optional[HttpUrl] = None
    updated_at: datetime
    
class TokenRequest(BaseModel):
    """Request for å få token"""
    code: str
    redirect_uri: HttpUrl
    
class TokenResponse(BaseModel):
    """Response med token"""
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: Optional[str] = None
    
class AuthResponse(BaseModel):
    """Komplett autentiseringsrespons"""
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user: UserProfile
    
class Permission(BaseModel):
    """Tillatelsesmodell"""
    name: str
    description: str
    
class Role(BaseModel):
    """Rollemodell"""
    id: str
    name: str
    description: Optional[str] = None
    permissions: List[Permission]
    
class UserPermissions(BaseModel):
    """Brukerens tillatelser"""
    user_id: str
    roles: List[Role]
    permissions: List[Permission]
    
class RoleAssignment(BaseModel):
    """Request for rolletildeling"""
    user_id: str
    role_name: str
    
class AuthSettings(BaseModel):
    """Autentiseringsinnstillinger"""
    auth0_domain: str
    auth0_client_id: str
    auth0_client_secret: str
    auth0_audience: str
    auth0_algorithm: str = "RS256"
    token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
class LoginRequest(BaseModel):
    """Login-forespørsel"""
    email: EmailStr
    password: str
    
class SignupRequest(BaseModel):
    """Registreringsforespørsel"""
    email: EmailStr
    password: str
    name: Optional[str] = None
    nickname: Optional[str] = None
    
class PasswordResetRequest(BaseModel):
    """Forespørsel om passordtilbakestilling"""
    email: EmailStr
    
class PasswordUpdateRequest(BaseModel):
    """Forespørsel om passordoppdatering"""
    user_id: str
    old_password: str
    new_password: str
    
class ProfileUpdateRequest(BaseModel):
    """Forespørsel om profiloppdatering"""
    user_id: str
    name: Optional[str] = None
    nickname: Optional[str] = None
    picture: Optional[HttpUrl] = None
    
class SecuritySettings(BaseModel):
    """Sikkerhetsinnstillinger for bruker"""
    user_id: str
    two_factor_enabled: bool = False
    two_factor_method: Optional[str] = None
    security_questions: Optional[Dict[str, str]] = None
    
class LoginAttempt(BaseModel):
    """Modell for innloggingsforsøk"""
    user_id: str
    timestamp: datetime
    ip_address: str
    success: bool
    user_agent: Optional[str] = None
    location: Optional[Dict[str, str]] = None
    
class SessionInfo(BaseModel):
    """Informasjon om aktiv økt"""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: Optional[str] = None
    
class AuthAuditLog(BaseModel):
    """Revisjonslogg for autentisering"""
    id: str
    timestamp: datetime
    event_type: str
    user_id: Optional[str] = None
    ip_address: str
    details: Dict
    severity: str