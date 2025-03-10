"""
Databasemodeller for brukere og autentisering
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, JSON, Text, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from typing import List, Optional

from ..database import Base

class UserRole(enum.Enum):
    """Brukerroller i systemet"""
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    GUEST = "guest"

class SubscriptionPlan(enum.Enum):
    """Tilgjengelige abonnementstyper"""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"

class User(Base):
    """Modell for brukere"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    
    # Hashet passord
    hashed_password = Column(String(255), nullable=False)
    
    # Personlige detaljer
    first_name = Column(String(100))
    last_name = Column(String(100))
    phone = Column(String(20))
    company = Column(String(100))
    job_title = Column(String(100))
    
    # Profilbilde
    profile_image = Column(String(255))
    
    # Rolle og tilganger
    role = Column(Enum(UserRole), default=UserRole.USER)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Abonnement
    subscription_plan = Column(Enum(SubscriptionPlan), default=SubscriptionPlan.FREE)
    subscription_expires = Column(DateTime, nullable=True)
    subscription_auto_renew = Column(Boolean, default=False)
    
    # API Nøkkel for programmatisk tilgang
    api_key = Column(String(100), unique=True, index=True, nullable=True)
    
    # Brukerpreferanser
    preferences = Column(JSON, default={})
    
    # Relasjoner
    properties = relationship("Property", back_populates="owner")
    payment_methods = relationship("PaymentMethod", back_populates="user")
    
    # Tokens for passordtilbakestilling, etc.
    tokens = relationship("UserToken", back_populates="user", cascade="all, delete-orphan")
    
    # Tidsstempler
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', role={self.role})>"

class PaymentMethod(Base):
    """Modell for betalingsmetoder"""
    __tablename__ = "payment_methods"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="payment_methods")
    
    # Type betalingsmetode
    payment_type = Column(String(50), nullable=False)  # "card", "paypal", "vipps", etc.
    
    # For kort (kryptert eller token)
    card_last4 = Column(String(4), nullable=True)
    card_brand = Column(String(20), nullable=True)
    card_exp_month = Column(Integer, nullable=True)
    card_exp_year = Column(Integer, nullable=True)
    
    # For andre betalingsmåter
    payment_token = Column(String(255), nullable=True)
    
    # Metadata
    is_default = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    
    # Tidsstempler
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<PaymentMethod(id={self.id}, user_id={self.user_id}, type='{self.payment_type}')>"

class UserToken(Base):
    """Modell for brukerens tokens (passordtilbakestilling, etc.)"""
    __tablename__ = "user_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="tokens")
    
    token = Column(String(255), unique=True, index=True, nullable=False)
    token_type = Column(String(50), nullable=False)  # "reset_password", "verification", etc.
    
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_used = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<UserToken(id={self.id}, user_id={self.user_id}, type='{self.token_type}')>" 