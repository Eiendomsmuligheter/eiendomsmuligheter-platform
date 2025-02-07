from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, ForeignKey, Enum, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .base import Base
import enum
import uuid
from datetime import datetime, timedelta

def generate_uuid():
    return str(uuid.uuid4())

class SubscriptionTier(enum.Enum):
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"

class SubscriptionStatus(enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    CANCELLED = "cancelled"
    PAST_DUE = "past_due"

class BillingInterval(enum.Enum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    # Abonnementsinformasjon
    tier = Column(Enum(SubscriptionTier), nullable=False)
    status = Column(Enum(SubscriptionStatus), default=SubscriptionStatus.PENDING)
    billing_interval = Column(Enum(BillingInterval), nullable=False)
    
    # Priser og fakturering
    base_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)  # Pris etter eventuelle rabatter
    currency = Column(String, default="NOK")
    tax_rate = Column(Float)
    
    # Stripe-spesifikk informasjon
    stripe_subscription_id = Column(String)
    stripe_customer_id = Column(String)
    stripe_price_id = Column(String)
    
    # Abonnementsperiode
    start_date = Column(DateTime(timezone=True), nullable=False)
    current_period_start = Column(DateTime(timezone=True))
    current_period_end = Column(DateTime(timezone=True))
    trial_end = Column(DateTime(timezone=True))
    cancelled_at = Column(DateTime(timezone=True))
    
    # Funksjonalitetsbegrensninger
    max_properties = Column(Integer)
    max_analyses = Column(Integer)
    max_documents = Column(Integer)
    features = Column(JSON)  # Liste over tilgjengelige funksjoner
    
    # Automatisk fornyelse
    auto_renew = Column(Boolean, default=True)
    renewal_reminder_sent = Column(Boolean, default=False)
    
    # Betalingshistorikk og metadata
    payment_history = Column(JSON)
    metadata = Column(JSON)
    
    # Tidsstempler
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relasjoner
    user = relationship("User", back_populates="subscription")
    payments = relationship("Payment", back_populates="subscription")
    
    def __repr__(self):
        return f"<Subscription {self.user_id} - {self.tier.value}>"
        
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "subscription": {
                "tier": self.tier.value,
                "status": self.status.value,
                "billing_interval": self.billing_interval.value,
                "auto_renew": self.auto_renew
            },
            "pricing": {
                "base_price": self.base_price,
                "current_price": self.current_price,
                "currency": self.currency,
                "tax_rate": self.tax_rate
            },
            "period": {
                "start_date": self.start_date,
                "current_period_start": self.current_period_start,
                "current_period_end": self.current_period_end,
                "trial_end": self.trial_end,
                "cancelled_at": self.cancelled_at
            },
            "limits": {
                "max_properties": self.max_properties,
                "max_analyses": self.max_analyses,
                "max_documents": self.max_documents,
                "features": self.features
            },
            "stripe": {
                "subscription_id": self.stripe_subscription_id,
                "customer_id": self.stripe_customer_id,
                "price_id": self.stripe_price_id
            }
        }

    @staticmethod
    def get_tier_features(tier: SubscriptionTier):
        """Henter funksjoner og begrensninger for et abonnementsnivå"""
        features = {
            SubscriptionTier.FREE: {
                "max_properties": 1,
                "max_analyses": 2,
                "max_documents": 3,
                "features": {
                    "basic_analysis": True,
                    "document_generation": False,
                    "energy_analysis": False,
                    "3d_visualization": False
                }
            },
            SubscriptionTier.BASIC: {
                "max_properties": 5,
                "max_analyses": 10,
                "max_documents": 15,
                "features": {
                    "basic_analysis": True,
                    "document_generation": True,
                    "energy_analysis": True,
                    "3d_visualization": False
                }
            },
            SubscriptionTier.PROFESSIONAL: {
                "max_properties": 20,
                "max_analyses": 50,
                "max_documents": 100,
                "features": {
                    "basic_analysis": True,
                    "document_generation": True,
                    "energy_analysis": True,
                    "3d_visualization": True,
                    "api_access": True
                }
            },
            SubscriptionTier.ENTERPRISE: {
                "max_properties": None,  # Ubegrenset
                "max_analyses": None,    # Ubegrenset
                "max_documents": None,   # Ubegrenset
                "features": {
                    "basic_analysis": True,
                    "document_generation": True,
                    "energy_analysis": True,
                    "3d_visualization": True,
                    "api_access": True,
                    "priority_support": True,
                    "custom_integration": True
                }
            }
        }
        return features.get(tier, features[SubscriptionTier.FREE])

    def update_subscription_limits(self):
        """Oppdaterer abonnementsgrenser basert på valgt nivå"""
        tier_features = self.get_tier_features(self.tier)
        self.max_properties = tier_features["max_properties"]
        self.max_analyses = tier_features["max_analyses"]
        self.max_documents = tier_features["max_documents"]
        self.features = tier_features["features"]

    def can_perform_action(self, action: str, current_usage: dict) -> bool:
        """Sjekker om en handling er tillatt under gjeldende abonnement"""
        if self.status != SubscriptionStatus.ACTIVE:
            return False
            
        if action == "add_property":
            return (self.max_properties is None or 
                    current_usage.get("properties", 0) < self.max_properties)
                    
        elif action == "perform_analysis":
            return (self.max_analyses is None or 
                    current_usage.get("analyses", 0) < self.max_analyses)
                    
        elif action == "generate_document":
            return (self.max_documents is None or 
                    current_usage.get("documents", 0) < self.max_documents)
                    
        elif action in self.features:
            return self.features[action]
            
        return False

    def calculate_next_billing_date(self) -> datetime:
        """Beregner neste faktureringsdato"""
        if not self.current_period_end:
            return None
            
        if self.billing_interval == BillingInterval.MONTHLY:
            return self.current_period_end + timedelta(days=30)
        elif self.billing_interval == BillingInterval.QUARTERLY:
            return self.current_period_end + timedelta(days=90)
        elif self.billing_interval == BillingInterval.YEARLY:
            return self.current_period_end + timedelta(days=365)
            
        return None

    def cancel_subscription(self, immediate: bool = False):
        """Kansellerer abonnementet"""
        self.cancelled_at = func.now()
        
        if immediate:
            self.status = SubscriptionStatus.CANCELLED
            self.current_period_end = func.now()
        else:
            # La abonnementet løpe ut perioden
            self.auto_renew = False
            
        # Her ville vi også integrere med Stripe for å kansellere der