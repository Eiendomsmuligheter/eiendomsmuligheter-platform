from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from ..database import Base
import enum

class PaymentStatus(enum.Enum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REFUNDED = "refunded"

class Payment(Base):
    __tablename__ = "payments"

    id = Column(Integer, primary_key=True, index=True)
    stripe_payment_id = Column(String, unique=True, index=True)
    amount = Column(Integer, nullable=False)  # Amount in øre
    currency = Column(String, nullable=False, default="nok")
    customer_id = Column(String, index=True)
    status = Column(Enum(PaymentStatus), nullable=False, default=PaymentStatus.PENDING)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=True)
    
    # Relasjoner
    user_id = Column(Integer, ForeignKey("users.id"))
    property_id = Column(Integer, ForeignKey("properties.id"))
    
    user = relationship("User", back_populates="payments")
    property = relationship("Property", back_populates="payments")

class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    stripe_subscription_id = Column(String, unique=True, index=True)
    status = Column(String, nullable=False)
    current_period_start = Column(DateTime, nullable=False)
    current_period_end = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False)
    canceled_at = Column(DateTime, nullable=True)
    
    # Relasjoner
    user_id = Column(Integer, ForeignKey("users.id"))
    plan_id = Column(Integer, ForeignKey("subscription_plans.id"))
    
    user = relationship("User", back_populates="subscriptions")
    plan = relationship("SubscriptionPlan", back_populates="subscriptions")

class SubscriptionPlan(Base):
    __tablename__ = "subscription_plans"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    stripe_price_id = Column(String, unique=True)
    price = Column(Integer, nullable=False)  # Price in øre
    currency = Column(String, nullable=False, default="nok")
    interval = Column(String, nullable=False)  # monthly, yearly
    features = Column(String, nullable=False)  # JSON string of features
    
    subscriptions = relationship("Subscription", back_populates="plan")