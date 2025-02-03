from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from ..database.base import Base

class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), unique=True)
    stripe_subscription_id = Column(String, unique=True)
    plan = Column(String, nullable=False)  # "basic", "pro", "enterprise"
    status = Column(String, nullable=False)  # "active", "cancelled", "past_due"
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    valid_until = Column(DateTime, nullable=True)

    # Relasjoner
    user = relationship("User", back_populates="subscription")