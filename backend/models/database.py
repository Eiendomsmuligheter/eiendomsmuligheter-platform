from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://dbuser:dbpassword@localhost/eiendomsmuligheter")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    subscription_id = Column(String, nullable=True)
    subscription_status = Column(String, nullable=True)
    subscription_end_date = Column(DateTime, nullable=True)
    
    analyses = relationship("PropertyAnalysis", back_populates="user")
    payments = relationship("Payment", back_populates="user")

class PropertyAnalysis(Base):
    __tablename__ = "property_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    address = Column(String)
    municipality = Column(String)
    gnr = Column(String)
    bnr = Column(String)
    property_data = Column(JSON)
    analysis_results = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String)  # pending, processing, completed, failed
    
    user = relationship("User", back_populates="analyses")
    documents = relationship("Document", back_populates="analysis")

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("property_analyses.id"))
    document_type = Column(String)  # building_application, situation_plan, etc.
    file_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String)  # generated, signed, submitted
    
    analysis = relationship("PropertyAnalysis", back_populates="documents")

class Payment(Base):
    __tablename__ = "payments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    stripe_payment_id = Column(String, unique=True)
    amount = Column(Float)
    currency = Column(String)
    status = Column(String)
    payment_method = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="payments")

class Subscription(Base):
    __tablename__ = "subscriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    stripe_subscription_id = Column(String, unique=True)
    stripe_customer_id = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))
    plan_id = Column(String)
    status = Column(String)
    current_period_start = Column(DateTime)
    current_period_end = Column(DateTime)
    cancel_at_period_end = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()