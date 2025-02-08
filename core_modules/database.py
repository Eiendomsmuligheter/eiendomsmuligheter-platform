from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base
import enum
from datetime import datetime
import os
from typing import Optional, List, Dict

# Database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost/eiendomsmuligheter")

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=True)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Create base class for declarative models
Base = declarative_base()

# Enums for database
class PropertyType(str, enum.Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    MIXED = "mixed"

class AnalysisStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class SubscriptionType(str, enum.Enum):
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"

# Database Models
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    auth0_id = Column(String, unique=True)
    name = Column(String)
    company = Column(String, nullable=True)
    stripe_customer_id = Column(String, nullable=True)
    subscription_type = Column(SQLEnum(SubscriptionType), default=SubscriptionType.BASIC)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    properties = relationship("Property", back_populates="owner")
    analyses = relationship("PropertyAnalysis", back_populates="user")
    documents = relationship("Document", back_populates="user")
    subscriptions = relationship("Subscription", back_populates="user")

class Property(Base):
    __tablename__ = "properties"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    address = Column(String, index=True)
    municipality = Column(String, index=True)
    property_id = Column(String, index=True)  # gnr/bnr
    property_type = Column(SQLEnum(PropertyType))
    area = Column(Float)
    lot_size = Column(Float)
    floors = Column(Integer)
    has_basement = Column(Boolean)
    has_attic = Column(Boolean)
    year_built = Column(Integer)
    last_renovated = Column(Integer, nullable=True)
    coordinates = Column(String)  # Format: "latitude,longitude"
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    owner = relationship("User", back_populates="properties")
    analyses = relationship("PropertyAnalysis", back_populates="property")
    documents = relationship("Document", back_populates="property")
    floor_plans = relationship("FloorPlan", back_populates="property")

class PropertyAnalysis(Base):
    __tablename__ = "property_analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    property_id = Column(Integer, ForeignKey("properties.id"))
    status = Column(SQLEnum(AnalysisStatus), default=AnalysisStatus.PENDING)
    analysis_type = Column(String)  # e.g., "rental", "development", "renovation"
    results = Column(JSON)  # Stores the complete analysis results
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(String, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="analyses")
    property = relationship("Property", back_populates="analyses")
    documents = relationship("Document", back_populates="analysis")

class FloorPlan(Base):
    __tablename__ = "floor_plans"

    id = Column(Integer, primary_key=True, index=True)
    property_id = Column(Integer, ForeignKey("properties.id"))
    floor_number = Column(Integer)
    file_path = Column(String)
    area = Column(Float)
    rooms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    analysis_results = Column(JSON, nullable=True)  # Stores room detection results etc.
    
    # Relationships
    property = relationship("Property", back_populates="floor_plans")

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    property_id = Column(Integer, ForeignKey("properties.id"))
    analysis_id = Column(Integer, ForeignKey("property_analyses.id"), nullable=True)
    document_type = Column(String)  # e.g., "report", "application", "drawing"
    file_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="documents")
    property = relationship("Property", back_populates="documents")
    analysis = relationship("PropertyAnalysis", back_populates="documents")

class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    stripe_subscription_id = Column(String, unique=True)
    subscription_type = Column(SQLEnum(SubscriptionType))
    status = Column(String)  # e.g., "active", "canceled", "past_due"
    current_period_start = Column(DateTime)
    current_period_end = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    canceled_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="subscriptions")

class Municipality(Base):
    __tablename__ = "municipalities"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    regulations = Column(JSON)  # Stores zoning rules and building regulations
    api_endpoint = Column(String, nullable=True)
    api_key = Column(String, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow)

class PropertyHistory(Base):
    __tablename__ = "property_history"

    id = Column(Integer, primary_key=True, index=True)
    property_id = Column(Integer, ForeignKey("properties.id"))
    event_type = Column(String)  # e.g., "analysis", "document_generation", "update"
    event_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))

# Database dependency
async def get_db():
    """Dependency for getting async database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Create all tables
async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Drop all tables
async def drop_db():
    """Drop all database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

# Database migrations
async def run_migrations():
    """Run database migrations using Alembic"""
    from alembic.config import Config
    from alembic import command
    
    # Load Alembic configuration
    alembic_cfg = Config("alembic.ini")
    
    # Run migrations
    command.upgrade(alembic_cfg, "head")