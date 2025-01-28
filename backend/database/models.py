"""
Database models for the platform
"""
from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, ForeignKey, Boolean, Table
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Property(Base):
    __tablename__ = "properties"

    id = Column(Integer, primary_key=True, index=True)
    address = Column(String, index=True)
    municipality = Column(String, index=True)
    gnr = Column(Integer)
    bnr = Column(Integer)
    property_type = Column(String)
    total_area = Column(Float)
    building_area = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    analyses = relationship("Analysis", back_populates="property")
    floor_plans = relationship("FloorPlan", back_populates="property")
    documents = relationship("Document", back_populates="property")

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    property_id = Column(Integer, ForeignKey("properties.id"))
    analysis_type = Column(String)  # development, rental, energy, etc.
    status = Column(String)  # pending, completed, failed
    result = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Relationships
    property = relationship("Property", back_populates="analyses")
    recommendations = relationship("Recommendation", back_populates="analysis")

class FloorPlan(Base):
    __tablename__ = "floor_plans"

    id = Column(Integer, primary_key=True, index=True)
    property_id = Column(Integer, ForeignKey("properties.id"))
    floor_number = Column(Integer)
    area = Column(Float)
    ceiling_height = Column(Float)
    file_path = Column(String)
    model_url = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    property = relationship("Property", back_populates="floor_plans")
    rooms = relationship("Room", back_populates="floor_plan")

class Room(Base):
    __tablename__ = "rooms"

    id = Column(Integer, primary_key=True, index=True)
    floor_plan_id = Column(Integer, ForeignKey("floor_plans.id"))
    name = Column(String)
    area = Column(Float)
    width = Column(Float)
    length = Column(Float)
    height = Column(Float)
    window_area = Column(Float)
    door_count = Column(Integer)
    room_type = Column(String)  # bedroom, bathroom, kitchen, etc.
    
    # Relationships
    floor_plan = relationship("FloorPlan", back_populates="rooms")

class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"))
    title = Column(String)
    description = Column(String)
    potential_value = Column(Float)
    complexity = Column(String)  # low, medium, high
    estimated_cost = Column(Float)
    estimated_timeframe = Column(String)
    requires_dispensation = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="recommendations")

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    property_id = Column(Integer, ForeignKey("properties.id"))
    document_type = Column(String)  # building_application, floor_plan, facade, etc.
    title = Column(String)
    file_path = Column(String)
    status = Column(String)  # draft, final
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    property = relationship("Property", back_populates="documents")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    analyses = relationship("Analysis", secondary="user_analyses")

# Association tables
user_analyses = Table('user_analyses', Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('analysis_id', Integer, ForeignKey('analyses.id'))
)