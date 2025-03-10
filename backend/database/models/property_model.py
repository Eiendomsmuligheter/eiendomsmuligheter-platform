"""
Databasemodeller for eiendommer og relaterte data
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON, Text, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from datetime import datetime
from typing import List, Optional

from ..database import Base

class PropertyType(enum.Enum):
    """Typer eiendommer"""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    LAND = "land"
    MIXED_USE = "mixed_use"

class PropertyStatus(enum.Enum):
    """Status for eiendommer"""
    ACTIVE = "active"
    PENDING = "pending"
    SOLD = "sold"
    ARCHIVED = "archived"

class Property(Base):
    """Modell for eiendommer"""
    __tablename__ = "properties"
    
    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(255), unique=True, index=True, nullable=True)
    address = Column(String(255), index=True, nullable=False)
    postal_code = Column(String(10), index=True)
    city = Column(String(100), index=True)
    municipality_id = Column(String(10), index=True)
    municipality_name = Column(String(100))
    
    property_type = Column(Enum(PropertyType), default=PropertyType.RESIDENTIAL)
    status = Column(Enum(PropertyStatus), default=PropertyStatus.ACTIVE)
    
    lot_size = Column(Float)
    building_size = Column(Float)
    floor_area_ratio = Column(Float)
    current_utilization = Column(Float)
    
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    zoning_category = Column(String(100), nullable=True)
    zoning_description = Column(Text, nullable=True)
    
    year_built = Column(Integer, nullable=True)
    last_renovation_year = Column(Integer, nullable=True)
    
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    owner = relationship("User", back_populates="properties")
    
    # GeoJSON-data for eiendomsgrenser
    geometry = Column(JSON, nullable=True)
    
    # Metadata (fleksibelt felt for ulike egenskaper)
    meta_data = Column(JSON, default={})
    
    # Images relatert til eiendommen (one-to-many)
    images = relationship("PropertyImage", back_populates="property", cascade="all, delete-orphan")
    
    # Analyser (one-to-many)
    analyses = relationship("PropertyAnalysis", back_populates="property", cascade="all, delete-orphan")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<Property(id={self.id}, address='{self.address}')>"

class PropertyImage(Base):
    """Modell for bilder knyttet til eiendommer"""
    __tablename__ = "property_images"
    
    id = Column(Integer, primary_key=True, index=True)
    property_id = Column(Integer, ForeignKey("properties.id"), nullable=False)
    property = relationship("Property", back_populates="images")
    
    file_path = Column(String(255), nullable=False)
    file_url = Column(String(255), nullable=True)
    file_type = Column(String(50), nullable=True)
    
    title = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)
    
    is_main_image = Column(Boolean, default=False)
    sort_order = Column(Integer, default=0)
    
    # Metadata (fleksibelt felt for ulike egenskaper)
    meta_data = Column(JSON, default={})
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<PropertyImage(id={self.id}, property_id={self.property_id})>"

class ZoningRegulation(Base):
    """Modell for reguleringer som gjelder eiendommer"""
    __tablename__ = "zoning_regulations"
    
    id = Column(Integer, primary_key=True, index=True)
    regulation_id = Column(String(100), index=True)
    municipality_id = Column(String(10), index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    status = Column(String(50), default="active")
    valid_from = Column(DateTime, nullable=False)
    valid_to = Column(DateTime, nullable=True)
    
    document_url = Column(String(255), nullable=True)
    
    # GeoJSON for reguleringens utstrekning
    geometry = Column(JSON, nullable=True)
    
    # Regler og verdier for reguleringen
    rules = Column(JSON, default=[])
    
    # Metadata (fleksibelt felt for ulike egenskaper)
    meta_data = Column(JSON, default={})
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<ZoningRegulation(id={self.id}, title='{self.title}')>" 