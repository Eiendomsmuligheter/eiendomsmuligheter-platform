from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .base import Base
import uuid

def generate_uuid():
    return str(uuid.uuid4())

class Property(Base):
    __tablename__ = "properties"

    id = Column(String, primary_key=True, default=generate_uuid)
    owner_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    # Eiendomsinformasjon
    address = Column(String, nullable=False)
    municipality_code = Column(String, nullable=False)
    municipality_name = Column(String)
    property_number = Column(String)  # gnr/bnr
    plot_size = Column(Float)
    total_area = Column(Float)
    build_year = Column(Integer)
    
    # Koordinater
    latitude = Column(Float)
    longitude = Column(Float)
    
    # Bygningsinformasjon
    building_type = Column(String)
    floors = Column(Integer)
    has_basement = Column(Boolean)
    has_attic = Column(Boolean)
    current_usage = Column(String)
    
    # Teknisk informasjon
    energy_rating = Column(String)
    energy_consumption = Column(Float)
    building_standard = Column(String)  # TEK10, TEK17, etc.
    
    # Regulering
    zoning_status = Column(String)
    allowed_utilization = Column(Float)  # %-BYA
    height_restrictions = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_analyzed_at = Column(DateTime(timezone=True))
    
    # Relasjoner
    owner = relationship("User", back_populates="properties")
    analyses = relationship("Analysis", back_populates="property")
    documents = relationship("Document", back_populates="property")
    
    # JSON-felter for detaljert informasjon
    building_details = Column(JSON)  # Detaljert bygningsinformasjon
    regulation_details = Column(JSON)  # Detaljerte reguleringsbestemmelser
    technical_details = Column(JSON)  # Tekniske detaljer
    
    def __repr__(self):
        return f"<Property {self.address}>"
        
    def to_dict(self):
        return {
            "id": self.id,
            "address": self.address,
            "municipality": {
                "code": self.municipality_code,
                "name": self.municipality_name
            },
            "property_number": self.property_number,
            "plot_size": self.plot_size,
            "total_area": self.total_area,
            "build_year": self.build_year,
            "location": {
                "latitude": self.latitude,
                "longitude": self.longitude
            },
            "building": {
                "type": self.building_type,
                "floors": self.floors,
                "has_basement": self.has_basement,
                "has_attic": self.has_attic,
                "current_usage": self.current_usage,
                "details": self.building_details
            },
            "technical": {
                "energy_rating": self.energy_rating,
                "energy_consumption": self.energy_consumption,
                "building_standard": self.building_standard,
                "details": self.technical_details
            },
            "regulation": {
                "zoning_status": self.zoning_status,
                "allowed_utilization": self.allowed_utilization,
                "height_restrictions": self.height_restrictions,
                "details": self.regulation_details
            }
        }