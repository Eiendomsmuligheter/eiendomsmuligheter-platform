from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .base import Base
import uuid
import enum

def generate_uuid():
    return str(uuid.uuid4())

class AnalysisStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class AnalysisType(enum.Enum):
    FULL = "full"
    DEVELOPMENT_POTENTIAL = "development_potential"
    ENERGY = "energy"
    REGULATION = "regulation"
    RENTAL_UNIT = "rental_unit"

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String, primary_key=True, default=generate_uuid)
    property_id = Column(String, ForeignKey("properties.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    # Analysetype og status
    analysis_type = Column(Enum(AnalysisType), nullable=False)
    status = Column(Enum(AnalysisStatus), default=AnalysisStatus.PENDING)
    
    # Analyseparametere og resultater
    parameters = Column(JSON)  # Input-parametere for analysen
    results = Column(JSON)  # Analyseresultater
    
    # Utviklingspotensial
    development_options = Column(JSON)  # Liste over utviklingsmuligheter
    estimated_costs = Column(JSON)  # Kostnadsestimater for hver opsjon
    potential_value = Column(JSON)  # Potensiell verdiøkning
    
    # Energianalyse
    energy_rating = Column(String)
    energy_consumption = Column(Float)
    energy_improvement_potential = Column(JSON)
    enova_support_options = Column(JSON)
    
    # Reguleringsanalyse
    zoning_analysis = Column(JSON)
    building_restrictions = Column(JSON)
    allowed_changes = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True))
    processing_time = Column(Float)  # Tid brukt på analysen i sekunder
    
    # Relasjoner
    property = relationship("Property", back_populates="analyses")
    user = relationship("User", back_populates="analyses")
    documents = relationship("Document", back_populates="analysis")
    
    def __repr__(self):
        return f"<Analysis {self.id} - {self.analysis_type.value}>"
        
    def to_dict(self):
        return {
            "id": self.id,
            "property_id": self.property_id,
            "analysis_type": self.analysis_type.value,
            "status": self.status.value,
            "development": {
                "options": self.development_options,
                "estimated_costs": self.estimated_costs,
                "potential_value": self.potential_value
            },
            "energy": {
                "rating": self.energy_rating,
                "consumption": self.energy_consumption,
                "improvement_potential": self.energy_improvement_potential,
                "enova_support": self.enova_support_options
            },
            "regulation": {
                "zoning_analysis": self.zoning_analysis,
                "building_restrictions": self.building_restrictions,
                "allowed_changes": self.allowed_changes
            },
            "metadata": {
                "created_at": self.created_at,
                "completed_at": self.completed_at,
                "processing_time": self.processing_time
            }
        }