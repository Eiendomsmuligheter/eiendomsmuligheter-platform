"""
Databasemodeller for analyser og resultater
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON, Text, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from typing import List, Optional

from ..database import Base

class AnalysisStatus(enum.Enum):
    """Status for en analyse"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"

class AnalysisType(enum.Enum):
    """Type analyse"""
    PROPERTY_POTENTIAL = "property_potential"
    REGULATION_CHECK = "regulation_check"
    MARKET_ANALYSIS = "market_analysis"
    ROI_CALCULATION = "roi_calculation"
    ENERGY_EFFICIENCY = "energy_efficiency"
    COMPREHENSIVE = "comprehensive"

class PropertyAnalysis(Base):
    """Modell for eiendomsanalyser"""
    __tablename__ = "property_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Relasjon til eiendom
    property_id = Column(Integer, ForeignKey("properties.id"), nullable=False)
    property = relationship("Property", back_populates="analyses")
    
    # Relasjon til bruker som kjørte analysen
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user = relationship("User")
    
    # Analysetype og status
    analysis_type = Column(Enum(AnalysisType), nullable=False)
    status = Column(Enum(AnalysisStatus), default=AnalysisStatus.PENDING)
    
    # Resultater
    results = Column(JSON, default={})
    
    # Inputparametere
    parameters = Column(JSON, default={})
    
    # Feilmeldinger (hvis analysen feiler)
    error_message = Column(Text, nullable=True)
    
    # Rapport-URL (hvis en rapport genereres)
    report_url = Column(String(255), nullable=True)
    
    # Metadata (fleksibelt felt for ulike egenskaper)
    meta_data = Column(JSON, default={})
    
    # Tidsstempler
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Visualiseringer (one-to-many)
    visualizations = relationship("AnalysisVisualization", back_populates="analysis", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<PropertyAnalysis(id={self.id}, property_id={self.property_id}, type={self.analysis_type})>"

class AnalysisVisualization(Base):
    """Modell for visualiseringer knyttet til analyser"""
    __tablename__ = "analysis_visualizations"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Relasjon til analyse
    analysis_id = Column(Integer, ForeignKey("property_analyses.id"), nullable=False)
    analysis = relationship("PropertyAnalysis", back_populates="visualizations")
    
    # Visualiseringstype
    visualization_type = Column(String(50), nullable=False)  # "3d_model", "heatmap", "chart", etc.
    
    # Tittel og beskrivelse
    title = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    
    # Data for visualiseringen
    data = Column(JSON, nullable=False)
    
    # Konfigurasjon for visualiseringen
    config = Column(JSON, default={})
    
    # Filer knyttet til visualiseringen
    file_path = Column(String(255), nullable=True)
    file_url = Column(String(255), nullable=True)
    
    # Metadata (fleksibelt felt for ulike egenskaper)
    meta_data = Column(JSON, default={})
    
    # Tidsstempler
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<AnalysisVisualization(id={self.id}, analysis_id={self.analysis_id}, type='{self.visualization_type}')>"

class RegulationCheckResult(Base):
    """Modell for resultater av reguleringssjekk"""
    __tablename__ = "regulation_check_results"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Relasjon til analyse
    analysis_id = Column(Integer, ForeignKey("property_analyses.id"), nullable=False)
    
    # Regulering som ble sjekket
    regulation_id = Column(Integer, ForeignKey("zoning_regulations.id"), nullable=False)
    
    # Resultat av sjekken
    is_compliant = Column(Boolean, nullable=False)
    
    # Detaljer
    details = Column(JSON, nullable=False)
    
    # Anbefalinger
    recommendations = Column(JSON, default=[])
    
    # Tidsstempler
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<RegulationCheckResult(id={self.id}, analysis_id={self.analysis_id}, is_compliant={self.is_compliant})>"

class BuildingPotential(Base):
    """Modell for bygningspotensial beregnet i en analyse"""
    __tablename__ = "building_potentials"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Relasjon til analyse
    analysis_id = Column(Integer, ForeignKey("property_analyses.id"), nullable=False)
    
    # Potensialdata
    max_buildable_area = Column(Float, nullable=False)
    max_height = Column(Float, nullable=False)
    max_units = Column(Integer, nullable=False)
    
    # Optimal konfigurasjon
    optimal_configuration = Column(String(255), nullable=True)
    
    # Begrensninger og anbefalinger
    constraints = Column(JSON, default=[])
    recommendations = Column(JSON, default=[])
    
    # Alternativer
    alternatives = Column(JSON, default=[])
    
    # ROI-beregninger
    estimated_construction_cost = Column(Float, nullable=True)
    estimated_market_value = Column(Float, nullable=True)
    estimated_roi = Column(Float, nullable=True)
    roi_details = Column(JSON, default={})
    
    # Tidsstempler
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<BuildingPotential(id={self.id}, analysis_id={self.analysis_id}, max_units={self.max_units})>" 