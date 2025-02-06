from sqlalchemy import Column, Integer, String, JSON, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .base import Base
import enum
import uuid

def generate_uuid():
    return str(uuid.uuid4())

class DocumentType(enum.Enum):
    BUILDING_APPLICATION = "building_application"
    ANALYSIS_REPORT = "analysis_report"
    FLOOR_PLAN = "floor_plan"
    FACADE_DRAWING = "facade_drawing"
    SITE_PLAN = "site_plan"
    TECHNICAL_DRAWING = "technical_drawing"
    ENERGY_CERTIFICATE = "energy_certificate"
    REGULATION_ASSESSMENT = "regulation_assessment"
    ZONING_MAP = "zoning_map"
    BIM_MODEL = "bim_model"

class DocumentStatus(enum.Enum):
    DRAFT = "draft"
    GENERATED = "generated"
    SIGNED = "signed"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    REJECTED = "rejected"

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=generate_uuid)
    property_id = Column(String, ForeignKey("properties.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    analysis_id = Column(String, ForeignKey("analyses.id"))
    
    # Dokumenttype og status
    document_type = Column(Enum(DocumentType), nullable=False)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.DRAFT)
    
    # Dokumentinformasjon
    title = Column(String, nullable=False)
    description = Column(String)
    version = Column(Integer, default=1)
    file_path = Column(String)  # Path til lagret dokument
    file_size = Column(Integer)  # Størrelse i bytes
    file_type = Column(String)  # PDF, DWG, IFC, etc.
    
    # Metadata
    metadata = Column(JSON)  # Ekstra metadata spesifikk for dokumenttypen
    municipality_requirements = Column(JSON)  # Krav fra kommunen for denne dokumenttypen
    validation_results = Column(JSON)  # Resultater fra validering mot krav
    
    # Signeringsinformasjon
    signature_required = Column(bool, default=False)
    signed_by = Column(String)
    signed_at = Column(DateTime(timezone=True))
    signature_data = Column(JSON)
    
    # Innsendingsinformasjon
    submission_id = Column(String)  # ID fra kommunens system
    submission_date = Column(DateTime(timezone=True))
    submission_status = Column(String)
    submission_feedback = Column(JSON)
    
    # Tidsstempler
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relasjoner
    property = relationship("Property", back_populates="documents")
    user = relationship("User", back_populates="documents")
    analysis = relationship("Analysis", back_populates="documents")
    
    def __repr__(self):
        return f"<Document {self.title} ({self.document_type.value})>"
        
    def to_dict(self):
        return {
            "id": self.id,
            "property_id": self.property_id,
            "document_type": self.document_type.value,
            "status": self.status.value,
            "title": self.title,
            "description": self.description,
            "version": self.version,
            "file_info": {
                "path": self.file_path,
                "size": self.file_size,
                "type": self.file_type
            },
            "metadata": self.metadata,
            "requirements": {
                "municipality": self.municipality_requirements,
                "validation": self.validation_results
            },
            "signature": {
                "required": self.signature_required,
                "signed_by": self.signed_by,
                "signed_at": self.signed_at,
                "data": self.signature_data
            },
            "submission": {
                "id": self.submission_id,
                "date": self.submission_date,
                "status": self.submission_status,
                "feedback": self.submission_feedback
            },
            "timestamps": {
                "created_at": self.created_at,
                "updated_at": self.updated_at
            }
        }

    def validate(self):
        """Validerer dokumentet mot kommunens krav"""
        validation_results = {}
        requirements = self.municipality_requirements or {}
        
        for req_key, requirement in requirements.items():
            if req_key == "file_type":
                validation_results[req_key] = {
                    "passed": self.file_type in requirement["allowed_types"],
                    "message": f"Filtype må være en av: {requirement['allowed_types']}"
                }
            elif req_key == "file_size":
                max_size = requirement["max_size"]
                validation_results[req_key] = {
                    "passed": self.file_size <= max_size,
                    "message": f"Filstørrelse må være mindre enn {max_size} bytes"
                }
            # Legg til flere valideringer etter behov
        
        self.validation_results = validation_results
        return all(result["passed"] for result in validation_results.values())

    def prepare_for_submission(self):
        """Forbereder dokumentet for innsending til kommunen"""
        if not self.validate():
            raise ValueError("Dokumentet oppfyller ikke alle krav")
            
        if self.signature_required and not self.signed_by:
            raise ValueError("Dokumentet krever signering")
            
        self.status = DocumentStatus.SUBMITTED
        self.submission_date = func.now()
        
        return {
            "document_id": self.id,
            "type": self.document_type.value,
            "content": self.file_path,
            "metadata": self.metadata
        }