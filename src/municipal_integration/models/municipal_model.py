from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class ZoneType(str, Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    MIXED = "mixed"
    AGRICULTURAL = "agricultural"
    CONSERVATION = "conservation"
    PUBLIC = "public"

class BuildingType(str, Enum):
    HOUSE = "house"
    APARTMENT = "apartment"
    GARAGE = "garage"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    ANNEXE = "annexe"

class PropertyStatus(str, Enum):
    EXISTING = "existing"
    PLANNED = "planned"
    UNDER_CONSTRUCTION = "under_construction"
    PROTECTED = "protected"

class Regulation(BaseModel):
    """Reguleringsbestemmelser for en eiendom"""
    zone_type: ZoneType
    max_utilization_rate: float = Field(..., description="Maksimal utnyttelsesgrad i %")
    max_height: float = Field(..., description="Maksimal byggehøyde i meter")
    min_distance_to_boundary: float = Field(..., description="Minimum avstand til nabogrense i meter")
    allowed_building_types: List[BuildingType]
    special_restrictions: Optional[List[str]] = []
    last_updated: datetime
    document_reference: str
    valid_from: datetime
    valid_to: Optional[datetime] = None

class BuildingRequirement(BaseModel):
    """Byggtekniske krav basert på TEK17/TEK10"""
    building_type: BuildingType
    min_ceiling_height: float = Field(..., description="Minimum takhøyde i meter")
    min_room_size: Dict[str, float] = Field(..., description="Minimum romstørrelse i m² for ulike romtyper")
    ventilation_requirements: Dict[str, str]
    fire_safety_requirements: Dict[str, str]
    accessibility_requirements: Dict[str, str]
    energy_requirements: Dict[str, Any]
    regulation_reference: str = Field(..., description="Referanse til relevant paragraf i TEK17/TEK10")

class PropertyData(BaseModel):
    """Grunndata for en eiendom"""
    property_id: str = Field(..., description="Gårds- og bruksnummer")
    municipality_code: str
    municipality_name: str
    address: str
    area: float = Field(..., description="Tomteareal i m²")
    existing_buildings: List[Dict[str, Any]] = []
    status: PropertyStatus
    ownership_history: List[Dict[str, Any]] = []
    cultural_heritage_status: Optional[str] = None
    environmental_restrictions: Optional[List[str]] = []

class BuildingApplication(BaseModel):
    """Data for byggesøknad"""
    application_id: str
    property_id: str
    applicant_info: Dict[str, str]
    application_type: str
    building_details: Dict[str, Any]
    attachments: List[str]
    status: str
    submitted_date: Optional[datetime]
    processing_deadline: Optional[datetime]
    case_handler: Optional[str]
    comments: List[Dict[str, Any]] = []

class MunicipalityAPI(BaseModel):
    """API-konfigurasjon for en kommune"""
    municipality_code: str
    municipality_name: str
    base_url: str
    api_key: Optional[str]
    endpoints: Dict[str, str]
    authentication_method: str
    rate_limits: Optional[Dict[str, Any]]
    contact_info: Dict[str, str]

class HistoricalCase(BaseModel):
    """Historisk byggesak"""
    case_id: str
    property_id: str
    case_type: str
    description: str
    decision: str
    decision_date: datetime
    documents: List[Dict[str, str]]
    related_cases: List[str] = []
    case_handler: str
    processing_time: int  # dager

class BuildingPermit(BaseModel):
    """Byggetillatelse"""
    permit_id: str
    application_id: str
    property_id: str
    permit_type: str
    granted_date: datetime
    valid_until: datetime
    conditions: List[str]
    responsible_enterprises: Dict[str, Dict[str, str]]
    supervision_requirements: List[str]
    inspection_plan: Dict[str, Any]

class MunicipalFees(BaseModel):
    """Kommunale gebyrer"""
    municipality_code: str
    valid_from: datetime
    valid_to: Optional[datetime]
    application_fees: Dict[str, float]
    processing_fees: Dict[str, float]
    connection_fees: Dict[str, float]
    inspection_fees: Dict[str, float]
    discount_rules: List[Dict[str, Any]] = []

class ZoningPlan(BaseModel):
    """Reguleringsplan"""
    plan_id: str
    plan_name: str
    municipality_code: str
    status: str
    valid_from: datetime
    valid_to: Optional[datetime]
    regulations: List[Regulation]
    affected_properties: List[str]
    documents: List[Dict[str, str]]
    map_references: List[str]
    last_modified: datetime

class PropertyAnalysis(BaseModel):
    """Analyse av eiendomspotensial"""
    property_id: str
    analysis_date: datetime
    current_regulations: Regulation
    development_potential: Dict[str, Any]
    restrictions: List[str]
    recommended_actions: List[Dict[str, Any]]
    estimated_processing_time: int  # dager
    estimated_costs: Dict[str, float]
    risk_assessment: Dict[str, Any]

# Konstanter og standardverdier
STANDARD_REQUIREMENTS = {
    BuildingType.HOUSE: BuildingRequirement(
        building_type=BuildingType.HOUSE,
        min_ceiling_height=2.4,
        min_room_size={
            "bedroom": 7.0,
            "living_room": 15.0,
            "kitchen": 6.0,
            "bathroom": 4.0
        },
        ventilation_requirements={
            "living_areas": "Minimum luftskifte 1.2 m³ per time per m²",
            "bathroom": "Minimum luftskifte 3.6 m³ per time per m²"
        },
        fire_safety_requirements={
            "escape_routes": "Maksimum 25 meter til rømningsvei",
            "fire_alarms": "Seriekoblede røykvarslere i alle etasjer"
        },
        accessibility_requirements={
            "entrance": "Trinnfri adkomst",
            "bathroom": "Snusirkel 1.5 meter diameter"
        },
        energy_requirements={
            "u_value_walls": 0.18,
            "u_value_roof": 0.13,
            "u_value_floor": 0.10,
            "u_value_windows": 0.80,
            "air_tightness": 0.6
        },
        regulation_reference="TEK17 §13-1"
    )
}

STANDARD_PROCESSING_TIMES = {
    "simple_application": 21,  # dager
    "complex_application": 84,  # dager
    "dispensation": 120  # dager
}
