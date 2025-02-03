from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class PropertyAnalysisRequest(BaseModel):
    address: str
    
class PropertyInfo(BaseModel):
    address: str
    municipality: str
    gnr: str
    bnr: str
    area: float
    building_type: str
    construction_year: Optional[int]
    
class BuildingHistory(BaseModel):
    case_number: str
    date: str
    description: str
    status: str
    documents: List[str]
    
class ZoningInfo(BaseModel):
    plan_id: str
    plan_name: str
    purpose: str
    coverage_rate: float
    max_height: float
    special_regulations: List[str]
    
class DevelopmentPotential(BaseModel):
    basement_rental: Optional[Dict[str, Any]]
    attic_conversion: Optional[Dict[str, Any]]
    property_division: Optional[Dict[str, Any]]
    recommendations: List[str]
    estimated_costs: Dict[str, float]
    estimated_value_increase: float
    
class ModelData(BaseModel):
    model_url: str
    textures: List[str]
    materials: List[Dict[str, Any]]
    
class EnovaSupport(BaseModel):
    eligible_measures: List[str]
    potential_support: float
    energy_savings: float
    requirements: List[str]
    
class PropertyAnalysisResponse(BaseModel):
    property_info: PropertyInfo
    building_history: List[BuildingHistory]
    zoning_info: ZoningInfo
    development_potential: DevelopmentPotential
    model_data: ModelData
    enova_support: EnovaSupport