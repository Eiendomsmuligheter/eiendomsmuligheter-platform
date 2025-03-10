from fastapi import APIRouter, Depends, HTTPException, Query, Body, status
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
import sys
import os
import time
import traceback
from functools import lru_cache
from datetime import datetime

# Legg til prosjektets rotmappe i PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importer modulene våre
try:
    from ai_modules.AlterraML import AlterraML, PropertyData
except ImportError:
    logger.error("Kunne ikke importere AlterraML, sjekk PYTHONPATH")
    
try:
    # Prøv først den relative stien
    from ..api.CommuneConnect import CommuneConnect
except ImportError:
    try:
        # Prøv deretter den absolutte stien
        from backend.api.CommuneConnect import CommuneConnect
    except ImportError:
        logger.error("Kunne ikke importere CommuneConnect, sjekk PYTHONPATH")

# Sett opp logging
logger = logging.getLogger(__name__)

# Opprette router
router = APIRouter(
    prefix="/api/property",
    tags=["property"],
    responses={
        404: {"description": "Ikke funnet"},
        500: {"description": "Serverfeil"}
    },
)

# Datamodeller
class PropertyRequest(BaseModel):
    property_id: Optional[str] = Field(None, description="Eiendoms-ID (hvis kjent)")
    address: str = Field(..., description="Adressen til eiendommen")
    municipality_id: Optional[str] = Field(None, description="Kommune-ID (hvis kjent)")
    zoning_category: Optional[str] = Field(None, description="Reguleringstype")
    lot_size: float = Field(..., description="Tomtestørrelse i kvadratmeter")
    current_utilization: float = Field(..., description="Nåværende utnyttelsesgrad")
    building_height: float = Field(..., description="Bygningshøyde i meter")
    floor_area_ratio: float = Field(..., description="BRA-faktor")
    images: Optional[List[str]] = Field(None, description="Liste med bilde-URLs")
    additional_data: Optional[Dict[str, Any]] = Field(None, description="Tilleggsdata")

class RegulationRule(BaseModel):
    id: str = Field(..., description="Regel-ID")
    rule_type: str = Field(..., description="Regeltype")
    value: Any = Field(..., description="Regelverdi")
    description: str = Field(..., description="Beskrivelse")
    unit: Optional[str] = Field(None, description="Måleenhet")
    category: Optional[str] = Field(None, description="Kategori")

class BuildingPotential(BaseModel):
    max_buildable_area: float = Field(..., description="Maksimalt byggbart areal")
    max_height: float = Field(..., description="Maksimal høyde")
    max_units: int = Field(..., description="Maksimalt antall enheter")
    optimal_configuration: str = Field(..., description="Optimal konfigurasjon")
    constraints: Optional[List[str]] = Field(None, description="Begrensninger")
    recommendations: Optional[List[str]] = Field(None, description="Anbefalinger")

class EnergyProfile(BaseModel):
    energy_class: str = Field(..., description="Energiklasse")
    heating_demand: float = Field(..., description="Oppvarmingsbehov")
    cooling_demand: float = Field(..., description="Kjølebehov")
    primary_energy_source: str = Field(..., description="Primær energikilde")
    recommendations: Optional[List[str]] = Field(None, description="Anbefalinger")

class PropertyAnalysisResponse(BaseModel):
    property_id: str = Field(..., description="Eiendoms-ID")
    address: str = Field(..., description="Adresse")
    regulations: List[RegulationRule] = Field(..., description="Reguleringsregler")
    building_potential: BuildingPotential = Field(..., description="Bygningspotensial")
    energy_profile: Optional[EnergyProfile] = Field(None, description="Energiprofil")
    roi_estimate: Optional[float] = Field(None, description="Estimert ROI")
    risk_assessment: Optional[Dict[str, Any]] = Field(None, description="Risikovurdering")
    recommendations: Optional[List[str]] = Field(None, description="Anbefalinger")

# Singleton-instanser med cache for å redusere oppstartstid og minnebruk
@lru_cache(maxsize=1)
def get_alterra_ml():
    """Returnerer en singleton-instans av AlterraML"""
    logger.info("Initialiserer AlterraML-instans")
    return AlterraML()

@lru_cache(maxsize=1)
def get_commune_connect():
    """Returnerer en singleton-instans av CommuneConnect"""
    logger.info("Initialiserer CommuneConnect-instans")
    return CommuneConnect.get_instance()

@router.post("/analyze", response_model=PropertyAnalysisResponse, status_code=status.HTTP_200_OK)
async def analyze_property(
    property_data: PropertyRequest,
    alterra: AlterraML = Depends(get_alterra_ml),
    commune: CommuneConnect = Depends(get_commune_connect)
):
    """
    Analyserer en eiendom basert på adresse og eiendomsdata.
    Kombinerer reguleringsdata fra kommuner med AI-drevet analyse for å finne potensial.
    """
    try:
        start_time = time.time()
        logger.info(f"Starter analyse av eiendom: {property_data.address}")
        
        # Ensure CommuneConnect is initialized
        commune.ensure_initialized()
        
        # Konverter datamodell til dict for AlterraML
        property_dict = property_data.dict()
        
        # Hent reguleringsplaner fra CommuneConnect
        regulations = None
        try:
            if property_data.municipality_id:
                logger.info(f"Henter reguleringsplaner for {property_data.address} i {property_data.municipality_id}")
                regulations = await commune.get_regulations_by_address(
                    property_data.address, property_data.municipality_id
                )
            else:
                logger.info(f"Henter reguleringsplaner for {property_data.address}")
                regulations = await commune.get_regulations_by_address(property_data.address)
                
            if regulations:
                logger.info(f"Hentet {len(regulations.regulations)} reguleringsplaner")
                property_dict["regulations"] = [r.__dict__ for r in regulations.regulations]
                if not property_dict.get("municipality_id"):
                    property_dict["municipality_id"] = regulations.municipality_id
            else:
                logger.warning(f"Ingen reguleringsplaner funnet for {property_data.address}")
                property_dict["regulations"] = []
                
        except Exception as e:
            logger.error(f"Feil ved henting av reguleringsdata: {str(e)}")
            logger.debug(traceback.format_exc())
            property_dict["regulations"] = []
            
        # Utfør analysen
        logger.info("Utfører AlterraML-analyse")
        property_analysis = await alterra.analyze_property(PropertyData(**property_dict))
        
        # Bygg responsen
        response = {
            "property_id": property_analysis.property_id,
            "address": property_analysis.address,
            "regulations": [
                {
                    "id": getattr(rule, 'rule_id', getattr(rule, 'id', '')),
                    "rule_type": getattr(rule, 'rule_type', ''),
                    "value": getattr(rule, 'value', None),
                    "description": getattr(rule, 'description', ''),
                    "unit": getattr(rule, 'unit', None),
                    "category": getattr(rule, 'category', '')
                }
                for rule in property_analysis.regulations
            ],
            "building_potential": {
                "max_buildable_area": getattr(property_analysis.building_potential, 'max_buildable_area', 0),
                "max_height": getattr(property_analysis.building_potential, 'max_height', 0),
                "max_units": getattr(property_analysis.building_potential, 'max_units', 0),
                "optimal_configuration": getattr(property_analysis.building_potential, 'optimal_configuration', ''),
                "constraints": getattr(property_analysis.building_potential, 'constraints', []),
                "recommendations": getattr(property_analysis.building_potential, 'recommendations', [])
            } if property_analysis.building_potential else {},
            "energy_profile": {
                "energy_class": getattr(property_analysis.energy_profile, 'energy_class', ''),
                "heating_demand": getattr(property_analysis.energy_profile, 'heating_demand', 0),
                "cooling_demand": getattr(property_analysis.energy_profile, 'cooling_demand', 0),
                "primary_energy_source": getattr(property_analysis.energy_profile, 'primary_energy_source', ''),
                "recommendations": getattr(property_analysis.energy_profile, 'recommendations', [])
            } if property_analysis.energy_profile else None,
            "roi_estimate": property_analysis.roi_estimate,
            "risk_assessment": property_analysis.risk_assessment,
            "recommendations": property_analysis.recommendations
        }
        
        elapsed_time = time.time() - start_time
        logger.info(f"Analyse fullført på {elapsed_time:.2f} sekunder")
        
        return response
        
    except Exception as e:
        logger.error(f"Feil under analyse av eiendom: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"En feil oppstod under analyseprosessen: {str(e)}"
        )

@router.get("/municipality/{municipality_id}/regulations", response_model=List[Dict[str, Any]], status_code=status.HTTP_200_OK)
async def get_municipality_regulations(
    municipality_id: str,
    commune: CommuneConnect = Depends(get_commune_connect)
):
    """Henter alle reguleringsplaner for en gitt kommune"""
    try:
        commune.ensure_initialized()
        regulations = await commune.get_all_regulations(municipality_id)
        return [reg.__dict__ for reg in regulations]
    except Exception as e:
        logger.error(f"Feil ved henting av reguleringsplaner for kommune {municipality_id}: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Kunne ikke hente reguleringsplaner: {str(e)}"
        )

@router.get("/municipality/{municipality_id}/contacts", response_model=List[Dict[str, Any]], status_code=status.HTTP_200_OK)
async def get_municipality_contacts(
    municipality_id: str,
    commune: CommuneConnect = Depends(get_commune_connect)
):
    """Henter kontaktinformasjon for en gitt kommune"""
    try:
        commune.ensure_initialized()
        contacts = await commune.get_municipality_contacts(municipality_id)
        return [contact.__dict__ for contact in contacts]
    except Exception as e:
        logger.error(f"Feil ved henting av kontakter for kommune {municipality_id}: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Kunne ikke hente kommunekontakter: {str(e)}"
        )

@router.get("/supported-municipalities", response_model=List[Dict[str, Any]], status_code=status.HTTP_200_OK)
async def get_supported_municipalities(
    commune: CommuneConnect = Depends(get_commune_connect)
):
    """Henter liste over støttede kommuner"""
    try:
        commune.ensure_initialized()
        return commune.get_supported_municipalities()
    except Exception as e:
        logger.error(f"Feil ved henting av støttede kommuner: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Kunne ikke hente støttede kommuner: {str(e)}"
        )

@router.get("/summary/{property_id}", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def get_property_summary(
    property_id: str,
    alterra: AlterraML = Depends(get_alterra_ml),
    commune: CommuneConnect = Depends(get_commune_connect)
):
    """
    Henter sammendrag av eiendomsdata basert på eiendoms-ID.
    """
    try:
        logger.info(f"Henter eiendomssammendrag for ID: {property_id}")
        
        # Bruk dummy-data for demonstrasjonsformål
        property_data = {
            "property_id": property_id,
            "address": "Moreneveien 37, 3058 Solbergmoen",
            "municipality_id": "3025",
            "municipality_name": "Drammen",
            "lot_size": 650.0,
            "current_utilization": 0.25,
            "building_height": 7.5,
            "floor_area_ratio": 0.5,
            "zoning_category": "Bolig",
            "created_at": datetime.now().isoformat()
        }
        
        # Hent reguleringsdata
        try:
            regulations = await commune.get_regulations_by_address(property_data["address"])
            if regulations and hasattr(regulations, 'regulations'):
                property_data["regulations"] = [
                    {
                        "id": getattr(rule, 'id', getattr(rule, 'regulation_id', '')),
                        "rule_type": getattr(rule, 'type', ''),
                        "value": getattr(rule, 'value', None),
                        "description": getattr(rule, 'description', ''),
                        "category": getattr(rule, 'category', '')
                    }
                    for rule in regulations.regulations
                ]
            else:
                property_data["regulations"] = []
        except Exception as reg_err:
            logger.error(f"Feil ved henting av reguleringsdata: {reg_err}")
            property_data["regulations"] = []
        
        # Legg til analyseverdier
        property_data["analysis"] = {
            "max_buildable_area": 325.0,
            "max_height": 9.0,
            "max_units": 3,
            "roi_estimate": 0.15,
            "energy_class": "C",
            "recommendations": [
                "Bygge rekkehus for optimal utnyttelse av tomten",
                "Vurdere solcellepaneler på taket",
                "Søk om dispensasjon for økt byggehøyde"
            ]
        }
        
        # Legg til visualiseringsdetaljer
        property_data["visualization"] = {
            "has_3d_model": True,
            "has_terrain": True,
            "terrain_url": f"/api/static/heightmaps/heightmap_{property_id}.png",
            "model_url": f"/api/static/models/building_{property_id}.glb",
            "texture_url": f"/api/static/textures/texture_{property_id}.jpg"
        }
        
        return property_data
        
    except Exception as e:
        logger.error(f"Feil ved henting av eiendomssammendrag: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Kunne ikke hente eiendomssammendrag: {str(e)}"
        ) 