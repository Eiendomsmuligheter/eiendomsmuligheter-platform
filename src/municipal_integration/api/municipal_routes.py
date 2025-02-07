from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Optional, Dict, Any
from ..services.municipal_service import MunicipalityService
from ..models.municipal_model import (
    PropertyData,
    Regulation,
    BuildingApplication,
    HistoricalCase,
    BuildingPermit,
    MunicipalFees,
    ZoningPlan,
    PropertyAnalysis
)
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

load_dotenv()
logger = logging.getLogger(__name__)

router = APIRouter()

# Lastinn konfigurasjon for kommunale API-er
MUNICIPALITY_CONFIG = {
    "municipalities": {
        "3005": {  # Drammen kommune
            "name": "Drammen",
            "base_url": "https://api.drammen.kommune.no/v1",
            "api_key": os.getenv("DRAMMEN_API_KEY"),
            "endpoints": {
                "property_data": "/properties",
                "regulations": "/regulations",
                "submit_application": "/applications",
                "historical_cases": "/cases",
                "zoning_plan": "/zoning"
            },
            "auth_method": "bearer",
            "rate_limits": {
                "requests_per_second": 10,
                "requests_per_day": 10000
            },
            "contact_info": {
                "email": "api@drammen.kommune.no",
                "phone": "+47 12345678"
            }
        }
        # Legg til flere kommuner her...
    }
}

async def get_municipality_service() -> MunicipalityService:
    """
    Dependency for å få tilgang til MunicipalityService
    """
    service = MunicipalityService(MUNICIPALITY_CONFIG)
    async with service:
        yield service

@router.get("/properties/{municipality_code}/{property_id}", response_model=PropertyData)
async def get_property_data(
    municipality_code: str,
    property_id: str,
    service: MunicipalityService = Depends(get_municipality_service)
):
    """
    Henter eiendomsdata fra kommunen
    """
    try:
        return await service.get_property_data(municipality_code, property_id)
    except Exception as e:
        logger.error(f"Feil ved henting av eiendomsdata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/regulations/{municipality_code}/{property_id}", response_model=List[Regulation])
async def get_regulations(
    municipality_code: str,
    property_id: str,
    service: MunicipalityService = Depends(get_municipality_service)
):
    """
    Henter reguleringsbestemmelser fra kommunen
    """
    try:
        return await service.get_regulations(municipality_code, property_id)
    except Exception as e:
        logger.error(f"Feil ved henting av reguleringsbestemmelser: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/applications/{municipality_code}", response_model=Dict[str, Any])
async def submit_building_application(
    municipality_code: str,
    application: BuildingApplication,
    background_tasks: BackgroundTasks,
    service: MunicipalityService = Depends(get_municipality_service)
):
    """
    Sender inn byggesøknad til kommunen
    """
    try:
        result = await service.submit_building_application(municipality_code, application)
        
        # Legg til bakgrunnsoppgave for å følge opp søknaden
        background_tasks.add_task(
            monitor_application_status,
            municipality_code,
            result["application_id"]
        )
        
        return result
    except Exception as e:
        logger.error(f"Feil ved innsending av byggesøknad: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical-cases/{municipality_code}/{property_id}", response_model=List[HistoricalCase])
async def get_historical_cases(
    municipality_code: str,
    property_id: str,
    service: MunicipalityService = Depends(get_municipality_service)
):
    """
    Henter historiske byggesaker for en eiendom
    """
    try:
        return await service.get_historical_cases(municipality_code, property_id)
    except Exception as e:
        logger.error(f"Feil ved henting av historiske saker: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/zoning-plan/{municipality_code}/{plan_id}", response_model=ZoningPlan)
async def get_zoning_plan(
    municipality_code: str,
    plan_id: str,
    service: MunicipalityService = Depends(get_municipality_service)
):
    """
    Henter reguleringsplan
    """
    try:
        return await service.get_zoning_plan(municipality_code, plan_id)
    except Exception as e:
        logger.error(f"Feil ved henting av reguleringsplan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis/{municipality_code}/{property_id}", response_model=PropertyAnalysis)
async def analyze_property_potential(
    municipality_code: str,
    property_id: str,
    service: MunicipalityService = Depends(get_municipality_service)
):
    """
    Utfører en komplett analyse av eiendomspotensial
    """
    try:
        return await service.analyze_property_potential(municipality_code, property_id)
    except Exception as e:
        logger.error(f"Feil ved analyse av eiendomspotensial: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fees/{municipality_code}", response_model=MunicipalFees)
async def get_municipal_fees(
    municipality_code: str,
    service: MunicipalityService = Depends(get_municipality_service)
):
    """
    Henter kommunale gebyrer og avgifter
    """
    try:
        return await service.get_municipal_fees(municipality_code)
    except Exception as e:
        logger.error(f"Feil ved henting av kommunale gebyrer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def monitor_application_status(municipality_code: str, application_id: str):
    """
    Bakgrunnsoppgave for å overvåke status på byggesøknad
    """
    try:
        # Implementer overvåkningslogikk her
        pass
    except Exception as e:
        logger.error(f"Feil ved overvåkning av søknadsstatus: {str(e)}")

# Websocket for sanntidsoppdateringer
@router.websocket("/ws/application-status/{application_id}")
async def application_status_websocket(websocket):
    """
    WebSocket-endepunkt for sanntidsoppdateringer av søknadsstatus
    """
    try:
        await websocket.accept()
        while True:
            # Implementer WebSocket-logikk her
            pass
    except Exception as e:
        logger.error(f"WebSocket-feil: {str(e)}")
        await websocket.close()
