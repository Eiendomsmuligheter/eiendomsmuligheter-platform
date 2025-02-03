from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from typing import Optional, Dict, Any
from ...services.property_analyzer_service import PropertyAnalyzerService
from ...services.municipality_service import MunicipalityService
from ...services.enova_service import EnovaService
from ...services.ai_service import AIService
from ...models.property import PropertyAnalysisRequest, PropertyAnalysisResponse
from ...auth.auth_bearer import JWTBearer

router = APIRouter()

# Dependency injection
def get_property_analyzer():
    return PropertyAnalyzerService()

def get_municipality_service():
    return MunicipalityService()

def get_enova_service():
    return EnovaService()

def get_ai_service():
    return AIService()

@router.post(
    "/analyze",
    response_model=PropertyAnalysisResponse,
    dependencies=[Depends(JWTBearer())]
)
async def analyze_property(
    request: PropertyAnalysisRequest,
    property_analyzer: PropertyAnalyzerService = Depends(get_property_analyzer)
):
    """
    Analyser en eiendom basert på adresse, bilde eller lenke
    """
    try:
        result = await property_analyzer.analyze_property(
            address=request.address,
            image_data=request.image_data,
            link=request.link
        )
        return PropertyAnalysisResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/image")
async def analyze_property_image(
    file: UploadFile = File(...),
    ai_service: AIService = Depends(get_ai_service)
):
    """
    Analyser et bilde av en eiendom
    """
    try:
        contents = await file.read()
        analysis = await ai_service.analyze_building(contents)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/municipalities/{municipality_code}/regulations")
async def get_municipality_regulations(
    municipality_code: str,
    municipality_service: MunicipalityService = Depends(get_municipality_service)
):
    """
    Hent byggeregler og forskrifter for en kommune
    """
    try:
        regulations = await municipality_service.get_regulations(municipality_code)
        return regulations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/properties/{property_id}/history")
async def get_property_history(
    property_id: str,
    municipality_service: MunicipalityService = Depends(get_municipality_service)
):
    """
    Hent byggesakshistorikk for en eiendom
    """
    try:
        history = await municipality_service.get_property_history(property_id)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/properties/{property_id}/enova-support")
async def get_enova_support_options(
    property_id: str,
    enova_service: EnovaService = Depends(get_enova_service)
):
    """
    Hent tilgjengelige ENOVA-støtteordninger for en eiendom
    """
    try:
        support_options = await enova_service.get_support_options(property_id)
        return support_options
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/properties/{property_id}/development-potential")
async def analyze_development_potential(
    property_id: str,
    ai_service: AIService = Depends(get_ai_service),
    municipality_service: MunicipalityService = Depends(get_municipality_service)
):
    """
    Analyser utviklingspotensial for en eiendom
    """
    try:
        # Hent eiendomsinformasjon og reguleringer
        property_info = await municipality_service.get_property_history(property_id)
        zoning_info = await municipality_service.check_zoning_restrictions(
            property_info["municipality_code"],
            property_info
        )

        # Analyser potensial
        potential = await ai_service.analyze_development_potential(
            property_info,
            zoning_info
        )
        return potential
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/properties/{property_id}/support-eligibility/{support_id}")
async def check_support_eligibility(
    property_id: str,
    support_id: str,
    enova_service: EnovaService = Depends(get_enova_service)
):
    """
    Sjekk om en eiendom kvalifiserer for en støtteordning
    """
    try:
        # Hent støtteordning og eiendomsinfo
        support_options = await enova_service.get_support_options(property_id)
        support_option = next(
            (opt for opt in support_options if opt["id"] == support_id),
            None
        )
        
        if not support_option:
            raise HTTPException(
                status_code=404,
                detail="Support option not found"
            )

        eligibility = await enova_service.check_eligibility(
            {"property_id": property_id},
            support_option
        )
        return eligibility
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))