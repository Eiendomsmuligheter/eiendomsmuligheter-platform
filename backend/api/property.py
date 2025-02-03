from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from ..services.PropertyAnalyzerService import PropertyAnalyzerService
from ..models.property import PropertyAnalysisRequest, PropertyAnalysisResponse

router = APIRouter()
property_analyzer = PropertyAnalyzerService()

@router.post("/analyze/address")
async def analyze_by_address(request: PropertyAnalysisRequest) -> PropertyAnalysisResponse:
    """
    Analyser eiendom basert på adresse
    """
    try:
        results = await property_analyzer.analyze_by_address(request.address)
        return PropertyAnalysisResponse(**results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/files")
async def analyze_files(files: List[UploadFile] = File(...)) -> PropertyAnalysisResponse:
    """
    Analyser eiendom basert på opplastede filer
    """
    try:
        file_contents = []
        for file in files:
            content = await file.read()
            file_contents.append(content)
        
        results = await property_analyzer.analyze_files(file_contents)
        return PropertyAnalysisResponse(**results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{property_id}")
async def get_property_details(property_id: str) -> PropertyAnalysisResponse:
    """
    Hent detaljert eiendomsinformasjon
    """
    try:
        results = await property_analyzer.get_property_details(property_id)
        return PropertyAnalysisResponse(**results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{property_id}/regulations")
async def get_regulations(property_id: str):
    """
    Hent reguleringsplan og kommunale bestemmelser
    """
    try:
        return await property_analyzer.get_regulations(property_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{property_id}/enova-support")
async def get_enova_support(property_id: str):
    """
    Beregn potensielle Enova-støtteordninger
    """
    try:
        return await property_analyzer.get_enova_support(property_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{property_id}/3d-model")
async def get_3d_model(property_id: str):
    """
    Hent 3D-modell av eiendommen
    """
    try:
        return await property_analyzer.get_3d_model(property_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{property_id}/documents")
async def generate_documents(property_id: str, document_types: List[str]):
    """
    Generer byggesaksdokumenter
    """
    try:
        return await property_analyzer.generate_documents(property_id, document_types)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))