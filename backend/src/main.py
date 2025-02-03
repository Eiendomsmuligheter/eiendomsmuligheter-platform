from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import logging

from services.property_analyzer import PropertyAnalyzer
from services.municipality_service import MunicipalityService
from services.document_generator import DocumentGenerator
from services.enova_service import EnovaService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Eiendomsmuligheter API",
    description="API for eiendomsanalyse og utviklingspotensial",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
property_analyzer = PropertyAnalyzer()
municipality_service = MunicipalityService()
document_generator = DocumentGenerator()
enova_service = EnovaService()

class PropertyAnalysisRequest(BaseModel):
    address: Optional[str] = None
    fileId: Optional[str] = None

class PropertyAnalysisResponse(BaseModel):
    property: Dict[str, Any]
    regulations: List[Dict[str, Any]]
    potential: Dict[str, Any]
    energyAnalysis: Dict[str, Any]
    documents: List[Dict[str, Any]]

@app.post("/api/property/upload")
async def upload_property_file(file: UploadFile = File(...)):
    try:
        file_id = await property_analyzer.process_upload(file)
        return {"fileId": file_id}
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/property/analyze", response_model=PropertyAnalysisResponse)
async def analyze_property(request: PropertyAnalysisRequest):
    try:
        # Analyze property based on address or fileId
        property_info = await property_analyzer.analyze(
            address=request.address,
            file_id=request.fileId
        )

        # Get municipality code from property info
        municipality_code = property_info.get("municipality_code")
        
        # Get regulations for the municipality
        regulations = await municipality_service.get_regulations(municipality_code)
        
        # Analyze development potential
        potential = await property_analyzer.analyze_potential(
            property_info,
            regulations
        )
        
        # Perform energy analysis
        energy_analysis = await property_analyzer.analyze_energy(property_info)
        
        # Generate relevant documents
        documents = await document_generator.generate_documents(
            property_info,
            potential
        )

        return {
            "property": property_info,
            "regulations": regulations,
            "potential": potential,
            "energyAnalysis": energy_analysis,
            "documents": documents
        }
    except Exception as e:
        logger.error(f"Error analyzing property: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/municipality/{code}/regulations")
async def get_municipality_regulations(code: str):
    try:
        regulations = await municipality_service.get_regulations(code)
        return regulations
    except Exception as e:
        logger.error(f"Error fetching regulations: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/property/{property_id}/documents/generate")
async def generate_property_documents(
    property_id: str,
    document_type: str
):
    try:
        document = await document_generator.generate_specific_document(
            property_id,
            document_type
        )
        return document
    except Exception as e:
        logger.error(f"Error generating document: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/property/{property_id}/enova-support")
async def get_enova_support_options(property_id: str):
    try:
        options = await enova_service.get_support_options(property_id)
        return options
    except Exception as e:
        logger.error(f"Error fetching Enova options: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/property/{property_id}/history")
async def get_property_history(property_id: str):
    try:
        history = await municipality_service.get_property_history(property_id)
        return history
    except Exception as e:
        logger.error(f"Error fetching property history: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)