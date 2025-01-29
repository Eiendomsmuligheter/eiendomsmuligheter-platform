from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
import uvicorn

from services.property_analyzer import PropertyAnalyzer
from services.municipality_service import MunicipalityService
from services.document_generator import DocumentGenerator
from services.enova_service import EnovaService

app = FastAPI(title="Eiendomsmuligheter API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services
property_analyzer = PropertyAnalyzer()
municipality_service = MunicipalityService()
document_generator = DocumentGenerator()
enova_service = EnovaService()

class PropertyAnalysisRequest(BaseModel):
    address: str
    images: Optional[List[str]] = None
    finn_link: Optional[str] = None

class PropertyAnalysisResponse(BaseModel):
    property_info: dict
    regulations: dict
    development_potential: dict
    energy_analysis: dict
    documents: List[str]

@app.post("/api/analyze", response_model=PropertyAnalysisResponse)
async def analyze_property(
    request: PropertyAnalysisRequest,
    files: Optional[List[UploadFile]] = File(None)
):
    try:
        # Analyze property
        property_info = await property_analyzer.analyze(
            address=request.address,
            images=request.images,
            finn_link=request.finn_link,
            uploaded_files=files
        )

        # Get municipality regulations
        regulations = await municipality_service.get_regulations(
            property_info["municipality"],
            property_info["gnr"],
            property_info["bnr"]
        )

        # Analyze development potential
        development_potential = await property_analyzer.analyze_development_potential(
            property_info,
            regulations
        )

        # Perform energy analysis
        energy_analysis = await enova_service.analyze_energy_potential(
            property_info,
            development_potential
        )

        # Generate documents
        documents = await document_generator.generate_documents(
            property_info,
            regulations,
            development_potential,
            energy_analysis
        )

        return PropertyAnalysisResponse(
            property_info=property_info,
            regulations=regulations,
            development_potential=development_potential,
            energy_analysis=energy_analysis,
            documents=documents
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/municipalities/{municipality_id}/regulations")
async def get_municipality_regulations(municipality_id: str):
    try:
        return await municipality_service.get_regulations(municipality_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/generate")
async def generate_documents(analysis_data: dict):
    try:
        return await document_generator.generate_documents(analysis_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/enova/calculate-support")
async def calculate_enova_support(property_data: dict):
    try:
        return await enova_service.calculate_support(property_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)