"""
Hovedapplikasjon for backend API
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import asyncio
import uvicorn
from pydantic import BaseModel
import aiofiles
import os
from datetime import datetime

# Import core modules
from core_modules.property_analyzer.property_analyzer import PropertyAnalyzer
from core_modules.municipality_client.municipality_client import MunicipalityAPIClient
from core_modules.regulations_analyzer.regulations_analyzer import RegulationsAnalyzer
from core_modules.floor_plan_analyzer.floor_plan_analyzer import FloorPlanAnalyzer
from core_modules.visualization_engine.omniverse_client import OmniverseClient

app = FastAPI(
    title="Eiendomsmuligheter API",
    description="API for eiendomsanalyse og utviklingspotensial",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # I produksjon bør dette begrenses til faktiske domener
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core components
property_analyzer = PropertyAnalyzer()
municipality_client = MunicipalityAPIClient()
regulations_analyzer = RegulationsAnalyzer()
floor_plan_analyzer = FloorPlanAnalyzer()
omniverse_client = OmniverseClient()

class AnalysisRequest(BaseModel):
    address: Optional[str] = None
    files: Optional[List[UploadFile]] = None

class AnalysisResponse(BaseModel):
    property_details: dict
    development_potential: dict
    recommendations: List[dict]
    model_url: Optional[str]

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_property(
    address: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None)
):
    """
    Analyser en eiendom basert på adresse og/eller opplastede filer
    """
    if not address and not files:
        raise HTTPException(
            status_code=400,
            detail="Enten adresse eller filer må oppgis"
        )

    try:
        # Create upload directory if it doesn't exist
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)

        # Save uploaded files
        saved_files = []
        if files:
            for file in files:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = os.path.join(
                    upload_dir,
                    f"{timestamp}_{file.filename}"
                )
                async with aiofiles.open(file_path, 'wb') as out_file:
                    content = await file.read()
                    await out_file.write(content)
                saved_files.append(file_path)

        # Perform property analysis
        analysis_result = await property_analyzer.analyze_property(
            address=address,
            image_paths=saved_files if saved_files else None
        )

        # Get municipal regulations
        if address:
            municipality_info = await municipality_client.get_property_info(address)
            regulations = await regulations_analyzer.analyze_regulations(
                municipality_info["municipality"],
                municipality_info["gnr"],
                municipality_info["bnr"]
            )
            analysis_result["regulations"] = regulations

        # Generate 3D model if floor plans are available
        model_url = None
        if saved_files:
            floor_plan_analysis = await floor_plan_analyzer.analyze_floor_plan(
                saved_files[0],  # Using first file for now
                building_type="residential"  # TODO: Detect building type
            )
            
            # Generate 3D visualization
            model_url = await omniverse_client.create_3d_model(floor_plan_analysis)
            analysis_result["floor_plan_analysis"] = floor_plan_analysis

        return {
            "property_details": analysis_result.get("property_details", {}),
            "development_potential": analysis_result.get("development_potential", {}),
            "recommendations": analysis_result.get("recommendations", []),
            "model_url": model_url
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analyse feilet: {str(e)}"
        )

@app.post("/api/analyze/potential")
async def analyze_development_potential(request: dict):
    """
    Analyser utviklingspotensial basert på eksisterende 3D-modell
    """
    try:
        model_url = request.get("modelUrl")
        if not model_url:
            raise HTTPException(
                status_code=400,
                detail="model_url er påkrevd"
            )

        # Analyze development potential
        potential = await floor_plan_analyzer.analyze_modification_potential(
            model_url,
            {}  # TODO: Add regulations
        )

        return {
            "potential": potential,
            "recommendations": []  # TODO: Generate recommendations
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analyse av utviklingspotensial feilet: {str(e)}"
        )

@app.post("/api/documents/generate")
async def generate_documents(request: dict):
    """
    Generer byggesaksdokumenter basert på analyse
    """
    try:
        analysis_id = request.get("analysis_id")
        if not analysis_id:
            raise HTTPException(
                status_code=400,
                detail="analysis_id er påkrevd"
            )

        # Generate documents
        documents = await floor_plan_analyzer.generate_building_documentation(
            analysis_id,
            request.get("municipality", "")
        )

        return {
            "documents": documents
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Dokumentgenerering feilet: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)