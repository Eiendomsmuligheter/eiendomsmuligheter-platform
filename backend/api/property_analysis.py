from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel, HttpUrl
import asyncio
import aiohttp
from ai_modules.property_analyzer.property_analyzer import PropertyAnalyzer
import logging
import json
from datetime import datetime

# Konfigurer logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Eiendomsmuligheter API",
    description="API for verdens beste eiendomsanalyse plattform",
    version="1.0.0"
)

# CORS konfigurasjon
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # I produksjon: spesifiser faktiske domener
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialiser property analyzer
property_analyzer = PropertyAnalyzer()

class AnalysisRequest(BaseModel):
    address: Optional[str] = None
    url: Optional[HttpUrl] = None

class AnalysisResponse(BaseModel):
    id: str
    timestamp: datetime
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None

@app.post("/api/analyze-property", response_model=AnalysisResponse)
async def analyze_property(
    files: Optional[List[UploadFile]] = File(None),
    address: Optional[str] = Form(None),
    url: Optional[HttpUrl] = Form(None)
):
    """
    Analyser en eiendom basert på opplastede bilder, adresse og/eller URL.
    
    - Støtter multiple bilder
    - Kan ta imot både adresse og URL til finn.no annonse
    - Returnerer komplett analyse med 3D-modell og anbefalinger
    """
    try:
        # Generer unik ID for denne analysen
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Logger innkommende forespørsel
        logger.info(f"Starting analysis {analysis_id}")
        logger.info(f"Address: {address}")
        logger.info(f"URL: {url}")
        logger.info(f"Number of files: {len(files) if files else 0}")

        # Validerer input
        if not any([files, address, url]):
            raise HTTPException(
                status_code=400,
                detail="Minst én av følgende må oppgis: bilder, adresse eller URL"
            )

        # Håndterer bilder hvis de er lastet opp
        image_paths = []
        if files:
            for file in files:
                # Lagre filen midlertidig
                file_path = f"/tmp/{analysis_id}_{file.filename}"
                with open(file_path, "wb") as buffer:
                    buffer.write(await file.read())
                image_paths.append(file_path)

        # Start analysen asynkront
        analysis_task = asyncio.create_task(
            _perform_analysis(
                analysis_id,
                image_paths,
                address,
                url
            )
        )

        # Returner umiddelbar respons
        return AnalysisResponse(
            id=analysis_id,
            timestamp=datetime.now(),
            status="processing"
        )

    except Exception as e:
        logger.error(f"Error in analyze_property: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"En feil oppstod under analyseprosessen: {str(e)}"
        )

@app.get("/api/analysis-status/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_status(analysis_id: str):
    """
    Hent status og resultater for en spesifikk analyse
    """
    try:
        # Hent status fra database eller cache
        status = await _get_analysis_status(analysis_id)
        
        return AnalysisResponse(
            id=analysis_id,
            timestamp=datetime.now(),
            status=status["status"],
            result=status.get("result"),
            error=status.get("error")
        )

    except Exception as e:
        logger.error(f"Error in get_analysis_status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Kunne ikke hente analysestatus: {str(e)}"
        )

async def _perform_analysis(
    analysis_id: str,
    image_paths: List[str],
    address: Optional[str],
    url: Optional[HttpUrl]
) -> dict:
    """
    Utfør selve analysen asynkront
    """
    try:
        # Oppdater status til "processing"
        await _update_analysis_status(analysis_id, "processing")

        # Analyser eiendommen
        result = await property_analyzer.analyze_property(
            image_paths=image_paths if image_paths else None,
            address=address,
            url=str(url) if url else None
        )

        # Lagre og returner resultatet
        await _update_analysis_status(
            analysis_id,
            "completed",
            result=result
        )

        return result

    except Exception as e:
        logger.error(f"Error in _perform_analysis: {str(e)}", exc_info=True)
        await _update_analysis_status(
            analysis_id,
            "failed",
            error=str(e)
        )
        raise

async def _update_analysis_status(
    analysis_id: str,
    status: str,
    result: Optional[dict] = None,
    error: Optional[str] = None
):
    """
    Oppdater status for en analyse i database/cache
    """
    # TODO: Implementer faktisk database/cache lagring
    pass

async def _get_analysis_status(analysis_id: str) -> dict:
    """
    Hent analysestatus fra database/cache
    """
    # TODO: Implementer faktisk database/cache henting
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)