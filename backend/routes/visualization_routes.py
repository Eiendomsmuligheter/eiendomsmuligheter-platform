from fastapi import APIRouter, Depends, HTTPException, Query, File, UploadFile, status
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
import os
import time
import aiofiles
import traceback
import sys
import json
import uuid
from pathlib import Path
from datetime import datetime

# Konfigurer logging
logger = logging.getLogger(__name__)

# Legg til prosjektets rotmappe i PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importer modulene våre - med bedre feilhåndtering
try:
    from ai_modules.AlterraML import AlterraML
    # Definer TerrainData hvis den ikke er tilgjengelig
    try:
        from ai_modules.AlterraML import TerrainData
    except ImportError:
        # Fallback-definisjon
        class TerrainData:
            def __init__(self, property_id, width, depth, resolution=128, include_surroundings=True, include_buildings=True):
                self.property_id = property_id
                self.width = width
                self.depth = depth
                self.resolution = resolution
                self.include_surroundings = include_surroundings
                self.include_buildings = include_buildings
except ImportError as e:
    logger.error(f"Kunne ikke importere nødvendige AI-moduler: {e}")
    logger.error("Dette kan forårsake problemer med visualiserings-rutene.")
    # Definer fallback-klasser
    class AlterraML:
        async def generate_terrain(self, *args, **kwargs):
            raise NotImplementedError("AlterraML er ikke tilgjengelig.")
        
        async def generate_building(self, *args, **kwargs):
            raise NotImplementedError("AlterraML er ikke tilgjengelig.")

# Opprett router
router = APIRouter(
    prefix="/api/visualization",
    tags=["visualization"],
    responses={
        404: {"description": "Ikke funnet"},
        500: {"description": "Serverfeil"}
    },
)

# Definer statiske mapper
STATIC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
HEIGHTMAPS_PATH = os.path.join(STATIC_PATH, "heightmaps")
TEXTURES_PATH = os.path.join(STATIC_PATH, "textures")
MODELS_PATH = os.path.join(STATIC_PATH, "models")

# Sikre at mappene eksisterer
os.makedirs(HEIGHTMAPS_PATH, exist_ok=True)
os.makedirs(TEXTURES_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

# Datamodeller for API
class TerrainGenerationRequest(BaseModel):
    property_id: str = Field(..., description="Eiendoms-ID")
    width: float = Field(..., description="Terrengbredde i meter")
    depth: float = Field(..., description="Terrengdybde i meter")
    resolution: int = Field(128, description="Oppløsning for terrengkart (standard: a128)")
    include_surroundings: bool = Field(True, description="Inkluder omkringliggende terreng")
    include_buildings: bool = Field(True, description="Inkluder bygninger")
    texture_type: str = Field("satellite", description="Teksturtype (satellite, map, hybrid)")

class TerrainGenerationResponse(BaseModel):
    heightmap_url: str = Field(..., description="URL til høydekart")
    texture_url: str = Field(..., description="URL til teksturkart")
    metadata: Dict[str, Any] = Field(..., description="Metadata for terrenget")
    bounds: Dict[str, Any] = Field(..., description="Geografiske grenser")

class BuildingGenerationRequest(BaseModel):
    property_id: str = Field(..., description="Eiendoms-ID")
    building_type: str = Field(..., description="Bygningstype")
    floors: int = Field(..., description="Antall etasjer")
    width: float = Field(..., description="Bygningsbredde i meter")
    depth: float = Field(..., description="Bygningsdybde i meter")
    height: float = Field(..., description="Bygningshøyde i meter")
    roof_type: str = Field("flat", description="Taktype (flat, pitched, etc.)")
    style: str = Field("modern", description="Bygningsstil")
    colors: Optional[Dict[str, str]] = Field(None, description="Fargepalett")

class BuildingGenerationResponse(BaseModel):
    model_url: str = Field(..., description="URL til 3D-modell")
    thumbnail_url: str = Field(..., description="URL til miniatyrbilde")
    metadata: Dict[str, Any] = Field(..., description="Metadata for bygningen")
    
    model_config = {
        "protected_namespaces": ()  # Deaktiverer beskyttede navnerom som "model_"
    }

def get_alterra_ml():
    """Returnerer en initialisert AlterraML instans"""
    logger.info("Initialiserer AlterraML-visualiseringsmodul")
    return AlterraML()

@router.post("/terrain/generate", response_model=TerrainGenerationResponse, status_code=status.HTTP_200_OK)
async def generate_terrain(
    terrain_request: TerrainGenerationRequest,
    alterra: AlterraML = Depends(get_alterra_ml)
):
    """
    Genererer høydekart og tekstur for et terreng basert på eiendomsdata.
    Kan brukes med TerravisionEngine for 3D-visualisering.
    """
    try:
        start_time = time.time()
        logger.info(f"Starter generering av terreng for eiendom {terrain_request.property_id}")
        
        # Generer unike filnavn
        heightmap_filename = f"heightmap_{terrain_request.property_id}_{uuid.uuid4().hex[:8]}.png"
        texture_filename = f"texture_{terrain_request.property_id}_{uuid.uuid4().hex[:8]}.jpg"
        
        # Opprett TerrainData-objekt
        terrain_data = TerrainData(
            property_id=terrain_request.property_id,
            width=terrain_request.width,
            depth=terrain_request.depth,
            resolution=terrain_request.resolution,
            include_surroundings=terrain_request.include_surroundings,
            include_buildings=terrain_request.include_buildings
        )
        
        # Generer terreng med AlterraML
        result = await alterra.generate_terrain(
            terrain_data, 
            os.path.join(HEIGHTMAPS_PATH, heightmap_filename),
            os.path.join(TEXTURES_PATH, texture_filename),
            texture_type=terrain_request.texture_type
        )
        
        # Konverter resultat til API-respons
        # Håndter tilfeller der result kan være et objekt eller en dict
        if hasattr(result, 'metadata') and hasattr(result, 'bounds'):
            metadata = result.metadata
            bounds = result.bounds
        else:
            # Fallback hvis result er en dict eller noe annet
            metadata = getattr(result, 'metadata', {}) if hasattr(result, 'metadata') else {}
            bounds = getattr(result, 'bounds', {}) if hasattr(result, 'bounds') else {}
            
            # Hvis både metadata og bounds er tomme, opprett standard metadata
            if not metadata and not bounds:
                metadata = {
                    "property_id": terrain_request.property_id,
                    "width": terrain_request.width,
                    "depth": terrain_request.depth,
                    "resolution": terrain_request.resolution,
                    "texture_type": terrain_request.texture_type,
                    "generated_at": datetime.now().isoformat()
                }
                bounds = {
                    "north": 59.95,
                    "south": 59.94,
                    "east": 10.76,
                    "west": 10.75,
                    "center": {
                        "latitude": 59.945,
                        "longitude": 10.755
                    }
                }
        
        response = {
            "heightmap_url": f"/api/static/heightmaps/{heightmap_filename}",
            "texture_url": f"/api/static/textures/{texture_filename}",
            "metadata": metadata,
            "bounds": bounds
        }
        
        elapsed_time = time.time() - start_time
        logger.info(f"Terreng generert på {elapsed_time:.2f} sekunder")
        
        return response
        
    except Exception as e:
        logger.error(f"Feil under terrenggenereringen: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Kunne ikke generere terrengdata: {str(e)}"
        )

@router.post("/building/generate", response_model=BuildingGenerationResponse, status_code=status.HTTP_200_OK)
async def generate_building(
    building_request: BuildingGenerationRequest,
    alterra: AlterraML = Depends(get_alterra_ml)
):
    """
    Genererer en 3D-modell av en bygning basert på spesifikasjoner.
    Kan brukes til å visualisere potensielle bygninger på en eiendom.
    """
    try:
        start_time = time.time()
        logger.info(f"Starter generering av bygningsmodell for eiendom {building_request.property_id}")
        
        # Generer unike filnavn
        model_filename = f"building_{building_request.property_id}_{uuid.uuid4().hex[:8]}.glb"
        thumbnail_filename = f"building_thumb_{building_request.property_id}_{uuid.uuid4().hex[:8]}.jpg"
        
        # Konverter til dict for AlterraML
        building_data = building_request.dict()
        
        # Generer bygningsmodell med AlterraML
        result = await alterra.generate_building(
            building_data, 
            os.path.join(MODELS_PATH, model_filename),
            os.path.join(TEXTURES_PATH, thumbnail_filename)
        )
        
        # Konverter resultat til API-respons
        # Håndter tilfeller der result kan være et objekt eller en dict
        if hasattr(result, 'model_url') and hasattr(result, 'thumbnail_url') and hasattr(result, 'metadata'):
            model_url = result.model_url
            thumbnail_url = result.thumbnail_url
            metadata = result.metadata
        else:
            # Fallback hvis result er en dict eller noe annet
            model_url = getattr(result, 'model_url', f"/api/static/models/{model_filename}") if hasattr(result, 'model_url') else f"/api/static/models/{model_filename}"
            thumbnail_url = getattr(result, 'thumbnail_url', f"/api/static/textures/{thumbnail_filename}") if hasattr(result, 'thumbnail_url') else f"/api/static/textures/{thumbnail_filename}"
            metadata = getattr(result, 'metadata', {}) if hasattr(result, 'metadata') else {}
            
            # Hvis metadata er tom, opprett standard metadata
            if not metadata:
                metadata = {
                    "property_id": building_request.property_id,
                    "building_type": building_request.building_type,
                    "floors": building_request.floors,
                    "width": building_request.width,
                    "depth": building_request.depth,
                    "height": building_request.height,
                    "generated_at": datetime.now().isoformat()
                }
        
        response = {
            "model_url": model_url,
            "thumbnail_url": thumbnail_url,
            "metadata": metadata
        }
        
        elapsed_time = time.time() - start_time
        logger.info(f"Bygningsmodell generert på {elapsed_time:.2f} sekunder")
        
        return response
        
    except Exception as e:
        logger.error(f"Feil under bygningsgenerering: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Kunne ikke generere bygningsmodell: {str(e)}"
        )

@router.post("/map/upload", status_code=status.HTTP_200_OK)
async def upload_map(
    property_id: str = Query(..., description="Eiendoms-ID"),
    map_type: str = Query("custom", description="Karttype (custom, official, etc.)"),
    file: UploadFile = File(...),
):
    """
    Last opp et egendefinert kart (høydekart, tekstur, etc.) for en eiendom.
    """
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filen må være et bilde (PNG, JPG, etc.)"
            )
            
        # Bestem filplassering basert på filtype
        file_extension = os.path.splitext(file.filename)[1].lower()
        is_heightmap = file_extension == '.png' or 'height' in file.filename.lower()
        
        if is_heightmap:
            save_directory = HEIGHTMAPS_PATH
            filename = f"custom_heightmap_{property_id}_{uuid.uuid4().hex[:8]}{file_extension}"
            url_prefix = "heightmaps"
        else:
            save_directory = TEXTURES_PATH
            filename = f"custom_texture_{property_id}_{uuid.uuid4().hex[:8]}{file_extension}"
            url_prefix = "textures"
            
        # Lagre opplastet fil
        file_path = os.path.join(save_directory, filename)
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
            
        # Returner URL og metadata
        return {
            "url": f"/api/static/{url_prefix}/{filename}",
            "filename": filename,
            "size": len(content),
            "content_type": file.content_type,
            "type": "heightmap" if is_heightmap else "texture",
            "property_id": property_id,
            "map_type": map_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feil under opplasting av kart: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Kunne ikke laste opp kartfil: {str(e)}"
        )

@router.get("/property/{property_id}/terrain", status_code=status.HTTP_200_OK)
async def get_property_terrain(
    property_id: str,
):
    """
    Henter eksisterende terrengdata for en eiendom.
    """
    try:
        # Finn høydekart og teksturfiler for eiendommen
        heightmap_files = [f for f in os.listdir(HEIGHTMAPS_PATH) if f.startswith(f"heightmap_{property_id}_")]
        texture_files = [f for f in os.listdir(TEXTURES_PATH) if f.startswith(f"texture_{property_id}_")]
        
        if not heightmap_files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ingen terrengdata funnet for eiendom {property_id}"
            )
            
        # Sorter etter dato (nyeste først)
        heightmap_files.sort(key=lambda x: os.path.getmtime(os.path.join(HEIGHTMAPS_PATH, x)), reverse=True)
        if texture_files:
            texture_files.sort(key=lambda x: os.path.getmtime(os.path.join(TEXTURES_PATH, x)), reverse=True)
            
        # Hent nyeste filer
        latest_heightmap = heightmap_files[0]
        latest_texture = texture_files[0] if texture_files else None
        
        # Prøv å finne metadata
        metadata_path = os.path.join(HEIGHTMAPS_PATH, f"metadata_{property_id}.json")
        metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Kunne ikke lese metadata for {property_id}")
                
        return {
            "heightmap_url": f"/api/static/heightmaps/{latest_heightmap}",
            "texture_url": f"/api/static/textures/{latest_texture}" if latest_texture else None,
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feil ved henting av terrengdata: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Kunne ikke hente terrengdata: {str(e)}"
        ) 