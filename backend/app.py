"""
Eiendomsmuligheter Platform Backend API
----------------------------------------

Dette er hovedapplikasjonen for backend-API-et.
Den importerer og registrerer alle API-ruter og håndterer konfigurasjonen av FastAPI.
"""
import logging
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import traceback

# Legg til prosjektets rotmappe i PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Last miljøvariabler
load_dotenv()

# Konfigurer logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("eiendomsmuligheter")

# Opprett FastAPI-app
app = FastAPI(
    title="Eiendomsmuligheter Platform API",
    description="API for Eiendomsmuligheter Platform",
    version="1.0.0"
)

# Konfigurer CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # I produksjon bør dette begrenses til faktiske domener
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Statiske filer
os.makedirs("static/heightmaps", exist_ok=True)
os.makedirs("static/textures", exist_ok=True)
os.makedirs("static/models", exist_ok=True)
os.makedirs("static/cache", exist_ok=True)
app.mount("/api/static", StaticFiles(directory="static"), name="static")

# Importer API-ruter
try:
    # Sjekk om vi kan importere direkte fra backend-mappen
    if os.path.exists(os.path.join(SCRIPT_DIR, "routes")):
        from routes.property_routes import router as property_router
        from routes.visualization_routes import router as visualization_router
        # Legg til payment_routes om den finnes
        payment_router = None
        try:
            from routes.payment_routes import router as payment_router
        except ImportError:
            logger.info("Payment ruter ikke funnet, hopper over")
        
        # Inkluder ruter
        app.include_router(property_router)
        app.include_router(visualization_router)
        if payment_router:
            app.include_router(payment_router)
        
        logger.info("API-ruter lastet vellykket (direkte import)")
    else:
        # Fallback til import via backend-prefix
        from backend.routes.property_routes import router as property_router
        from backend.routes.visualization_routes import router as visualization_router
        
        # Inkluder ruter
        app.include_router(property_router)
        app.include_router(visualization_router)
        
        logger.info("API-ruter lastet vellykket (backend-prefix import)")
except ImportError as e:
    logger.warning(f"Kunne ikke importere alle ruter: {str(e)}")
    logger.warning("API vil kjøre med begrenset funksjonalitet")
    logger.debug(f"Import exception: {traceback.format_exc()}")

@app.get("/")
async def root():
    return {"message": "Velkommen til Eiendomsmuligheter Platform API"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

# Kjøres hvis denne filen er hovedskriptet
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starter Eiendomsmuligheter Platform API på {host}:{port}")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        workers=int(os.getenv("WORKERS", "1"))
    ) 