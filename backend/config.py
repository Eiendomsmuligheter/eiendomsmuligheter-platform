"""
Konfigurasjon for backend
"""
from pydantic import BaseSettings
from typing import Dict, Optional
import os

class Settings(BaseSettings):
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/eiendomsdb")
    
    # NVIDIA Omniverse settings
    OMNIVERSE_URL: str = os.getenv("OMNIVERSE_URL", "omniverse://localhost")
    OMNIVERSE_TOKEN: Optional[str] = os.getenv("OMNIVERSE_TOKEN")
    
    # File storage
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    
    # API settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Eiendomsmuligheter Platform"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Municipal API endpoints
    MUNICIPAL_APIS: Dict[str, Dict] = {
        "drammen": {
            "innsyn": "https://innsyn2020.drammen.kommune.no/",
            "kart": "https://kart.drammen.kommune.no/",
            "byggesak": "https://www.drammen.kommune.no/tjenester/byggesak/"
        }
    }
    
    # Enova API
    ENOVA_API_URL: Optional[str] = os.getenv("ENOVA_API_URL")
    ENOVA_API_KEY: Optional[str] = os.getenv("ENOVA_API_KEY")
    
    # Analysis settings
    DEFAULT_ANALYSIS_SETTINGS = {
        "min_room_height": 2.4,  # meters
        "min_window_area_ratio": 0.10,  # 10% of floor area
        "min_door_width": 0.9,  # meters
        "min_parking_space": 18,  # square meters
        "min_outdoor_area": 25,  # square meters per unit
    }
    
    # 3D visualization settings
    VISUALIZATION_SETTINGS = {
        "default_material_library": "standard_materials",
        "render_quality": "high",
        "max_texture_resolution": 4096,
        "enable_rtx": True,
    }
    
    class Config:
        case_sensitive = True

settings = Settings()