"""
Core configuration file for the Eiendomsmuligheter platform.
"""
import os
from pathlib import Path
from typing import Dict, Any

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

# API configurations
API_CONFIG = {
    "title": "Eiendomsmuligheter API",
    "description": "World's best property analysis platform",
    "version": "1.0.0",
    "prefix": "/api/v1"
}

# Database configurations
DATABASE_CONFIG = {
    "default": {
        "ENGINE": "postgresql",
        "NAME": os.getenv("DB_NAME", "eiendomsmuligheter"),
        "USER": os.getenv("DB_USER", "postgres"),
        "PASSWORD": os.getenv("DB_PASSWORD", ""),
        "HOST": os.getenv("DB_HOST", "localhost"),
        "PORT": os.getenv("DB_PORT", "5432"),
    }
}

# External API endpoints
EXTERNAL_APIS = {
    "kartverket": {
        "base_url": "https://ws.geonorge.no/",
        "api_key": os.getenv("KARTVERKET_API_KEY", "")
    },
    "kommune": {
        "base_url": "https://innsyn2020.drammen.kommune.no/",
        "api_key": os.getenv("KOMMUNE_API_KEY", "")
    },
    "enova": {
        "base_url": "https://api.enova.no/",
        "api_key": os.getenv("ENOVA_API_KEY", "")
    }
}

# NVIDIA Omniverse configuration
NVIDIA_CONFIG = {
    "enabled": True,
    "api_key": os.getenv("NVIDIA_API_KEY", ""),
    "endpoint": "https://api.omniverse.nvidia.com/",
    "version": "2.0"
}

# AI Model configurations
AI_CONFIG = {
    "floor_plan": {
        "model_path": "models/floor_plan_analyzer",
        "confidence_threshold": 0.85
    },
    "property_analyzer": {
        "model_path": "models/property_analyzer",
        "features": ["rooms", "windows", "doors", "stairs"]
    }
}

# Payment configuration (Stripe)
PAYMENT_CONFIG = {
    "stripe_public_key": os.getenv("STRIPE_PUBLIC_KEY", ""),
    "stripe_secret_key": os.getenv("STRIPE_SECRET_KEY", ""),
    "webhook_secret": os.getenv("STRIPE_WEBHOOK_SECRET", "")
}

def get_config(key: str) -> Dict[str, Any]:
    """Get configuration by key."""
    config_map = {
        "api": API_CONFIG,
        "database": DATABASE_CONFIG,
        "external_apis": EXTERNAL_APIS,
        "nvidia": NVIDIA_CONFIG,
        "ai": AI_CONFIG,
        "payment": PAYMENT_CONFIG
    }
    return config_map.get(key, {})