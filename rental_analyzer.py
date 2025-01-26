import requests
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import json
import re
from pathlib import Path
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from datetime import datetime
import cv2
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

# AI-modeller og analyseverktøy
from ai_modules.floor_plan_analyzer import FloorPlanAnalyzer
from ai_modules.economic_analyzer import EconomicAnalyzer
from ai_modules.visualization_engine import VisualizationEngine
import cv2
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

# Konfigurer logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PropertyMetrics:
    area: float
    rooms: int
    floor: Optional[int]
    build_year: int
    condition: str
    parking: bool
    elevator: bool
    balcony: bool
    estimated_rent: float
    confidence_score: float

class BuildingType(Enum):
    APARTMENT = "leilighet"
    HOUSE = "enebolig"
    ROW_HOUSE = "rekkehus"
    BASEMENT = "kjellerleilighet"
    ATTIC = "loftsleilighet"
    ANNEXE = "anneks"
    DUPLEX = "tomannsbolig"
    TRIPLEX = "tremannsbolig"
    COMMERCIAL = "næringseiendom"
    
@dataclass
class RentalAnalyzer:
    def __init__(self):
        self.floor_plan_analyzer = FloorPlanAnalyzer()
        self.economic_analyzer = EconomicAnalyzer()
        self.visualization_engine = VisualizationEngine()
        self.requirements = RentalRequirements()
        self.model = self._initialize_ml_model()
        
    def _initialize_ml_model(self) -> RandomForestRegressor:
        """Initialiser maskinlæringsmodell for prisestimering"""
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        # TODO: Last inn og tren modell med historiske data
        return model
        
    async def analyze_property(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Utfør fullstendig analyse av eiendom"""
        try:
            # Analyser plantegning
            floor_plan_analysis = await self._analyze_floor_plan(property_data)
            
            # Utfør økonomisk analyse
            economic_analysis = await self._perform_economic_analysis(property_data)
            
            # Generer 3D-visualisering
            visualization = await self._generate_visualization(floor_plan_analysis)
            
            # Sammenstill resultater
            return {
                "floor_plan_analysis": floor_plan_analysis,
                "economic_analysis": economic_analysis,
                "visualization": visualization,
                "recommendations": self._generate_recommendations(floor_plan_analysis, economic_analysis),
                "compliance": self._check_compliance(floor_plan_analysis)
            }
            
        except Exception as e:
            logger.error(f"Feil ved analyse av eiendom: {str(e)}")
            return {"error": str(e)}
            
    async def _analyze_floor_plan(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyser plantegning"""
        try:
            # Last plantegning
            if "floor_plan_image" in property_data:
                analysis = self.floor_plan_analyzer.analyze_floor_plan(
                    property_data["floor_plan_image"]
                )
                return analysis
            else:
                raise ValueError("Mangler plantegning")
                
        except Exception as e:
            logger.error(f"Feil ved analyse av plantegning: {str(e)}")
            return {"error": str(e)}
            
    async def _perform_economic_analysis(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Utfør økonomisk analyse"""
        try:
            analysis = self.economic_analyzer.analyze_investment(property_data)
            return analysis
            
        except Exception as e:
            logger.error(f"Feil ved økonomisk analyse: {str(e)}")
            return {"error": str(e)}
            
    async def _generate_visualization(self, floor_plan_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generer 3D-visualisering"""
        try:
            visualization = self.visualization_engine.create_3d_model(floor_plan_analysis)
            return visualization
            
        except Exception as e:
            logger.error(f"Feil ved generering av visualisering: {str(e)}")
            return {"error": str(e)}
            
    def _generate_recommendations(self, floor_plan: Dict[str, Any], 
                                economics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generer anbefalinger basert på analyser"""
        recommendations = []
        
        # Analyse av planløsning
        if "rooms" in floor_plan:
            self._analyze_room_optimization(floor_plan, recommendations)
            
        # Økonomiske anbefalinger
        if "roi" in economics:
            self._analyze_economic_optimization(economics, recommendations)
            
        return recommendations
        
    def _analyze_room_optimization(self, floor_plan: Dict[str, Any], 
                                 recommendations: List[Dict[str, Any]]) -> None:
        """Analyser optimalisering av romløsning"""
        try:
            # Sjekk rommenes funksjonalitet
            for room in floor_plan.get("rooms", []):
                if room["area"] < self.requirements.min_room_area:
                    recommendations.append({
                        "type": "room_optimization",
                        "severity": "high",
                        "description": f"Rom er for lite ({room['area']}m²)",
                        "suggestion": "Vurder å slå sammen med tilstøtende rom"
                    })
                    
                if room.get("window_area", 0) < room["area"] * self.requirements.min_window_area:
                    recommendations.append({
                        "type": "lighting",
                        "severity": "medium",
                        "description": "Utilstrekkelig dagslys",
                        "suggestion": "Vurder å legge til vinduer eller takvinduer"
                    })
                    
        except Exception as e:
            logger.error(f"Feil ved romoptimalisering: {str(e)}")
            
    def _analyze_economic_optimization(self, economics: Dict[str, Any], 
                                    recommendations: List[Dict[str, Any]]) -> None:
        """Analyser økonomisk optimalisering"""
        try:
            if economics["roi"] < 0.10:  # Under 10% ROI
                recommendations.append({
                    "type": "economic",
                    "severity": "high",
                    "description": "Lav forventet avkastning",
                    "suggestion": "Vurder kostnadsreduserende tiltak eller økt leiepris"
                })
                
            if economics.get("vacancy_risk", 0) > 0.15:  # Over 15% risiko
                recommendations.append({
                    "type": "economic",
                    "severity": "medium",
                    "description": "Høy risiko for ledighet",
                    "suggestion": "Vurder tiltak for å øke attraktivitet"
                })
                
        except Exception as e:
            logger.error(f"Feil ved økonomisk optimalisering: {str(e)}")
            
    def _check_compliance(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Sjekk samsvar med krav og reguleringer"""
        compliance = {
            "compliant": True,
            "issues": [],
            "requirements_met": []
        }
        
        try:
            # Sjekk takhøyde
            if analysis.get("ceiling_height", 0) < self.requirements.min_ceiling_height:
                compliance["compliant"] = False
                compliance["issues"].append({
                    "type": "ceiling_height",
                    "description": "Takhøyde under minimumskrav",
                    "current": analysis["ceiling_height"],
                    "required": self.requirements.min_ceiling_height
                })
                
            # Sjekk vindusareal
            if analysis.get("window_ratio", 0) < self.requirements.min_window_area:
                compliance["compliant"] = False
                compliance["issues"].append({
                    "type": "natural_light",
                    "description": "Utilstrekkelig dagslys",
                    "current": analysis["window_ratio"],
                    "required": self.requirements.min_window_area
                })
                
            # Sjekk ventilasjon
            if not analysis.get("ventilation_compliant", False):
                compliance["compliant"] = False
                compliance["issues"].append({
                    "type": "ventilation",
                    "description": "Manglende eller utilstrekkelig ventilasjon",
                    "required": "Mekanisk ventilasjon påkrevd"
                })
                
        except Exception as e:
            logger.error(f"Feil ved samsvarskontroll: {str(e)}")
            compliance["compliant"] = False
            compliance["issues"].append({
                "type": "error",
                "description": f"Feil ved samsvarskontroll: {str(e)}"
            })
            
        return compliance

class RentalRequirements:
    min_ceiling_height: float = 2.4
    min_window_area: float = 0.1  # 10% av gulvareal
    ventilation_required: bool = True
    fire_escape_required: bool = True
    separate_entrance: bool = True
    bathroom_required: bool = True
    kitchen_required: bool = True
