import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json
import os
import torch
from torch import nn
import segmentation_models_pytorch as smp
from PIL import Image
import yaml
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
import re
from shapely.geometry import Polygon, box
import pandas as pd

# Konfigurer logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RoomFeatures:
    """Dataklasse for romegenskaper"""
    area: float
    width: float
    length: float
    perimeter: float
    room_type: str
    meets_requirements: Dict[str, bool]
    windows: List[Dict[str, Any]]
    doors: List[Dict[str, Any]]
    ceiling_height: float
    natural_light_score: float
    ventilation_score: float
    accessibility_score: float

@dataclass
class AnalysisResult:
    """Dataklasse for analyseresultater"""
    rooms: List[RoomFeatures]
    total_area: float
    floor_count: int
    building_footprint: float
    suggested_improvements: List[str]
    compliance_score: float
    accessibility_rating: float
    energy_efficiency: float
    natural_light_rating: float
    renovation_potential: float
    estimated_renovation_cost: float
    visualization_data: Dict[str, Any]

class FloorPlanAnalyzer:
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialiserer FloorPlanAnalyzer med avanserte AI-modeller og analyseverktøy
        
        Args:
            config_path: Sti til konfigurasjonsfil (valgfri)
        """
        # Last konfigurasjon
        self.config = self._load_config(config_path)
        
        # Initialiser AI-modeller
        self.models = self._initialize_models()
        
        # Initialiserer prosessorer og verktøy
        self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
        self.layout_model = LayoutLMv3ForSequenceClassification.from_pretrained(
            "microsoft/layoutlmv3-base", num_labels=len(self.config["room_types"])
        )
        
        # Metrics og logging
        self.metrics = self._initialize_metrics()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Laster konfigurasjon fra fil eller bruker standardverdier"""
        default_config = {
            "models": {
                "segmentation": "deeplabv3plus_resnet101",
                "detection": "faster_rcnn_resnet101",
                "room_classifier": "efficientnet_b4"
            },
            "room_types": [
                "stue", "kjøkken", "soverom", "bad", "gang", "bod", 
                "vaskerom", "kontor", "verksted"
            ],
            "min_room_size": {
                "soverom": 7.0,
                "stue": 15.0,
                "kjøkken": 6.0,
                "bad": 4.0
            },
            "requirements": {
                "ceiling_height": 2.4,
                "window_ratio": 0.10,
                "ventilation": {
                    "bathroom": "mechanical",
                    "kitchen": "mechanical",
                    "bedroom": "natural"
                }
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                return {**default_config, **user_config}
        return default_config

    def _initialize_models(self) -> Dict[str, Any]:
        """Initialiserer og laster alle nødvendige AI-modeller"""
        models = {}
        
        # Segmenteringsmodell
        models["segmentation"] = smp.DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            classes=len(self.config["room_types"]),
            activation="softmax"
        )
        
        # Objektdeteksjonsmodell for vinduer og dører
        models["detection"] = torch.hub.load('ultralytics/yolov5', 'custom',
            path='models/object_detection.pt')
        
        # Romklassifiseringsmodell
        models["room_classifier"] = tf.keras.models.load_model(
            'models/room_classifier.h5')
        
        # 3D-rekonstruksjonsmodell
        models["3d_reconstruction"] = self._load_3d_model()
        
        return models

    def _initialize_metrics(self) -> Dict[str, Any]:
        """Initialiserer metrikker for analyseytelse"""
        return {
            "processed_images": 0,
            "average_processing_time": 0,
            "success_rate": 1.0,
            "accuracy_scores": {
                "room_detection": [],
                "measurement": [],
                "classification": []
            }
        }

    async def analyze_floor_plan(
        self, 
        image_path: Union[str, Path, np.ndarray],
        include_3d: bool = True,
        detailed_analysis: bool = True
    ) -> AnalysisResult:
        """
        Utfører omfattende analyse av plantegning med avansert AI
        
        Args:
            image_path: Sti til bilde eller numpy array
            include_3d: Generer 3D-modell
            detailed_analysis: Utfør detaljert analyse
            
        Returns:
            AnalysisResult: Komplett analyserapport
        """
        try:
            # Last og preprosesser bilde
            image = self._load_and_preprocess_image(image_path)
            
            # Parallell prosessering av ulike analyseaspekter
            with ThreadPoolExecutor() as executor:
                # Start alle analyser parallelt
                room_future = executor.submit(self._detect_and_analyze_rooms, image)
                structure_future = executor.submit(self._analyze_building_structure, image)
                measure_future = executor.submit(self._calculate_measurements, image)
                if include_3d:
                    model_future = executor.submit(self._generate_3d_model, image)
                
                # Samle resultater
                rooms = room_future.result()
                structure = structure_future.result()
                measurements = measure_future.result()
                model_data = model_future.result() if include_3d else None
            
            # Analyser bygningskrav og reguleringer
            compliance = self._analyze_compliance(rooms, structure)
            
            # Generer forbedringsforslag
            improvements = self._generate_improvement_suggestions(
                rooms, compliance, measurements)
            
            # Beregn kostnader og potensial
            renovation = self._calculate_renovation_details(
                rooms, improvements, measurements)
            
            # Kompiler resultat
            result = AnalysisResult(
                rooms=rooms,
                total_area=measurements["total_area"],
                floor_count=structure["floor_count"],
                building_footprint=measurements["footprint"],
                suggested_improvements=improvements,
                compliance_score=compliance["overall_score"],
                accessibility_rating=compliance["accessibility_score"],
                energy_efficiency=structure["energy_efficiency"],
                natural_light_rating=self._calculate_light_score(rooms),
                renovation_potential=renovation["potential_score"],
                estimated_renovation_cost=renovation["estimated_cost"],
                visualization_data=model_data if include_3d else {}
            )
            
            # Oppdater metrikker
            self._update_metrics(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Feil under analyse av plantegning: {str(e)}")
            raise

    def _detect_and_analyze_rooms(self, image: np.ndarray) -> List[RoomFeatures]:
        """
        Detekterer og analyserer rom med avansert AI-segmentering
        """
        # Utfør semantisk segmentering
        mask = self.models["segmentation"](torch.from_numpy(image).unsqueeze(0))
        room_masks = self._process_segmentation_mask(mask)
        
        rooms = []
        for mask in room_masks:
            # Analyser romgeometri
            geometry = self._analyze_room_geometry(mask)
            
            # Detekter vinduer og dører
            windows = self._detect_windows(image, mask)
            doors = self._detect_doors(image, mask)
            
            # Klassifiser romtype
            room_type = self._classify_room_type(image, mask)
            
            # Analyser romkvaliteter
            qualities = self._analyze_room_qualities(
                image, mask, windows, doors, geometry)
            
            # Sjekk krav
            requirements = self._check_room_requirements(
                qualities, room_type, geometry)
            
            rooms.append(RoomFeatures(
                area=geometry["area"],
                width=geometry["width"],
                length=geometry["length"],
                perimeter=geometry["perimeter"],
                room_type=room_type,
                meets_requirements=requirements,
                windows=windows,
                doors=doors,
                ceiling_height=qualities["ceiling_height"],
                natural_light_score=qualities["natural_light"],
                ventilation_score=qualities["ventilation"],
                accessibility_score=qualities["accessibility"]
            ))
        
        return rooms

    def _analyze_building_structure(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyserer bygningsstruktur og tekniske detaljer"""
        # Detekter strukturelle elementer
        walls = self._detect_walls(image)
        floors = self._detect_floors(image)
        
        # Analyser bæresystem
        structural_system = self._analyze_structural_system(walls)
        
        # Beregn energieffektivitet
        energy_efficiency = self._calculate_energy_efficiency(
            walls, floors, self.config["requirements"])
        
        return {
            "wall_structure": walls,
            "floor_structure": floors,
            "structural_system": structural_system,
            "energy_efficiency": energy_efficiency,
            "floor_count": len(floors)
        }

    def _generate_3d_model(self, image: np.ndarray) -> Dict[str, Any]:
        """Genererer detaljert 3D-modell fra plantegning"""
        try:
            # Konverter 2D til 3D
            model = self.models["3d_reconstruction"](image)
            
            # Legg til teksturer og materialer
            textured_model = self._add_textures(model)
            
            # Generer møblering
            furniture = self._generate_furniture_layout(model)
            
            # Optimaliser for web-visning
            web_model = self._optimize_for_web(textured_model)
            
            return {
                "model": web_model,
                "furniture": furniture,
                "textures": self._get_texture_maps(),
                "lighting": self._calculate_lighting(),
                "camera_positions": self._suggest_camera_positions()
            }
        except Exception as e:
            logger.error(f"Feil ved 3D-modellering: {str(e)}")
            return {}

    def _calculate_measurements(self, 
        image: np.ndarray,
        reference_scale: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Beregner nøyaktige mål med maskinlæring og referansekalibrering
        """
        # Kalibrer målestokk
        scale = reference_scale or self._detect_scale(image)
        
        # Konverter pikselkoordinater til metriske mål
        measurements = {}
        
        # Beregn totalareal og omkrets
        measurements["total_area"] = self._calculate_total_area(image, scale)
        measurements["total_perimeter"] = self._calculate_perimeter(image, scale)
        
        # Beregn bygningsavtrykk
        measurements["footprint"] = self._calculate_footprint(image, scale)
        
        # Detaljerte rommål
        measurements["room_dimensions"] = self._calculate_room_dimensions(
            image, scale)
        
        # Vegg- og vindushøyder
        measurements["wall_heights"] = self._detect_wall_heights(image, scale)
        measurements["window_dimensions"] = self._measure_windows(image, scale)
        
        return measurements

    def _check_room_requirements(
        self,
        qualities: Dict[str, Any],
        room_type: str,
        geometry: Dict[str, float]
    ) -> Dict[str, bool]:
        """
        Sjekker om rommet oppfyller alle tekniske krav
        """
        requirements = {
            "size": self._check_size_requirements(
                geometry["area"], room_type),
            "ceiling_height": qualities["ceiling_height"] >= 
                self.config["requirements"]["ceiling_height"],
            "natural_light": qualities["natural_light"] >= 
                self.config["requirements"]["window_ratio"],
            "ventilation": self._check_ventilation_requirements(
                qualities["ventilation"], room_type),
            "accessibility": qualities["accessibility"] >= 0.8,
            "fire_safety": self._check_fire_safety(
                room_type, geometry, qualities),
            "sound_insulation": self._check_sound_requirements(
                room_type, qualities),
            "moisture_protection": self._check_moisture_protection(
                room_type, qualities)
        }
        
        return requirements

    def generate_report(self, analysis: AnalysisResult) -> Dict[str, Any]:
        """
        Genererer omfattende rapport med anbefalinger og visualiseringer
        """
        report = {
            "sammendrag": self._generate_summary(analysis),
            "tekniske_detaljer": self._generate_technical_details(analysis),
            "forbedringer": self._generate_improvement_details(analysis),
            "kostnadsestimater": self._generate_cost_estimates(analysis),
            "visualiseringer": self._generate_visualizations(analysis),
            "lovkrav": self._generate_regulatory_compliance(analysis),
            "energi_og_miljo": self._generate_environmental_analysis(analysis)
        }
        
        return report