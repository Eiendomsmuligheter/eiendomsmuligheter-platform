#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PropertyAnalyzer - Implementerer analyselogikk for AlterraML
------------------------------------------------------------
Denne modulen inneholder implementasjon av faktiske analysemetoder
som brukes av AlterraML-klassen.
"""

import os
import sys
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
# Bruk opencv-python-headless istedenfor full opencv
import cv2
# Lazy import av tensorflow og torch for å redusere oppstartstid og minnebruk
# import tensorflow as tf
# import torch
from PIL import Image
import logging
import yaml
import json
import requests
import asyncio
from datetime import datetime
import re
import time
from pathlib import Path
import traceback
from bs4 import BeautifulSoup
# Lazy import av transformers for å redusere oppstartstid
# from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from functools import lru_cache
import uuid
from collections import namedtuple

# Sett opp logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sett miljøvariabler for å redusere ressursbruk
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduser TensorFlow-logging
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Dynamisk GPU-minneallokering
os.environ["NVIDIA_VISIBLE_DEVICES"] = "none"  # Deaktiver GPU som standard

class PropertyAnalyzer:
    """
    Hovedklasse for analyse av eiendommer basert på bilder, adresse eller lenker.
    Integrerer alle analysemoduler og håndterer koordinering mellom dem.
    
    Optimalisert for lavere ressursbruk:
    - Lazy loading av modeller
    - Bruk av lettere modeller
    - Caching av resultater
    - Progressiv prosessering
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialiserer PropertyAnalyzer med konfigurasjon og nødvendige modeller
        
        Args:
            config_path: Sti til konfigurasjonsfil
        """
        self.config = self._load_config(config_path)
        self.models = {}  # Lazy loading av modeller
        self.processors = {}  # Lazy loading av prosessorer
        self.cache = {}  # Enkel hurtigbuffer for å unngå unødvendige API-kall
        self.model_paths = self._setup_model_paths()
        
        logger.info("PropertyAnalyzer initialisert med optimalisert konfigurasjon")
        
    def _load_config(self, config_path: str) -> Dict:
        """Laster konfigurasjon fra YAML-fil"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    config = yaml.safe_load(file)
                logger.info(f"Konfigurasjon lastet fra {config_path}")
                return config
            else:
                logger.warning(f"Konfigurasjonsfil {config_path} ikke funnet. Bruker standardverdier.")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Feil ved lasting av konfigurasjon: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Returnerer standardkonfigurasjon hvis konfigurasjonsfil ikke finnes"""
        # Sjekk om GPU er tilgjengelig, men default til CPU for lavere ressursbruk
        use_gpu = False
        try:
            # Lazy import for å unngå unødvendig avhengighet
            # import torch
            use_gpu = False
        except ImportError:
            use_gpu = False
            
        return {
            "api_keys": {
                "google_maps": os.environ.get("GOOGLE_MAPS_API_KEY", ""),
                "finn_api": os.environ.get("FINN_API_KEY", ""),
                "municipality_api": os.environ.get("MUNICIPALITY_API_KEY", "")
            },
            "models": {
                "image_analyzer": "models/image_analyzer_lite",  # Bruk lettere modeller
                "floor_plan_analyzer": "models/floor_plan_analyzer_lite",
                "text_analyzer": "models/text_analyzer_lite",
                "document_analyzer": "models/document_analyzer_lite"
            },
            "paths": {
                "temp_dir": "temp",
                "output_dir": "output",
                "models_dir": "models",
                "data_dir": "data"
            },
            "municipality_data": {
                "drammen": {
                    "api_url": "https://innsyn2020.drammen.kommune.no/api",
                    "map_url": "https://kart.drammen.kommune.no"
                }
            },
            "enova": {
                "api_url": "https://api.enova.no",
                "support_rates": {
                    "heat_pump": 10000,
                    "insulation": 500,  # per m²
                    "windows": 1000     # per window
                }
            },
            "use_gpu": use_gpu,
            "processing": {
                "max_image_size": 1280,  # Redusert fra 1920 for lavere minnebruk
                "image_quality": 85,     # Litt lavere kvalitet for mindre størrelse
                "use_caching": True,
                "parallel_processing": True,
                "progressive_loading": True  # Ny funksjon for progressiv lasting
            },
            "optimization": {
                "use_lite_models": True,  # Bruk lettere modeller
                "quantize_models": True,  # Kvantiser modeller for lavere minnebruk
                "lazy_loading": True,     # Last modeller kun ved behov
                "batch_processing": True  # Prosesser i batches for bedre ytelse
            }
        }
    
    def _setup_model_paths(self) -> Dict[str, str]:
        """Setter opp stier til modellfilene"""
        base_dir = self.config.get("paths", {}).get("models_dir", "models")
        
        # Bruk lite-versjoner av modellene hvis konfigurert
        suffix = "_lite" if self.config.get("optimization", {}).get("use_lite_models", True) else ""
        
        return {
            "image_analyzer": os.path.join(base_dir, f"image_analyzer{suffix}"),
            "floor_plan_analyzer": os.path.join(base_dir, f"floor_plan_analyzer{suffix}"),
            "document_analyzer": os.path.join(base_dir, f"document_analyzer{suffix}"),
            "region_segmentation": os.path.join(base_dir, f"region_segmentation{suffix}"),
            "3d_model_generator": os.path.join(base_dir, f"3d_model_generator{suffix}")
        }
    
    def _get_model(self, model_name: str) -> Any:
        """Lazy loading av modeller - laster kun når de faktisk trengs"""
        if model_name in self.models:
            return self.models[model_name]
            
        logger.info(f"Laster modell: {model_name}")
        
        try:
            if model_name == "document_analyzer":
                # Bruk en lettere modell for dokumentanalyse
                if self.config.get("optimization", {}).get("use_lite_models", True):
                    # Bruk ONNX Runtime for raskere inferens
                    import onnxruntime as ort
                    model_path = os.path.join(self.model_paths["document_analyzer"], "model.onnx")
                    if os.path.exists(model_path):
                        session = ort.InferenceSession(model_path)
                        self.models[model_name] = {"session": session}
                        return self.models[model_name]
                
                # Fallback til standard modell hvis ONNX ikke er tilgjengelig
                try:
                    # Lazy import for å redusere oppstartstid
                    # from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
                    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", 
                                                                   use_fast=True)
                model = LayoutLMv3ForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base")
                    
                    # Kvantiser modellen hvis konfigurert
                    if self.config.get("optimization", {}).get("quantize_models", True):
                        # Lazy import for å redusere oppstartstid
                        # import torch
                        model = torch.quantization.quantize_dynamic(
                            model, {torch.nn.Linear}, dtype=torch.qint8
                        )
                    
                    self.models[model_name] = {
                    "processor": processor,
                    "model": model
                }
                except ImportError:
                    logger.warning("Transformers ikke tilgjengelig, bruker enklere dokumentanalyse")
                    self.models[model_name] = {"type": "simple_analyzer"}
            
            elif model_name == "image_analyzer":
                # Bruk OpenCV-basert bildeanalyse istedenfor tunge ML-modeller
                self.models[model_name] = {"type": "opencv_analyzer"}
                
            elif model_name == "floor_plan_analyzer":
                # Bruk enklere geometrisk analyse for plantegninger
                self.models[model_name] = {"type": "geometric_analyzer"}
                
            # Legg til flere modeller etter behov
                
            return self.models.get(model_name, {})
            
        except Exception as e:
            logger.error(f"Feil ved lasting av modell {model_name}: {str(e)}")
            # Returner en tom dict som fallback
            return {}
    
    def _get_processor(self, processor_name: str) -> Any:
        """Lazy loading av prosessorer - laster kun når de faktisk trengs"""
        if processor_name in self.processors:
            return self.processors[processor_name]
            
        logger.info(f"Initialiserer prosessor: {processor_name}")
        
        try:
            # Import moduler kun når de trengs
            if processor_name == "image_analyzer":
                from image_analyzer.lite import ImageAnalyzerLite
                self.processors[processor_name] = ImageAnalyzerLite()
                
            elif processor_name == "floor_plan_analyzer":
                from floor_plan_analyzer.lite import FloorPlanAnalyzerLite
                self.processors[processor_name] = FloorPlanAnalyzerLite()
                
            elif processor_name == "price_estimator":
                from price_estimator.lite import PriceEstimatorLite
                self.processors[processor_name] = PriceEstimatorLite()
                
            elif processor_name == "energy_analyzer":
                from energy_analyzer.lite import EnergyAnalyzerLite
                self.processors[processor_name] = EnergyAnalyzerLite()
                
            # Legg til flere prosessorer etter behov
                
            return self.processors.get(processor_name)
            
        except ImportError as e:
            logger.warning(f"Kunne ikke importere prosessor {processor_name}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Feil ved initialisering av prosessor {processor_name}: {str(e)}")
            return None
    
    @lru_cache(maxsize=32)
    def _get_municipality_data(self, municipality_name: str) -> Dict:
        """Henter kommunedata med caching for bedre ytelse"""
        cache_key = f"municipality_{municipality_name}"
        
        if cache_key in self.cache:
            # Sjekk om cachen er gyldig (mindre enn 24 timer gammel)
            cache_time = self.cache[cache_key].get("timestamp", 0)
            if time.time() - cache_time < 86400:  # 24 timer
                return self.cache[cache_key].get("data", {})
        
        # Hent data fra API eller lokal kilde
        try:
            municipality_config = self.config.get("municipality_data", {}).get(municipality_name, {})
            if not municipality_config:
                logger.warning(f"Ingen konfigurasjon funnet for kommune: {municipality_name}")
                return {}
                
            api_url = municipality_config.get("api_url")
            if not api_url:
                logger.warning(f"Ingen API URL funnet for kommune: {municipality_name}")
                return {}
                
            # Hent data fra API
            response = requests.get(f"{api_url}/regulations", timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Lagre i cache
                self.cache[cache_key] = {
                    "data": data,
                    "timestamp": time.time()
                }
                
                return data
            else:
                logger.error(f"Feil ved henting av kommunedata: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Feil ved henting av kommunedata for {municipality_name}: {str(e)}")
            return {}
    
    async def analyze_property(self, 
                        image_path: Optional[str] = None,
                        address: Optional[str] = None,
                        url: Optional[str] = None,
                        floor_plan_path: Optional[str] = None,
                        documents: Optional[List[str]] = None,
                        client_preferences: Optional[Dict] = None) -> Dict:
        """
        Hovedmetode for analyse av eiendom. Kan ta imot bilde, adresse eller URL.
        Optimalisert for progressiv prosessering og lavere ressursbruk.
        
        Args:
            image_path: Sti til bilde av eiendommen (eller liste med bilder)
            address: Eiendomsadresse
            url: URL til eiendomsannonse (f.eks. Finn.no)
            floor_plan_path: Sti til plantegning
            documents: Liste med stier til relevante dokumenter
            client_preferences: Kundeprefereranser for analysen
            
        Returns:
            Dict med komplett analyserapport inkludert:
            - Bygningsdetaljer
            - Reguleringsinfo
            - Utviklingspotensial
            - 3D-modell
            - Energianalyse
            - Kostnadskalkyle
        """
        start_time = time.time()
        
        # Initialiser resultatobjekt
        result = {
            "id": f"analysis_{int(time.time())}",
                "timestamp": datetime.now().isoformat(),
            "status": "in_progress",
            "progress": 0,
            "address": address,
            "results": {},
            "errors": []
        }
        
        try:
            # Steg 1: Hent adresseinformasjon hvis ikke oppgitt
            if not address and (image_path or url):
                address = await self._extract_address(image_path, url)
                result["address"] = address
                result["progress"] = 10
            
            if not address:
                raise ValueError("Kunne ikke finne adresse. Vennligst oppgi en gyldig adresse.")
            
            # Steg 2: Hent grunnleggende eiendomsinformasjon
            property_info = await self._get_property_info(address)
            result["results"]["property_info"] = property_info
            result["progress"] = 20
            
            # Steg 3: Analyser bilder hvis tilgjengelig (asynkront)
            image_analysis_task = None
            if image_path:
                image_analysis_task = asyncio.create_task(self._analyze_images(image_path))
            
            # Steg 4: Analyser plantegning hvis tilgjengelig (asynkront)
            floor_plan_task = None
        if floor_plan_path:
                floor_plan_task = asyncio.create_task(self._analyze_floor_plan(floor_plan_path))
            
            # Steg 5: Hent kommunale reguleringer
            municipality = property_info.get("municipality", "")
            if municipality:
                regulations = await self._get_regulations(municipality, property_info)
                result["results"]["regulations"] = regulations
                result["progress"] = 40
            
            # Steg 6: Vent på bildeanalyse og plantegningsanalyse
            if image_analysis_task:
                image_results = await image_analysis_task
                result["results"]["image_analysis"] = image_results
            
            if floor_plan_task:
                floor_plan_results = await floor_plan_task
                result["results"]["floor_plan"] = floor_plan_results
            
            result["progress"] = 60
            
            # Steg 7: Generer 3D-modell (forenklet versjon for lavere ressursbruk)
            model_data = await self._generate_3d_model(
                property_info, 
                result["results"].get("floor_plan", {}),
                result["results"].get("image_analysis", {})
            )
            result["results"]["model_data"] = model_data
            result["progress"] = 80
            
            # Steg 8: Beregn utviklingspotensial og energianalyse
            potential = await self._calculate_potential(
                property_info,
                result["results"].get("regulations", {}),
                result["results"].get("floor_plan", {})
            )
            result["results"]["development_potential"] = potential
            
            energy = await self._analyze_energy(
                property_info,
                result["results"].get("floor_plan", {}),
                client_preferences
            )
            result["results"]["energy_analysis"] = energy
            
            # Steg 9: Ferdigstill analysen
            result["status"] = "completed"
            result["progress"] = 100
            result["processing_time"] = time.time() - start_time
            
            logger.info(f"Analyse fullført på {result['processing_time']:.2f} sekunder")
        return result
            
        except Exception as e:
            error_msg = f"Feil under analyse: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            result["status"] = "error"
            result["errors"].append(error_msg)
            result["processing_time"] = time.time() - start_time
        
        return result
    
    def _log_error(self, message: str):
        """Logger feilmelding"""
        logger.error(message)

async def analyze_property(property_data) -> Dict[str, Any]:
    """Analyser en eiendom og returner resultater"""
    logger.info(f"Analyserer eiendom: {property_data.address}")
    
    # Generere dummy-resultater
    from collections import namedtuple
    
    # Opprett basic dummy-strukturer
    RegulationRule = namedtuple('RegulationRule', 'rule_id rule_type value description unit category')
    BuildingPotential = namedtuple('BuildingPotential', 'max_buildable_area max_height max_units optimal_configuration constraints recommendations')
    EnergyProfile = namedtuple('EnergyProfile', 'energy_class heating_demand cooling_demand primary_energy_source recommendations')
    
    # Lag reguleringsregler
    regulations = []
    
    # Sjekk om vi har eksisterende regler i property_data
    if hasattr(property_data, 'regulations') and property_data.regulations:
        for reg in property_data.regulations:
            regulations.append(RegulationRule(
                rule_id=reg.get('rule_id', 'RULE-' + str(uuid.uuid4())[:8]),
                rule_type=reg.get('rule_type', 'unknown'),
                value=reg.get('value', 0),
                description=reg.get('description', 'Ukjent regel'),
                unit=reg.get('unit', None),
                category=reg.get('category', None)
            ))
    
    # Legg til standard-regler hvis ingen er definert
    if not regulations:
        regulations = [
            RegulationRule(
                rule_id="REG-001", 
                rule_type="floor_area_ratio", 
                value=0.7, 
                description="Maks tillatt BRA faktor", 
                unit="ratio", 
                category="utilization"
            ),
            RegulationRule(
                rule_id="REG-002", 
                rule_type="max_height", 
                value=12.0, 
                description="Maks byggehøyde", 
                unit="meter", 
                category="height"
            ),
            RegulationRule(
                rule_id="REG-003", 
                rule_type="min_distance", 
                value=4.0, 
                description="Minimum avstand til nabobygg", 
                unit="meter", 
                category="placement"
            )
        ]
    
    # Beregn byggepotensial basert på tomtestørrelse og utnyttelsesgrad
    lot_size = property_data.lot_size or 1000.0
    far = property_data.floor_area_ratio or 0.5
    current_util = property_data.current_utilization or 0.2
    max_height = property_data.building_height or 8.0
    
    # Beregne verdier
    max_buildable_area = lot_size * far
    available_area = max_buildable_area - (lot_size * current_util)
    max_units = max(1, int(available_area / 70))  # Antatt 70 m² per boenhet
    
    # Lag byggepotensial
    building_potential = BuildingPotential(
        max_buildable_area=max_buildable_area,
        max_height=max_height,
        max_units=max_units,
        optimal_configuration=f"{max_units} enheter fordelt på {max(1, int(max_height/3))} etasjer",
        constraints=["Avstand til nabo: 4m", "Maks takvinkel: 45°"],
        recommendations=["Plasser bygget mot nord for god solforhold", "Vurder underetasje for å utnytte terreng"]
    )
    
    # Lag energiprofil
    energy_profile = EnergyProfile(
        energy_class="B",
        heating_demand=75.0,
        cooling_demand=20.0,
        primary_energy_source="Electricity",
        recommendations=["Vurder solceller på sørvendt tak", "Installér varmepumpe for effektiv oppvarming"]
    )
    
    # Beregn ROI
    roi_estimate = 0.15  # 15% avkastning
    
    # Lag risikovurdering
    risk_assessment = {
        "market_risk": "low",
        "regulatory_risk": "medium",
        "construction_risk": "low",
        "financial_risk": "medium",
        "environmental_risk": "low"
    }
    
    # Lag anbefalinger
    recommendations = [
        "Bygg rekkehus for optimal utnyttelse av tomten",
        "Inkluder grønne tak for bedre miljøprofil",
        "Planlegg for parkeringsløsning under bakken",
        "Optimalisér for sydvendte balkonger"
    ]
    
    # Sett sammen analysen
    analysis_result = {
        "property_id": property_data.property_id,
        "address": property_data.address,
        "regulations": regulations,
        "building_potential": building_potential,
        "energy_profile": energy_profile,
        "roi_estimate": roi_estimate,
        "risk_assessment": risk_assessment,
        "recommendations": recommendations
    }
    
    logger.info(f"Analyse fullført for {property_data.address}")
    return analysis_result

async def generate_terrain(terrain_data, heightmap_path: str, texture_path: str, texture_type: str = "satellite") -> Dict[str, Any]:
    """Generér terrengdata for visualisering"""
    logger.info(f"Genererer terreng for eiendom {terrain_data.property_id}")
    
    # I en ekte implementasjon ville vi generert faktiske høydekart og teksturer
    # For nå, lager vi bare tomme bildefiler
    
    # Lag en tom PNG for høydekartet (hvit farge)
    try:
        # Prøv å bruke PIL hvis det er tilgjengelig
        try:
            from PIL import Image
            
            # Opprett et hvitt bilde som høydekart
            heightmap_img = Image.new('L', (terrain_data.resolution, terrain_data.resolution), 255)
            heightmap_img.save(heightmap_path)
            
            # Opprett et grønt bilde som tekstur
            texture_img = Image.new('RGB', (terrain_data.resolution, terrain_data.resolution), (100, 200, 100))
            texture_img.save(texture_path)
            
            logger.info(f"Lagret høydekart til {heightmap_path} og tekstur til {texture_path}")
        except ImportError:
            # Fallback til å bare opprette tomme filer
            with open(heightmap_path, 'wb') as f:
                f.write(b'DUMMY HEIGHTMAP DATA')
            
            with open(texture_path, 'wb') as f:
                f.write(b'DUMMY TEXTURE DATA')
            
            logger.info(f"Lagret dummy-høydekart til {heightmap_path} og dummy-tekstur til {texture_path}")
            
        except Exception as e:
        logger.error(f"Kunne ikke generere terrengfiler: {e}")
        # Fortsatt returnere et resultat for å unngå feil i API-et
    
    # Returner metadata om terrenget
    terrain_result = {
        "metadata": {
            "property_id": terrain_data.property_id,
            "width": terrain_data.width,
            "depth": terrain_data.depth,
            "resolution": terrain_data.resolution,
            "include_surroundings": terrain_data.include_surroundings,
            "include_buildings": terrain_data.include_buildings,
            "texture_type": texture_type,
            "generated_at": datetime.now().isoformat()
        },
        "bounds": {
            "north": 59.956,  # Dummy koordinater
            "south": 59.942,
            "east": 10.789,
            "west": 10.768,
            "min_height": 10.0,
            "max_height": 45.0
        }
    }
    
    logger.info(f"Terrengdata generert for eiendom {terrain_data.property_id}")
    return terrain_result
    
async def generate_building(building_data: Dict[str, Any], model_path: str, thumbnail_path: str) -> Dict[str, Any]:
    """Generér bygningsmodell for visualisering"""
    property_id = building_data.get('property_id', 'unknown')
    logger.info(f"Genererer bygningsmodell for eiendom {property_id}")
    
    # I en ekte implementasjon ville vi generert faktiske 3D-modeller
    # For nå, lager vi bare tomme filer
    
    try:
        # Opprett tomme filer
        with open(model_path, 'wb') as f:
            f.write(b'DUMMY 3D MODEL DATA')
        
        # Generer et thumbnail hvis mulig
        try:
            from PIL import Image
            
            # Opprett et enkelt thumbnail
            thumb_img = Image.new('RGB', (200, 200), (150, 150, 150))
            thumb_img.save(thumbnail_path)
            
            logger.info(f"Lagret 3D-modell til {model_path} og thumbnail til {thumbnail_path}")
        except ImportError:
            # Fallback til å bare opprette en tom fil
            with open(thumbnail_path, 'wb') as f:
                f.write(b'DUMMY THUMBNAIL DATA')
            
            logger.info(f"Lagret dummy 3D-modell til {model_path} og dummy-thumbnail til {thumbnail_path}")
            
        except Exception as e:
        logger.error(f"Kunne ikke generere bygningsfiler: {e}")
        # Fortsatt returnere et resultat for å unngå feil i API-et
    
    # Returner metadata om bygningen
    building_result = {
        "metadata": {
            "property_id": property_id,
            "building_type": building_data.get('building_type', 'default'),
            "floors": building_data.get('floors', 2),
            "dimensions": {
                "width": building_data.get('width', 10.0),
                "depth": building_data.get('depth', 10.0),
                "height": building_data.get('height', 6.0)
            },
            "style": building_data.get('style', 'modern'),
            "roof_type": building_data.get('roof_type', 'flat'),
            "generated_at": datetime.now().isoformat()
        }
    }
    
    logger.info(f"Bygningsmodell generert for eiendom {property_id}")
    return building_result
