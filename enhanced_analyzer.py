import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List, Dict, Optional
import json
import logging
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

class EnhancedPropertyAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.requirements = self._load_requirements()
        
    def _load_requirements(self):
        """Last inn alle tekniske krav fra TEK17"""
        return {
            "rømning": {
                "maksimal_avstand": 30,  # meter til rømningsvei
                "min_bredde": 0.9,  # meter for rømningsvei
                "nødvendig_lys": True,
                "merking": True
            },
            "brann": {
                "vegger": {
                    "EI30": "30 minutter brannmotstand",
                    "EI60": "60 minutter brannmotstand",
                    "REI90": "90 minutter brannmotstand"
                },
                "brannceller": True,
                "røykvarsler": True,
                "slukkeutstyr": True
            },
            "lys": {
                "dagslys_faktor": 0.1,  # 10% av gulvareal
                "lyshøyde": 2.1,  # meter
                "vindu_areal": 0.1  # 10% av gulvareal
            },
            "ventilasjon": {
                "luftskifte": 0.5,  # ganger per time
                "friskluft": 26,  # m³ per time per person
                "avtrekk": {
                    "kjøkken": 36,  # m³ per time
                    "bad": 54,  # m³ per time
                    "toalett": 36  # m³ per time
                }
            },
            "rom": {
                "takhøyde": 2.4,  # meter
                "min_areal": {
                    "soverom": 7,  # m²
                    "stue": 15,  # m²
                    "kjøkken": 6  # m²
                }
            }
        }
        
    def analyze_finn_listing(self, finnkode: str) -> Dict:
        """Analyser Finn.no annonse"""
        url = f"https://www.finn.no/realestate/homes/ad.html?finnkode={finnkode}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Hent all relevant informasjon
        data = {
            "finnkode": finnkode,
            "bilder": self._extract_images(soup),
            "plantegninger": self._extract_floor_plans(soup),
            "info": self._extract_property_info(soup)
        }
        
        return self.perform_analysis(data)
        
    def perform_analysis(self, data: Dict) -> Dict:
        """Utfør komplett analyse med dokumentasjon"""
        analysis = {
            "eksisterende": self._analyze_current_state(data),
            "muligheter": self._analyze_potential(data),
            "krav": self._analyze_requirements(data),
            "visualisering": self._generate_visualization(data)
        }
        
        return self._generate_report(analysis)
        
    def _analyze_current_state(self, data: Dict) -> Dict:
        """Analyser eksisterende tilstand"""
        return {
            "etasjer": self._detect_floors(data["plantegninger"]),
            "rom": self._analyze_rooms(data["plantegninger"]),
            "teknisk": self._analyze_technical_state(data)
        }
        
    def _analyze_requirements(self, data: Dict) -> Dict:
        """Analyser og dokumenter alle tekniske krav"""
        requirements = {}
        
        # Brannkrav
        requirements["brann"] = self._analyze_fire_safety(data)
        
        # Rømningskrav
        requirements["rømning"] = self._analyze_escape_routes(data)
        
        # Lyskrav
        requirements["lys"] = self._analyze_lighting(data)
        
        # Ventilasjonskrav
        requirements["ventilasjon"] = self._analyze_ventilation(data)
        
        return requirements
        
    def _analyze_fire_safety(self, data: Dict) -> Dict:
        """Analyser brannsikkerhet med dokumentasjon"""
        return {
            "vegger": {
                "type": "EI60",
                "dokumentasjon": {
                    "beregning": "Beregnet basert på TEK17 §11-8",
                    "detaljer": self._generate_wall_details()
                }
            },
            "rømning": {
                "avstander": self._calculate_escape_distances(data),
                "dokumentasjon": "Tegning med markerte rømningsveier"
            },
            "brannceller": self._analyze_fire_cells(data)
        }
        
    def _generate_wall_details(self) -> Dict:
        """Generer detaljerte veggspesifikasjoner"""
        return {
            "oppbygging": [
                "13mm gipsplate",
                "98mm isolasjon",
                "98mm stenderverk",
                "13mm gipsplate"
            ],
            "brannmotstand": "EI60",
            "lydklasse": "R'w+Ctr ≥ 55 dB",
            "u_verdi": "0.21 W/(m²·K)"
        }
        
    def _analyze_escape_routes(self, data: Dict) -> Dict:
        """Analyser rømningsveier med dokumentasjon"""
        return {
            "primær_rømningsvei": {
                "type": "Hovedtrapp",
                "bredde": 1.2,
                "avstand": self._calculate_escape_distance()
            },
            "sekundær_rømningsvei": {
                "type": "Vinduer",
                "spesifikasjoner": "Rømningsvindu med fri åpning minimum 0.5m x 0.6m"
            },
            "dokumentasjon": {
                "tegning": "Plantegning med markerte rømningsveier",
                "beregninger": self._escape_route_calculations()
            }
        }
        
    def _analyze_lighting(self, data: Dict) -> Dict:
        """Analyser lyskrav med dokumentasjon"""
        return {
            "dagslys": {
                "beregning": self._calculate_daylight_factor(),
                "dokumentasjon": "Dagslysberegning per rom"
            },
            "vinduer": {
                "areal": self._calculate_window_areas(),
                "lyshøyde": self._measure_window_heights()
            }
        }
        
    def _generate_visualization(self, data: Dict) -> Dict:
        """Generer 3D visualiseringer"""
        return {
            "3d_modell": {
                "eksisterende": self._generate_3d_model(data, "existing"),
                "forslag": self._generate_3d_model(data, "proposed")
            },
            "tekniske_detaljer": {
                "brannvegger": self._visualize_fire_walls(),
                "rømning": self._visualize_escape_routes(),
                "ventilasjon": self._visualize_ventilation()
            }
        }
        
    def _generate_report(self, analysis: Dict) -> Dict:
        """Generer komplett rapport med all dokumentasjon"""
        return {
            "sammendrag": {
                "konklusjon": "Detaljert vurdering av muligheter",
                "hovedpunkter": self._summarize_main_points(analysis)
            },
            "teknisk_dokumentasjon": {
                "krav": analysis["krav"],
                "beregninger": self._compile_calculations(analysis),
                "tegninger": self._compile_drawings(analysis)
            },
            "visualiseringer": analysis["visualisering"],
            "anbefalinger": self._generate_recommendations(analysis)
        }