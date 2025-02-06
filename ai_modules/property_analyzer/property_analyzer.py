import os
import sys
from typing import Dict, List, Optional
import numpy as np
import cv2
import tensorflow as tf
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
import torch
from PIL import Image

class PropertyAnalyzer:
    """
    Hovedklasse for analyse av eiendommer basert på bilder, adresse eller lenker.
    Integrerer alle analysemoduler og håndterer koordinering mellom dem.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.models = self._initialize_models()
        self.processors = self._initialize_processors()
        
    def analyze_property(self, 
                        image_path: Optional[str] = None,
                        address: Optional[str] = None,
                        url: Optional[str] = None) -> Dict:
        """
        Hovedmetode for analyse av eiendom. Kan ta imot bilde, adresse eller URL.
        
        Args:
            image_path: Sti til bilde av eiendommen
            address: Eiendomsadresse
            url: URL til eiendomsannonse (f.eks. Finn.no)
            
        Returns:
            Dict med komplett analyserapport inkludert:
            - Bygningsdetaljer
            - Reguleringsinfo
            - Utviklingspotensial
            - 3D-modell
            - Energianalyse
            - Kostnadskalkyle
        """
        try:
            # Samle inn all grunnleggende informasjon
            property_data = self._gather_property_info(image_path, address, url)
            
            # Analysere bygningsstruktur og potensial
            structure_analysis = self._analyze_building_structure(property_data)
            
            # Sjekke kommunale regler og forskrifter
            regulation_info = self._check_regulations(property_data)
            
            # Analysere utviklingspotensial
            development_potential = self._analyze_development_potential(
                structure_analysis, 
                regulation_info
            )
            
            # Generere 3D-modell med NVIDIA Omniverse
            model_3d = self._generate_3d_model(property_data, structure_analysis)
            
            # Utføre energianalyse
            energy_analysis = self._perform_energy_analysis(property_data)
            
            # Generere kostnadsestimater
            cost_estimation = self._estimate_costs(development_potential)
            
            return {
                "property_info": property_data,
                "structure_analysis": structure_analysis,
                "regulations": regulation_info,
                "development_potential": development_potential,
                "3d_model": model_3d,
                "energy_analysis": energy_analysis,
                "cost_estimation": cost_estimation,
                "recommendations": self._generate_recommendations(locals())
            }
            
        except Exception as e:
            self._log_error(f"Error in property analysis: {str(e)}")
            raise
            
    def _gather_property_info(self, image_path, address, url) -> Dict:
        """Samler inn all tilgjengelig informasjon om eiendommen"""
        property_info = {}
        
        if image_path:
            property_info.update(self._analyze_images(image_path))
        
        if address:
            property_info.update(self._fetch_address_info(address))
            
        if url:
            property_info.update(self._scrape_property_listing(url))
            
        return property_info
        
    def _analyze_building_structure(self, property_data: Dict) -> Dict:
        """Analyserer bygningsstruktur og identifiserer muligheter"""
        return {
            "floors": self._analyze_floors(property_data),
            "basement": self._analyze_basement(property_data),
            "attic": self._analyze_attic(property_data),
            "measurements": self._get_measurements(property_data),
            "construction_type": self._identify_construction_type(property_data)
        }
        
    def _check_regulations(self, property_data: Dict) -> Dict:
        """Sjekker gjeldende reguleringsplan og byggtekniske forskrifter"""
        municipality = self._get_municipality(property_data)
        return {
            "zoning_plan": self._fetch_zoning_plan(municipality, property_data),
            "building_regulations": self._get_building_regulations(municipality),
            "restrictions": self._check_restrictions(municipality, property_data),
            "requirements": self._get_requirements(municipality, property_data)
        }
        
    def _analyze_development_potential(self, 
                                     structure: Dict, 
                                     regulations: Dict) -> Dict:
        """Analyserer utviklingspotensial basert på struktur og reguleringer"""
        return {
            "rental_units": self._analyze_rental_potential(structure, regulations),
            "expansion_possibilities": self._find_expansion_options(structure, regulations),
            "property_division": self._analyze_division_potential(structure, regulations),
            "renovation_needs": self._identify_renovation_needs(structure)
        }
        
    def _generate_3d_model(self, 
                          property_data: Dict, 
                          structure_analysis: Dict) -> Dict:
        """Genererer detaljert 3D-modell med NVIDIA Omniverse"""
        return {
            "model_url": self._create_3d_model(property_data, structure_analysis),
            "floor_plans": self._generate_floor_plans(structure_analysis),
            "facade_drawings": self._generate_facade_drawings(structure_analysis),
            "site_plan": self._generate_site_plan(property_data)
        }
        
    def _perform_energy_analysis(self, property_data: Dict) -> Dict:
        """Utfører energianalyse og identifiserer forbedringspotensial"""
        return {
            "current_rating": self._calculate_energy_rating(property_data),
            "potential_rating": self._calculate_potential_rating(property_data),
            "improvement_measures": self._identify_energy_improvements(property_data),
            "enova_support": self._calculate_enova_support(property_data)
        }
        
    def _estimate_costs(self, development_potential: Dict) -> Dict:
        """Estimerer kostnader for ulike utviklingsmuligheter"""
        return {
            "renovation_costs": self._calculate_renovation_costs(development_potential),
            "conversion_costs": self._calculate_conversion_costs(development_potential),
            "expansion_costs": self._calculate_expansion_costs(development_potential),
            "potential_revenue": self._estimate_revenue(development_potential)
        }
        
    def _generate_recommendations(self, analysis_data: Dict) -> List[Dict]:
        """Genererer prioriterte anbefalinger basert på all analysert data"""
        recommendations = []
        
        # Analyser muligheter og sorter etter kost/nytte
        opportunities = self._identify_opportunities(analysis_data)
        prioritized_actions = self._prioritize_actions(opportunities)
        
        for action in prioritized_actions:
            recommendations.append({
                "title": action["title"],
                "description": action["description"],
                "cost": action["estimated_cost"],
                "benefit": action["estimated_benefit"],
                "roi": action["roi"],
                "timeline": action["estimated_timeline"],
                "requirements": action["requirements"],
                "next_steps": action["next_steps"]
            })
            
        return recommendations