from typing import Dict, List, Optional
import numpy as np
import logging
from datetime import datetime
import aiohttp
import json
import math

logger = logging.getLogger(__name__)

class EnergyAnalyzer:
    """
    Klasse for analyse av energibruk og beregning av Enova-støtte.
    Funksjoner:
    - Energimerkeberegning
    - Varmetapsberegning
    - Oppgraderingsanbefalinger
    - Enova-støtteberegning
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.enova_api = EnovaAPI(self.config.get("enova_api_key"))
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Last konfigurasjon"""
        if config_path:
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            "energy_calculation_method": "NS3031",
            "default_climate_zone": "Oslo",
            "include_detailed_simulation": True
        }
        
    async def analyze_energy_performance(self,
                                       building_data: Dict,
                                       construction_details: Dict) -> Dict:
        """
        Utfør komplett energianalyse av bygningen
        """
        try:
            # Beregn nåværende energiytelse
            current_performance = await self._calculate_energy_performance(
                building_data,
                construction_details
            )
            
            # Identifiser forbedringspotensial
            improvement_potential = await self._identify_improvements(
                current_performance,
                construction_details
            )
            
            # Beregn potensiell energimerking etter forbedringer
            potential_performance = await self._calculate_potential_performance(
                current_performance,
                improvement_potential
            )
            
            # Beregn støttemuligheter fra Enova
            enova_support = await self.enova_api.calculate_support(
                current_performance,
                improvement_potential
            )
            
            return {
                "current_performance": current_performance,
                "improvement_potential": improvement_potential,
                "potential_performance": potential_performance,
                "enova_support": enova_support,
                "recommendations": self._generate_recommendations(locals())
            }
            
        except Exception as e:
            logger.error(f"Feil ved energianalyse: {str(e)}")
            raise
            
    async def _calculate_energy_performance(self,
                                          building_data: Dict,
                                          construction_details: Dict) -> Dict:
        """
        Beregn nåværende energiytelse
        """
        try:
            # Beregn U-verdier
            u_values = self._calculate_u_values(construction_details)
            
            # Beregn varmetap
            heat_loss = self._calculate_heat_loss(
                building_data,
                u_values
            )
            
            # Beregn energibehov
            energy_demand = self._calculate_energy_demand(
                heat_loss,
                building_data
            )
            
            # Beregn energimerke
            energy_rating = self._calculate_energy_rating(
                energy_demand,
                building_data
            )
            
            return {
                "u_values": u_values,
                "heat_loss": heat_loss,
                "energy_demand": energy_demand,
                "energy_rating": energy_rating
            }
            
        except Exception as e:
            logger.error(f"Feil ved beregning av energiytelse: {str(e)}")
            raise
            
    async def _identify_improvements(self,
                                   current_performance: Dict,
                                   construction_details: Dict) -> Dict:
        """
        Identifiser mulige energiforbedringer
        """
        improvements = {
            "insulation": self._analyze_insulation_improvements(
                current_performance,
                construction_details
            ),
            "windows": self._analyze_window_improvements(
                current_performance,
                construction_details
            ),
            "heating_system": self._analyze_heating_system_improvements(
                current_performance,
                construction_details
            ),
            "ventilation": self._analyze_ventilation_improvements(
                current_performance,
                construction_details
            )
        }
        
        return {
            "measures": improvements,
            "priorities": self._prioritize_improvements(improvements),
            "cost_estimates": self._estimate_improvement_costs(improvements)
        }
        
    def _calculate_u_values(self, construction_details: Dict) -> Dict:
        """
        Beregn U-verdier for bygningsdeler
        """
        u_values = {}
        
        for component, details in construction_details.items():
            if component == "walls":
                u_values[component] = self._calculate_wall_u_value(details)
            elif component == "roof":
                u_values[component] = self._calculate_roof_u_value(details)
            elif component == "floor":
                u_values[component] = self._calculate_floor_u_value(details)
            elif component == "windows":
                u_values[component] = self._calculate_window_u_value(details)
            
        return u_values
        
    def _calculate_heat_loss(self,
                           building_data: Dict,
                           u_values: Dict) -> Dict:
        """
        Beregn varmetap for bygningen
        """
        # Beregn transmisjonstap
        transmission_loss = self._calculate_transmission_loss(
            building_data,
            u_values
        )
        
        # Beregn infiltrasjonstap
        infiltration_loss = self._calculate_infiltration_loss(
            building_data
        )
        
        # Beregn ventilasjonstap
        ventilation_loss = self._calculate_ventilation_loss(
            building_data
        )
        
        return {
            "transmission": transmission_loss,
            "infiltration": infiltration_loss,
            "ventilation": ventilation_loss,
            "total": sum([
                transmission_loss,
                infiltration_loss,
                ventilation_loss
            ])
        }
        
    def _calculate_energy_demand(self,
                               heat_loss: Dict,
                               building_data: Dict) -> Dict:
        """
        Beregn energibehov
        """
        # Beregn oppvarmingsbehov
        heating_demand = self._calculate_heating_demand(
            heat_loss,
            building_data
        )
        
        # Beregn kjølebehov
        cooling_demand = self._calculate_cooling_demand(
            heat_loss,
            building_data
        )
        
        # Beregn varmtvannsbehov
        hot_water_demand = self._calculate_hot_water_demand(
            building_data
        )
        
        # Beregn elektrisitetsbehov
        electricity_demand = self._calculate_electricity_demand(
            building_data
        )
        
        return {
            "heating": heating_demand,
            "cooling": cooling_demand,
            "hot_water": hot_water_demand,
            "electricity": electricity_demand,
            "total": sum([
                heating_demand,
                cooling_demand,
                hot_water_demand,
                electricity_demand
            ])
        }
        
    def _calculate_energy_rating(self,
                               energy_demand: Dict,
                               building_data: Dict) -> str:
        """
        Beregn energimerke basert på energibehov
        """
        # Beregn spesifikk energibruk
        specific_energy = energy_demand["total"] / building_data["heated_area"]
        
        # Definer energimerke-grenser
        limits = {
            "A": 95,  # kWh/m²
            "B": 120,
            "C": 145,
            "D": 175,
            "E": 205,
            "F": 250,
            "G": float("inf")
        }
        
        # Returner energimerke
        for rating, limit in limits.items():
            if specific_energy <= limit:
                return rating
                
        return "G"
        
    def _generate_recommendations(self, analysis_data: Dict) -> List[Dict]:
        """
        Generer prioriterte anbefalinger for energiforbedringer
        """
        recommendations = []
        
        # Analyser forbedringspotensial
        improvements = analysis_data["improvement_potential"]["measures"]
        costs = analysis_data["improvement_potential"]["cost_estimates"]
        
        for category, measures in improvements.items():
            for measure in measures:
                cost = costs[f"{category}_{measure['id']}"]
                savings = measure["energy_savings"]
                roi = savings["yearly_value"] / cost["total"]
                
                recommendations.append({
                    "category": category,
                    "measure": measure["description"],
                    "cost": cost,
                    "savings": savings,
                    "roi": roi,
                    "priority": self._calculate_priority(roi, savings),
                    "enova_support": measure.get("enova_support", 0),
                    "implementation_time": measure["implementation_time"],
                    "technical_requirements": measure["technical_requirements"]
                })
        
        # Sorter anbefalinger etter prioritet
        recommendations.sort(key=lambda x: x["priority"], reverse=True)
        
        return recommendations
        
    def _calculate_priority(self, roi: float, savings: Dict) -> float:
        """
        Beregn prioritet for et tiltak basert på ROI og besparelser
        """
        # Vektlegging av faktorer
        roi_weight = 0.4
        energy_weight = 0.3
        co2_weight = 0.3
        
        # Normaliser verdier
        roi_score = min(roi / 0.5, 1.0)  # Maks score ved 50% ROI
        energy_score = min(savings["yearly_kwh"] / 10000, 1.0)  # Maks ved 10000 kWh
        co2_score = min(savings["yearly_co2"] / 5000, 1.0)  # Maks ved 5000 kg CO2
        
        # Beregn vektet score
        return (
            roi_score * roi_weight +
            energy_score * energy_weight +
            co2_score * co2_weight
        )