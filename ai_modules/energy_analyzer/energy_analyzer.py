from typing import Dict, List, Optional, Any, Union, TypedDict
import numpy as np
import logging
from datetime import datetime
import aiohttp
import json
import math
from functools import lru_cache

logger = logging.getLogger(__name__)

class EnergyPerformance(TypedDict):
    u_values: Dict[str, float]
    heat_loss: Dict[str, float]
    energy_demand: Dict[str, float]
    energy_rating: str

class EnovaAPI:
    """Håndterer kommunikasjon med Enova API for støtteberegninger"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.enova.no/v1"
        
    async def calculate_support(self, 
                               current_performance: Dict[str, Any], 
                               improvement_potential: Dict[str, Any]) -> Dict[str, Any]:
        """Beregner potensielle støttebeløp fra Enova basert på foreslåtte tiltak"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "currentPerformance": current_performance,
                    "improvementPotential": improvement_potential
                }
                
                async with session.post(
                    f"{self.base_url}/support/calculate",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        logger.error(f"Enova API feil: {response.status}")
                        return {
                            "success": False,
                            "error": f"API feil {response.status}"
                        }
                    
                    result = await response.json()
                    return {
                        "success": True,
                        "total_support": result.get("totalSupport", 0),
                        "measures": result.get("supportedMeasures", []),
                        "application_url": result.get("applicationUrl", "")
                    }
        except Exception as e:
            logger.error(f"Feil ved beregning av Enova-støtte: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

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
        self.enova_api = EnovaAPI(self.config.get("enova_api_key", ""))
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Last konfigurasjon"""
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"Kunne ikke laste konfigurasjon: {str(e)}")
        
        return {
            "energy_calculation_method": "NS3031",
            "default_climate_zone": "Oslo",
            "include_detailed_simulation": True,
            "enova_api_key": ""
        }
        
    async def analyze_energy_performance(self,
                                       building_data: Dict[str, Any],
                                       construction_details: Dict[str, Any]) -> Dict[str, Any]:
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
            
            # Samle analyse-data for anbefalingsgenerering
            analysis_data = {
                "current_performance": current_performance,
                "improvement_potential": improvement_potential,
                "potential_performance": potential_performance,
                "enova_support": enova_support
            }
            
            return {
                **analysis_data,
                "recommendations": self._generate_recommendations(analysis_data)
            }
            
        except Exception as e:
            logger.error(f"Feil ved energianalyse: {str(e)}")
            raise
            
    async def _calculate_energy_performance(self,
                                          building_data: Dict[str, Any],
                                          construction_details: Dict[str, Any]) -> EnergyPerformance:
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
    
    async def _calculate_potential_performance(self,
                                             current_performance: Dict[str, Any],
                                             improvement_potential: Dict[str, Any]) -> EnergyPerformance:
        """
        Beregn potensiell energiytelse etter implementering av anbefalte tiltak
        """
        # Kopier gjeldende ytelse som utgangspunkt
        potential = current_performance.copy()
        
        # Oppdater U-verdier basert på foreslåtte tiltak
        updated_u_values = current_performance["u_values"].copy()
        for category, measures in improvement_potential["measures"].items():
            if category == "insulation" or category == "windows":
                for measure in measures:
                    if "updated_u_value" in measure:
                        component = measure.get("component", "")
                        if component in updated_u_values:
                            updated_u_values[component] = measure["updated_u_value"]
        
        # Oppdater varmetap med nye U-verdier
        updated_heat_loss = self._calculate_heat_loss(
            # Bruker original bygningsdata
            {k: v for k, v in self._extract_building_data_from_heat_loss(current_performance).items()},
            updated_u_values
        )
        
        # Oppdater energibehov med nytt varmetap
        updated_energy_demand = self._calculate_energy_demand(
            updated_heat_loss,
            # Bruker original bygningsdata
            {k: v for k, v in self._extract_building_data_from_energy_demand(current_performance).items()}
        )
        
        # Beregn nytt energimerke
        updated_energy_rating = self._calculate_energy_rating(
            updated_energy_demand,
            {k: v for k, v in self._extract_building_data_from_energy_demand(current_performance).items()}
        )
        
        return {
            "u_values": updated_u_values,
            "heat_loss": updated_heat_loss,
            "energy_demand": updated_energy_demand,
            "energy_rating": updated_energy_rating
        }
    
    def _extract_building_data_from_heat_loss(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """Hjelpemetode for å ekstrahere bygningsdata fra ytelsesdata"""
        # Denne metoden ville normalt rekonstruere bygningsdata fra varmetapsberegninger
        # Forenklet implementasjon for demo
        return {
            "heated_area": 150.0,  # Example value
            "volume": 375.0,       # Example value
            "climate_zone": "Oslo"
        }
    
    def _extract_building_data_from_energy_demand(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """Hjelpemetode for å ekstrahere bygningsdata fra energibehovsberegninger"""
        # Denne metoden ville normalt rekonstruere bygningsdata fra energibehovsberegninger
        # Forenklet implementasjon for demo
        return {
            "heated_area": 150.0,   # Example value
            "occupants": 4,         # Example value
            "climate_zone": "Oslo"
        }
            
    async def _identify_improvements(self,
                                   current_performance: Dict[str, Any],
                                   construction_details: Dict[str, Any]) -> Dict[str, Any]:
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
    
    def _analyze_insulation_improvements(self, 
                                        current_performance: Dict[str, Any],
                                        construction_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyser forbedringspotensial for isolasjon"""
        improvements = []
        u_values = current_performance["u_values"]
        
        # Sjekk veggisolasjon
        if "walls" in u_values and u_values["walls"] > 0.22:
            improvements.append({
                "id": "wall_insulation",
                "component": "walls",
                "description": "Etterisolering av yttervegger",
                "current_u_value": u_values["walls"],
                "updated_u_value": 0.18,
                "energy_savings": {
                    "yearly_kwh": self._estimate_insulation_savings("walls", u_values["walls"], 0.18, construction_details),
                    "yearly_value": 0,  # Beregnes i neste steg
                    "yearly_co2": 0     # Beregnes i neste steg
                },
                "implementation_time": "2-4 uker",
                "technical_requirements": ["Fagperson kreves", "Byggesøknad kan være nødvendig"]
            })
        
        # Lignende sjekker for tak, gulv, etc.
        # ...
        
        # Beregn økonomisk verdi og CO2-besparelser
        for improvement in improvements:
            improvement["energy_savings"]["yearly_value"] = improvement["energy_savings"]["yearly_kwh"] * 1.5  # kr/kWh
            improvement["energy_savings"]["yearly_co2"] = improvement["energy_savings"]["yearly_kwh"] * 0.017  # kg CO2/kWh
            
        return improvements
    
    def _analyze_window_improvements(self, 
                                    current_performance: Dict[str, Any],
                                    construction_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyser forbedringspotensial for vinduer"""
        # Implementasjon ville være lignende som for isolasjon
        return []
    
    def _analyze_heating_system_improvements(self, 
                                           current_performance: Dict[str, Any],
                                           construction_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyser forbedringspotensial for varmesystem"""
        # Implementasjon ville vurdere oppgraderinger som varmepumpe, etc.
        return []
    
    def _analyze_ventilation_improvements(self, 
                                        current_performance: Dict[str, Any],
                                        construction_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyser forbedringspotensial for ventilasjon"""
        # Implementasjon ville vurdere balansert ventilasjon med varmegjenvinning, etc.
        return []
    
    def _estimate_insulation_savings(self, 
                                    component: str, 
                                    old_u: float, 
                                    new_u: float, 
                                    construction_details: Dict[str, Any]) -> float:
        """Estimer energibesparelse ved isolasjonsoppgradering"""
        # Forenklet beregning - i virkeligheten ville dette være mer komplekst
        if component == "walls":
            area = construction_details.get("wall_area", 0)
            if area == 0:
                # Fallback hvis areal ikke er definert
                area = 100  # m²
            
            # Forenklet beregning av varmetap-reduksjon
            delta_u = old_u - new_u
            degree_hours = 100000  # Typisk verdi for Oslo (°C·h)
            
            return delta_u * area * degree_hours
        
        return 0
        
    def _prioritize_improvements(self, improvements: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """
        Prioriter forbedringer basert på ROI, kostnader og energibesparelse
        """
        # Flat liste med alle forbedringer
        all_improvements = []
        for category, measures in improvements.items():
            for measure in measures:
                all_improvements.append({
                    "id": f"{category}_{measure['id']}",
                    "category": category,
                    "measure": measure,
                    "energy_savings": measure["energy_savings"]["yearly_kwh"],
                    # Kostnader estimeres i neste steg, bruker dummy-verdi her
                    "estimated_cost": 10000
                })
        
        # Sorter etter energibesparelse / kostnad (enkel ROI)
        all_improvements.sort(key=lambda x: x["energy_savings"] / x["estimated_cost"], reverse=True)
        
        return [improvement["id"] for improvement in all_improvements]
    
    def _estimate_improvement_costs(self, improvements: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
        """
        Estimer kostnader for hver forbedringsmulighet
        """
        cost_estimates = {}
        
        for category, measures in improvements.items():
            for measure in measures:
                measure_id = f"{category}_{measure['id']}"
                
                # Estimat basert på kategori og spesifikke faktorer
                if category == "insulation":
                    # Estimer basert på areal, etc.
                    base_cost = 1000  # Kr per m²
                    area = 100  # m² (dette burde hentes fra faktiske data)
                    total_cost = base_cost * area
                elif category == "windows":
                    # Estimer basert på antall vinduer, etc.
                    window_cost = 15000  # Kr per vindu
                    count = 8  # Antall vinduer (dette burde hentes fra faktiske data)
                    total_cost = window_cost * count
                elif category == "heating_system":
                    # Fast estimat basert på systemtype
                    total_cost = 80000
                elif category == "ventilation":
                    # Fast estimat basert på boligstørrelse
                    total_cost = 120000
                else:
                    total_cost = 50000  # Default fallback
                
                cost_estimates[measure_id] = {
                    "materials": total_cost * 0.6,
                    "labor": total_cost * 0.4,
                    "total": total_cost
                }
        
        return cost_estimates
        
    @lru_cache(maxsize=128)
    def _calculate_u_values(self, construction_details: Dict[str, Any]) -> Dict[str, float]:
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
    
    def _calculate_wall_u_value(self, details: Dict[str, Any]) -> float:
        """Beregn U-verdi for vegg"""
        # Forenklet implementasjon - i virkeligheten ville dette være mer komplekst
        if "u_value" in details:
            return float(details["u_value"])
        
        # Beregn basert på lagoppbygging
        if "layers" in details:
            r_total = 0.13  # Innvendig + utvendig overgangsmotstand
            for layer in details["layers"]:
                thickness = layer.get("thickness", 0) / 1000  # mm til m
                lambda_value = layer.get("lambda", 0.04)  # W/mK
                r_total += thickness / lambda_value
            
            return 1.0 / r_total if r_total > 0 else 1.0
        
        # Fallback basert på byggeår
        construction_year = details.get("construction_year", 1980)
        if construction_year < 1970:
            return 0.9
        elif construction_year < 1987:
            return 0.4
        elif construction_year < 1997:
            return 0.3
        elif construction_year < 2007:
            return 0.25
        elif construction_year < 2017:
            return 0.22
        else:
            return 0.18
    
    def _calculate_roof_u_value(self, details: Dict[str, Any]) -> float:
        """Beregn U-verdi for tak"""
        # Lignende implementasjon som for vegger
        return 0.18  # Forenklet returnering av standardverdi
    
    def _calculate_floor_u_value(self, details: Dict[str, Any]) -> float:
        """Beregn U-verdi for gulv"""
        # Lignende implementasjon som for vegger
        return 0.18  # Forenklet returnering av standardverdi
    
    def _calculate_window_u_value(self, details: Dict[str, Any]) -> float:
        """Beregn U-verdi for vinduer"""
        # Lignende implementasjon som for vegger
        return 1.2  # Forenklet returnering av standardverdi
        
    def _calculate_heat_loss(self,
                           building_data: Dict[str, Any],
                           u_values: Dict[str, float]) -> Dict[str, float]:
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
            "total": transmission_loss + infiltration_loss + ventilation_loss
        }
    
    def _calculate_transmission_loss(self, 
                                   building_data: Dict[str, Any], 
                                   u_values: Dict[str, float]) -> float:
        """Beregn transmisjonstap"""
        # Forenklet implementasjon for demo
        return 10000.0  # W
    
    def _calculate_infiltration_loss(self, 
                                   building_data: Dict[str, Any]) -> float:
        """Beregn infiltrasjonstap"""
        # Forenklet implementasjon for demo
        return 3000.0  # W
    
    def _calculate_ventilation_loss(self, 
                                  building_data: Dict[str, Any]) -> float:
        """Beregn ventilasjonstap"""
        # Forenklet implementasjon for demo
        return 5000.0  # W
        
    def _calculate_energy_demand(self,
                               heat_loss: Dict[str, float],
                               building_data: Dict[str, Any]) -> Dict[str, float]:
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
            "total": heating_demand + cooling_demand + hot_water_demand + electricity_demand
        }
    
    def _calculate_heating_demand(self, 
                                heat_loss: Dict[str, float], 
                                building_data: Dict[str, Any]) -> float:
        """Beregn oppvarmingsbehov"""
        # Forenklet implementasjon for demo
        return 15000.0  # kWh/år
    
    def _calculate_cooling_demand(self, 
                                heat_loss: Dict[str, float], 
                                building_data: Dict[str, Any]) -> float:
        """Beregn kjølebehov"""
        # Forenklet implementasjon for demo
        return 1000.0  # kWh/år
    
    def _calculate_hot_water_demand(self, 
                                  building_data: Dict[str, Any]) -> float:
        """Beregn varmtvannsbehov"""
        # Forenklet implementasjon for demo
        return 4000.0  # kWh/år
    
    def _calculate_electricity_demand(self, 
                                    building_data: Dict[str, Any]) -> float:
        """Beregn elektrisitetsbehov"""
        # Forenklet implementasjon for demo
        return 5000.0  # kWh/år
        
    def _calculate_energy_rating(self,
                               energy_demand: Dict[str, float],
                               building_data: Dict[str, Any]) -> str:
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
        
    def _generate_recommendations(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generer prioriterte anbefalinger for energiforbedringer
        """
        recommendations = []
        
        # Analyser forbedringspotensial
        improvements = analysis_data["improvement_potential"]["measures"]
        costs = analysis_data["improvement_potential"]["cost_estimates"]
        
        for category, measures in improvements.items():
            for measure in measures:
                measure_id = measure["id"]
                cost_key = f"{category}_{measure_id}"
                
                if cost_key in costs:
                    cost = costs[cost_key]
                    savings = measure["energy_savings"]
                    roi = savings["yearly_value"] / cost["total"] if cost["total"] > 0 else 0
                    
                    recommendations.append({
                        "category": category,
                        "measure": measure["description"],
                        "cost": cost,
                        "savings": savings,
                        "roi": roi,
                        "priority": self._calculate_priority(roi, savings),
                        "enova_support": self._get_enova_support_for_measure(measure_id, analysis_data),
                        "implementation_time": measure.get("implementation_time", "N/A"),
                        "technical_requirements": measure.get("technical_requirements", [])
                    })
        
        # Sorter anbefalinger etter prioritet
        recommendations.sort(key=lambda x: x["priority"], reverse=True)
        
        return recommendations
    
    def _get_enova_support_for_measure(self, measure_id: str, analysis_data: Dict[str, Any]) -> float:
        """Hent ut Enova-støtte for et spesifikt tiltak"""
        if "enova_support" in analysis_data and "measures" in analysis_data["enova_support"]:
            for measure in analysis_data["enova_support"]["measures"]:
                if measure.get("id") == measure_id:
                    return measure.get("support_amount", 0)
        return 0
        
    def _calculate_priority(self, roi: float, savings: Dict[str, float]) -> float:
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
