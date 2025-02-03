from typing import Dict, Any, List
import logging
import aiohttp
from fastapi import HTTPException

class EnovaService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.enova_api_url = "https://api.enova.no"  # Eksempel URL
        
    async def analyze_support_options(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyser støttemuligheter fra Enova basert på eiendomsdata
        """
        try:
            energy_analysis = await self._perform_energy_analysis(property_data)
            eligible_measures = await self._find_eligible_measures(energy_analysis)
            support_calculation = await self._calculate_support(eligible_measures)
            
            return {
                "energy_analysis": energy_analysis,
                "eligible_measures": eligible_measures,
                "support_calculation": support_calculation,
                "total_support": sum(measure["support_amount"] 
                                   for measure in support_calculation)
            }
        
        except Exception as e:
            self.logger.error(f"Feil ved analyse av støttemuligheter: {str(e)}")
            raise HTTPException(status_code=500, 
                             detail="Kunne ikke analysere støttemuligheter")
    
    async def _perform_energy_analysis(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Utfør energianalyse av bygningen
        """
        try:
            current_consumption = self._estimate_current_consumption(property_data)
            improvement_potential = self._calculate_improvement_potential(property_data)
            
            return {
                "current_consumption": current_consumption,
                "improvement_potential": improvement_potential,
                "energy_label": self._calculate_energy_label(current_consumption),
                "co2_emissions": self._calculate_co2_emissions(current_consumption)
            }
        
        except Exception as e:
            self.logger.error(f"Feil ved energianalyse: {str(e)}")
            raise
    
    async def _find_eligible_measures(self, energy_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Finn støtteberettigede tiltak basert på energianalyse
        """
        measures = []
        
        # Varmepumpe
        if self._is_eligible_for_heat_pump(energy_analysis):
            measures.append({
                "type": "heat_pump",
                "description": "Installasjon av varmepumpe",
                "estimated_saving": 8000,  # kWh/år
                "estimated_cost": 120000,  # NOK
                "support_rate": 0.35  # 35% støtte
            })
        
        # Etterisolering
        if self._needs_insulation(energy_analysis):
            measures.append({
                "type": "insulation",
                "description": "Etterisolering av vegger og tak",
                "estimated_saving": 12000,  # kWh/år
                "estimated_cost": 200000,  # NOK
                "support_rate": 0.40  # 40% støtte
            })
        
        # Solceller
        if self._is_suitable_for_solar(energy_analysis):
            measures.append({
                "type": "solar_panels",
                "description": "Installasjon av solcellepanel",
                "estimated_saving": 5000,  # kWh/år
                "estimated_cost": 150000,  # NOK
                "support_rate": 0.35  # 35% støtte
            })
        
        return measures
    
    async def _calculate_support(self, eligible_measures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Beregn støttebeløp for hvert tiltak
        """
        return [{
            **measure,
            "support_amount": measure["estimated_cost"] * measure["support_rate"]
        } for measure in eligible_measures]
    
    def _estimate_current_consumption(self, property_data: Dict[str, Any]) -> float:
        """
        Estimer nåværende energiforbruk
        """
        # Implementer estimering av energiforbruk
        base_consumption = property_data.get("area", 0) * 200  # kWh/m²/år
        
        # Juster for byggeår
        year_factor = self._get_year_factor(property_data.get("construction_year", 1950))
        
        # Juster for bygningstype
        type_factor = self._get_building_type_factor(property_data.get("building_type", "house"))
        
        return base_consumption * year_factor * type_factor
    
    def _calculate_improvement_potential(self, property_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Beregn potensial for energiforbedring
        """
        current_consumption = self._estimate_current_consumption(property_data)
        
        return {
            "insulation": current_consumption * 0.30,  # 30% besparelse
            "windows": current_consumption * 0.15,     # 15% besparelse
            "heat_pump": current_consumption * 0.40,   # 40% besparelse
            "ventilation": current_consumption * 0.20,  # 20% besparelse
            "total": current_consumption * 0.60        # 60% total besparelse
        }
    
    def _calculate_energy_label(self, consumption: float) -> str:
        """
        Beregn energimerke basert på forbruk
        """
        # Implementer energimerkeberegning
        pass
    
    def _calculate_co2_emissions(self, consumption: float) -> float:
        """
        Beregn CO2-utslipp basert på energiforbruk
        """
        # Implementer CO2-beregning
        pass
    
    def _is_eligible_for_heat_pump(self, energy_analysis: Dict[str, Any]) -> bool:
        """
        Sjekk om bygningen er egnet for varmepumpe
        """
        # Implementer varmepumpesjekk
        pass
    
    def _needs_insulation(self, energy_analysis: Dict[str, Any]) -> bool:
        """
        Sjekk om bygningen trenger etterisolering
        """
        # Implementer isolasjonssjekk
        pass
    
    def _is_suitable_for_solar(self, energy_analysis: Dict[str, Any]) -> bool:
        """
        Sjekk om bygningen er egnet for solceller
        """
        # Implementer solcellesjekk
        pass
    
    def _get_year_factor(self, year: int) -> float:
        """
        Beregn faktor basert på byggeår
        """
        if year >= 2010:
            return 0.7
        elif year >= 1990:
            return 0.9
        elif year >= 1970:
            return 1.1
        else:
            return 1.3
    
    def _get_building_type_factor(self, building_type: str) -> float:
        """
        Beregn faktor basert på bygningstype
        """
        factors = {
            "house": 1.0,
            "apartment": 0.8,
            "cabin": 1.2,
            "office": 1.5
        }
        return factors.get(building_type, 1.0)