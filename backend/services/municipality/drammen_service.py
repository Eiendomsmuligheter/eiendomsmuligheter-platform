"""
Drammen Municipality Service - Handles all interaction with Drammen kommune's systems.
"""
from typing import Dict, Any, List, Optional
import aiohttp
import logging
from datetime import datetime

class DrammenMunicipalityService:
    """Service for interacting with Drammen kommune's systems."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://innsyn2020.drammen.kommune.no/"
        self.case_archive_url = f"{self.base_url}/postjournal-v2/"
        self.map_service_url = "https://kart.drammen.kommune.no/"
    
    async def get_property_cases(self, gnr: str, bnr: str) -> List[Dict[str, Any]]:
        """
        Fetches all cases related to a property from before 2020.
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Construct the search URL
                search_url = f"{self.case_archive_url}/fb851964-3185-43eb-81ba-9ac75226dfa8"
                params = {
                    "gnr": gnr,
                    "bnr": bnr,
                    "type": "advanced_search"
                }
                
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        cases = await response.json()
                        return self._process_cases(cases)
                    else:
                        self.logger.error(f"Failed to fetch cases: {response.status}")
                        return []
        
        except Exception as e:
            self.logger.error(f"Error fetching property cases: {str(e)}")
            raise
    
    async def get_zoning_plan(self, gnr: str, bnr: str) -> Dict[str, Any]:
        """
        Fetches current zoning plan for the property.
        """
        try:
            # First check if property has specific regulation plan
            regulation_plan = await self._get_regulation_plan(gnr, bnr)
            
            if regulation_plan:
                return regulation_plan
            
            # If no specific regulation plan, get municipal plan
            return await self._get_municipal_plan(gnr, bnr)
            
        except Exception as e:
            self.logger.error(f"Error fetching zoning plan: {str(e)}")
            raise
    
    async def get_building_restrictions(self, gnr: str, bnr: str) -> Dict[str, Any]:
        """
        Gets all building restrictions for the property.
        """
        try:
            restrictions = {
                "max_utilization": await self._get_max_utilization(gnr, bnr),
                "building_height": await self._get_height_restrictions(gnr, bnr),
                "distance_requirements": await self._get_distance_requirements(gnr, bnr),
                "special_considerations": await self._get_special_considerations(gnr, bnr)
            }
            
            return restrictions
            
        except Exception as e:
            self.logger.error(f"Error fetching building restrictions: {str(e)}")
            raise
    
    async def calculate_development_potential(self, 
                                           property_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates development potential based on regulations and property data.
        """
        try:
            # Get current BYA
            current_bya = self._calculate_bya(
                built_area=property_data['built_area'],
                parking_area=property_data['parking_area']
            )
            
            # Get maximum allowed BYA
            max_bya = await self._get_max_utilization(
                property_data['gnr'],
                property_data['bnr']
            )
            
            # Calculate remaining potential
            remaining_bya = max_bya - current_bya
            
            return {
                "current_bya": current_bya,
                "max_bya": max_bya,
                "remaining_bya": remaining_bya,
                "can_split_plot": await self._can_split_plot(property_data),
                "potential_developments": await self._get_potential_developments(
                    property_data,
                    remaining_bya
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating development potential: {str(e)}")
            raise
    
    async def get_building_code_requirements(self, 
                                          construction_year: int) -> Dict[str, Any]:
        """
        Determines which building code (TEK10/TEK17) applies and returns requirements.
        """
        try:
            if construction_year >= 2017:
                code_version = "TEK17"
                requirements = await self._get_tek17_requirements()
            else:
                code_version = "TEK10"
                requirements = await self._get_tek10_requirements()
            
            return {
                "code_version": code_version,
                "requirements": requirements,
                "applies_to_changes": True,
                "exemptions": await self._get_code_exemptions(construction_year)
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching building code requirements: {str(e)}")
            raise
    
    def _calculate_bya(self, built_area: float, parking_area: float) -> float:
        """
        Calculates BYA (Bebygd areal) according to Drammen municipality's rules.
        
        %-BYA = (Bebygd areal / Tomtearealet) x 100
        """
        return built_area + parking_area
    
    async def _get_regulation_plan(self, gnr: str, bnr: str) -> Optional[Dict[str, Any]]:
        """Fetches specific regulation plan if it exists."""
        # Implementation for fetching regulation plan
        pass
    
    async def _get_municipal_plan(self, gnr: str, bnr: str) -> Dict[str, Any]:
        """Fetches municipal plan details."""
        # Implementation for fetching municipal plan
        pass
    
    async def _get_max_utilization(self, gnr: str, bnr: str) -> float:
        """Gets maximum allowed utilization for the property."""
        # Implementation for getting max utilization
        pass
    
    async def _get_height_restrictions(self, gnr: str, bnr: str) -> Dict[str, float]:
        """Gets height restrictions for the property."""
        # Implementation for getting height restrictions
        pass
    
    async def _get_distance_requirements(self, gnr: str, bnr: str) -> Dict[str, float]:
        """Gets distance requirements for the property."""
        # Implementation for getting distance requirements
        pass
    
    async def _get_special_considerations(self, gnr: str, bnr: str) -> List[str]:
        """Gets any special considerations for the property."""
        # Implementation for getting special considerations
        pass
    
    async def _can_split_plot(self, property_data: Dict[str, Any]) -> bool:
        """Determines if a plot can be split based on regulations and size."""
        # Implementation for determining if plot can be split
        pass
    
    async def _get_potential_developments(self, 
                                       property_data: Dict[str, Any],
                                       remaining_bya: float) -> List[Dict[str, Any]]:
        """Gets list of potential developments for the property."""
        # Implementation for getting potential developments
        pass
    
    def _process_cases(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes and formats case data."""
        # Implementation for processing cases
        pass
    
    async def _get_tek17_requirements(self) -> Dict[str, Any]:
        """Gets TEK17 requirements."""
        # Implementation for getting TEK17 requirements
        pass
    
    async def _get_tek10_requirements(self) -> Dict[str, Any]:
        """Gets TEK10 requirements."""
        # Implementation for getting TEK10 requirements
        pass
    
    async def _get_code_exemptions(self, construction_year: int) -> List[str]:
        """Gets applicable building code exemptions."""
        # Implementation for getting code exemptions
        pass