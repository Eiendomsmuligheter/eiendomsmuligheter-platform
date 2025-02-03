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
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.map_service_url}/api/reguleringsplan"
                params = {
                    "gnr": gnr,
                    "bnr": bnr,
                    "kommune": "3005"  # Drammen kommune
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        plan_data = await response.json()
                        
                        if not plan_data:
                            return None
                            
                        return {
                            "plan_id": plan_data["planident"],
                            "plan_navn": plan_data["plannavn"],
                            "vedtaksdato": plan_data["vedtaksdato"],
                            "arealformål": plan_data["arealformål"],
                            "utnyttingsgrad": plan_data["utnyttingsgrad"],
                            "høydebestemmelser": plan_data["høydebestemmelser"],
                            "byggegrenser": plan_data["byggegrenser"],
                            "spesielle_bestemmelser": plan_data["spesielle_bestemmelser"]
                        }
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error fetching regulation plan: {str(e)}")
            return None
    
    async def _get_municipal_plan(self, gnr: str, bnr: str) -> Dict[str, Any]:
        """Fetches municipal plan details."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.map_service_url}/api/kommuneplan"
                params = {
                    "gnr": gnr,
                    "bnr": bnr,
                    "kommune": "3005"  # Drammen kommune
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        plan_data = await response.json()
                        
                        return {
                            "område_type": plan_data["område_type"],
                            "arealformål": plan_data["arealformål"],
                            "utnyttingsgrad": plan_data["utnyttingsgrad"],
                            "etasjer_maks": plan_data["etasjer_maks"],
                            "høyde_maks": plan_data["høyde_maks"],
                            "spesielle_hensyn": plan_data["spesielle_hensyn"]
                        }
                        
                    raise Exception(f"Failed to fetch municipal plan: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Error fetching municipal plan: {str(e)}")
            raise
    
    async def _get_max_utilization(self, gnr: str, bnr: str) -> float:
        """Gets maximum allowed utilization for the property."""
        try:
            # First check regulation plan
            regulation_plan = await self._get_regulation_plan(gnr, bnr)
            if regulation_plan and "utnyttingsgrad" in regulation_plan:
                return float(regulation_plan["utnyttingsgrad"].replace("%", ""))
            
            # If no regulation plan, check municipal plan
            municipal_plan = await self._get_municipal_plan(gnr, bnr)
            return float(municipal_plan["utnyttingsgrad"].replace("%", ""))
            
        except Exception as e:
            self.logger.error(f"Error getting max utilization: {str(e)}")
            raise
    
    async def _get_height_restrictions(self, gnr: str, bnr: str) -> Dict[str, float]:
        """Gets height restrictions for the property."""
        try:
            # First check regulation plan
            regulation_plan = await self._get_regulation_plan(gnr, bnr)
            if regulation_plan and "høydebestemmelser" in regulation_plan:
                return {
                    "mønehøyde_maks": regulation_plan["høydebestemmelser"]["mønehøyde"],
                    "gesimshøyde_maks": regulation_plan["høydebestemmelser"]["gesimshøyde"],
                    "etasjer_maks": regulation_plan["høydebestemmelser"]["etasjer"]
                }
            
            # If no regulation plan, check municipal plan
            municipal_plan = await self._get_municipal_plan(gnr, bnr)
            return {
                "mønehøyde_maks": municipal_plan["høyde_maks"]["mønehøyde"],
                "gesimshøyde_maks": municipal_plan["høyde_maks"]["gesimshøyde"],
                "etasjer_maks": municipal_plan["etasjer_maks"]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting height restrictions: {str(e)}")
            raise
    
    async def _get_distance_requirements(self, gnr: str, bnr: str) -> Dict[str, float]:
        """Gets distance requirements for the property."""
        try:
            # First check regulation plan
            regulation_plan = await self._get_regulation_plan(gnr, bnr)
            if regulation_plan and "byggegrenser" in regulation_plan:
                return {
                    "nabogrense": regulation_plan["byggegrenser"]["nabogrense"],
                    "vei_senterlinje": regulation_plan["byggegrenser"]["vei_senterlinje"],
                    "jernbane": regulation_plan["byggegrenser"].get("jernbane", 30.0),
                    "vassdrag": regulation_plan["byggegrenser"].get("vassdrag", 20.0)
                }
            
            # If no regulation plan, use standard distances from municipal plan
            return {
                "nabogrense": 4.0,  # Standard fra PBL
                "vei_senterlinje": 15.0,  # Standard for kommunal vei
                "jernbane": 30.0,  # Standard for jernbane
                "vassdrag": 20.0  # Standard for vassdrag
            }
            
        except Exception as e:
            self.logger.error(f"Error getting distance requirements: {str(e)}")
            raise
    
    async def _get_special_considerations(self, gnr: str, bnr: str) -> List[str]:
        """Gets any special considerations for the property."""
        try:
            considerations = []
            
            # Check regulation plan
            regulation_plan = await self._get_regulation_plan(gnr, bnr)
            if regulation_plan and "spesielle_bestemmelser" in regulation_plan:
                considerations.extend(regulation_plan["spesielle_bestemmelser"])
            
            # Check municipal plan
            municipal_plan = await self._get_municipal_plan(gnr, bnr)
            if "spesielle_hensyn" in municipal_plan:
                considerations.extend(municipal_plan["spesielle_hensyn"])
            
            # Add standard considerations if applicable
            if await self._is_in_cultural_heritage_area(gnr, bnr):
                considerations.append("Eiendommen ligger i kulturhistorisk område")
            
            if await self._is_in_flood_zone(gnr, bnr):
                considerations.append("Eiendommen ligger i flomutsatt område")
            
            if await self._has_contaminated_ground(gnr, bnr):
                considerations.append("Det er registrert forurenset grunn på eiendommen")
            
            return considerations
            
        except Exception as e:
            self.logger.error(f"Error getting special considerations: {str(e)}")
            raise
    
    async def _is_in_cultural_heritage_area(self, gnr: str, bnr: str) -> bool:
        """Check if property is in cultural heritage area."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.map_service_url}/api/kulturminner"
                params = {
                    "gnr": gnr,
                    "bnr": bnr,
                    "kommune": "3005"
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("in_cultural_zone", False)
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error checking cultural heritage area: {str(e)}")
            return False
    
    async def _is_in_flood_zone(self, gnr: str, bnr: str) -> bool:
        """Check if property is in flood zone."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.map_service_url}/api/flomfare"
                params = {
                    "gnr": gnr,
                    "bnr": bnr,
                    "kommune": "3005"
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("in_flood_zone", False)
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error checking flood zone: {str(e)}")
            return False
    
    async def _has_contaminated_ground(self, gnr: str, bnr: str) -> bool:
        """Check if property has registered contaminated ground."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.map_service_url}/api/forurensning"
                params = {
                    "gnr": gnr,
                    "bnr": bnr,
                    "kommune": "3005"
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("has_contamination", False)
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error checking contaminated ground: {str(e)}")
            return False
    
    async def _can_split_plot(self, property_data: Dict[str, Any]) -> bool:
        """Determines if a plot can be split based on regulations and size."""
        try:
            # Get regulation data
            regulation_plan = await self._get_regulation_plan(
                property_data['gnr'],
                property_data['bnr']
            )
            
            # Get municipal plan
            municipal_plan = await self._get_municipal_plan(
                property_data['gnr'],
                property_data['bnr']
            )
            
            # Check plot size
            min_plot_size = 600  # Standard minimum plot size in Drammen
            if regulation_plan and "min_tomtestørrelse" in regulation_plan:
                min_plot_size = regulation_plan["min_tomtestørrelse"]
            elif municipal_plan and "min_tomtestørrelse" in municipal_plan:
                min_plot_size = municipal_plan["min_tomtestørrelse"]
            
            # Calculate potential new plot sizes
            total_area = property_data['plot_size']
            potential_plot_1 = total_area * 0.5
            potential_plot_2 = total_area * 0.5
            
            # Check if both potential plots meet minimum size
            if potential_plot_1 < min_plot_size or potential_plot_2 < min_plot_size:
                return False
            
            # Check existing building placement
            existing_building_area = property_data['building_footprint']
            if existing_building_area / total_area > 0.3:  # Max 30% coverage
                return False
            
            # Check access requirements
            has_road_access = self._check_road_access(property_data)
            if not has_road_access:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking plot split possibility: {str(e)}")
            return False
    
    def _check_road_access(self, property_data: Dict[str, Any]) -> bool:
        """Check if property has sufficient road access for splitting."""
        try:
            # Check road width
            if property_data.get('road_width', 0) < 4.0:  # Minimum 4m road width
                return False
            
            # Check if there's space for necessary infrastructure
            if property_data.get('distance_to_road', 0) > 50.0:  # Max 50m to public road
                return False
            
            # Check if terrain allows for road construction
            if property_data.get('max_slope', 0) > 12.5:  # Max 1:8 slope for road
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking road access: {str(e)}")
            return False
    
    async def _get_potential_developments(self, 
                                       property_data: Dict[str, Any],
                                       remaining_bya: float) -> List[Dict[str, Any]]:
        """Gets list of potential developments for the property."""
        developments = []
        
        try:
            # Get regulations
            regulations = await self.get_building_restrictions(
                property_data['gnr'],
                property_data['bnr']
            )
            
            # Check basement conversion potential
            if property_data.get('has_basement', False):
                basement_potential = await self._analyze_basement_potential(
                    property_data,
                    regulations
                )
                if basement_potential['is_possible']:
                    developments.append({
                        'type': 'basement_conversion',
                        'description': 'Konvertering av kjeller til utleieenhet',
                        'potential_area': basement_potential['area'],
                        'estimated_cost': basement_potential['cost'],
                        'estimated_value': basement_potential['value'],
                        'requirements': basement_potential['requirements']
                    })
            
            # Check attic conversion potential
            if property_data.get('has_attic', False):
                attic_potential = await self._analyze_attic_potential(
                    property_data,
                    regulations
                )
                if attic_potential['is_possible']:
                    developments.append({
                        'type': 'attic_conversion',
                        'description': 'Utbygging av loft',
                        'potential_area': attic_potential['area'],
                        'estimated_cost': attic_potential['cost'],
                        'estimated_value': attic_potential['value'],
                        'requirements': attic_potential['requirements']
                    })
            
            # Check extension potential
            if remaining_bya > 20:  # Minimum 20m² for meaningful extension
                extension_potential = await self._analyze_extension_potential(
                    property_data,
                    regulations,
                    remaining_bya
                )
                if extension_potential['is_possible']:
                    developments.append({
                        'type': 'extension',
                        'description': 'Tilbygg',
                        'potential_area': extension_potential['area'],
                        'estimated_cost': extension_potential['cost'],
                        'estimated_value': extension_potential['value'],
                        'requirements': extension_potential['requirements']
                    })
            
            # Check plot division potential
            if await self._can_split_plot(property_data):
                plot_division = await self._analyze_plot_division(
                    property_data,
                    regulations
                )
                developments.append({
                    'type': 'plot_division',
                    'description': 'Tomtedeling',
                    'potential_plots': plot_division['potential_plots'],
                    'estimated_cost': plot_division['cost'],
                    'estimated_value': plot_division['value'],
                    'requirements': plot_division['requirements']
                })
            
            # Check garage/carport potential
            if property_data.get('parking_need', 0) > property_data.get('parking_spaces', 0):
                garage_potential = await self._analyze_garage_potential(
                    property_data,
                    regulations,
                    remaining_bya
                )
                if garage_potential['is_possible']:
                    developments.append({
                        'type': 'garage',
                        'description': 'Garasje/carport',
                        'potential_area': garage_potential['area'],
                        'estimated_cost': garage_potential['cost'],
                        'estimated_value': garage_potential['value'],
                        'requirements': garage_potential['requirements']
                    })
            
            # Sort developments by estimated ROI
            return sorted(
                developments,
                key=lambda x: (x.get('estimated_value', 0) - x.get('estimated_cost', 0)) / x.get('estimated_cost', 1),
                reverse=True
            )
            
        except Exception as e:
            self.logger.error(f"Error getting potential developments: {str(e)}")
            return []
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