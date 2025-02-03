"""
Basement Analysis Service - Analyzes basement conversion potential.
"""
from typing import Dict, Any
import logging
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class BasementRequirements:
    """Requirements for basement conversion."""
    min_ceiling_height: float = 2.2  # meters
    min_window_size: float = 0.6  # m²
    min_window_width: float = 0.5  # meters
    max_window_sill_height: float = 1.5  # meters from floor
    min_room_size: float = 6.0  # m²
    min_living_area: float = 15.0  # m²
    min_bathroom_size: float = 4.0  # m²
    min_kitchen_size: float = 6.0  # m²
    required_ventilation: str = "balanced"
    required_insulation: Dict[str, float] = {
        "walls": 0.22,  # U-value W/(m²·K)
        "floor": 0.18,  # U-value W/(m²·K)
        "windows": 1.2   # U-value W/(m²·K)
    }

class BasementAnalyzer:
    """Service for analyzing basement conversion potential."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.requirements = BasementRequirements()
    
    async def analyze_basement_potential(self,
                                      property_data: Dict[str, Any],
                                      regulations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the potential for converting a basement to a rental unit.
        """
        try:
            # Check if basement exists and get basic data
            if not property_data.get('has_basement', False):
                return self._create_negative_result("Ingen kjeller funnet")
            
            basement_data = property_data['basement_data']
            
            # Check ceiling height
            if basement_data['ceiling_height'] < self.requirements.min_ceiling_height:
                return self._create_negative_result(
                    f"Takhøyde på {basement_data['ceiling_height']}m er under minimumskrav på {self.requirements.min_ceiling_height}m"
                )
            
            # Check windows
            window_assessment = self._assess_windows(basement_data['windows'])
            if not window_assessment['meets_requirements']:
                return self._create_negative_result(window_assessment['message'])
            
            # Check moisture conditions
            moisture_assessment = self._assess_moisture(basement_data['moisture_measurements'])
            if not moisture_assessment['is_acceptable']:
                return self._create_negative_result(moisture_assessment['message'])
            
            # Check terrain and drainage
            terrain_assessment = self._assess_terrain(property_data['terrain_data'])
            if not terrain_assessment['is_suitable']:
                return self._create_negative_result(terrain_assessment['message'])
            
            # If all basic requirements are met, calculate potential
            conversion_potential = self._calculate_conversion_potential(
                basement_data,
                property_data,
                regulations
            )
            
            # Calculate costs and value
            economics = self._calculate_economics(conversion_potential, property_data['location'])
            
            return {
                'is_possible': True,
                'area': conversion_potential['usable_area'],
                'potential_rooms': conversion_potential['potential_rooms'],
                'cost': economics['estimated_cost'],
                'value': economics['estimated_value'],
                'roi': economics['roi'],
                'requirements': self._compile_requirements(conversion_potential),
                'recommendations': self._compile_recommendations(conversion_potential)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing basement potential: {str(e)}")
            return self._create_negative_result("Feil under analyse av kjeller")
    
    def _assess_windows(self, windows_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess if windows meet requirements for living space."""
        try:
            compliant_windows = []
            non_compliant_windows = []
            
            for window in windows_data:
                area = window['width'] * window['height']
                if (area >= self.requirements.min_window_size and
                    window['width'] >= self.requirements.min_window_width and
                    window['sill_height'] <= self.requirements.max_window_sill_height):
                    compliant_windows.append(window)
                else:
                    non_compliant_windows.append(window)
            
            if not compliant_windows:
                return {
                    'meets_requirements': False,
                    'message': "Ingen vinduer oppfyller kravene til rom for varig opphold"
                }
            
            return {
                'meets_requirements': True,
                'compliant_windows': compliant_windows,
                'non_compliant_windows': non_compliant_windows,
                'message': "Tilstrekkelige vinduer for beboelse"
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing windows: {str(e)}")
            raise
    
    def _assess_moisture(self, moisture_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess moisture conditions in basement."""
        try:
            # Check relative humidity
            if moisture_data['relative_humidity'] > 70:
                return {
                    'is_acceptable': False,
                    'message': f"For høy relativ fuktighet: {moisture_data['relative_humidity']}%"
                }
            
            # Check wall moisture content
            if moisture_data['wall_moisture'] > 15:
                return {
                    'is_acceptable': False,
                    'message': f"For høyt fuktnivå i vegger: {moisture_data['wall_moisture']}%"
                }
            
            # Check floor moisture content
            if moisture_data['floor_moisture'] > 15:
                return {
                    'is_acceptable': False,
                    'message': f"For høyt fuktnivå i gulv: {moisture_data['floor_moisture']}%"
                }
            
            return {
                'is_acceptable': True,
                'measurements': moisture_data,
                'message': "Akseptable fuktnivåer"
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing moisture: {str(e)}")
            raise
    
    def _assess_terrain(self, terrain_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess terrain conditions around basement."""
        try:
            problems = []
            
            # Check ground slope away from building
            if terrain_data['slope'] < 1/50:  # Minimum 1:50 fall away from building
                problems.append("Utilstrekkelig fall fra bygning")
            
            # Check drainage conditions
            if not terrain_data['has_functioning_drainage']:
                problems.append("Manglende eller utilstrekkelig drenering")
            
            # Check soil type
            if terrain_data['soil_type'] in ['clay', 'silt']:
                problems.append("Ugunstig grunnforhold (leire/silt)")
            
            if problems:
                return {
                    'is_suitable': False,
                    'message': ". ".join(problems)
                }
            
            return {
                'is_suitable': True,
                'message': "Gunstige terrengforhold"
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing terrain: {str(e)}")
            raise
    
    def _calculate_conversion_potential(self,
                                     basement_data: Dict[str, Any],
                                     property_data: Dict[str, Any],
                                     regulations: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the potential for basement conversion."""
        try:
            # Calculate usable area
            gross_area = basement_data['area']
            deductions = sum([
                basement_data.get('technical_room_area', 0),
                basement_data.get('storage_requirement', 0),
                basement_data.get('access_area', 0)
            ])
            usable_area = gross_area - deductions
            
            # Calculate potential room layout
            room_layout = self._calculate_room_layout(usable_area)
            
            # Calculate technical requirements
            technical_requirements = {
                'ventilation': self._calculate_ventilation_requirements(usable_area),
                'insulation': self._calculate_insulation_requirements(basement_data),
                'electrical': self._calculate_electrical_requirements(room_layout),
                'plumbing': self._calculate_plumbing_requirements(room_layout)
            }
            
            return {
                'usable_area': usable_area,
                'potential_rooms': room_layout,
                'technical_requirements': technical_requirements,
                'parking_impact': self._assess_parking_impact(property_data, regulations)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating conversion potential: {str(e)}")
            raise
    
    def _calculate_economics(self,
                           conversion_potential: Dict[str, Any],
                           location_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate costs and potential value of conversion."""
        try:
            # Calculate costs
            area = conversion_potential['usable_area']
            base_cost_per_m2 = Decimal('15000')  # NOK per m²
            
            costs = {
                'construction': area * base_cost_per_m2,
                'ventilation': self._calculate_ventilation_cost(conversion_potential),
                'electrical': self._calculate_electrical_cost(conversion_potential),
                'plumbing': self._calculate_plumbing_cost(conversion_potential),
                'windows': self._calculate_windows_cost(conversion_potential),
                'planning': Decimal('50000')  # Fixed cost for planning and permits
            }
            
            total_cost = sum(costs.values())
            
            # Calculate potential rental income
            monthly_rent = self._estimate_rental_income(
                area,
                location_data,
                conversion_potential['potential_rooms']
            )
            
            # Calculate value increase
            value_increase = self._calculate_value_increase(
                area,
                location_data,
                monthly_rent
            )
            
            return {
                'estimated_cost': total_cost,
                'estimated_value': value_increase,
                'monthly_rent': monthly_rent,
                'roi': (value_increase - total_cost) / total_cost,
                'detailed_costs': costs
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating economics: {str(e)}")
            raise
    
    def _create_negative_result(self, message: str) -> Dict[str, Any]:
        """Create a standardized negative result."""
        return {
            'is_possible': False,
            'area': 0,
            'cost': 0,
            'value': 0,
            'roi': 0,
            'message': message,
            'requirements': [],
            'recommendations': []
        }
    
    def _compile_requirements(self, conversion_potential: Dict[str, Any]) -> list:
        """Compile list of requirements for conversion."""
        requirements = []
        
        # Building requirements
        requirements.extend([
            f"Minimum takhøyde: {self.requirements.min_ceiling_height}m",
            f"Minimum vindusareal: {self.requirements.min_window_size}m²",
            f"Maksimal brystningshøyde: {self.requirements.max_window_sill_height}m"
        ])
        
        # Technical requirements
        tech_reqs = conversion_potential['technical_requirements']
        requirements.extend([
            f"Ventilasjon: {tech_reqs['ventilation']}",
            f"Isolasjon vegger: U-verdi {self.requirements.required_insulation['walls']}",
            f"Isolasjon gulv: U-verdi {self.requirements.required_insulation['floor']}",
            f"Vinduer: U-verdi {self.requirements.required_insulation['windows']}"
        ])
        
        # Room requirements
        requirements.extend([
            f"Minimum størrelse oppholdsrom: {self.requirements.min_room_size}m²",
            f"Minimum størrelse bad: {self.requirements.min_bathroom_size}m²",
            f"Minimum størrelse kjøkken: {self.requirements.min_kitchen_size}m²"
        ])
        
        return requirements
    
    def _compile_recommendations(self, conversion_potential: Dict[str, Any]) -> list:
        """Compile list of recommendations for optimal conversion."""
        recommendations = []
        
        # Window recommendations
        if conversion_potential['usable_area'] > 30:
            recommendations.append(
                "Vurder å installere flere vinduer for bedre dagslys"
            )
        
        # Moisture control
        recommendations.append(
            "Installer fuktsperre og god drenering rundt grunnmur"
        )
        
        # Ventilation
        recommendations.append(
            "Installer balansert ventilasjon med varmegjenvinning"
        )
        
        # Insulation
        recommendations.append(
            "Vurder innvendig etterisolering med dampsperre"
        )
        
        # Sound insulation
        recommendations.append(
            "Installer trinnlyddemping i etasjeskillet"
        )
        
        return recommendations