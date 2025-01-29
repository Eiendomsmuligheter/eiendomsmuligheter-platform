from typing import Dict, List
import aiohttp
import json
from datetime import datetime

class EnovaService:
    def __init__(self):
        self.base_url = "https://api.enova.no"  # Example URL
        self.session = None
        self.support_programs = self._load_support_programs()

    def _load_support_programs(self) -> Dict:
        """
        Load Enova support programs and criteria
        """
        return {
            "residential": {
                "energy_upgrade": {
                    "name": "Oppgradering av bolig",
                    "max_support": 150000,
                    "requirements": [
                        "Minimum to tiltak",
                        "Dokumentert energibesparelse",
                        "Profesjonell utførelse"
                    ],
                    "measures": {
                        "insulation": {
                            "walls": 400,  # NOK per m²
                            "roof": 250,   # NOK per m²
                            "floor": 300   # NOK per m²
                        },
                        "windows": 4000,   # NOK per window
                        "doors": 5000,     # NOK per door
                        "heating": {
                            "heat_pump_air": 25000,
                            "heat_pump_ground": 45000,
                            "solar": 35000
                        },
                        "ventilation": 25000
                    }
                },
                "heating_conversion": {
                    "name": "Konvertering til fornybar oppvarming",
                    "max_support": 100000,
                    "requirements": [
                        "Erstatte fossil oppvarming",
                        "Professjonell installasjon",
                        "Dokumentert effekt"
                    ]
                }
            }
        }

    async def _init_session(self):
        """
        Initialize aiohttp session
        """
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def analyze_energy_potential(
        self,
        property_info: Dict,
        development_potential: Dict
    ) -> Dict:
        """
        Analyze energy improvement potential and calculate Enova support
        """
        results = {
            "current_state": await self._analyze_current_state(property_info),
            "improvement_potential": await self._analyze_improvement_potential(
                property_info,
                development_potential
            ),
            "support_schemes": await self._find_applicable_support(
                property_info,
                development_potential
            ),
            "recommendations": await self._generate_recommendations(
                property_info,
                development_potential
            )
        }

        return results

    async def _analyze_current_state(self, property_info: Dict) -> Dict:
        """
        Analyze current energy state of the property
        """
        try:
            # Calculate current energy rating
            energy_consumption = self._calculate_energy_consumption(property_info)
            heating_source = property_info.get('heating_source', 'electric')
            
            return {
                "energy_rating": self._calculate_energy_rating(
                    energy_consumption,
                    property_info['area']
                ),
                "energy_consumption": energy_consumption,
                "heating_source": heating_source,
                "insulation_level": self._estimate_insulation_level(property_info),
                "window_quality": self._assess_window_quality(property_info),
                "ventilation_type": property_info.get('ventilation_type', 'natural')
            }
        except Exception as e:
            print(f"Error analyzing current state: {str(e)}")
            return {}

    async def _analyze_improvement_potential(
        self,
        property_info: Dict,
        development_potential: Dict
    ) -> Dict:
        """
        Analyze potential for energy improvements
        """
        try:
            current_state = await self._analyze_current_state(property_info)
            
            improvements = {
                "insulation": self._analyze_insulation_potential(
                    property_info,
                    current_state
                ),
                "windows": self._analyze_window_potential(
                    property_info,
                    current_state
                ),
                "heating": self._analyze_heating_potential(
                    property_info,
                    current_state
                ),
                "ventilation": self._analyze_ventilation_potential(
                    property_info,
                    current_state
                )
            }
            
            return {
                "improvements": improvements,
                "total_savings": self._calculate_total_savings(improvements),
                "new_energy_rating": self._calculate_new_energy_rating(
                    property_info,
                    improvements
                ),
                "co2_reduction": self._calculate_co2_reduction(improvements)
            }
        except Exception as e:
            print(f"Error analyzing improvement potential: {str(e)}")
            return {}

    def _calculate_energy_consumption(self, property_info: Dict) -> float:
        """
        Calculate current energy consumption based on property information
        """
        base_consumption = property_info.get('area', 0) * 200  # kWh/m² basis
        
        # Justeringsfaktorer
        age_factor = self._get_age_factor(property_info.get('year_built', 1900))
        insulation_factor = self._get_insulation_factor(property_info)
        heating_factor = self._get_heating_factor(
            property_info.get('heating_source', 'electric')
        )
        
        return base_consumption * age_factor * insulation_factor * heating_factor

    def _calculate_energy_rating(
        self,
        energy_consumption: float,
        area: float
    ) -> str:
        """
        Calculate energy rating (A-G) based on consumption per m²
        """
        consumption_per_m2 = energy_consumption / area
        
        if consumption_per_m2 <= 95:
            return 'A'
        elif consumption_per_m2 <= 120:
            return 'B'
        elif consumption_per_m2 <= 145:
            return 'C'
        elif consumption_per_m2 <= 175:
            return 'D'
        elif consumption_per_m2 <= 205:
            return 'E'
        elif consumption_per_m2 <= 250:
            return 'F'
        else:
            return 'G'

    def _analyze_insulation_potential(
        self,
        property_info: Dict,
        current_state: Dict
    ) -> Dict:
        """
        Analyze potential for improved insulation
        """
        potential = {
            "walls": self._analyze_wall_insulation(property_info),
            "roof": self._analyze_roof_insulation(property_info),
            "floor": self._analyze_floor_insulation(property_info),
            "savings": 0,
            "cost": 0,
            "support": 0
        }
        
        # Calculate savings and costs
        for area, details in potential.items():
            if isinstance(details, dict):
                potential["savings"] += details.get("savings", 0)
                potential["cost"] += details.get("cost", 0)
                potential["support"] += details.get("support", 0)
        
        return potential

    def _analyze_wall_insulation(self, property_info: Dict) -> Dict:
        """
        Analyze wall insulation potential
        """
        wall_area = self._calculate_wall_area(property_info)
        current_u_value = property_info.get('wall_u_value', 0.8)
        target_u_value = 0.18
        
        savings = self._calculate_insulation_savings(
            wall_area,
            current_u_value,
            target_u_value
        )
        
        cost = wall_area * 1200  # NOK per m²
        support = wall_area * self.support_programs['residential']['energy_upgrade']['measures']['insulation']['walls']
        
        return {
            "area": wall_area,
            "current_u_value": current_u_value,
            "target_u_value": target_u_value,
            "savings": savings,
            "cost": cost,
            "support": support
        }

    def _calculate_wall_area(self, property_info: Dict) -> float:
        """
        Calculate total wall area
        """
        height = property_info.get('height', 2.4)
        perimeter = property_info.get('perimeter', 0)
        window_area = property_info.get('window_area', 0)
        door_area = property_info.get('door_area', 0)
        
        return (height * perimeter) - window_area - door_area

    def _calculate_insulation_savings(
        self,
        area: float,
        current_u: float,
        target_u: float
    ) -> float:
        """
        Calculate annual energy savings from improved insulation
        """
        degree_days = 4000  # Oslo-klima
        hours_per_year = 24 * 365
        energy_price = 1.5  # NOK per kWh
        
        energy_saving = (
            area * (current_u - target_u) *
            degree_days * 24 / 1000  # kWh per year
        )
        
        return energy_saving * energy_price  # NOK per year

    async def calculate_support(self, property_data: Dict) -> Dict:
        """
        Calculate potential Enova support for energy measures
        """
        support = {
            "total_support": 0,
            "measures": [],
            "requirements": [],
            "next_steps": []
        }

        try:
            # Calculate support for each measure
            for measure in property_data.get('planned_measures', []):
                measure_support = self._calculate_measure_support(measure)
                if measure_support['eligible']:
                    support['total_support'] += measure_support['amount']
                    support['measures'].append(measure_support)

            # Add requirements and next steps
            support['requirements'] = self._get_support_requirements(
                property_data,
                support['measures']
            )
            support['next_steps'] = self._generate_next_steps(
                property_data,
                support['measures']
            )

        except Exception as e:
            print(f"Error calculating support: {str(e)}")

        return support

    def _calculate_measure_support(self, measure: Dict) -> Dict:
        """
        Calculate support for a specific measure
        """
        measure_type = measure.get('type')
        measure_details = measure.get('details', {})
        
        if measure_type in self.support_programs['residential']['energy_upgrade']['measures']:
            base_support = self.support_programs['residential']['energy_upgrade']['measures'][measure_type]
            
            if isinstance(base_support, dict):
                support_amount = self._calculate_detailed_support(
                    measure_type,
                    measure_details,
                    base_support
                )
            else:
                support_amount = base_support
            
            return {
                "type": measure_type,
                "eligible": True,
                "amount": support_amount,
                "requirements": self._get_measure_requirements(measure_type)
            }
        
        return {
            "type": measure_type,
            "eligible": False,
            "amount": 0,
            "requirements": []
        }

    def _calculate_detailed_support(
        self,
        measure_type: str,
        details: Dict,
        support_rates: Dict
    ) -> float:
        """
        Calculate detailed support amount for complex measures
        """
        total_support = 0
        
        if measure_type == "insulation":
            for area_type, area_size in details.items():
                if area_type in support_rates:
                    total_support += area_size * support_rates[area_type]
        
        elif measure_type == "heating":
            heating_type = details.get('type')
            if heating_type in support_rates:
                total_support = support_rates[heating_type]
        
        return min(
            total_support,
            self.support_programs['residential']['energy_upgrade']['max_support']
        )

    def _get_measure_requirements(self, measure_type: str) -> List[str]:
        """
        Get specific requirements for a measure
        """
        base_requirements = [
            "Tiltaket må utføres av fagfolk",
            "Dokumentasjon på utført arbeid må fremlegges",
            "Før- og etterbilder kreves"
        ]
        
        measure_specific = {
            "insulation": [
                "Minimum 10 cm isolasjon",
                "U-verdi må forbedres med minimum 30%"
            ],
            "windows": [
                "U-verdi må være 0.8 eller bedre",
                "Dokumentert U-verdi fra leverandør"
            ],
            "heating": [
                "Varmepumpe må være Enova-godkjent",
                "Årsvarmefaktor (SCOP) må dokumenteres"
            ]
        }
        
        return base_requirements + measure_specific.get(measure_type, [])