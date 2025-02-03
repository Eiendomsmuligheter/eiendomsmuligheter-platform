"""
PropertyAnalyzer - Hovedmodul for eiendomsanalyse
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

@dataclass
class PropertyDetails:
    address: str
    gnr: int
    bnr: int
    municipality: str
    property_type: str
    total_area: float
    building_area: float
    floors: int
    basement: bool
    attic: bool
    zoning_plan: str
    building_year: int

class PropertyAnalyzer:
    def __init__(self):
        self.municipality_client = None
        self.regulations_analyzer = None
        self.floor_plan_analyzer = None
        self.facade_analyzer = None

    async def analyze_property(self, 
                             address: str = None, 
                             image_url: str = None, 
                             finn_url: str = None) -> Dict:
        """
        Hovedmetode for eiendomsanalyse
        """
        property_details = await self._gather_property_details(address, image_url, finn_url)
        zoning_analysis = await self._analyze_zoning_regulations(property_details)
        development_potential = await self._analyze_development_potential(property_details, zoning_analysis)
        
        return {
            "property_details": property_details,
            "zoning_analysis": zoning_analysis,
            "development_potential": development_potential,
            "recommendations": await self._generate_recommendations(development_potential)
        }

    async def _gather_property_details(self, 
                                     address: Optional[str], 
                                     image_url: Optional[str], 
                                     finn_url: Optional[str]) -> PropertyDetails:
        """
        Henter detaljert informasjon om eiendommen
        """
        property_data = {}
        
        if address:
            # Hent data fra kartverket
            kartverket_data = await self.municipality_client.get_property_data(address)
            property_data.update(kartverket_data)
            
        if finn_url:
            # Hent data fra Finn.no
            finn_data = await self.municipality_client.get_finn_data(finn_url)
            property_data.update(finn_data)
            
        if image_url:
            # Analyser bilder
            image_data = await self.floor_plan_analyzer.analyze_images(image_url)
            property_data.update(image_data)
            
        # Hent historiske data fra kommunen
        municipal_data = await self.municipality_client.get_historical_data(
            property_data.get('gnr'),
            property_data.get('bnr')
        )
        
        return PropertyDetails(
            address=property_data.get('address'),
            gnr=property_data.get('gnr'),
            bnr=property_data.get('bnr'),
            municipality=property_data.get('municipality'),
            property_type=property_data.get('property_type'),
            total_area=property_data.get('total_area'),
            building_area=property_data.get('building_area'),
            floors=property_data.get('floors'),
            basement=property_data.get('has_basement', False),
            attic=property_data.get('has_attic', False),
            zoning_plan=property_data.get('zoning_plan'),
            building_year=property_data.get('building_year')
        )

    async def _analyze_zoning_regulations(self, 
                                        property_details: PropertyDetails) -> Dict:
        """
        Analyserer gjeldende reguleringsplan og forskrifter
        """
        # Hent reguleringsplan
        zoning_plan = await self.municipality_client.get_zoning_plan(
            property_details.gnr,
            property_details.bnr
        )
        
        # Hent kommuneplan hvis ingen reguleringsplan finnes
        if not zoning_plan:
            zoning_plan = await self.municipality_client.get_municipal_plan(
                property_details.gnr,
                property_details.bnr
            )
            
        # Hent gjeldende byggtekniske forskrifter
        building_code = await self.regulations_analyzer.get_building_code_requirements(
            property_details.building_year
        )
        
        # Analyser tillatt utnyttelse
        utilization = await self.regulations_analyzer.analyze_utilization(
            property_details,
            zoning_plan
        )
        
        # Hent byggegrenser og avstander
        boundaries = await self.regulations_analyzer.get_boundary_requirements(
            property_details,
            zoning_plan
        )
        
        # Sjekk spesielle hensyn
        special_considerations = await self.regulations_analyzer.get_special_considerations(
            property_details.gnr,
            property_details.bnr
        )
        
        return {
            "zoning_plan": zoning_plan,
            "building_code": building_code,
            "utilization": utilization,
            "boundaries": boundaries,
            "special_considerations": special_considerations,
            "allowed_usage_types": await self.regulations_analyzer.get_allowed_usage_types(
                zoning_plan
            )
        }

    async def _analyze_development_potential(self, 
                                           property_details: PropertyDetails,
                                           zoning_analysis: Dict) -> Dict:
        """
        Analyserer utviklingspotensialet for eiendommen
        """
        potential = {
            "basement_conversion": self._analyze_basement_potential(property_details),
            "attic_conversion": self._analyze_attic_potential(property_details),
            "property_division": self._analyze_division_potential(property_details, zoning_analysis),
            "height_extension": self._analyze_height_potential(property_details, zoning_analysis),
            "rental_units": self._analyze_rental_potential(property_details)
        }
        
        return potential

    async def _generate_recommendations(self, development_potential: Dict) -> List[Dict]:
        """
        Genererer anbefalinger basert på analysene
        """
        recommendations = []
        
        # Analyser kjellerpotensial
        if development_potential.get("basement_conversion", {}).get("has_potential", False):
            basement_rec = {
                "type": "basement_conversion",
                "title": "Utvikling av kjeller",
                "description": "Kjelleren har potensial for utvikling til utleieenhet",
                "potential_income": development_potential["basement_conversion"].get("estimated_rent", 0),
                "estimated_cost": development_potential["basement_conversion"].get("conversion_cost", 0),
                "requirements": development_potential["basement_conversion"].get("required_improvements", []),
                "priority": self._calculate_priority(
                    development_potential["basement_conversion"]
                )
            }
            recommendations.append(basement_rec)
            
        # Analyser loftspotensial
        if development_potential.get("attic_conversion", {}).get("has_potential", False):
            attic_rec = {
                "type": "attic_conversion",
                "title": "Utvikling av loft",
                "description": "Loftet kan utvikles til beboelig areal",
                "potential_income": development_potential["attic_conversion"].get("estimated_rent", 0),
                "estimated_cost": development_potential["attic_conversion"].get("conversion_cost", 0),
                "requirements": development_potential["attic_conversion"].get("required_improvements", []),
                "priority": self._calculate_priority(
                    development_potential["attic_conversion"]
                )
            }
            recommendations.append(attic_rec)
            
        # Analyser høydepotensial
        if development_potential.get("height_extension", {}).get("has_potential", False):
            height_rec = {
                "type": "height_extension",
                "title": "Påbygg i høyden",
                "description": "Bygningen kan utvides med ekstra etasje(r)",
                "potential_area": development_potential["height_extension"].get("potential_new_area", 0),
                "estimated_cost": development_potential["height_extension"].get("cost_estimate", 0),
                "requirements": development_potential["height_extension"].get("technical_requirements", []),
                "priority": self._calculate_priority(
                    development_potential["height_extension"]
                )
            }
            recommendations.append(height_rec)
            
        # Analyser tomtedelingspotensial
        if development_potential.get("property_division", {}).get("has_potential", False):
            division_rec = {
                "type": "property_division",
                "title": "Tomtedeling",
                "description": "Tomten kan deles for ytterligere utvikling",
                "potential_value": self._estimate_division_value(
                    development_potential["property_division"]
                ),
                "requirements": development_potential["property_division"].get("requirements", []),
                "priority": self._calculate_priority(
                    development_potential["property_division"]
                )
            }
            recommendations.append(division_rec)
            
        # Analyser utleiepotensial
        if development_potential.get("rental_units", {}).get("has_potential", False):
            rental_rec = {
                "type": "rental_units",
                "title": "Utleieenheter",
                "description": "Potensial for utvikling av utleieenheter",
                "potential_income": development_potential["rental_units"].get("total_potential_income", 0),
                "conversion_cost": development_potential["rental_units"].get("total_conversion_cost", 0),
                "units": development_potential["rental_units"].get("potential_units", []),
                "priority": self._calculate_priority(
                    development_potential["rental_units"]
                )
            }
            recommendations.append(rental_rec)
            
        # Sorter anbefalinger etter prioritet
        recommendations.sort(key=lambda x: x["priority"], reverse=True)
        
        # Legg til sammendrag av total utviklingspotensial
        total_potential = {
            "type": "summary",
            "title": "Totalt utviklingspotensial",
            "description": "Sammendrag av alle utviklingsmuligheter",
            "total_potential_value": sum(
                rec.get("potential_value", 0) for rec in recommendations
            ),
            "total_potential_income": sum(
                rec.get("potential_income", 0) for rec in recommendations
            ),
            "total_estimated_cost": sum(
                rec.get("estimated_cost", 0) for rec in recommendations
            ),
            "roi_analysis": self._calculate_total_roi(recommendations)
        }
        
        recommendations.insert(0, total_potential)
        
        return recommendations
    
    def _calculate_priority(self, potential: Dict) -> float:
        """
        Beregner prioritet for en utviklingsmulighet basert på:
        - Estimert ROI
        - Kompleksitet i gjennomføring
        - Risiko
        - Tidsperspektiv
        """
        roi = potential.get("roi_percentage", 0)
        complexity = len(potential.get("required_improvements", [])) * 10
        risk_factor = self._calculate_risk_factor(potential)
        time_factor = self._estimate_time_factor(potential)
        
        # Vektet score (høyere er bedre)
        priority_score = (
            (roi * 0.4) +                    # 40% vekt på ROI
            ((100 - complexity) * 0.25) +    # 25% vekt på kompleksitet (invertert)
            ((100 - risk_factor) * 0.20) +   # 20% vekt på risiko (invertert)
            ((100 - time_factor) * 0.15)     # 15% vekt på tid (invertert)
        )
        
        return priority_score
        
    def _calculate_risk_factor(self, potential: Dict) -> float:
        """
        Beregner risikofaktor for en utviklingsmulighet
        """
        risk_score = 0
        
        # Vurder teknisk kompleksitet
        if potential.get("technical_requirements"):
            risk_score += len(potential["technical_requirements"]) * 5
            
        # Vurder regulatoriske krav
        if potential.get("legal_requirements"):
            risk_score += len(potential["legal_requirements"]) * 7
            
        # Vurder kostnadsusikkerhet
        if potential.get("estimated_cost", 0) > 1000000:
            risk_score += 20
            
        # Maksimum risikoscore er 100
        return min(risk_score, 100)
        
    def _estimate_time_factor(self, potential: Dict) -> float:
        """
        Estimerer tidsfaktor for gjennomføring
        """
        base_time = 0
        
        # Grunnleggende tidsfaktorer
        if potential.get("requires_planning_permission", False):
            base_time += 30  # Søknadsprosess tar tid
            
        if potential.get("technical_requirements"):
            base_time += len(potential["technical_requirements"]) * 5
            
        if potential.get("legal_requirements"):
            base_time += len(potential["legal_requirements"]) * 3
            
        # Prosjektspesifikke faktorer
        if potential.get("type") == "property_division":
            base_time += 40  # Tomtedeling er tidkrevende
        elif potential.get("type") == "height_extension":
            base_time += 25  # Påbygg er komplekst
            
        # Maksimum tidsfaktor er 100
        return min(base_time, 100)
        
    def _calculate_total_roi(self, recommendations: List[Dict]) -> Dict:
        """
        Beregner total ROI for alle anbefalinger
        """
        total_income = sum(rec.get("potential_income", 0) for rec in recommendations)
        total_cost = sum(rec.get("estimated_cost", 0) for rec in recommendations)
        total_value = sum(rec.get("potential_value", 0) for rec in recommendations)
        
        if total_cost == 0:
            return {
                "roi_percentage": 0,
                "payback_period": float('inf'),
                "risk_level": "N/A"
            }
            
        roi_percentage = ((total_income + total_value) / total_cost - 1) * 100
        payback_period = total_cost / (total_income + total_value) if (total_income + total_value) > 0 else float('inf')
        
        return {
            "roi_percentage": roi_percentage,
            "payback_period": payback_period,
            "risk_level": self._determine_risk_level(roi_percentage, payback_period)
        }
        
    def _determine_risk_level(self, roi_percentage: float, payback_period: float) -> str:
        """
        Bestemmer risikonivå basert på ROI og tilbakebetalingstid
        """
        if roi_percentage > 30 and payback_period < 5:
            return "Lav"
        elif roi_percentage > 15 and payback_period < 8:
            return "Moderat"
        elif roi_percentage > 0:
            return "Høy"
        else:
            return "Svært høy"

    def _analyze_basement_potential(self, property_details: PropertyDetails) -> Dict:
        """
        Analyserer potensial for utleie/ombygging av kjeller
        """
        if not property_details.basement:
            return {
                "has_potential": False,
                "reason": "Ingen kjeller registrert"
            }
            
        # Analyser kjellerens egnethet for utleie
        ceiling_height = self.floor_plan_analyzer.get_basement_height(property_details)
        window_area = self.floor_plan_analyzer.get_basement_window_area(property_details)
        moisture_level = self.floor_plan_analyzer.get_basement_moisture(property_details)
        
        # Sjekk mot TEK17 krav
        meets_height = ceiling_height >= 2.2  # Minimum takhøyde for oppholdsrom
        meets_window = window_area >= (property_details.building_area * 0.1)  # 10% av gulvareal
        meets_moisture = moisture_level <= 0.5  # Akseptabelt fuktnivå
        
        potential = {
            "has_potential": all([meets_height, meets_window, meets_moisture]),
            "current_status": {
                "ceiling_height": ceiling_height,
                "window_area": window_area,
                "moisture_level": moisture_level
            },
            "requirements_met": {
                "height": meets_height,
                "window": meets_window,
                "moisture": meets_moisture
            },
            "required_improvements": []
        }
        
        # Legg til nødvendige forbedringer
        if not meets_height:
            potential["required_improvements"].append({
                "type": "ceiling_height",
                "current": ceiling_height,
                "required": 2.2,
                "description": "Senking av kjellergulv eller andre tiltak for å øke takhøyden"
            })
            
        if not meets_window:
            potential["required_improvements"].append({
                "type": "window_area",
                "current": window_area,
                "required": property_details.building_area * 0.1,
                "description": "Installering av større vinduer eller lysgraver"
            })
            
        if not meets_moisture:
            potential["required_improvements"].append({
                "type": "moisture",
                "current": moisture_level,
                "required": 0.5,
                "description": "Fuktsikring og drenering rundt kjeller"
            })
            
        return potential

    def _analyze_attic_potential(self, property_details: PropertyDetails) -> Dict:
        """
        Analyserer potensial for loftsutbygging
        """
        if not property_details.attic:
            return {
                "has_potential": False,
                "reason": "Ingen loft registrert"
            }
            
        # Analyser loftets egnethet for utbygging
        ceiling_height = self.floor_plan_analyzer.get_attic_height(property_details)
        floor_area = self.floor_plan_analyzer.get_attic_floor_area(property_details)
        roof_angle = self.floor_plan_analyzer.get_roof_angle(property_details)
        
        # Sjekk mot TEK17 krav
        meets_height = ceiling_height >= 2.2  # Minimum takhøyde
        meets_area = floor_area >= 15  # Minimum 15m² med full høyde
        meets_angle = roof_angle >= 30  # Minimum takvinkel for god utnyttelse
        
        potential = {
            "has_potential": all([meets_height, meets_area, meets_angle]),
            "current_status": {
                "ceiling_height": ceiling_height,
                "floor_area": floor_area,
                "roof_angle": roof_angle
            },
            "requirements_met": {
                "height": meets_height,
                "area": meets_area,
                "angle": meets_angle
            },
            "required_improvements": []
        }
        
        # Legg til nødvendige forbedringer
        if not meets_height:
            potential["required_improvements"].append({
                "type": "ceiling_height",
                "current": ceiling_height,
                "required": 2.2,
                "description": "Heving av tak eller senking av bjelkelag"
            })
            
        if not meets_area:
            potential["required_improvements"].append({
                "type": "floor_area",
                "current": floor_area,
                "required": 15,
                "description": "Strukturelle endringer for å øke areal med full høyde"
            })
            
        if not meets_angle:
            potential["required_improvements"].append({
                "type": "roof_angle",
                "current": roof_angle,
                "required": 30,
                "description": "Ombygging av takkonstruksjon"
            })
            
        # Sjekk for andre muligheter
        potential["additional_options"] = []
        
        if self.floor_plan_analyzer.can_add_dormers(property_details):
            potential["additional_options"].append({
                "type": "dormers",
                "description": "Mulighet for takopplett/arker",
                "estimated_area_increase": self.floor_plan_analyzer.calculate_dormer_area_gain(property_details)
            })
            
        if self.floor_plan_analyzer.can_raise_roof(property_details):
            potential["additional_options"].append({
                "type": "raise_roof",
                "description": "Mulighet for takheving",
                "estimated_height_gain": self.floor_plan_analyzer.calculate_roof_height_gain(property_details)
            })
            
        return potential

    def _analyze_division_potential(self, 
                                  property_details: PropertyDetails,
                                  zoning_analysis: Dict) -> Dict:
        """
        Analyserer potensial for tomtedeling
        """
        # Hent relevante data
        total_area = property_details.total_area
        min_plot_size = zoning_analysis.get('minimum_plot_size', 600)  # Standard minimum tomtestørrelse
        current_utilization = (property_details.building_area / total_area) * 100
        max_utilization = zoning_analysis.get('utilization', {}).get('max_bya', 30)
        
        # Sjekk om tomten er stor nok for deling
        can_divide = total_area >= (min_plot_size * 2)
        
        if not can_divide:
            return {
                "has_potential": False,
                "reason": f"Tomten er for liten for deling (minimum {min_plot_size * 2}m² kreves)",
                "current_area": total_area
            }
            
        # Analyser mulige delingsalternativer
        potential_divisions = self._calculate_potential_divisions(
            property_details,
            min_plot_size,
            max_utilization
        )
        
        # Sjekk adkomstmuligheter for nye tomter
        access_analysis = self._analyze_access_possibilities(
            property_details,
            potential_divisions
        )
        
        # Analyser infrastrukturbehov
        infrastructure = self._analyze_infrastructure_requirements(
            property_details,
            potential_divisions
        )
        
        return {
            "has_potential": bool(potential_divisions),
            "current_status": {
                "total_area": total_area,
                "current_utilization": current_utilization,
                "max_utilization": max_utilization
            },
            "requirements": {
                "minimum_plot_size": min_plot_size,
                "required_access_width": 4.0,  # Standard minimum veibredde
                "infrastructure_requirements": infrastructure
            },
            "potential_divisions": potential_divisions,
            "access_analysis": access_analysis,
            "recommended_division": self._get_optimal_division(
                potential_divisions,
                access_analysis,
                infrastructure
            )
        }
        
    def _calculate_potential_divisions(self, 
                                    property_details: PropertyDetails,
                                    min_plot_size: float,
                                    max_utilization: float) -> List[Dict]:
        """
        Beregner mulige delingsalternativer for tomten
        """
        divisions = []
        
        # Analyser mulige delingslinjer
        plot_geometry = self.floor_plan_analyzer.get_plot_geometry(property_details)
        
        # Prøv ulike delingsstrategier
        strategies = [
            self._try_vertical_division,
            self._try_horizontal_division,
            self._try_diagonal_division
        ]
        
        for strategy in strategies:
            possible_division = strategy(
                plot_geometry,
                min_plot_size,
                max_utilization
            )
            if possible_division:
                divisions.append(possible_division)
                
        return divisions
        
    def _analyze_access_possibilities(self, 
                                   property_details: PropertyDetails,
                                   potential_divisions: List[Dict]) -> Dict:
        """
        Analyserer adkomstmuligheter for potensielle nye tomter
        """
        return {
            "possible_access_points": self.floor_plan_analyzer.find_access_points(property_details),
            "road_connection_points": self.floor_plan_analyzer.find_road_connections(property_details),
            "required_easements": self._calculate_required_easements(
                property_details,
                potential_divisions
            )
        }
        
    def _analyze_infrastructure_requirements(self,
                                         property_details: PropertyDetails,
                                         potential_divisions: List[Dict]) -> Dict:
        """
        Analyserer behov for infrastruktur ved tomtedeling
        """
        return {
            "water_connection": self._analyze_water_requirements(property_details, potential_divisions),
            "sewage": self._analyze_sewage_requirements(property_details, potential_divisions),
            "electricity": self._analyze_electricity_requirements(property_details, potential_divisions),
            "estimated_costs": self._estimate_infrastructure_costs(property_details, potential_divisions)
        }

    def _analyze_height_potential(self,
                                property_details: PropertyDetails,
                                zoning_analysis: Dict) -> Dict:
        """
        Analyserer potensial for påbygg i høyden
        """
        # Hent gjeldende høyder og begrensninger
        current_height = property_details.total_height
        max_height = zoning_analysis.get('height_restrictions', {}).get('max_height', 9.0)
        max_stories = zoning_analysis.get('height_restrictions', {}).get('max_stories', 3)
        current_stories = property_details.floors
        
        # Sjekk om påbygg er mulig
        height_difference = max_height - current_height
        story_difference = max_stories - current_stories
        
        if height_difference <= 0 and story_difference <= 0:
            return {
                "has_potential": False,
                "reason": "Bygget har allerede nådd maksimal tillatt høyde/etasjer"
            }
            
        # Analyser konstruksjonsmessige muligheter
        structural_analysis = self.floor_plan_analyzer.analyze_structural_capacity(property_details)
        
        # Beregn potensielt nytt areal
        potential_new_area = self._calculate_potential_new_area(
            property_details,
            story_difference,
            structural_analysis
        )
        
        # Analyser tekniske krav
        technical_requirements = self._analyze_technical_requirements(
            property_details,
            story_difference
        )
        
        return {
            "has_potential": True,
            "current_status": {
                "current_height": current_height,
                "current_stories": current_stories,
                "max_allowed_height": max_height,
                "max_allowed_stories": max_stories
            },
            "potential": {
                "additional_height": height_difference,
                "additional_stories": story_difference,
                "potential_new_area": potential_new_area
            },
            "structural_analysis": {
                "foundation_capacity": structural_analysis.get("foundation_capacity"),
                "wall_capacity": structural_analysis.get("wall_capacity"),
                "required_reinforcement": structural_analysis.get("required_reinforcement", [])
            },
            "technical_requirements": technical_requirements,
            "cost_estimate": self._estimate_height_extension_costs(
                potential_new_area,
                structural_analysis,
                technical_requirements
            ),
            "recommended_approach": self._get_recommended_height_extension(
                height_difference,
                story_difference,
                structural_analysis
            )
        }
        
    def _calculate_potential_new_area(self,
                                   property_details: PropertyDetails,
                                   story_difference: int,
                                   structural_analysis: Dict) -> float:
        """
        Beregner potensielt nytt areal ved påbygg
        """
        base_area = property_details.building_area
        if structural_analysis.get("requires_setback", False):
            # Reduser areal med 20% per etasje for tilbaketrukket påbygg
            return base_area * story_difference * 0.8
        return base_area * story_difference
        
    def _analyze_technical_requirements(self,
                                    property_details: PropertyDetails,
                                    story_difference: int) -> Dict:
        """
        Analyserer tekniske krav for påbygg
        """
        return {
            "elevator": {
                "required": (property_details.floors + story_difference) > 3,
                "estimated_cost": 1000000 if (property_details.floors + story_difference) > 3 else 0
            },
            "fire_safety": {
                "required_measures": self._get_fire_safety_requirements(
                    property_details.floors + story_difference
                ),
                "estimated_cost": self._estimate_fire_safety_costs(
                    property_details.floors + story_difference
                )
            },
            "ventilation": {
                "requires_upgrade": story_difference > 0,
                "estimated_cost": 150000 * story_difference
            }
        }
        
    def _get_recommended_height_extension(self,
                                      height_difference: float,
                                      story_difference: int,
                                      structural_analysis: Dict) -> Dict:
        """
        Gir anbefaling om beste tilnærming til påbygg
        """
        if structural_analysis.get("foundation_capacity", 0) < 0.7:
            return {
                "type": "limited",
                "recommendation": "Begrenset påbygg med 1 etasje",
                "reason": "Begrenset bæreevne i fundamentering"
            }
            
        if height_difference >= 3.0 and story_difference >= 1:
            return {
                "type": "full",
                "recommendation": f"Fullt påbygg med {story_difference} etasjer",
                "reason": "God margin på høyde og etasjer"
            }
            
        return {
            "type": "partial",
            "recommendation": "Delvis påbygg med tilbaketrukket øverste etasje",
            "reason": "Optimal utnyttelse innenfor begrensninger"
        }

    def _analyze_rental_potential(self, property_details: PropertyDetails) -> Dict:
        """
        Analyserer potensial for utleieenheter
        """
        potential_units = []
        total_potential_income = 0
        
        # Analyser kjellerpotensial
        basement_potential = self._analyze_basement_potential(property_details)
        if basement_potential.get("has_potential", False):
            potential_units.append({
                "type": "basement",
                "area": property_details.building_area * 0.8,  # Estimate 80% of ground floor
                "estimated_rent": self._estimate_rental_income("basement", property_details.building_area * 0.8),
                "conversion_cost": self._estimate_conversion_cost("basement", basement_potential),
                "requirements": basement_potential.get("required_improvements", [])
            })
            
        # Analyser loftspotensial
        attic_potential = self._analyze_attic_potential(property_details)
        if attic_potential.get("has_potential", False):
            potential_units.append({
                "type": "attic",
                "area": property_details.building_area * 0.6,  # Estimate 60% of ground floor
                "estimated_rent": self._estimate_rental_income("attic", property_details.building_area * 0.6),
                "conversion_cost": self._estimate_conversion_cost("attic", attic_potential),
                "requirements": attic_potential.get("required_improvements", [])
            })
            
        # Analyser mulighet for hybel i hovedetasje
        main_floor_potential = self._analyze_main_floor_division(property_details)
        if main_floor_potential.get("has_potential", False):
            potential_units.append({
                "type": "main_floor_unit",
                "area": main_floor_potential.get("potential_area", 0),
                "estimated_rent": self._estimate_rental_income("main_floor", main_floor_potential.get("potential_area", 0)),
                "conversion_cost": self._estimate_conversion_cost("main_floor", main_floor_potential),
                "requirements": main_floor_potential.get("required_improvements", [])
            })
            
        # Kalkuler total potensiell leieinntekt
        for unit in potential_units:
            total_potential_income += unit["estimated_rent"]
            
        # Analyser parkeringskrav
        parking_requirements = self._analyze_parking_requirements(len(potential_units))
        
        return {
            "has_potential": bool(potential_units),
            "potential_units": potential_units,
            "total_potential_income": total_potential_income,
            "total_conversion_cost": sum(unit["conversion_cost"] for unit in potential_units),
            "parking_requirements": parking_requirements,
            "roi_analysis": self._calculate_rental_roi(
                total_potential_income,
                sum(unit["conversion_cost"] for unit in potential_units)
            ),
            "recommended_approach": self._get_recommended_rental_approach(potential_units),
            "legal_requirements": self._get_rental_legal_requirements(property_details, len(potential_units))
        }
        
    def _analyze_main_floor_division(self, property_details: PropertyDetails) -> Dict:
        """
        Analyser mulighet for å dele hovedetasje i separate enheter
        """
        # Implementer logikk for å analysere oppdeling av hovedetasje
        floor_plan = self.floor_plan_analyzer.get_floor_plan(property_details, floor=1)
        
        min_unit_size = 25  # minimum størrelse for hybel
        current_layout = self.floor_plan_analyzer.analyze_layout(floor_plan)
        
        division_possible = self.floor_plan_analyzer.can_divide_floor(
            floor_plan,
            min_unit_size
        )
        
        if not division_possible:
            return {
                "has_potential": False,
                "reason": "Hovedetasje er for liten eller uegnet for oppdeling"
            }
            
        potential_area = self.floor_plan_analyzer.calculate_division_area(floor_plan)
        
        return {
            "has_potential": True,
            "potential_area": potential_area,
            "required_improvements": [
                {
                    "type": "walls",
                    "description": "Nye skillevegger med lydkrav",
                    "estimated_cost": 100000
                },
                {
                    "type": "entrance",
                    "description": "Separat inngang",
                    "estimated_cost": 50000
                },
                {
                    "type": "kitchen",
                    "description": "Nytt kjøkken",
                    "estimated_cost": 150000
                }
            ]
        }
        
    def _estimate_rental_income(self, unit_type: str, area: float) -> float:
        """
        Estimerer potensiell leieinntekt basert på type enhet og areal
        """
        # Basert på gjennomsnittlige leiepriser i området
        base_rates = {
            "basement": 150,  # kr per m2
            "attic": 180,    # kr per m2
            "main_floor": 200 # kr per m2
        }
        
        return area * base_rates.get(unit_type, 150)
        
    def _estimate_conversion_cost(self, unit_type: str, potential: Dict) -> float:
        """
        Estimerer kostnad for ombygging
        """
        base_cost = {
            "basement": 15000,  # kr per m2
            "attic": 20000,    # kr per m2
            "main_floor": 10000 # kr per m2
        }
        
        total_cost = 0
        for improvement in potential.get("required_improvements", []):
            total_cost += improvement.get("estimated_cost", 0)
            
        return total_cost
        
    def _analyze_parking_requirements(self, num_units: int) -> Dict:
        """
        Analyser parkeringskrav for antall enheter
        """
        return {
            "required_spaces": num_units,  # Typisk 1 per enhet
            "available_spaces": self.floor_plan_analyzer.count_parking_spaces(),
            "potential_new_spaces": self.floor_plan_analyzer.find_potential_parking_spaces(),
            "estimated_cost": self._estimate_parking_space_cost(
                max(0, num_units - self.floor_plan_analyzer.count_parking_spaces())
            )
        }
        
    def _calculate_rental_roi(self, annual_income: float, total_cost: float) -> Dict:
        """
        Beregn ROI for utleieprosjekt
        """
        # Anta 5% årlig vedlikeholdskostnad av total ombyggingskostnad
        maintenance_cost = total_cost * 0.05
        
        # Beregn netto årlig inntekt
        net_annual_income = annual_income - maintenance_cost
        
        # Beregn tilbakebetalingstid og ROI
        payback_period = total_cost / net_annual_income if net_annual_income > 0 else float('inf')
        roi_percentage = (net_annual_income / total_cost) * 100 if total_cost > 0 else 0
        
        return {
            "annual_income": annual_income,
            "maintenance_cost": maintenance_cost,
            "net_annual_income": net_annual_income,
            "payback_period_years": payback_period,
            "roi_percentage": roi_percentage,
            "recommendation": self._get_roi_recommendation(roi_percentage)
        }
        
    def _get_recommended_rental_approach(self, potential_units: List[Dict]) -> Dict:
        """
        Gir anbefaling om beste tilnærming til utleie
        """
        if not potential_units:
            return {
                "recommendation": "Ingen utleiepotensial funnet",
                "reason": "Bygningen er ikke egnet for oppdeling i utleieenheter"
            }
            
        # Sorter enheter etter ROI
        sorted_units = sorted(
            potential_units,
            key=lambda x: x["estimated_rent"] / x["conversion_cost"] if x["conversion_cost"] > 0 else 0,
            reverse=True
        )
        
        return {
            "recommendation": f"Prioriter {sorted_units[0]['type']} utleieenhet først",
            "sequence": [unit["type"] for unit in sorted_units],
            "reason": "Beste forhold mellom kostnad og potensiell leieinntekt"
        }
        
    def _get_rental_legal_requirements(self, property_details: PropertyDetails, num_units: int) -> List[Dict]:
        """
        Henter lovkrav for utleieenheter
        """
        return [
            {
                "category": "Brannkrav",
                "requirements": [
                    "Brannskiller mellom enheter",
                    "Rømningsveier",
                    "Brannvarslingsanlegg"
                ]
            },
            {
                "category": "Ventilasjon",
                "requirements": [
                    "Separat ventilasjon for hver enhet",
                    "Avtrekk fra bad og kjøkken"
                ]
            },
            {
                "category": "Bod",
                "requirements": [
                    "Min. 5m² bod per enhet",
                    "Sportsbod min. 2.5m² per enhet"
                ]
            },
            {
                "category": "Parkering",
                "requirements": [
                    f"Minimum {num_units} parkeringsplasser",
                    "HC-parkering hvis mer enn 4 enheter"
                ]
            }
        ]