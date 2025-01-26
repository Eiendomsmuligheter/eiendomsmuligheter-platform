from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime
import emoji
import logging

logger = logging.getLogger(__name__)

@dataclass
class MunicipalityCheckpoint:
    """Detaljert sjekkpunkt for kommunale krav"""
    id: str
    name: str
    description: str
    required: bool
    status: str  # "completed", "pending", "not_applicable", "failed"
    documentation_needed: List[str]
    relevant_regulations: List[str]
    municipality_specific_rules: Dict[str, Any]
    validation_steps: List[str]

class ComprehensiveChecklist:
    def __init__(self):
        self.pre_analysis_checks = [
            MunicipalityCheckpoint(
                id="CHECK001",
                name="Planbestemmelser",
                description="Sjekk av gjeldende reguleringsplan og kommuneplan",
                required=True,
                status="pending",
                documentation_needed=["Reguleringsplan", "Kommuneplan", "Bestemmelser"],
                relevant_regulations=["Plan- og bygningsloven § 12-4"],
                municipality_specific_rules={},
                validation_steps=[
                    "Last ned gjeldende reguleringsplan",
                    "Sjekk kommuneplanens arealdel",
                    "Verifiser tomtens reguleringsformål",
                    "Identifiser alle relevante bestemmelser"
                ]
            ),
            MunicipalityCheckpoint(
                id="CHECK002",
                name="Grad av utnytting",
                description="Beregning av tillatt og eksisterende utnyttelse",
                required=True,
                status="pending",
                documentation_needed=["Situasjonskart", "Arealoversikt"],
                relevant_regulations=["TEK17 § 5-1", "TEK17 § 5-4"],
                municipality_specific_rules={},
                validation_steps=[
                    "Beregn tomtens størrelse",
                    "Beregn eksisterende BYA",
                    "Beregn tillatt BYA",
                    "Vurder parkeringskrav"
                ]
            ),
            MunicipalityCheckpoint(
                id="CHECK003",
                name="Avstandskrav",
                description="Sjekk av avstander til nabogrenser, vei og andre bygninger",
                required=True,
                status="pending",
                documentation_needed=["Situasjonskart med målsatte avstander"],
                relevant_regulations=["Plan- og bygningsloven § 29-4"],
                municipality_specific_rules={},
                validation_steps=[
                    "Mål avstand til nabogrense",
                    "Mål avstand til vei",
                    "Mål avstand til andre bygninger",
                    "Sjekk brannkrav ved mindre avstander"
                ]
            )
        ]

        self.technical_requirements = [
            MunicipalityCheckpoint(
                id="TECH001",
                name="Brannsikkerhet",
                description="Kontroll av brannskiller og rømningsveier",
                required=True,
                status="pending",
                documentation_needed=["Branntegninger", "Brannkonsept"],
                relevant_regulations=["TEK17 kapittel 11"],
                municipality_specific_rules={},
                validation_steps=[
                    "Verifiser brannskiller",
                    "Kontroller rømningsveier",
                    "Sjekk krav til brannvarsling",
                    "Vurder behov for sprinkleranlegg"
                ]
            ),
            MunicipalityCheckpoint(
                id="TECH002",
                name="Ventilasjonskrav",
                description="Kontroll av ventilasjonskrav for ulike rom",
                required=True,
                status="pending",
                documentation_needed=["Ventilasjonstegninger"],
                relevant_regulations=["TEK17 § 13-1", "TEK17 § 13-2"],
                municipality_specific_rules={},
                validation_steps=[
                    "Beregn luftmengdebehov",
                    "Vurder type ventilasjonssystem",
                    "Sjekk krav til avtrekk",
                    "Kontroller plassering av ventiler"
                ]
            )
        ]

        self.documentation_requirements = [
            MunicipalityCheckpoint(
                id="DOC001",
                name="Situasjonskart",
                description="Oppdatert situasjonskart med alle nødvendige mål",
                required=True,
                status="pending",
                documentation_needed=["Situasjonskart fra kommunen"],
                relevant_regulations=["SAK10 § 5-4"],
                municipality_specific_rules={},
                validation_steps=[
                    "Bestill situasjonskart fra kommunen",
                    "Påfør alle mål og avstander",
                    "Marker planlagt tiltak",
                    "Vis parkering og uteområder"
                ]
            ),
            MunicipalityCheckpoint(
                id="DOC002",
                name="Plantegninger",
                description="Detaljerte plantegninger av alle berørte etasjer",
                required=True,
                status="pending",
                documentation_needed=["Målsatte plantegninger"],
                relevant_regulations=["SAK10 § 5-4"],
                municipality_specific_rules={},
                validation_steps=[
                    "Lag plantegning før tiltak",
                    "Lag plantegning etter tiltak",
                    "Mål opp alle rom",
                    "Marker alle endringer"
                ]
            ),
            MunicipalityCheckpoint(
                id="DOC003",
                name="3D-visualiseringer",
                description="3D-modeller og visualiseringer av planlagt tiltak",
                required=True,
                status="pending",
                documentation_needed=[
                    "3D-modell av eksisterende bygg",
                    "3D-modell av planlagt tiltak",
                    "Renderinger fra ulike vinkler"
                ],
                relevant_regulations=[],
                municipality_specific_rules={},
                validation_steps=[
                    "Generer 3D-modell av eksisterende bygg",
                    "Lag 3D-modell av planlagt tiltak",
                    "Vis før/etter sammenligninger",
                    "Inkluder målangivelser i 3D"
                ]
            )
        ]

    def generate_checklist_report(self, municipality: str) -> Dict[str, Any]:
        """Genererer en fullstendig rapport med alle sjekkpunkter"""
        report = {
            "municipality": municipality,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "summary": {
                "total_checks": len(self.pre_analysis_checks) + 
                              len(self.technical_requirements) + 
                              len(self.documentation_requirements),
                "completed": 0,
                "pending": 0,
                "failed": 0
            },
            "sections": {
                "pre_analysis": self._format_section_report(self.pre_analysis_checks),
                "technical": self._format_section_report(self.technical_requirements),
                "documentation": self._format_section_report(self.documentation_requirements)
            },
            "recommendations": self._generate_recommendations()
        }
        return report

    def analyze_property_options(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyserer alle mulige alternativer for eiendommen"""
        analysis = {
            "existing_building": self._analyze_existing_building(property_data),
            "options": {
                "bruksendring": self._analyze_usage_change(property_data),
                "tilbygg": self._analyze_extension(property_data),
                "påbygg": self._analyze_additional_floor(property_data),
                "riving_nybygg": self._analyze_rebuild(property_data)
            },
            "recommended_option": None,
            "recommendation_reasons": [],
            "economic_analysis": {}
        }

        # Beregn optimal løsning
        best_option = self._calculate_best_option(analysis["options"])
        analysis["recommended_option"] = best_option["option"]
        analysis["recommendation_reasons"] = best_option["reasons"]
        analysis["economic_analysis"] = self._calculate_economic_impact(
            best_option["option"],
            property_data
        )

        return analysis

    def _analyze_existing_building(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyserer eksisterende bygning"""
        return {
            "total_area": property_data.get("area", 0),
            "rooms": self._analyze_rooms(property_data),
            "technical_state": self._assess_technical_state(property_data),
            "current_usage": self._analyze_current_usage(property_data),
            "limitations": self._identify_limitations(property_data)
        }

    def _analyze_usage_change(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Vurderer muligheter for bruksendring"""
        return {
            "possible": True,  # Må implementeres med faktisk logikk
            "potential_areas": self._identify_conversion_areas(property_data),
            "requirements": self._list_conversion_requirements(property_data),
            "estimated_cost": self._estimate_conversion_cost(property_data),
            "potential_income": self._estimate_rental_income(property_data),
            "complexity": "medium",
            "timeline": "3-6 months"
        }

    def _analyze_extension(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Vurderer muligheter for tilbygg"""
        return {
            "possible": True,  # Må implementeres med faktisk logikk
            "max_area": self._calculate_max_extension_area(property_data),
            "optimal_placement": self._find_optimal_extension_placement(property_data),
            "requirements": self._list_extension_requirements(property_data),
            "estimated_cost": self._estimate_extension_cost(property_data),
            "potential_income": self._estimate_extension_income(property_data),
            "complexity": "high",
            "timeline": "6-12 months"
        }

    def _analyze_additional_floor(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Vurderer muligheter for påbygg"""
        return {
            "possible": True,  # Må implementeres med faktisk logikk
            "structural_feasibility": self._assess_structural_capacity(property_data),
            "max_height": self._calculate_max_height(property_data),
            "requirements": self._list_vertical_extension_requirements(property_data),
            "estimated_cost": self._estimate_vertical_extension_cost(property_data),
            "potential_income": self._estimate_vertical_extension_income(property_data),
            "complexity": "very_high",
            "timeline": "8-14 months"
        }

    def _analyze_rebuild(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Vurderer muligheter for riving og nybygg"""
        return {
            "possible": True,  # Må implementeres med faktisk logikk
            "max_potential": self._calculate_max_building_potential(property_data),
            "optimal_design": self._suggest_optimal_design(property_data),
            "requirements": self._list_rebuild_requirements(property_data),
            "estimated_cost": self._estimate_rebuild_cost(property_data),
            "potential_income": self._estimate_rebuild_income(property_data),
            "complexity": "extreme",
            "timeline": "12-24 months"
        }

    def _calculate_best_option(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Beregner beste alternativ basert på alle faktorer"""
        scores = {}
        for option_name, option_data in options.items():
            scores[option_name] = self._calculate_option_score(option_data)

        best_option = max(scores.items(), key=lambda x: x[1]["total_score"])
        return {
            "option": best_option[0],
            "score": best_option[1]["total_score"],
            "reasons": best_option[1]["reasons"]
        }

    def _calculate_option_score(self, option_data: Dict[str, Any]) -> Dict[str, Any]:
        """Beregner score for et alternativ basert på multiple faktorer"""
        score = {
            "roi_score": self._calculate_roi_score(option_data),
            "complexity_score": self._calculate_complexity_score(option_data),
            "timeline_score": self._calculate_timeline_score(option_data),
            "risk_score": self._calculate_risk_score(option_data),
            "market_score": self._calculate_market_score(option_data)
        }

        total_score = sum(score.values()) / len(score)
        reasons = self._generate_score_reasons(score)

        return {
            "total_score": total_score,
            "breakdown": score,
            "reasons": reasons
        }

    def _generate_recommendations(self) -> List[str]:
        """Genererer anbefalinger basert på sjekklistestatus"""
        recommendations = []
        for checkpoint in (self.pre_analysis_checks + 
                         self.technical_requirements + 
                         self.documentation_requirements):
            if checkpoint.status == "pending":
                recommendations.append(f"Fullfør {checkpoint.name}: {checkpoint.description}")
            elif checkpoint.status == "failed":
                recommendations.append(f"VIKTIG: Korriger {checkpoint.name}: {checkpoint.description}")
        return recommendations