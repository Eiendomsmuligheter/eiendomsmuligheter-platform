from typing import Dict, List, Any, Optional
import logging
from .municipality_service import MunicipalityService
from .ai_service import AIService
from .enova_service import EnovaService

class PropertyAnalyzerService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.municipality_service = MunicipalityService()
        self.ai_service = AIService()
        self.enova_service = EnovaService()

    async def analyze_property(
        self,
        address: str,
        image_data: Optional[bytes] = None,
        link: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Utfør en komplett analyse av en eiendom basert på adresse og evt. bilder/link
        """
        try:
            # 1. Hent grunnleggende eiendomsinformasjon
            property_info = await self._get_property_info(address)

            # 2. Analyser bygningen med AI hvis bilde er tilgjengelig
            building_analysis = None
            if image_data:
                building_analysis = await self.ai_service.analyze_building(image_data)
            elif link:
                building_analysis = await self.ai_service.analyze_building_from_link(link)

            # 3. Hent kommunale regler og historikk
            municipal_data = await self._get_municipal_data(property_info)

            # 4. Analyser utviklingsmuligheter
            development_potential = await self._analyze_development_potential(
                property_info,
                municipal_data,
                building_analysis
            )

            # 5. Generer energianalyse og ENOVA-støttemuligheter
            energy_analysis = await self._analyze_energy_potential(
                property_info,
                building_analysis
            )

            return {
                "property_info": property_info,
                "building_analysis": building_analysis,
                "municipal_data": municipal_data,
                "development_potential": development_potential,
                "energy_analysis": energy_analysis,
                "recommendations": await self._generate_recommendations(
                    development_potential,
                    energy_analysis
                )
            }

        except Exception as e:
            self.logger.error(f"Error in property analysis: {str(e)}")
            raise

    async def _get_property_info(self, address: str) -> Dict[str, Any]:
        """Hent grunnleggende eiendomsinformasjon fra Kartverket"""
        # TODO: Implementer Kartverket API integrasjon
        return {
            "address": address,
            "municipality_code": "3005",  # Drammen
            "property_id": "3005-1-1",  # Example
            "coordinates": {
                "lat": 59.744225,
                "lon": 10.204458
            }
        }

    async def _get_municipal_data(
        self,
        property_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Hent all relevant kommunal informasjon"""
        municipality_code = property_info["municipality_code"]
        property_id = property_info["property_id"]

        return {
            "regulations": await self.municipality_service.get_regulations(
                municipality_code
            ),
            "property_history": await self.municipality_service.get_property_history(
                property_id
            ),
            "zoning": await self.municipality_service.check_zoning_restrictions(
                municipality_code,
                property_info
            )
        }

    async def _analyze_development_potential(
        self,
        property_info: Dict[str, Any],
        municipal_data: Dict[str, Any],
        building_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyser utviklingsmuligheter basert på all tilgjengelig informasjon"""
        zoning = municipal_data["zoning"]
        regulations = municipal_data["regulations"]

        # Beregn maksimal utnyttelse
        max_bya = zoning["max_bya"]
        lot_size = building_analysis.get("lot_size", 1000)  # default hvis ikke tilgjengelig
        max_buildable_area = (lot_size * max_bya) / 100

        # Analyser muligheter
        potential = {
            "current_utilization": {
                "bya": building_analysis.get("bya", 0) if building_analysis else 0,
                "bra": building_analysis.get("bra", 0) if building_analysis else 0
            },
            "max_potential": {
                "bya": max_buildable_area,
                "additional_bya": max_buildable_area - (
                    building_analysis.get("bya", 0) if building_analysis else 0
                )
            },
            "possibilities": []
        }

        # Vurder ulike utviklingsmuligheter
        if potential["max_potential"]["additional_bya"] > 50:
            potential["possibilities"].extend([
                {
                    "type": "extension",
                    "description": "Tilbygg mulig",
                    "potential_size": min(50, potential["max_potential"]["additional_bya"])
                }
            ])

        if building_analysis and building_analysis.get("has_basement"):
            potential["possibilities"].append({
                "type": "basement_apartment",
                "description": "Potensial for utleiedel i kjeller",
                "requirements": [
                    "Minimum takhøyde 2.2m",
                    "Separate inngang",
                    "Brannskille mot hovedetasje"
                ]
            })

        if lot_size > 600:  # Eksempel på tomtedelingsvurdering
            potential["possibilities"].append({
                "type": "lot_division",
                "description": "Mulighet for tomtedeling",
                "minimum_size": "600m²"
            })

        return potential

    async def _analyze_energy_potential(
        self,
        property_info: Dict[str, Any],
        building_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyser energibesparelsespotensial og støttemuligheter"""
        return await self.enova_service.analyze_property(
            property_info,
            building_analysis
        )

    async def _generate_recommendations(
        self,
        development_potential: Dict[str, Any],
        energy_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generer anbefalinger basert på analysene"""
        recommendations = []

        # Utviklingsanbefalinger
        for possibility in development_potential["possibilities"]:
            recommendations.append({
                "type": "development",
                "title": possibility["description"],
                "description": await self._generate_possibility_description(possibility),
                "priority": "high" if possibility["type"] in ["basement_apartment", "extension"] else "medium"
            })

        # Energianbefalinger
        for measure in energy_analysis.get("recommended_measures", []):
            recommendations.append({
                "type": "energy",
                "title": measure["title"],
                "description": measure["description"],
                "cost_estimate": measure.get("cost_estimate"),
                "energy_saving": measure.get("energy_saving"),
                "support_amount": measure.get("support_amount"),
                "priority": "high" if measure.get("roi", 0) < 5 else "medium"
            })

        return sorted(
            recommendations,
            key=lambda x: 0 if x["priority"] == "high" else 1
        )

    async def _generate_possibility_description(
        self,
        possibility: Dict[str, Any]
    ) -> str:
        """Generer detaljert beskrivelse av en utviklingsmulighet"""
        if possibility["type"] == "extension":
            return f"""
                Mulighet for tilbygg på opptil {possibility['potential_size']}m².
                Dette vil øke boligens bruksareal og verdi.
                Husk å søke om byggetillatelse.
            """.strip()
        elif possibility["type"] == "basement_apartment":
            return """
                Kjelleren kan potensielt gjøres om til en separat utleieenhet.
                Dette krever:
                - Minimum takhøyde på 2.2m
                - Egen inngang
                - Brannskille mot hovedetasjen
                - Tilfredsstillende lysforhold
                - Våtrom som oppfyller tekniske krav
            """.strip()
        elif possibility["type"] == "lot_division":
            return f"""
                Tomten kan potensielt deles.
                Minimumskrav til tomtestørrelse er {possibility['minimum_size']}.
                Sjekk reguleringsplan for spesifikke krav.
            """.strip()
        return possibility["description"]