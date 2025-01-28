import logging
from typing import List, Dict, Any
import aiohttp
from datetime import datetime

class EnovaService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_base_url = "https://api.enova.no"  # Example URL
        self.cache = {}

    async def get_support_options(
        self,
        property_id: str,
        energy_info: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Get available Enova support options for a property"""
        try:
            # Check cache first
            cache_key = f"support_options_{property_id}"
            if cache_key in self.cache:
                return self.cache[cache_key]

            # In a real implementation, this would call Enova's API
            # For now, return predefined support options
            options = await self._get_predefined_options(energy_info)
            
            # Cache the results
            self.cache[cache_key] = options
            return options
        except Exception as e:
            self.logger.error(f"Error getting support options: {str(e)}")
            raise

    async def _get_predefined_options(
        self,
        energy_info: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Get predefined support options"""
        current_year = datetime.now().year
        
        options = [
            {
                "title": "Luft-til-luft varmepumpe",
                "description": """
                    Installasjon av luft-til-luft varmepumpe kan gi betydelige
                    besparelser på oppvarmingskostnadene. Enova støtter
                    installasjon av varmepumper som oppfyller gitte krav.
                """.strip(),
                "amount": 5000,
                "requirements": [
                    "Varmepumpen må være Energy Star-sertifisert",
                    "Installasjon må utføres av godkjent installatør",
                    "Minimum SCOP på 4.0"
                ],
                "benefits": [
                    "20-25% reduksjon i oppvarmingskostnader",
                    "Bedre inneklima",
                    "Økt boligverdi"
                ],
                "valid_until": f"{current_year}-12-31"
            },
            {
                "title": "Etterisolering av yttervegg",
                "description": """
                    Etterisolering av yttervegger kan gi betydelige
                    energibesparelser. Enova støtter tiltak som oppfyller
                    gjeldende isolasjonskrav.
                """.strip(),
                "amount": 25000,
                "requirements": [
                    "Minimum 20 cm total isolasjonstykkelse",
                    "U-verdi på maksimalt 0.18 W/m²K",
                    "Dokumentert utførelse av fagfolk"
                ],
                "benefits": [
                    "15-20% reduksjon i varmetap",
                    "Bedre inneklima",
                    "Økt boligverdi"
                ],
                "valid_until": f"{current_year}-12-31"
            },
            {
                "title": "Installasjon av balansert ventilasjon",
                "description": """
                    Balansert ventilasjon med varmegjenvinning sikrer god
                    luftkvalitet og reduserer varmetap. Enova støtter
                    installasjon av moderne ventilasjonsanlegg.
                """.strip(),
                "amount": 20000,
                "requirements": [
                    "Minimum varmegjenvinningsgrad på 80%",
                    "SFP-faktor lavere enn 2.0",
                    "Installasjon av fagfolk"
                ],
                "benefits": [
                    "Opptil 40% reduksjon i ventilasjonvarmetap",
                    "Bedre inneklima",
                    "Redusert fuktproblematikk"
                ],
                "valid_until": f"{current_year}-12-31"
            },
            {
                "title": "Utskifting av vinduer",
                "description": """
                    Gamle vinduer er ofte en betydelig kilde til varmetap.
                    Enova støtter utskifting til moderne energieffektive vinduer.
                """.strip(),
                "amount": 15000,
                "requirements": [
                    "U-verdi på maksimalt 0.8 W/m²K",
                    "Dokumentert kvalitet og installasjon",
                    "Minimum 3-lags glass"
                ],
                "benefits": [
                    "Opptil 20% reduksjon i oppvarmingsbehov",
                    "Bedre lydisolering",
                    "Mindre kondensproblemer"
                ],
                "valid_until": f"{current_year}-12-31"
            },
            {
                "title": "Vannbåren varme",
                "description": """
                    Vannbåren varme gir effektiv og komfortabel oppvarming.
                    Enova støtter konvertering til vannbåren varme.
                """.strip(),
                "amount": 30000,
                "requirements": [
                    "Komplett system med styring",
                    "Dokumentert energieffektivitet",
                    "Profesjonell installasjon"
                ],
                "benefits": [
                    "Fleksibel oppvarming",
                    "Jevn varmefordeling",
                    "Mulighet for flere varmekilder"
                ],
                "valid_until": f"{current_year}-12-31"
            }
        ]

        # If we have energy info, adjust recommendations
        if energy_info:
            current_rating = energy_info.get("rating", "G")
            consumption = energy_info.get("consumption", 0)
            
            # Add specific recommendations based on current status
            if current_rating in ["F", "G"]:
                options.append({
                    "title": "Helhetlig energioppgradering",
                    "description": """
                        Din bolig har et betydelig potensial for
                        energieffektivisering. En helhetlig oppgradering kan
                        gi store besparelser.
                    """.strip(),
                    "amount": 100000,
                    "requirements": [
                        "Minimum to energitiltak",
                        "Dokumentert energibesparelse",
                        "Profesjonell prosjektering"
                    ],
                    "benefits": [
                        "Betydelig reduksjon i energiforbruk",
                        "Økt boligverdi",
                        "Bedre bokomfort"
                    ],
                    "valid_until": f"{current_year}-12-31"
                })

        return options

    async def calculate_potential_savings(
        self,
        property_info: Dict[str, Any],
        support_option: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate potential savings for a support option"""
        try:
            # In a real implementation, this would use proper calculations
            # For now, return estimated values
            annual_consumption = property_info.get("energy_consumption", 20000)
            
            if support_option["title"] == "Luft-til-luft varmepumpe":
                savings = annual_consumption * 0.25
            elif support_option["title"] == "Etterisolering av yttervegg":
                savings = annual_consumption * 0.15
            elif support_option["title"] == "Installasjon av balansert ventilasjon":
                savings = annual_consumption * 0.20
            else:
                savings = annual_consumption * 0.10

            return {
                "annual_savings_kwh": savings,
                "annual_savings_nok": savings * 1.5,  # Assuming 1.5 NOK per kWh
                "co2_reduction": savings * 0.17,  # kg CO2 per kWh
                "payback_years": support_option["amount"] / (savings * 1.5)
            }
        except Exception as e:
            self.logger.error(f"Error calculating savings: {str(e)}")
            raise

    async def get_support_history(
        self,
        property_id: str
    ) -> List[Dict[str, Any]]:
        """Get history of Enova support for a property"""
        try:
            # In a real implementation, this would call Enova's API
            # For now, return empty list
            return []
        except Exception as e:
            self.logger.error(f"Error getting support history: {str(e)}")
            raise

    async def check_eligibility(
        self,
        property_info: Dict[str, Any],
        support_option: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if a property is eligible for a support option"""
        try:
            # In a real implementation, this would check actual criteria
            # For now, return mock eligibility check
            return {
                "eligible": True,
                "requirements_met": support_option["requirements"],
                "missing_requirements": [],
                "next_steps": [
                    "Kontakt godkjent installatør",
                    "Innhent tilbud",
                    "Søk støtte via Enova's nettside"
                ]
            }
        except Exception as e:
            self.logger.error(f"Error checking eligibility: {str(e)}")
            raise