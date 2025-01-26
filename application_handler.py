from typing import Dict, List, Optional
import requests
from dataclasses import dataclass
import json
import logging
from pathlib import Path

@dataclass
class ApplicationDocument:
    """Representerer et søknadsdokument"""
    form_type: str
    form_id: str
    required_attachments: List[str]
    municipality: str
    
class ApplicationHandler:
    """Håndterer søknadsprosessen for utleieenheter"""
    
    def __init__(self, municipality: str):
        self.municipality = municipality
        self.logger = logging.getLogger(__name__)
        self.forms = self._get_municipality_forms()
        
    def _get_municipality_forms(self) -> Dict[str, str]:
        """Henter alle relevante skjemaer fra kommunen"""
        # Dette ville normalt hente fra kommunens API
        return {
            "Oslo": {
                "bruksendring": "https://www.oslo.kommune.no/skjema/bruksendring",
                "utleietillatelse": "https://www.oslo.kommune.no/skjema/utleie",
                "nabovarsling": "https://www.oslo.kommune.no/skjema/nabovarsel"
            }
        }.get(self.municipality, {})
        
    def prepare_application(self, property_data: Dict) -> Dict:
        """Forbereder komplett søknad med all nødvendig dokumentasjon"""
        application = {
            "søknadsskjema": self._fill_application_form(property_data),
            "vedlegg": self._prepare_attachments(property_data),
            "sjekkliste": self._generate_checklist(),
            "erklæringer": self._prepare_declarations()
        }
        
        return application
        
    def _fill_application_form(self, property_data: Dict) -> Dict:
        """Fyller ut søknadsskjema automatisk"""
        return {
            "eiendom": {
                "gårds_og_bruksnummer": property_data["gnr_bnr"],
                "adresse": property_data["address"],
                "kommune": self.municipality
            },
            "tiltakshaver": {
                "navn": property_data["owner"],
                "kontakt": property_data["contact"]
            },
            "tiltak": {
                "type": "bruksendring",
                "beskrivelse": "Etablering av utleieenhet",
                "areal": property_data["area"]
            },
            "ansvarlig_søker": {
                "foretak": "AutoPlan AS",
                "org_nummer": "123456789",
                "ansvarlig": "Per Planlegger"
            }
        }
        
    def _prepare_attachments(self, property_data: Dict) -> List[Dict]:
        """Forbereder alle nødvendige vedlegg"""
        return [
            {
                "type": "situasjonsplan",
                "innhold": self._generate_site_plan(property_data),
                "format": "PDF"
            },
            {
                "type": "plantegning",
                "innhold": self._generate_floor_plans(property_data),
                "format": "PDF"
            },
            {
                "type": "brannkonsept",
                "innhold": self._generate_fire_concept(property_data),
                "format": "PDF"
            },
            {
                "type": "ventilasjon",
                "innhold": self._generate_ventilation_docs(property_data),
                "format": "PDF"
            }
        ]
        
    def _generate_checklist(self) -> List[Dict]:
        """Genererer sjekkliste for komplett søknad"""
        return [
            {
                "kategori": "Søknadsskjema",
                "elementer": [
                    "Søknadsskjema for bruksendring",
                    "Gjenpart nabovarsel",
                    "Kvittering for nabovarsel"
                ]
            },
            {
                "kategori": "Tegninger",
                "elementer": [
                    "Situasjonsplan",
                    "Eksisterende plantegninger",
                    "Nye plantegninger",
                    "Snittegninger",
                    "Fasadetegninger"
                ]
            },
            {
                "kategori": "Dokumentasjon",
                "elementer": [
                    "Brannkonsept",
                    "Ventilasjonsplan",
                    "Dagslysberegninger",
                    "Støyberegninger",
                    "Statiske beregninger"
                ]
            }
        ]
        
    def _prepare_declarations(self) -> List[Dict]:
        """Forbereder nødvendige erklæringer"""
        return [
            {
                "type": "Ansvarsrett",
                "rolle": "SØK",
                "foretak": "AutoPlan AS",
                "tiltaksklasse": 1
            },
            {
                "type": "Ansvarsrett",
                "rolle": "PRO",
                "foretak": "AutoPlan AS",
                "tiltaksklasse": 1
            },
            {
                "type": "Samsvarserlæring",
                "beskrivelse": "Prosjektering av brannsikkerhet",
                "foretak": "AutoPlan AS"
            }
        ]
        
    def validate_application(self, application: Dict) -> Dict:
        """Validerer at søknaden er komplett og korrekt"""
        validation_result = {
            "er_komplett": True,
            "mangler": [],
            "advarsler": []
        }
        
        # Sjekk søknadsskjema
        if not self._validate_application_form(application["søknadsskjema"]):
            validation_result["er_komplett"] = False
            validation_result["mangler"].append("Ufullstendig søknadsskjema")
            
        # Sjekk vedlegg
        missing_attachments = self._validate_attachments(application["vedlegg"])
        if missing_attachments:
            validation_result["er_komplett"] = False
            validation_result["mangler"].extend(missing_attachments)
            
        # Sjekk erklæringer
        if not self._validate_declarations(application["erklæringer"]):
            validation_result["er_komplett"] = False
            validation_result["mangler"].append("Manglende eller ufullstendige erklæringer")
            
        return validation_result
        
    def submit_application(self, application: Dict) -> Dict:
        """Sender inn søknaden til kommunen"""
        # Dette ville normalt integrere med kommunens API
        submission_result = {
            "status": "success",
            "søknadsnummer": "2024-12345",
            "mottatt_dato": "2024-03-01",
            "estimert_behandlingstid": "3 uker"
        }
        
        return submission_result