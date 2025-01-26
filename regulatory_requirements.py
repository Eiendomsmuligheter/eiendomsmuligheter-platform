from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class RegulatoryFramework:
    """Komplett rammeverk for norske lover og forskrifter relevant for utleieenheter"""
    
    def __init__(self):
        self.regulations = {
            "PLAN_OG_BYGNINGSLOV": {
                "reference": "LOV-2008-06-27-71",
                "sections": {
                    "§ 20-1": "Tiltak som krever søknad og tillatelse",
                    "§ 29-3": "Krav til universell utforming og forsvarlighet",
                    "§ 31-2": "Tiltak på eksisterende byggverk"
                }
            },
            "TEK17": {
                "reference": "FOR-2017-06-19-840",
                "chapters": {
                    "Kapittel 8": {
                        "title": "Opparbeidet uteareal",
                        "sections": {
                            "§ 8-1": "Uteareal",
                            "§ 8-4": "Parkering og annet oppstillingsareal"
                        }
                    },
                    "Kapittel 11": {
                        "title": "Sikkerhet ved brann",
                        "sections": {
                            "§ 11-1": "Sikkerhet ved brann",
                            "§ 11-2": "Risikoklasser",
                            "§ 11-4": "Bæreevne og stabilitet",
                            "§ 11-5": "Sikkerhet ved eksplosjon",
                            "§ 11-6": "Tiltak mot brannspredning mellom byggverk",
                            "§ 11-7": "Brannseksjoner",
                            "§ 11-8": "Brannceller",
                            "§ 11-11": "Generelle krav om rømning og redning",
                            "§ 11-13": "Utgang fra branncelle",
                            "§ 11-14": "Rømningsvei",
                            "§ 11-17": "Tilrettelegging for rednings- og slokkemannskap"
                        }
                    },
                    "Kapittel 12": {
                        "title": "Planløsning og bygningsdeler",
                        "sections": {
                            "§ 12-1": "Krav til planløsning og universell utforming",
                            "§ 12-2": "Krav til tilgjengelig boenhet",
                            "§ 12-5": "Sikkerhet i bruk",
                            "§ 12-7": "Krav til rom og annet oppholdsareal",
                            "§ 12-8": "Entre og garderobe",
                            "§ 12-9": "Bad og toalett",
                            "§ 12-10": "Bod og oppbevaringsplass",
                            "§ 12-11": "Balkong, terrasse og uteplass",
                            "§ 12-13": "Dør, port",
                            "§ 12-14": "Trapp",
                            "§ 12-15": "Utforming av rekkverk",
                            "§ 12-16": "Rampe",
                            "§ 12-17": "Vindu og andre glassfelt"
                        }
                    },
                    "Kapittel 13": {
                        "title": "Miljø og helse",
                        "sections": {
                            "§ 13-1": "Generelle krav til ventilasjon",
                            "§ 13-2": "Ventilasjon i boenhet",
                            "§ 13-3": "Ventilasjon i byggverk for publikum og arbeidsbygg",
                            "§ 13-4": "Termisk inneklima",
                            "§ 13-5": "Radon",
                            "§ 13-6": "Lyd og vibrasjoner",
                            "§ 13-7": "Lys",
                            "§ 13-8": "Utsyn",
                            "§ 13-9": "Generelle krav om fukt",
                            "§ 13-10": "Fukt fra grunnen",
                            "§ 13-12": "Nedbør",
                            "§ 13-13": "Oversvømmelse",
                            "§ 13-14": "Fuktsikring av våtrom og rom med vanninstallasjoner",
                            "§ 13-15": "Rengjøring før bygning tas i bruk",
                            "§ 13-16": "Inneklima/luftkvalitet",
                            "§ 13-17": "Tilføring av friskluft",
                            "§ 13-18": "Filterkvalitet",
                            "§ 13-19": "CO2- og temperaturstyring",
                            "§ 13-20": "Forsert ventilasjon i kjøkken"
                        }
                    },
                    "Kapittel 14": {
                        "title": "Energi",
                        "sections": {
                            "§ 14-1": "Generelle krav",
                            "§ 14-2": "Krav til energieffektivitet",
                            "§ 14-3": "Minimumskrav til energieffektivitet",
                            "§ 14-4": "Krav til løsninger for energiforsyning"
                        }
                    }
                }
            },
            "SAK10": {
                "reference": "FOR-2010-03-26-488",
                "sections": {
                    "§ 2-1": "Tiltak som krever søknad og tillatelse",
                    "§ 3-1": "Kvalifikasjonskrav",
                    "§ 5-4": "Dokumentasjon for søknad om tillatelse til tiltak"
                }
            },
            "BRANN": {
                "reference": "FOR-2002-06-26-847",
                "sections": {
                    "Kapittel 2": "Generelle krav til eier",
                    "Kapittel 3": "Generelle krav til bruker",
                    "Kapittel 4": "Krav til organisatoriske tiltak"
                }
            },
            "HUSLEIELOVEN": {
                "reference": "LOV-1999-03-26-17",
                "sections": {
                    "Kapittel 2": "Overlevering og krav til husrommet",
                    "Kapittel 3": "Vedlikehold",
                    "Kapittel 4": "Husleie"
                }
            },
            "BOLIGUTLEIE": {
                "reference": "Kommunale forskrifter",
                "requirements": {
                    "Søknadsprosess": "Krav til søknad om bruksendring",
                    "Tekniske_krav": "Spesifikke tekniske krav for utleieenheter",
                    "Brannsikring": "Særskilte brannsikringskrav for utleieboliger"
                }
            }
        }
        
    def get_detailed_requirements(self, regulation_type: str) -> Dict:
        """Henter detaljerte krav for en spesifikk regulering"""
        if regulation_type == "BRANN":
            return {
                "rømningsvei": {
                    "krav": {
                        "bredde": "Minimum 0.9 meter",
                        "høyde": "Minimum 2.1 meter",
                        "avstand": "Maksimum 25 meter til utgang",
                        "merking": "Alle rømningsveier skal være tydelig merket",
                        "belysning": "Nødlys ved strømbrudd"
                    },
                    "dokumentasjon": [
                        "Plantegning med rømningsveier",
                        "Beregning av rømningsavstander",
                        "Spesifikasjon av dører i rømningsvei",
                        "Belysningsplan"
                    ]
                },
                "brannskiller": {
                    "krav": {
                        "leilighet_leilighet": "EI 60",
                        "boenhet_fellesareal": "EI 60",
                        "kjeller_bolig": "REI 60",
                        "trapperom": "EI 30"
                    },
                    "dokumentasjon": [
                        "Detaljerte veggoppbygginger",
                        "Brannmotstandsberegninger",
                        "Materialspesifikasjoner"
                    ]
                }
            }
        elif regulation_type == "LYS":
            return {
                "dagslys": {
                    "krav": {
                        "dagslys_faktor": "Minimum 2% i oppholdsrom",
                        "vindu_areal": "Minimum 10% av gulvareal",
                        "lyshøyde": "Minimum 2.1 meter",
                        "utsyn": "Fri sikt mot det fri"
                    },
                    "dokumentasjon": [
                        "Dagslysberegninger",
                        "Vindusarealberegninger",
                        "Utsynsanalyse"
                    ]
                }
            }
        # Fortsetter med flere detaljerte krav...
        return {}

    def generate_compliance_checklist(self) -> List[Dict]:
        """Genererer en komplett sjekkliste for regelverksetterlevelse"""
        return [
            {
                "kategori": "Brannsikkerhet",
                "krav": [
                    "Rømningsveier",
                    "Brannskiller",
                    "Brannvarsling",
                    "Slokkeutstyr"
                ],
                "dokumentasjonskrav": [
                    "Branntegninger",
                    "Brannkonsept",
                    "Produktdokumentasjon"
                ]
            },
            {
                "kategori": "Ventilasjon",
                "krav": [
                    "Luftmengder",
                    "Avtrekk",
                    "Filtrering"
                ],
                "dokumentasjonskrav": [
                    "Ventilasjonstegninger",
                    "Luftmengdeberegninger"
                ]
            },
            # Fortsetter med flere kategorier...
        ]

class MunicipalityRequirements:
    """Håndterer kommunespesifikke krav og søknadsprosesser"""
    
    def __init__(self, municipality_name: str):
        self.municipality = municipality_name
        self.requirements = self._load_municipality_requirements()
        
    def _load_municipality_requirements(self) -> Dict:
        """Laster inn spesifikke krav for kommunen"""
        # Dette ville normalt hentes fra en database eller API
        return {
            "Oslo": {
                "søknadsskjemaer": {
                    "bruksendring": "https://www.oslo.kommune.no/skjema/123",
                    "utleietillatelse": "https://www.oslo.kommune.no/skjema/456"
                },
                "særkrav": {
                    "parkering": "Minimum 1 plass per boenhet",
                    "uteoppholdsareal": "Minimum 20m² per boenhet"
                }
            },
            # Andre kommuner...
        }
        
    def get_application_forms(self) -> List[str]:
        """Henter relevante søknadsskjemaer for kommunen"""
        return list(self.requirements.get(self.municipality, {}).get("søknadsskjemaer", {}).values())
        
    def get_specific_requirements(self) -> Dict:
        """Henter kommunespesifikke krav"""
        return self.requirements.get(self.municipality, {}).get("særkrav", {})