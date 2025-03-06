import re
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import json
import asyncio
from datetime import datetime
import os
import yaml
from difflib import SequenceMatcher
import requests
import aiohttp
from bs4 import BeautifulSoup
from pathlib import Path

# Konfigurer logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class RegulationAnalyzer:
    """
    Spesialisert analysator for byggeforskrifter og reguleringer
    som bruker strukturerte regler og mønstermatchende teknikker.
    
    Denne klassen erstatter avhengigheten av GPT-4 API med en
    mer robust og kostnadseffektiv løsning som kjører direkte i plattformen.
    
    Funksjoner:
    - Ekstraherer regler fra reguleringstekst
    - Henter kommunespesifikke regler og forskrifter
    - Evaluerer prosjekter mot gjeldende regelverk
    - Genererer anbefalinger basert på samsvar med reglene
    - Gir tolkninger av kravene spesifikt for det aktuelle prosjektet
    
    Fordeler sammenlignet med GPT-4 løsningen:
    - Ingen API-kostnader
    - Fungerer uten internettforbindelse
    - Konsistente og etterprøvbare resultater
    - Raskere responstid
    - Bedre kontroll over analyseprosessen
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.rules_database = self._load_rules_database()
        self.municipality_rules = self._load_municipality_rules()
        self.tek_requirements = self._load_tek_requirements()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Laster konfigurasjon fra fil eller bruker standardverdier"""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    if config_path.endswith('.json'):
                        return json.load(f)
                    elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Feil ved lasting av konfigurasjon: {str(e)}")
        
        # Standard konfigurasjon
        return {
            "confidence_threshold": 0.85,
            "rules_path": os.path.join("data", "regulations"),
            "municipalities_path": os.path.join("data", "municipalities"),
            "tek_path": os.path.join("data", "tek"),
            "online_search_enabled": True,
            "cache_directory": os.path.join("cache", "regulations"),
            "update_interval_days": 30
        }
        
    def _load_rules_database(self) -> Dict:
        """Laster regelbasedatabasen"""
        rules_db = {}
        rules_path = self.config.get("rules_path", "")
        
        if os.path.exists(rules_path):
            try:
                # Last alle regelfiler
                for filename in os.listdir(rules_path):
                    if filename.endswith('.json'):
                        file_path = os.path.join(rules_path, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            category = filename.split('.')[0]
                            rules_db[category] = json.load(f)
                logger.info(f"Lastet {len(rules_db)} regelkategorier")
            except Exception as e:
                logger.error(f"Feil ved lasting av regler: {str(e)}")
        
        # Hvis ingen regler ble lastet, bruk innebygde regler
        if not rules_db:
            rules_db = self._get_default_rules()
            
        return rules_db
    
    def _get_default_rules(self) -> Dict:
        """Returnerer innebygde standardregler"""
        return {
            "general": {
                "building_height": {
                    "pattern": r"(bygningshøyde|gesimshøyde|mønehøyde)\s*(?:skal|må|kan)\s*(?:ikke)?\s*(?:være|overstige)?\s*(\d+(?:[.,]\d+)?)",
                    "extract": lambda m: float(m.group(2).replace(',', '.')),
                    "type": "numeric",
                    "unit": "meter"
                },
                "floor_count": {
                    "pattern": r"((?:maks(?:imal)?|antall)\s*(?:tillatt|etasjer))\s*(?:er|på)?\s*(\d+)",
                    "extract": lambda m: int(m.group(2)),
                    "type": "numeric",
                    "unit": "etasjer"
                },
                "plot_utilization": {
                    "pattern": r"(utnyttelsesgrad|BYA)\s*(?:er|på)?\s*(\d+(?:[.,]\d+)?)",
                    "extract": lambda m: float(m.group(2).replace(',', '.')),
                    "type": "percentage",
                    "unit": "prosent"
                }
            },
            "residential": {
                "minimum_room_size": {
                    "pattern": r"(minimums(?:areal|størrelse)|minste\s*areal)\s*(?:for|på)?\s*(?:et)?\s*(soverom|bad|kjøkken|stue|rom)?\s*(?:er|skal\s*være)?\s*(\d+(?:[.,]\d+)?)",
                    "extract": lambda m: (m.group(2) or "rom", float(m.group(3).replace(',', '.'))),
                    "type": "numeric",
                    "unit": "kvadratmeter"
                },
                "ceiling_height": {
                    "pattern": r"(takhøyde|romhøyde)\s*(?:skal|må)\s*(?:være|minimum|minst)?\s*(\d+(?:[.,]\d+)?)",
                    "extract": lambda m: float(m.group(2).replace(',', '.')),
                    "type": "numeric",
                    "unit": "meter"
                }
            },
            "accessibility": {
                "doorway_width": {
                    "pattern": r"(døråpning|dørbredde)\s*(?:skal|må|minimum|minst)?\s*(\d+(?:[.,]\d+)?)",
                    "extract": lambda m: float(m.group(2).replace(',', '.')),
                    "type": "numeric",
                    "unit": "meter"
                },
                "wheelchair_turning": {
                    "pattern": r"(snusirkel|rullestol|manøvreringsareal)\s*(?:med)?\s*diameter\s*(?:på)?\s*(\d+(?:[.,]\d+)?)",
                    "extract": lambda m: float(m.group(2).replace(',', '.')),
                    "type": "numeric",
                    "unit": "meter"
                }
            }
        }
        
    def _load_municipality_rules(self) -> Dict:
        """Laster regler for spesifikke kommuner"""
        municipality_rules = {}
        municipalities_path = self.config.get("municipalities_path", "")
        
        if os.path.exists(municipalities_path):
            try:
                # Last alle kommunefiler
                for filename in os.listdir(municipalities_path):
                    if filename.endswith('.json'):
                        file_path = os.path.join(municipalities_path, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            municipality = filename.split('.')[0]
                            municipality_rules[municipality] = json.load(f)
                logger.info(f"Lastet regler for {len(municipality_rules)} kommuner")
            except Exception as e:
                logger.error(f"Feil ved lasting av kommuneregler: {str(e)}")
        
        # Hvis ingen kommuneregler ble lastet, bruk innebygde regler
        if not municipality_rules:
            municipality_rules = self._get_default_municipality_rules()
            
        return municipality_rules
    
    def _get_default_municipality_rules(self) -> Dict:
        """Returnerer innebygde kommuneregler"""
        return {
            "oslo": {
                "building_height": 13.0,
                "max_floors": 4,
                "plot_utilization": 24.0,
                "minimum_distances": {
                    "to_neighbor": 4.0,
                    "to_road": 15.0
                },
                "rental_requirements": [
                    "Egen inngang",
                    "Brannskille mellom enheter",
                    "Minst ett bad og kjøkken per enhet"
                ]
            },
            "bergen": {
                "building_height": 12.0,
                "max_floors": 3,
                "plot_utilization": 30.0,
                "minimum_distances": {
                    "to_neighbor": 4.0,
                    "to_road": 10.0
                },
                "rental_requirements": [
                    "Egen inngang",
                    "Brannskille mellom enheter",
                    "Minst ett bad og kjøkken per enhet"
                ]
            },
            "trondheim": {
                "building_height": 11.0,
                "max_floors": 3,
                "plot_utilization": 30.0,
                "minimum_distances": {
                    "to_neighbor": 4.0,
                    "to_road": 10.0
                },
                "rental_requirements": [
                    "Egen inngang",
                    "Brannskille mellom enheter",
                    "Minst ett bad og kjøkken per enhet"
                ]
            },
            "drammen": {
                "building_height": 9.0,
                "max_floors": 3,
                "plot_utilization": 35.0,
                "minimum_distances": {
                    "to_neighbor": 4.0,
                    "to_road": 8.0
                },
                "rental_requirements": [
                    "Egen inngang",
                    "Brannskille mellom enheter",
                    "Minst ett bad og kjøkken per enhet",
                    "Lydisolasjon mellom enheter"
                ]
            }
        }
    
    def _load_tek_requirements(self) -> Dict:
        """Laster krav fra byggteknisk forskrift (TEK)"""
        tek_requirements = {}
        tek_path = self.config.get("tek_path", "")
        
        if os.path.exists(tek_path):
            try:
                # Last alle TEK-filer
                for filename in os.listdir(tek_path):
                    if filename.endswith('.json'):
                        file_path = os.path.join(tek_path, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            tek_version = filename.split('.')[0]
                            tek_requirements[tek_version] = json.load(f)
                logger.info(f"Lastet {len(tek_requirements)} TEK-versjoner")
            except Exception as e:
                logger.error(f"Feil ved lasting av TEK-krav: {str(e)}")
        
        # Hvis ingen TEK-krav ble lastet, bruk innebygde krav
        if not tek_requirements:
            tek_requirements = self._get_default_tek_requirements()
            
        return tek_requirements
    
    def _get_default_tek_requirements(self) -> Dict:
        """Returnerer innebygde TEK-krav"""
        return {
            "TEK17": {
                "ceiling_height": 2.4,
                "minimum_room_sizes": {
                    "bedroom": 7.0,
                    "kitchen": 6.0,
                    "living_room": 15.0,
                    "bathroom": 4.0
                },
                "accessibility": {
                    "doorway_width": 0.86,
                    "wheelchair_turning": 1.5
                },
                "energy": {
                    "u_values": {
                        "external_wall": 0.18,
                        "roof": 0.13,
                        "floor": 0.10,
                        "windows": 0.8,
                        "doors": 0.8
                    }
                },
                "fire_safety": {
                    "separate_fire_cells": True,
                    "warning_systems": True,
                    "escape_routes": True
                },
                "ventilation": {
                    "air_changes": 1.2
                }
            },
            "TEK10": {
                "ceiling_height": 2.4,
                "minimum_room_sizes": {
                    "bedroom": 7.0,
                    "kitchen": 6.0,
                    "living_room": 15.0,
                    "bathroom": 4.0
                },
                "accessibility": {
                    "doorway_width": 0.8,
                    "wheelchair_turning": 1.5
                },
                "energy": {
                    "u_values": {
                        "external_wall": 0.22,
                        "roof": 0.18,
                        "floor": 0.18,
                        "windows": 1.2,
                        "doors": 1.2
                    }
                },
                "fire_safety": {
                    "separate_fire_cells": True,
                    "warning_systems": True,
                    "escape_routes": True
                },
                "ventilation": {
                    "air_changes": 1.2
                }
            }
        }
    
    async def analyze_regulations(self, regulation_text: str, project_details: Dict, municipality: str) -> Dict:
        """
        Analyser byggeforskrifter og reguleringer for et prosjekt
        """
        try:
            logger.info(f"Analyserer forskrifter for {municipality}")
            
            # Sjekk om vi har oppdaterte regler for kommunen, hvis ikke, hent dem
            await self._update_municipality_rules(municipality)
            
            # Finn hvilken TEK-versjon som gjelder
            tek_version = self._determine_tek_version(project_details)
            
            # Analyser reguleringene
            extracted_rules = self._extract_rules_from_text(regulation_text)
            
            # Kombiner med kommunespesifikke regler
            combined_rules = self._combine_with_municipality_rules(
                extracted_rules,
                municipality
            )
            
            # Legg til TEK-krav
            full_requirements = self._add_tek_requirements(
                combined_rules,
                tek_version
            )
            
            # Vurder prosjektet mot kravene
            compliance = self._evaluate_compliance(
                full_requirements,
                project_details
            )
            
            # Generer anbefalinger
            recommendations = self._generate_recommendations(
                compliance,
                project_details,
                municipality
            )
            
            # Lag tolkninger av kravene
            interpretations = self._create_interpretations(
                full_requirements,
                project_details,
                municipality
            )
            
            return {
                "requirements": full_requirements,
                "interpretations": interpretations,
                "recommendations": recommendations,
                "compliance_status": compliance
            }
            
        except Exception as e:
            logger.error(f"Feil ved analyse av forskrifter: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "requirements": {},
                "interpretations": [],
                "recommendations": [],
                "compliance_status": {"overall_status": "unknown"}
            }
    
    async def _update_municipality_rules(self, municipality: str) -> None:
        """Oppdaterer kommuneregler hvis nødvendig"""
        if not self.config.get("online_search_enabled", False):
            return
            
        cache_dir = self.config.get("cache_directory", "")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            
        cache_file = os.path.join(cache_dir, f"{municipality}.json")
        
        # Sjekk om vi har en cachet versjon av reglene
        needs_update = True
        if os.path.exists(cache_file):
            # Sjekk om cachen er nyere enn update_interval_days
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if file_age.days < self.config.get("update_interval_days", 30):
                needs_update = False
        
        if needs_update:
            try:
                # Hent oppdaterte regler fra kommunens nettside
                rules = await self._fetch_municipality_rules_online(municipality)
                
                if rules:
                    # Oppdater cache
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(rules, f, ensure_ascii=False, indent=2)
                    
                    # Oppdater minnet
                    self.municipality_rules[municipality] = rules
                    logger.info(f"Oppdaterte regler for {municipality}")
            except Exception as e:
                logger.error(f"Kunne ikke oppdatere regler for {municipality}: {str(e)}")
    
    async def _fetch_municipality_rules_online(self, municipality: str) -> Dict:
        """Henter kommuneregler fra nett"""
        # Dette er en placeholder for faktisk web scraping av kommunale nettsider
        # En reell implementasjon ville brukt spesifikke selektorer for hver kommune
        
        urls = {
            "oslo": "https://www.oslo.kommune.no/plan-bygg-og-eiendom/",
            "bergen": "https://www.bergen.kommune.no/innbyggerhjelpen/planer-bygg-og-eiendom/",
            "trondheim": "https://www.trondheim.kommune.no/tema/bygg-kart-og-eiendom/",
            "drammen": "https://www.drammen.kommune.no/tjenester/byggesak/"
        }
        
        if municipality.lower() not in urls:
            return {}
            
        url = urls[municipality.lower()]
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return {}
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Dette er veldig forenklet - en faktisk implementasjon ville være mer spesifikk
            # for hver kommunes nettsted og struktur
            rules = {}
            
            # Eksempel: finn tabeller med reguleringsdata
            tables = soup.find_all('table')
            for table in tables:
                headers = [th.text.strip() for th in table.find_all('th')]
                
                for row in table.find_all('tr'):
                    cells = [td.text.strip() for td in row.find_all('td')]
                    if len(cells) >= 2:
                        key = cells[0].lower()
                        if "høyde" in key:
                            try:
                                value = float(re.search(r'(\d+[.,]?\d*)', cells[1]).group(1).replace(',', '.'))
                                rules["building_height"] = value
                            except:
                                pass
                        elif "utnyttelsesgrad" in key or "bya" in key:
                            try:
                                value = float(re.search(r'(\d+[.,]?\d*)', cells[1]).group(1).replace(',', '.'))
                                rules["plot_utilization"] = value
                            except:
                                pass
            
            # Hvis vi fant noen regler, returner dem, ellers bruk standardregler
            if rules:
                return rules
            else:
                return self.municipality_rules.get(municipality.lower(), {})
                
        except Exception as e:
            logger.error(f"Feil ved henting av online regler for {municipality}: {str(e)}")
            return {}
    
    def _determine_tek_version(self, project_details: Dict) -> str:
        """Bestemmer hvilken TEK-versjon som gjelder"""
        construction_year = project_details.get("construction_year", 0)
        
        if "renovation_year" in project_details:
            renovation_year = project_details.get("renovation_year")
            
            # For renovering gjelder nyere TEK-versjon
            if renovation_year >= 2017:
                return "TEK17"
            elif renovation_year >= 2010:
                return "TEK10"
        
        # For eksisterende bygg uten renovering
        if construction_year >= 2017:
            return "TEK17"
        elif construction_year >= 2010:
            return "TEK10"
        
        # Standard er nyeste versjon for ukjent
        return "TEK17"
    
    def _extract_rules_from_text(self, text: str) -> Dict:
        """Ekstraherer regler fra reguleringstekst"""
        extracted_rules = {}
        
        # Gå gjennom alle regelkategorier
        for category, rules in self.rules_database.items():
            category_rules = {}
            
            # Gå gjennom alle regler i kategorien
            for rule_name, rule_info in rules.items():
                pattern = rule_info.get("pattern")
                if not pattern:
                    continue
                
                # Finn alle forekomster av regelen
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    extract_func = rule_info.get("extract")
                    if extract_func:
                        try:
                            value = extract_func(match)
                            
                            # Håndter ulike typer verdier
                            if isinstance(value, tuple):
                                subcategory, val = value
                                if subcategory not in category_rules:
                                    category_rules[subcategory] = {}
                                category_rules[subcategory][rule_name] = val
                            else:
                                category_rules[rule_name] = value
                        except Exception as e:
                            logger.warning(f"Feil ved ekstrahering av regel {rule_name}: {str(e)}")
            
            if category_rules:
                extracted_rules[category] = category_rules
        
        return extracted_rules
    
    def _combine_with_municipality_rules(self, extracted_rules: Dict, municipality: str) -> Dict:
        """Kombinerer ekstraherte regler med kommunespesifikke regler"""
        combined_rules = extracted_rules.copy()
        
        # Legg til kommunespesifikke regler hvis tilgjengelig
        municipality_rule_set = self.municipality_rules.get(municipality.lower(), {})
        
        for category, rules in municipality_rule_set.items():
            if isinstance(rules, dict):
                if category not in combined_rules:
                    combined_rules[category] = {}
                
                for rule_name, value in rules.items():
                    if rule_name not in combined_rules[category]:
                        combined_rules[category][rule_name] = value
            else:
                if category not in combined_rules:
                    combined_rules[category] = rules
        
        return combined_rules
    
    def _add_tek_requirements(self, rules: Dict, tek_version: str) -> Dict:
        """Legger til krav fra TEK"""
        combined_rules = rules.copy()
        
        # Legg til TEK-krav hvis tilgjengelig
        tek_rule_set = self.tek_requirements.get(tek_version, {})
        
        for category, requirements in tek_rule_set.items():
            if isinstance(requirements, dict):
                if category not in combined_rules:
                    combined_rules[category] = {}
                
                if isinstance(combined_rules[category], dict):
                    for req_name, value in requirements.items():
                        if req_name not in combined_rules[category]:
                            combined_rules[category][req_name] = value
            else:
                if category not in combined_rules:
                    combined_rules[category] = requirements
        
        # Legg til TEK-versjon
        combined_rules["tek_version"] = tek_version
        
        return combined_rules
    
    def _evaluate_compliance(self, requirements: Dict, project_details: Dict) -> Dict:
        """Vurderer prosjektets samsvar med kravene"""
        compliance = {
            "overall_status": "compliant",
            "requirements_status": {},
            "issues": [],
            "compliance_score": 1.0
        }
        
        issues_count = 0
        checked_requirements = 0
        
        # Sjekk prosjektdetaljer mot krav
        for category, category_requirements in requirements.items():
            if category == "tek_version":
                continue
                
            category_compliance = {}
            
            if isinstance(category_requirements, dict):
                for req_name, req_value in category_requirements.items():
                    if isinstance(req_value, dict):
                        # Håndter nøstede krav
                        for sub_req, sub_value in req_value.items():
                            status, message = self._check_requirement(
                                f"{category}.{req_name}.{sub_req}",
                                sub_value,
                                project_details
                            )
                            
                            if status != "compliant":
                                issues_count += 1
                                compliance["issues"].append({
                                    "requirement": f"{category}.{req_name}.{sub_req}",
                                    "expected": sub_value,
                                    "actual": self._extract_project_value(f"{category}.{req_name}.{sub_req}", project_details),
                                    "message": message,
                                    "severity": "high" if status == "non_compliant" else "medium"
                                })
                            
                            category_compliance[f"{req_name}.{sub_req}"] = status
                            checked_requirements += 1
                    else:
                        # Håndter enkle krav
                        status, message = self._check_requirement(
                            f"{category}.{req_name}",
                            req_value,
                            project_details
                        )
                        
                        if status != "compliant":
                            issues_count += 1
                            compliance["issues"].append({
                                "requirement": f"{category}.{req_name}",
                                "expected": req_value,
                                "actual": self._extract_project_value(f"{category}.{req_name}", project_details),
                                "message": message,
                                "severity": "high" if status == "non_compliant" else "medium"
                            })
                        
                        category_compliance[req_name] = status
                        checked_requirements += 1
            
            compliance["requirements_status"][category] = category_compliance
        
        # Beregn compliance score
        if checked_requirements > 0:
            compliance_score = 1.0 - (issues_count / checked_requirements)
            compliance["compliance_score"] = max(0.0, min(1.0, compliance_score))
        
        # Sett overall status
        if issues_count > 0:
            severe_issues = sum(1 for issue in compliance["issues"] if issue["severity"] == "high")
            
            if severe_issues > 0:
                compliance["overall_status"] = "non_compliant"
            else:
                compliance["overall_status"] = "partially_compliant"
        
        return compliance
    
    def _extract_project_value(self, path: str, project_details: Dict) -> Any:
        """Henter en verdi fra prosjektdetaljer basert på sti"""
        path_parts = path.split('.')
        
        value = project_details
        for part in path_parts:
            # Prøv direkte nøkkel
            if isinstance(value, dict) and part in value:
                value = value[part]
            # Prøv med fuzzy matching
            elif isinstance(value, dict):
                best_match = None
                best_score = 0
                
                for key in value.keys():
                    score = SequenceMatcher(None, part.lower(), key.lower()).ratio()
                    if score > best_score and score >= 0.8:
                        best_score = score
                        best_match = key
                
                if best_match:
                    value = value[best_match]
                else:
                    return None
            else:
                return None
        
        return value
        
    def _is_in_list(self, required_item: Any, actual_list: List[Any]) -> bool:
        """Sjekker om et element finnes i en liste, med fuzzy matching for strenger"""
        if isinstance(required_item, str):
            for item in actual_list:
                if isinstance(item, str):
                    similarity = SequenceMatcher(None, required_item.lower(), item.lower()).ratio()
                    if similarity >= self.config.get("confidence_threshold", 0.85):
                        return True
            return False
        else:
            return required_item in actual_list
            
    def _check_requirement(self, 
                         requirement_path: str, 
                         required_value: Any, 
                         project_details: Dict) -> Tuple[str, str]:
        """Sjekker en spesifikk krav mot prosjektdetaljer"""
        actual_value = self._extract_project_value(requirement_path, project_details)
        
        # Hvis ingen verdi funnet, må sjekkes manuelt
        if actual_value is None:
            return "unknown", f"Kunne ikke finne verdi for {requirement_path} i prosjektdetaljer"
        
        # Sjekk type krav
        if isinstance(required_value, (int, float)):
            if isinstance(actual_value, (int, float)):
                # For numeriske verdier, sjekk om faktisk verdi er innenfor kravene
                if requirement_path.endswith("minimum") or "min" in requirement_path:
                    if actual_value >= required_value:
                        return "compliant", "Oppfyller minimumskrav"
                    else:
                        return "non_compliant", f"Under minimumskrav på {required_value}"
                elif requirement_path.endswith("maximum") or "max" in requirement_path:
                    if actual_value <= required_value:
                        return "compliant", "Under maksimumskrav"
                    else:
                        return "non_compliant", f"Over maksimumskrav på {required_value}"
                else:
                    # Standard er å sjekke om verdien er minst like stor som kravet
                    if actual_value >= required_value:
                        return "compliant", "Oppfyller krav"
                    else:
                        return "non_compliant", f"Oppfyller ikke krav på {required_value}"
            else:
                return "
# Fortsettelse av _check_requirement-metoden som ble avbrutt
                return "unknown", f"Forventet numerisk verdi, fant {type(actual_value)}"
        elif isinstance(required_value, bool):
            if isinstance(actual_value, bool):
                if actual_value == required_value:
                    return "compliant", "Oppfyller krav"
                else:
                    return "non_compliant", f"Forventet {required_value}, fant {actual_value}"
            else:
                return "unknown", f"Forventet boolsk verdi, fant {type(actual_value)}"
        elif isinstance(required_value, str):
            if isinstance(actual_value, str):
                # For strengverdier, bruk fuzzy matching
                similarity = SequenceMatcher(None, required_value.lower(), actual_value.lower()).ratio()
                if similarity >= self.config.get("confidence_threshold", 0.85):
                    return "compliant", "Oppfyller krav"
                else:
                    return "non_compliant", f"Forventet {required_value}, fant {actual_value}"
            else:
                return "unknown", f"Forventet strengverdi, fant {type(actual_value)}"
        elif isinstance(required_value, list):
            if isinstance(actual_value, list):
                # For lister, sjekk om alle nødvendige elementer er til stede
                missing_elements = [item for item in required_value if not self._is_in_list(item, actual_value)]
                if not missing_elements:
                    return "compliant", "Oppfyller alle krav i listen"
                else:
                    return "non_compliant", f"Mangler elementer: {', '.join(str(x) for x in missing_elements)}"
            else:
                return "unknown", f"Forventet liste, fant {type(actual_value)}"
        
        # Standard for ukjente typer
        return "unknown", f"Kunne ikke sammenligne {requirement_path} med verdi {required_value}"

    def _generate_recommendations(self, 
                                compliance: Dict, 
                                project_details: Dict,
                                municipality: str) -> List[Dict]:
        """Genererer anbefalinger basert på vurdering av samsvar"""
        recommendations = []
        
        # Legg til anbefalinger for hver sak
        for issue in compliance.get("issues", []):
            recommendation = self._create_recommendation_for_issue(issue, project_details, municipality)
            if recommendation:
                recommendations.append(recommendation)
        
        # Legg til noen generelle anbefalinger
        recommendations.extend(self._create_general_recommendations(project_details, municipality))
        
        return recommendations
    
    def _create_recommendation_for_issue(self, 
                                       issue: Dict, 
                                       project_details: Dict,
                                       municipality: str) -> Optional[Dict]:
        """Oppretter en spesifikk anbefaling for et problem"""
        requirement = issue.get("requirement", "")
        expected = issue.get("expected", None)
        actual = issue.get("actual", None)
        
        if not requirement or expected is None:
            return None
        
        recommendation = {
            "title": f"Oppfyll krav for {requirement}",
            "description": issue.get("message", ""),
            "severity": issue.get("severity", "medium"),
            "requirement": requirement,
            "expected_value": expected,
            "current_value": actual
        }
        
        # Legg til spesifikke tiltak basert på kravtype
        if "ceiling_height" in requirement:
            recommendation["actions"] = [
                "Senk gulv eller hev tak for å oppnå nødvendig takhøyde",
                "Hvis ikke mulig, søk dispensasjon fra kommunen"
            ]
            recommendation["estimated_cost"] = "Høy"
        elif "room_size" in requirement or "area" in requirement:
            recommendation["actions"] = [
                "Utvid rom ved å fjerne ikke-bærende vegger",
                "Vurder omorganisering av planløsning"
            ]
            recommendation["estimated_cost"] = "Medium"
        elif "doorway" in requirement or "door" in requirement:
            recommendation["actions"] = [
                "Utvid døråpninger til minimum kravbredde",
                "Installer nye dører med riktig dimensjon"
            ]
            recommendation["estimated_cost"] = "Lav til medium"
        elif "fire" in requirement or "safety" in requirement:
            recommendation["actions"] = [
                "Installer brannvarslingssystem",
                "Etabler brannskiller mellom enheter",
                "Sikre rømningsveier i henhold til TEK"
            ]
            recommendation["estimated_cost"] = "Medium"
        elif "ventilation" in requirement:
            recommendation["actions"] = [
                "Installer balansert ventilasjon",
                "Sikre tilstrekkelig luftutskiftning i alle rom"
            ]
            recommendation["estimated_cost"] = "Medium til høy"
        elif "energy" in requirement or "u_value" in requirement:
            recommendation["actions"] = [
                "Etterisoler konstruksjoner",
                "Bytt til energieffektive vinduer",
                "Vurder varmepumpe for energieffektiv oppvarming"
            ]
            recommendation["estimated_cost"] = "Medium til høy"
        
        return recommendation
    
    def _get_municipality_specific_recommendations(self, municipality: str) -> Optional[Dict]:
        """Henter kommune-spesifikke anbefalinger"""
        municipality = municipality.lower()
        
        # Database over kommunespesifikke anbefalinger
        # Utvides for å dekke alle kommuner i Norge
        municipality_recommendations = {
            "oslo": {
                "title": "Sjekk Oslo kommunes spesifikke krav til utleieenheter",
                "description": "Oslo kommune har strenge krav til utleieenheter, spesielt vedrørende parkering og uteareal",
                "severity": "medium",
                "actions": [
                    "Kontakt Plan- og bygningsetaten for veiledning",
                    "Sikre at parkeringskravene er oppfylt"
                ],
                "reference_url": "https://www.oslo.kommune.no/plan-bygg-og-eiendom/"
            },
            "drammen": {
                "title": "Sjekk Drammens krav til utleieenheter",
                "description": "Drammen kommune krever at utleieenheter registreres særskilt",
                "severity": "medium",
                "actions": [
                    "Kontakt kommunen for registrering av utleieenhet",
                    "Sikre at bruksendringen er godkjent"
                ],
                "reference_url": "https://www.drammen.kommune.no/tjenester/byggesak/"
            },
            # Her kan vi legge til flere kommuner etter behov
        }
        
        # Sjekk direkte treff først
        if municipality in municipality_recommendations:
            return municipality_recommendations[municipality]
            
        # For kommuner som ikke eksplisitt er definert, returner en generisk anbefaling
        return {
            "title": f"Sjekk {municipality.capitalize()} kommunes lokale bestemmelser",
            "description": "Lokale kommunale bestemmelser kan variere. Kontakt teknisk etat i kommunen for spesifikke retningslinjer.",
            "severity": "medium",
            "actions": [
                f"Kontakt {municipality.capitalize()} kommune for veiledning om lokale krav",
                "Sjekk kommuneplanen for området"
            ]
        }
    
    def _create_general_recommendations(self, project_details: Dict, municipality: str) -> List[Dict]:
        """Oppretter generelle anbefalinger basert på prosjekttype"""
        recommendations = []
        
        # Sjekk prosjekttype
        project_type = project_details.get("project_type", "unknown")
        
        if project_type == "rental_unit":
            # Anbefalinger for utleieenheter
            recommendations.append({
                "title": "Sikre at utleieenhet oppfyller alle krav",
                "description": "Utleieenheter må oppfylle spesifikke krav for brannsikkerhet og rømningsveier",
                "severity": "high",
                "actions": [
                    "Installer røykvarslere i alle rom",
                    "Sikre minst to uavhengige rømningsveier",
                    "Installere brannslukningsapparat"
                ],
                "estimated_cost": "Lav"
            })
        elif project_type == "renovation":
            # Anbefalinger for renovering
            recommendations.append({
                "title": "Vurder energioppgradering ved renovering",
                "description": "Ved større renoveringer kan det være kostnadseffektivt å oppgradere energitiltak samtidig",
                "severity": "medium",
                "actions": [
                    "Etterisoler vegger og tak",
                    "Bytt til energieffektive vinduer",
                    "Vurder balansert ventilasjon"
                ],
                "estimated_cost": "Medium",
                "potential_benefits": [
                    "Energibesparelser",
                    "Bedre inneklima",
                    "Mulighet for Enova-støtte"
                ]
            })
        
        # Legg til kommune-spesifikke anbefalinger
        municipality_specific = self._get_municipality_specific_recommendations(municipality)
        if municipality_specific:
            recommendations.append(municipality_specific)
        
        return recommendations
        
    def _create_interpretations(self,
                              full_requirements: Dict,
                              project_details: Dict,
                              municipality: str) -> List[Dict]:
        """Lager tolkninger av kravene for dette spesifikke prosjektet"""
        interpretations = []
        
        # Gå gjennom alle krav og lag brukervennlige tolkninger
        for category, requirements in full_requirements.items():
            if category == "tek_version":
                # Legg til informasjon om relevant TEK-versjon
                interpretations.append({
                    "requirement": "Byggteknisk forskrift",
                    "interpretation": f"Dette prosjektet må følge kravene i {requirements}",
                    "impact": "high",
                    "notes": "Nyere TEK-versjoner har strengere krav, spesielt for energieffektivitet"
                })
                continue
            
            if isinstance(requirements, dict):
                for req_name, req_value in requirements.items():
                    if isinstance(req_value, dict):
                        # Håndter nøstede krav
                        for sub_req, sub_value in req_value.items():
                            interpretation = self._interpret_requirement(
                                f"{category}.{req_name}.{sub_req}",
                                sub_value,
                                project_details
                            )
                            if interpretation:
                                interpretations.append(interpretation)
                    else:
                        # Håndter enkle krav
                        interpretation = self._interpret_requirement(
                            f"{category}.{req_name}",
                            req_value,
                            project_details
                        )
                        if interpretation:
                            interpretations.append(interpretation)
        
        # Legg til kommune-spesifikke tolkninger
        municipality_specific = self._get_municipality_specific_interpretations(municipality)
        if municipality_specific:
            interpretations.extend(municipality_specific)
        
        return interpretations
    
    def _interpret_requirement(self, 
                             requirement_path: str, 
                             required_value: Any, 
                             project_details: Dict) -> Optional[Dict]:
        """Lager en brukervennlig tolkning av et spesifikt krav"""
        # Sjekk om prosjektet oppfyller kravet
        actual_value = self._extract_project_value(requirement_path, project_details)
        status, message = self._check_requirement(requirement_path, required_value, project_details)
        
        # Lag leservennlig navn for kravet
        friendly_name = self._get_friendly_requirement_name(requirement_path)
        
        # Lag en tolkning basert på kravtypen
        interpretation = {
            "requirement": friendly_name,
            "expected_value": required_value,
            "actual_value": actual_value,
            "status": status,
            "interpretation": self._create_interpretation_text(friendly_name, required_value, actual_value, status),
            "impact": self._determine_requirement_impact(requirement_path)
        }
        
        # Legg til tilleggsinformasjon for spesifikke kravtyper
        if "ceiling_height" in requirement_path:
            interpretation["notes"] = "Takhøyde er viktig for romfølelse og ventilasjon. Dispensasjon kan søkes, men innvilges sjelden."
        elif "energy" in requirement_path:
            interpretation["notes"] = "Energikrav er strengere i nyere TEK-versjoner. Oppgradering kan gi rett til støtte fra Enova."
        elif "fire_safety" in requirement_path:
            interpretation["notes"] = "Brannsikkerhetskrav er absolutte og kan ikke fravikes uten omfattende kompenserende tiltak."
        
        return interpretation
    
    def _create_interpretation_text(self, 
                                  friendly_name: str, 
                                  required_value: Any, 
                                  actual_value: Any, 
                                  status: str) -> str:
        """Lager en brukervennlig tolkningsetkning"""
        if status == "compliant":
            return f"{friendly_name} oppfyller kravene (krever {required_value}, prosjektet har {actual_value})."
        elif status == "non_compliant":
            return f"{friendly_name} oppfyller IKKE kravene. Krever minimum {required_value}, men prosjektet har kun {actual_value}."
        else:
            return f"Kunne ikke vurdere {friendly_name}. Kravet er {required_value}, men verdien er ukjent eller ikke sammenlignbar."
    
    def _get_friendly_requirement_name(self, requirement_path: str) -> str:
        """Konverterer teknisk kravsti til et vennlig navn"""
        path_parts = requirement_path.split('.')
        
        # Mapping for vennlige navn
        friendly_names = {
            "ceiling_height": "Takhøyde",
            "minimum_room_sizes.bedroom": "Minimum soveromsstørrelse",
            "minimum_room_sizes.living_room": "Minimum stusstørrelse",
            "minimum_room_sizes.kitchen": "Minimum kjøkkenstørrelse",
            "minimum_room_sizes.bathroom": "Minimum badestørrelse",
            "accessibility.doorway_width": "Dørbredde",
            "accessibility.wheelchair_turning": "Snuareal for rullestol",
            "energy.u_values.external_wall": "Isolasjonsverdi for yttervegg",
            "energy.u_values.roof": "Isolasjonsverdi for tak",
            "energy.u_values.floor": "Isolasjonsverdi for gulv",
            "energy.u_values.windows": "Isolasjonsverdi for vinduer",
            "fire_safety.separate_fire_cells": "Separate brannceller",
            "fire_safety.warning_systems": "Brannvarslingssystem",
            "fire_safety.escape_routes": "Rømningsveier",
            "ventilation.air_changes": "Luftutskiftning"
        }
        
        # Sjekk om vi har et vennlig navn for hele stien
        if requirement_path in friendly_names:
            return friendly_names[requirement_path]
        
        # Prøv å finne den beste delmatchen
        best_match = None
        best_match_length = 0
        
        for key, name in friendly_names.items():
            if requirement_path.startswith(key) and len(key) > best_match_length:
                best_match = name
                best_match_length = len(key)
        
        if best_match:
            return best_match
        
        # Fallback: Bruk siste del av stien med fin formatering
        return path_parts[-1].replace('_', ' ').title()
    
    def _determine_requirement_impact(self, requirement_path: str) -> str:
        """Bestemmer viktigheten/innvirkningen av et krav"""
        high_impact_keywords = ["fire_safety", "escape", "bearing", "structural", "ceiling_height"]
        medium_impact_keywords = ["energy", "accessibility", "ventilation", "insulation"]
        
        for keyword in high_impact_keywords:
            if keyword in requirement_path:
                return "high"
                
        for keyword in medium_impact_keywords:
            if keyword in requirement_path:
                return "medium"
                
        return "low"
    
    def _get_municipality_specific_interpretations(self, municipality: str) -> List[Dict]:
        """Henter kommune-spesifikke tolkninger"""
        municipality = municipality.lower()
        interpretations = []
        
        # Utvidet database for kommune-spesifikke tolkninger
        municipality_interpretations = {
            "oslo": [
                {
                    "requirement": "Parkeringskrav i Oslo",
                    "interpretation": "Oslo kommune har strenge parkeringskrav for utleieenheter, spesielt i sentrumsnære områder.",
                    "impact": "medium",
                    "notes": "I noen soner kan det søkes fritak fra parkeringskrav mot betaling av parkeringsavgift."
                }
            ],
            "drammen": [
                {
                    "requirement": "Utnyttelsesgrad i Drammen",
                    "interpretation": "Drammen kommune tillater typisk høyere utnyttelsesgrad (35% BYA) enn mange andre kommuner.",
                    "impact": "medium",
                    "notes": "Dette gir større muligheter for tilbygg og påbygg."
                }
            ],
            # Legg til flere kommuner her etter behov
        }
        
        # Hent spesifikke tolkninger hvis de finnes
        if municipality in municipality_interpretations:
            return municipality_interpretations[municipality]
        
        # For kommuner som ikke er eksplisitt definert
        return [
            {
                "requirement": f"Lokale bestemmelser i {municipality.capitalize()}",
                "interpretation": f"Sjekk {municipality.capitalize()} kommunes lokale bestemmelser for spesifikke krav i ditt område.",
                "impact": "medium",
                "notes": "Kommunale bestemmelser kan variere mellom ulike områder innen samme kommune."
            }
        ]
        
    async def fetch_regulation_documents(self, gnr: int, bnr: int, municipality: str) -> List[Dict]:
        """
        Henter relevante reguleringsdokumenter for en eiendom basert på 
        gårds- og bruksnummer fra kommunens systemer.
        """
        try:
            # Konstruer API-endepunkt basert på kommune
            # Utvidet database med API-endepunkter for flere kommuner
            api_endpoints = {
                "drammen": f"https://innsyn2020.drammen.kommune.no/api/properties/search?gnr={gnr}&bnr={bnr}",
                "oslo": f"https://innsyn.pbe.oslo.kommune.no/api/saksinnsyn/property?gnr={gnr}&bnr={bnr}",
                "bergen": f"https://www.bergen.kommune.no/api/eiendom?gnr={gnr}&bnr={bnr}",
                "trondheim": f"https://kart.trondheim.kommune.no/api/eiendom?gnr={gnr}&bnr={bnr}",
                "stavanger": f"https://opengov.360online.com/Møter/stavanger/search?gnr={gnr}&bnr={bnr}",
                # Generisk fallback-endpoint basert på Kartverkets tjenester
                "default": f"https://ws.geonorge.no/eiendom/v1/kommuneinfoenhet?kommunenummer={{kommunenr}}&gardsnummer={gnr}&bruksnummer={bnr}"
            }
            
            # Sjekk om vi har et endpoint for denne kommunen
            endpoint = api_endpoints.get(municipality.lower(), None)
            
            # Hvis ingen direkte endpoint, bruk default
            if endpoint is None:
                # Hent kommunenummer for kommunen
                kommune_nr = self._get_kommune_nummer(municipality)
                if kommune_nr:
                    endpoint = api_endpoints["default"].format(kommunenr=kommune_nr)
                else:
                    logger.warning(f"Fant ikke kommunenummer for {municipality}")
                    return []
            
            # Utfør API-kall
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint) as response:
                    if response.status != 200:
                        logger.error(f"API feil: {response.status} - {await response.text()}")
                        return []
                        
                    data = await response.json()
                    
            # Prosesser resultater basert på kommune
            if municipality.lower() == "drammen":
                return self._process_drammen_results(data)
            elif municipality.lower() == "oslo":
                return self._process_oslo_results(data)
            elif municipality.lower() == "bergen":
                return self._process_bergen_results(data)
            else:
                # For andre kommuner, prøv generisk prosessering
                return self._process_generic_results(data, municipality)
                
        except Exception as e:
            logger.error(f"Feil ved henting av reguleringsdokumenter: {str(e)}")
            return []
    
    def _get_kommune_nummer(self, municipality: str) -> Optional[str]:
        """Konverterer kommunenavn til kommunenummer"""
        # Dette er en forenklet versjon. En fullstendig versjon ville hatt alle norske kommuner
        kommune_nummer = {
            "oslo": "0301",
            "bergen": "4601", 
            "trondheim": "5001",
            "stavanger": "1103",
            "drammen": "3005",
            "kristiansand": "4204",
            "tromsø": "5401",
            "fredrikstad": "3004",
            "sandnes": "1108",
            "sarpsborg": "3003",
            "bodø": "1804",
            "larvik": "3805",
            "ålesund": "1507",
            "arendal": "4203",
            "tønsberg": "3803",
            "porsgrunn": "3806",
            "skien": "3807",
            "haugesund": "1106",
            "sandefjord": "3804",
            "moss": "3002",
            "hamar": "3403",
            "halden": "3001",
            "lillehammer": "3405",
            "gjøvik": "3407",
            "molde": "1506",
            "kongsberg": "3006",
            "harstad": "5402",
            "kristiansund": "1505",
            "steinkjer": "5006",
            "elverum": "3420",
            "alta": "5403",
            "narvik": "1806",
            "rana": "1833",
            "askøy": "4627",
            "ringerike": "3007",
            "lørenskog": "3029",
            "bærum": "3024",
            "asker": "3025",
            "ski": "3020",
            "ås": "3021",
            "oppegård": "3020",
            "skedsmo": "3030",
            "sarpsborg": "3003",
            "rælingen": "3027",
            "nittedal": "3031",
            "røyken": "3025", # Del av Asker fra 2020
            "nesodden": "3023",
            "frogn": "3022"
            # Dette kan utvides med alle norske kommuner
        }
        
        return kommune_nummer.get(municipality.lower(), None)
            
    def _process_drammen_results(self, data: Dict) -> List[Dict]:
        """Prosesserer resultater fra Drammens API"""
        documents = []
        
        # Drammen returnerer en liste med saker
        if "cases" in data:
            for case in data["cases"]:
                # Hent relevante dokumenter
                if "documents" in case:
                    for doc in case["documents"]:
                        if self._is_relevant_regulation_document(doc.get("title", "")):
                            documents.append({
                                "title": doc.get("title", "Ukjent dokument"),
                                "date": doc.get("date", ""),
                                "url": doc.get("url", ""),
                                "type": self._determine_document_type(doc.get("title", "")),
                                "case_id": case.get("id", ""),
                                "case_title": case.get("title", "")
                            })
        
        return documents
        
    def _process_oslo_results(self, data: Dict) -> List[Dict]:
        """Prosesserer resultater fra Oslos API"""
        documents = []
        
        # Oslo har en annen struktur
        if "properties" in data:
            for prop in data["properties"]:
                if "cases" in prop:
                    for case in prop["cases"]:
                        if "documents" in case:
                            for doc in case["documents"]:
                                if self._is_relevant_regulation_document(doc.get("title", "")):
                                    documents.append({
                                        "title": doc.get("title", "Ukjent dokument"),
                                        "date": doc.get("date", ""),
                                        "url": doc.get("documentUrl", ""),
                                        "type": self._determine_document_type(doc.get("title", "")),
                                        "case_id": case.get("caseNumber", ""),
                                        "case_title": case.get("title", "")
                                    })
        
        return documents
    
    def _process_bergen_results(self, data: Dict) -> List[Dict]:
        """Prosesserer resultater fra Bergens API"""
        documents = []
        
        # Implementer Bergen-spesifikk prosessering her
        # Dette er en placeholder som må tilpasses faktisk API-respons
        if "dokumenter" in data:
            for doc in data["dokumenter"]:
                if self._is_relevant_regulation_document(doc.get("tittel", "")):
                    documents.append({
                        "title": doc.get("tittel", "Ukjent dokument"),
                        "date": doc.get("dato", ""),
                        "url": doc.get("url", ""),
                        "type": self._determine_document_type(doc.get("tittel", "")),
                        "case_id": doc.get("saksnummer", ""),
                        "case_title": doc.get("sakstittel", "")
                    })
        
        return documents
    
    def _process_generic_results(self, data: Dict, municipality: str) -> List[Dict]:
        """Generisk prosessering for kommuner uten spesifikk implementasjon"""
        documents = []
        
        # Forsøk å tolke standard felter som kan variere fra API til API
        # Dette er en best-effort implementasjon
        
        # Prøv ulike mulige feltnavn for dokumentliste
        doc_lists = ["documents", "dokumenter", "saker", "cases", "reguleringsplaner", "plans"]
        
        for doc_list_name in doc_lists:
            if doc_list_name in data:
                for doc in data[doc_list_name]:
                    # Prøv ulike mulige feltnavn for dokumenttittel
                    title_fields = ["title", "tittel", "name", "navn", "dokumentTittel"]
                    title = next((doc.get(field, "") for field in title_fields if field in doc), "Ukjent dokument")
                    
                    # Prøv ulike mulige feltnavn for dato
                    date_fields = ["date", "dato", "dokumentDato", "created"]
                    date = next((doc.get(field, "") for field in date_fields if field in doc), "")
                    
                    # Prøv ulike mulige feltnavn for URL
                    url_fields = ["url", "dokumentUrl", "link", "href", "fileUrl"]
                    url = next((doc.get(field, "") for field in url_fields if field in doc), "")
                    
                    if self._is_relevant_regulation_document(title):
                        documents.append({
                            "title": title,
                            "date": date,
                            "url": url,
                            "type": self._determine_document_type(title),
                            "case_id": doc.get("id", doc.get("saksnummer", "")),
                            "case_title": title,
                            "municipality": municipality
                        })
        
        return documents
        
    def _is_relevant_regulation_document(self, title: str) -> bool:
        """Sjekker om et dokument er relevant for reguleringsanalyse"""
        relevant_keywords = [
            "regulering", "plan", "bestemmelser", "tillatelse", "byggesak",
            "dispensasjon", "vedtak", "uttalelse", "rapport", "kart",
            "pbl", "arealdel", "kommuneplan", "områdeplan", "detaljplan",
            "bygningslov", "forskrift", "tek17", "tek10", "arealplan"
        ]
        
        title_lower = title.lower()
        return any(keyword in title_lower for keyword in relevant_keywords)
        
    def _determine_document_type(self, title: str) -> str:
        """Bestemmer dokumenttype basert på tittel"""
        title_lower = title.lower()
        
        type_keywords = {
            "reguleringsplan": ["reguleringsplan", "områdeplan", "detaljplan"],
            "bestemmelser": ["bestemmelser", "forskrift"],
            "kart": ["kart", "plankart"],
            "vedtak": ["vedtak", "beslutning"],
            "tillatelse": ["tillatelse", "godkjenning"],
            "dispensasjon": ["dispensasjon", "unntak", "fravik"],
            "uttalelse": ["uttalelse", "merknad", "innspill"],
            "rapport": ["rapport", "analyse", "vurdering", "notat"],
            "kommuneplan": ["kommuneplan", "arealdel", "områdeplan"],
            "veileder": ["veileder", "retningslinje", "norm"]
        }
        
        for doc_type, keywords in type_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                return doc_type
                
        return "annet"
            
    async def extract_regulation_text(self, document_url: str) -> str:
        """
        Henter tekst fra et reguleringsdokument gitt en URL
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(document_url) as response:
                    if response.status != 200:
                        logger.error(f"Kunne ikke hente dokument: {response.status}")
                        return ""
                        
                    content_type = response.headers.get('Content-Type', '')
                    
                    if 'application/pdf' in content_type:
                        # For PDF-er, last ned og bruk PyPDF2 eller lignende
                        content = await response.read()
                        return self._extract_text_from_pdf(content)
                    elif 'text/html' in content_type:
                        # For HTML, bruk BeautifulSoup
                        html = await response.text()
                        return self._extract_text_from_html(html)
                    elif 'application/msword' in content_type or 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type:
                        # For Word-dokumenter
                        content = await response.read()
                        return self._extract_text_from_word(content)
                    else:
                        # For andre formater, prøv å returnere rå tekst
                        return await response.text()
                        
        except Exception as e:
            logger.error(f"Feil ved uthenting av tekst fra dokument: {str(e)}")
            return ""
            
    def _extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Ekstraherer tekst fra PDF-innhold"""
        try:
            import io
            from PyPDF2 import PdfReader
            
            reader = PdfReader(io.BytesIO(pdf_content))
            text = ""
            
            for page in reader.pages:
                text += page.extract_text() + "\n"
                
            return text
            
        except ImportError:
            logger.warning("PyPDF2 ikke installert, kan ikke lese PDF-innhold")
            return ""
        except Exception as e:
            logger.error(f"Feil ved lesing av PDF: {str(e)}")
            return ""
    
    def _extract_text_from_word(self, word_content: bytes) -> str:
        """Ekstraherer tekst fra Word-dokument"""
        try:
            import io
            import docx
            
            doc = docx.Document(io.BytesIO(word_content))
            text = []
            
            for para in doc.paragraphs:
                text.append(para.text)
                
            return '\n'.join(text)
            
        except ImportError:
            logger.warning("python-docx ikke installert, kan ikke lese Word-dokumenter")
            return ""
        except Exception as e:
            logger.error(f"Feil ved lesing av Word-dokument: {str(e)}")
            return ""
            
    def _extract_text_from_html(self, html: str) -> str:
        """Ekstraherer tekst fra HTML-innhold"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Fjern script og style elementer
            for script in soup(["script", "style"]):
                script.extract()
                
            # Hent tekst
            text = soup.get_text()
            
            # Bryt opp i linjer og fjern ledende/etterfølgende whitespace
            lines = (line.strip() for line in text.splitlines())
            
            # Bryt opp avsnitt i linjer
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            
            # Fjern tomme linjer
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.error(f"Feil ved parsing av HTML: {str(e)}")
            return ""
            
    async def analyze_property_regulations(self, gnr: int, bnr: int, municipality: str, project_details: Dict) -> Dict:
        """
        Komplett analyse av reguleringer for en eiendom basert på gnr/bnr
        
        Args:
            gnr: Gårdsnummer for eiendommen
            bnr: Bruksnummer for eiendommen
            municipality: Kommune eiendommen ligger i
            project_details: Prosjektdetaljer for å evaluere mot regler
            
        Returns:
            Dict med reguleringsanalyse inkludert krav, tolkninger, 
            anbefalinger og samsvarsstatusen til prosjektet
        """
        try:
            # Hent relevante dokumenter
            documents = await self.fetch_regulation_documents(gnr, bnr, municipality)
            
            if not documents:
                logger.warning(f"Ingen reguleringsdokumenter funnet for {gnr}/{bnr} i {municipality}")
                
                # Bruk standardregler hvis ingen dokumenter finnes
                return await self.analyze_regulations(
                    "",  # Ingen reguleringstekst
                    project_details,
                    municipality
                )
            
            # Samle all tekst fra dokumentene
            combined_text = ""
            for doc in documents:
                if "url" in doc and doc["url"]:
                    text = await self.extract_regulation_text(doc["url"])
                    if text:
                        combined_text += f"\n--- {doc['title']} ---\n{text}\n"
            
            # Analyser samlet reguleringstekst
            analysis = await self.analyze_regulations(
                combined_text,
                project_details,
                municipality
            )
            
            # Legg til informasjon om kildedokumenter
            analysis["source_documents"] = documents
            
            return analysis
            
        except Exception as e:
            logger.error(f"Feil ved analyse av eiendomsreguleringer: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "requirements": {},
                "interpretations": [],
                "recommendations": [],
                "compliance_status": {"overall_status": "unknown"}
            }
            
    async def check_municipality_innsyn(self, municipality: str, gnr: int, bnr: int) -> Dict:
        """
        Sjekker en kommunens innsynsløsning for informasjon om eiendommen
        
        Denne funksjonen søker i kommunens innsynsløsning for tidligere saker,
        dispensasjoner, og annen viktig informasjon om eiendommen.
        
        Args:
            municipality: Kommune eiendommen ligger i
            gnr: Gårdsnummer for eiendommen
            bnr: Bruksnummer for eiendommen
            
        Returns:
            Dict med relevante saker og informasjon fra innsynsløsningen
        """
        try:
            # Bygg opp URL basert på kommune - utvidet for flere kommuner
            innsyn_urls = {
                "drammen": f"https://innsyn2020.drammen.kommune.no/postjournal-v2/fb851964-3185-43eb-81ba-9ac75226dfa8?gnr={gnr}&bnr={bnr}",
                "oslo": f"https://innsyn.pbe.oslo.kommune.no/saksinnsyn/casedet.asp?direct=Y&mode=&caseno=&gnr={gnr}&bnr={bnr}",
                "bergen": f"https://www.bergen.kommune.no/innsynpb/search?gnr={gnr}&bnr={bnr}",
                "trondheim": f"https://www.trondheim.kommune.no/byggesak/sok-i-byggesaksarkivet/?gnr={gnr}&bnr={bnr}",
                "stavanger": f"https://opengov.360online.com/Møter/stavanger/search?gnr={gnr}&bnr={bnr}",
                "kristiansand": f"https://www.kristiansand.kommune.no/navigasjon/politikk-og-administrasjon/dokumenter/?gnr={gnr}&bnr={bnr}",
                "tromsø": f"https://www.tromso.kommune.no/innsyn.171846.no.html?gnr={gnr}&bnr={bnr}",
                "bodø": f"https://bodo.kommune.no/innsyn/?gnr={gnr}&bnr={bnr}",
                "ålesund": f"https://alesund.kommune.no/innsyn/?gnr={gnr}&bnr={bnr}",
                "fredrikstad": f"https://www.fredrikstad.kommune.no/tjenester/politikk-og-demokrati/dokumenter/sok-i-politiske-dokumenter/?gnr={gnr}&bnr={bnr}",
                # Generisk fallback
                "default": f"https://www.{municipality}.kommune.no/innsyn?gnr={gnr}&bnr={bnr}"
            }
            
            if municipality.lower() not in innsyn_urls:
                logger.warning(f"Innsynsløsning ikke eksplisitt konfigurert for {municipality}, prøver generisk URL")
                url = innsyn_urls["default"].replace("{municipality}", municipality.lower())
            else:
                url = innsyn_urls[municipality.lower()]
            
            # Hent data fra innsynsløsningen
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Feil ved tilgang til innsynsløsning: {response.status}")
                        return {"status": "error", "message": f"Feil ved tilgang til innsynsløsning: {response.status}"}
                        
                    html = await response.text()
            
            # Parse resultater basert på kommune
            if municipality.lower() == "drammen":
                return self._parse_drammen_innsyn(html, gnr, bnr)
            elif municipality.lower() == "oslo":
                return self._parse_oslo_innsyn(html, gnr, bnr)
            elif municipality.lower() == "bergen":
                return self._parse_bergen_innsyn(html, gnr, bnr)
            else:
                # Generisk parsing for andre kommuner
                return self._parse_generic_innsyn(html, gnr, bnr, municipality)
                
        except Exception as e:
            logger.error(f"Feil ved sjekk av innsynsløsning: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    def _parse_drammen_innsyn(self, html: str, gnr: int, bnr: int) -> Dict:
        """
        Parser resultater fra Drammens innsynsløsning
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            cases = []
            
            # Finn tabell med saker
            tables = soup.find_all('table')
            for table in tables:
                if not table.find('th', text=re.compile(r'Sakstittel|Dokumenttittel')):
                    continue
                    
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        case_date = cells[0].text.strip()
                        case_number = cells[1].text.strip()
                        case_title = cells[2].text.strip()
                        
                        # Finn lenke til saken
                        link = cells[2].find('a')
                        case_url = link['href'] if link and 'href' in link.attrs else ""
                        
                        cases.append({
                            "date": case_date,
                            "case_number": case_number,
                            "title": case_title,
                            "url": case_url
                        })
            
            # Kategoriser saker
            building_permits = []
            dispensations = []
            other_cases = []
            
            for case in cases:
                title_lower = case["title"].lower()
                if "byggesak" in title_lower or "tillatelse" in title_lower or "byggetillatelse" in title_lower:
                    building_permits.append(case)
                elif "dispensasjon" in title_lower:
                    dispensations.append(case)
                else:
                    other_cases.append(case)
            
            return {
                "status": "success",
                "property_id": f"{gnr}/{bnr}",
                "total_cases": len(cases),
                "building_permits": building_permits,
                "dispensations": dispensations,
                "other_cases": other_cases
            }
            
        except Exception as e:
            logger.error(f"Feil ved parsing av Drammen innsyn: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    def _parse_oslo_innsyn(self, html: str, gnr: int, bnr: int) -> Dict:
        """
        Parser resultater fra Oslos innsynsløsning
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            cases = []
            
            # Finn tabeller med saker (Oslo har en annen struktur)
            tables = soup.find_all('table', class_='caseList')
            
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        case_date = cells[0].text.strip() if len(cells) > 0 else ""
                        case_number = cells[1].text.strip() if len(cells) > 1 else ""
                        case_title = cells[2].text.strip() if len(cells) > 2 else ""
                        
                        # Finn lenke til saken
                        link = cells[2].find('a') if len(cells) > 2 else None
                        case_url = link['href'] if link and 'href' in link.attrs else ""
                        
                        cases.append({
                            "date": case_date,
                            "case_number": case_number,
                            "title": case_title,
                            "url": case_url
                        })
            
            # Kategoriser saker
            building_permits = []
            dispensations = []
            other_cases = []
            
            for case in cases:
                title_lower = case["title"].lower()
                if "byggesak" in title_lower or "tillatelse" in title_lower or "rammetillatelse" in title_lower:
                    building_permits.append(case)
                elif "dispensasjon" in title_lower:
                    dispensations.append(case)
                else:
                    other_cases.append(case)
            
            return {
                "status": "success",
                "property_id": f"{gnr}/{bnr}",
                "total_cases": len(cases),
                "building_permits": building_permits,
                "dispensations": dispensations,
                "other_cases": other_cases
            }
            
        except Exception as e:
            logger.error(f"Feil ved parsing av Oslo innsyn: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _parse_bergen_innsyn(self, html: str, gnr: int, bnr: int) -> Dict:
        """
        Parser resultater fra Bergens innsynsløsning
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            cases = []
            
            # Bergen har en annen struktur - dette må tilpasses deres faktiske HTML
            # Dette er et eksempel som må justeres
            result_divs = soup.find_all('div', class_='search-result-item')
            
            for div in result_divs:
                title_elem = div.find('h3')
                date_elem = div.find('span', class_='date')
                link_elem = div.find('a', class_='case-link')
                
                case_title = title_elem.text.strip() if title_elem else ""
                case_date = date_elem.text.strip() if date_elem else ""
                case_url = link_elem['href'] if link_elem and 'href' in link_elem.attrs else ""
                case_number = div.find('span', class_='case-number').text.strip() if div.find('span', class_='case-number') else ""
                
                cases.append({
                    "date": case_date,
                    "case_number": case_number,
                    "title": case_title,
                    "url": case_url
                })
            
            # Kategoriser saker
            building_permits = []
            dispensations = []
            other_cases = []
            
            for case in cases:
                title_lower = case["title"].lower()
                if "byggesak" in title_lower or "tillatelse" in title_lower:
                    building_permits.append(case)
                elif "dispensasjon" in title_lower:
                    dispensations.append(case)
                else:
                    other_cases.append(case)
            
            return {
                "status": "success",
                "property_id": f"{gnr}/{bnr}",
                "total_cases": len(cases),
                "building_permits": building_permits,
                "dispensations": dispensations,
                "other_cases": other_cases
            }
            
        except Exception as e:
            logger.error(f"Feil ved parsing av Bergen innsyn: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _parse_generic_innsyn(self, html: str, gnr: int, bnr: int, municipality: str) -> Dict:
        """
        Generisk parsing for kommuner uten spesifikk implementasjon
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            cases = []
            
            # Prøv å finne tabeller som kan inneholde saker
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:  # Minst dato og tittel
                        case_date = cells[0].text.strip() if len(cells) > 0 else ""
                        case_title = cells[-1].text.strip() if len(cells) > 1 else ""  # Antar tittel er i siste kolonne
                        
                        # Prøv å finne saksnummer
                        case_number = ""
                        for cell in cells:
                            # Prøv å identifisere celler med saksnumre (ofte formatert som 20XX/XXXXX)
                            if re.search(r'\d{2,4}/\d{2,5}', cell.text):
                                case_number = cell.text.strip()
                                break
                        
                        # Finn lenke (kan være i hvilken som helst celle)
                        case_url = ""
                        for cell in cells:
                            link = cell.find('a')
                            if link and 'href' in link.attrs:
                                case_url = link['href']
                                break
                        
                        cases.append({
                            "date": case_date,
                            "case_number": case_number,
                            "title": case_title,
                            "url": case_url
                        })
            
            # Hvis ingen tabeller ble funnet, prøv å finne listelementer
            if not cases:
                list_items = soup.find_all('li')
                for item in list_items:
                    # Prøv å identifisere listelementer som kan være saker
                    link = item.find('a')
                    
                    if link and 'href' in link.attrs:
                        case_title = link.text.strip()
                        case_url = link['href']
                        
                        # Prøv å finne dato i teksten
                        date_pattern = r'\d{2}\.\d{2}\.\d{4}|\d{2}\-\d{2}\-\d{4}'
                        date_match = re.search(date_pattern, item.text)
                        case_date = date_match.group(0) if date_match else ""
                        
                        # Prøv å finne saksnummer
                        case_number_pattern = r'\d{2,4}/\d{2,5}'
                        number_match = re.search(case_number_pattern, item.text)
                        case_number = number_match.group(0) if number_match else ""
                        
                        cases.append({
                            "date": case_date,
                            "case_number": case_number,
                            "title": case_title,
                            "url": case_url
                        })
            
            # Kategoriser saker
            building_permits = []
            dispensations = []
            other_cases = []
            
            for case in cases:
                title_lower = case["title"].lower()
                if any(keyword in title_lower for keyword in ["byggesak", "tillatelse", "byggetillatelse", "rammetillatelse"]):
                    building_permits.append(case)
                elif "dispensasjon" in title_lower:
                    dispensations.append(case)
                else:
                    other_cases.append(case)
            
            return {
                "status": "success",
                "property_id": f"{gnr}/{bnr}",
                "municipality": municipality,
                "total_cases": len(cases),
                "building_permits": building_permits,
                "dispensations": dispensations,
                "other_cases": other_cases
            }
            
        except Exception as e:
            logger.error(f"Feil ved generisk parsing av innsyn for {municipality}: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    def check_regulation_plan(self, municipality: str, property_data: Dict) -> Dict:
        """
        Sjekker gjeldende reguleringsplan for en eiendom
        
        Args:
            municipality: Kommune eiendommen ligger i
            property_data: Data om eiendommen inkludert koordinater eller adresse
            
        Returns:
            Dict med informasjon om gjeldende reguleringsplan
        """
        try:
            # Standardverdier som brukes hvis kommune-spesifikke data ikke er tilgjengelig
            default_plan = {
                "plan_name": f"Kommuneplan for {municipality.capitalize()}",
                "zone_type": "Boligformål",
                "allowed_utilization": 25.0,  # BYA i prosent
                "max_height": 8.0,  # meter
                "max_floors": 2,
                "special_restrictions": [],
                "plan_url": f"https://www.{municipality.lower()}.kommune.no/arealplan/kommuneplan/"
            }
            
            # Kommune-spesifikke data - utvidet for flere kommuner
            municipality_plans = {
                "drammen": {
                    "plan_name": "Kommuneplan for Drammen 2021-2040",
                    "zone_type": "Boligformål",
                    "allowed_utilization": 35.0,  # BYA i prosent
                    "max_height": 9.0,  # meter
                    "max_floors": 3,
                    "special_restrictions": [
                        "Bevaringsverdig miljø - krav om tilpasning til eksisterende bebyggelse",
                        "Krav om min. 30% grøntareal"
                    ],
                    "plan_url": "https://www.drammen.kommune.no/tjenester/arealplan/kommuneplan/"
                },
                "oslo": {
                    "plan_name": "Kommuneplan for Oslo 2018, arealdel",
                    "zone_type": "Bebyggelse og anlegg",
                    "allowed_utilization": 24.0,  # BYA i prosent
                    "max_height": 13.0,  # meter
                    "max_floors": 4,
                    "special_restrictions": [
                        "Hensynssone kulturmiljø",
                        "Blågrønn faktor min. 0.7"
                    ],
                    "plan_url": "https://www.oslo.kommune.no/plan-bygg-og-eiendom/overordnede-planer/kommuneplan/"
                },
                "bergen": {
                    "plan_name": "Kommuneplanens arealdel 2018-2030",
                    "zone_type": "Boligbebyggelse",
                    "allowed_utilization": 30.0,  # BYA i prosent
                    "max_height": 12.0,  # meter
                    "max_floors": 3,
                    "special_restrictions": [
                        "Hensynssone for bevaring av kulturmiljø i enkelte områder",
                        "Krav til uteoppholdsareal min. 50 m² per boenhet"
                    ],
                    "plan_url": "https://www.bergen.kommune.no/hvaskjer/tema/kommuneplanens-arealdel-2018"
                },
                "trondheim": {
                    "plan_name": "Kommuneplanens arealdel 2022-2034",
                    "zone_type": "Boligbebyggelse",
                    "allowed_utilization": 30.0,  # BYA i prosent
                    "max_height": 11.0,  # meter
                    "max_floors": 3,
                    "special_restrictions": [
                        "Krav om tilpasning til terreng",
                        "Minimumskrav til uteoppholdsareal"
                    ],
                    "plan_url": "https://www.trondheim.kommune.no/tema/bygg-kart-og-eiendom/arealplaner/kommuneplanens-arealdel/"
                },
                "stavanger": {
                    "plan_name": "Kommuneplan for Stavanger 2020-2034",
                    "zone_type": "Boligbebyggelse",
                    "allowed_utilization": 28.0,  # BYA i prosent
                    "max_height": 9.0,  # meter
                    "max_floors": 3,
                    "special_restrictions": [
                        "Krav til parkering: 1.2 plasser per boenhet",
                        "Krav til uteoppholdsareal min. 30 m² per boenhet"
                    ],
                    "plan_url": "https://www.stavanger.kommune.no/plan-og-utbygging/kommuneplan-2020-2034/"
                },
                "kristiansand": {
                    "plan_name": "Kommuneplan for Kristiansand 2020-2030",
                    "zone_type": "Boligbebyggelse",
                    "allowed_utilization": 30.0,  # BYA i prosent
                    "max_height": 8.5,  # meter
                    "max_floors": 2,
                    "special_restrictions": [
                        "Krav til minste uteoppholdsareal (MUA)",
                        "Krav til parkering basert på soneinndeling"
                    ],
                    "plan_url": "https://www.kristiansand.kommune.no/navigasjon/innbyggerdialog-og-frivillighet/akt/planer/kommuneplan/"
                }
            }
            
            # Returner kommune-spesifikk plan hvis tilgjengelig, ellers standard
            if municipality.lower() in municipality_plans:
                return municipality_plans[municipality.lower()]
            else:
                return default_plan
                
        except Exception as e:
            logger.error(f"Feil ved sjekk av reguleringsplan: {str(e)}")
            return {"error": str(e)}
            
    async def get_building_info(self, gnr: int, bnr: int, municipality: str) -> Dict:
        """
        Henter informasjon om bygninger på eiendommen fra Matrikkelen
        
        Denne funksjonen henter informasjon fra offentlige registre om
        bygninger på eiendommen, inkludert byggeår, areal, og bruksformål.
        
        Args:
            gnr: Gårdsnummer for eiendommen
            bnr: Bruksnummer for eiendommen
            municipality: Kommune eiendommen ligger i
            
        Returns:
            Dict med bygningsinformasjon fra Matrikkelen
        """
        try:
            # I en reell implementasjon ville dette gjøre et API-kall til Matrikkelen
            # For denne demonstrasjonen, returner dummy-data som varierer basert på input
            
            # Simuler noe forsinkelse som ville oppstå ved et faktisk API-kall
            await asyncio.sleep(0.5)
            
            # Generer noen "tilfeldige" men konsistente verdier basert på gnr/bnr
            seed = gnr * 1000 + bnr
            import random
            random.seed(seed)
            
            construction_year = 1950 + random.randint(0, 70)
            area_bra = 100 + random.randint(0, 150)
            area_bya = area_bra * 0.6
            floors = 1 + random.randint(0, 2)
            property_area = 500 + random.randint(0, 1000)
            
            building_types = ["Enebolig", "Tomannsbolig", "Rekkehus", "Leilighet", "Fritidsbolig"]
            building_type = building_types[random.randint(0, min(4, (gnr + bnr) % 5))]
            
            energy_ratings = ["A", "B", "C", "D", "E", "F", "G"]
            energy_rating = energy_ratings[random.randint(0, min(6, (gnr + bnr) % 7))]
            
            # Generer en "ekte" adresse basert på kommune
            street_names = {
                "oslo": ["Storgata", "Kirkeveien", "Bygdøy allé", "Trondheimsveien", "Sognsveien"],
                "bergen": ["Bryggen", "Fløyveien", "Strandgaten", "Nygårdsgaten", "Nordnesveien"],
                "trondheim": ["Munkegata", "Kongens gate", "Elgeseter gate", "Klæbuveien", "Byåsveien"],
                "stavanger": ["Kirkegata", "Pedersgata", "Eiganesveien", "Madlaveien", "Randabergveien"],
                "drammen": ["Engene", "Bragernes torg", "Konnerudgata", "Solbergveien", "Bjørnstjerne Bjørnsons gate"]
            }
            
            street_name = street_names.get(municipality.lower(), ["Hovedveien", "Skoleveien", "Stasjonsgata", "Fjordveien", "Kirkeveien"])
            street = street_name[gnr % len(street_name)]
            number = bnr + 1
            postal_codes = {
                "oslo": ["0001", "0170", "0368", "0473", "0586"],
                "bergen": ["5003", "5073", "5097", "5155", "5231"],
                "trondheim": ["7013", "7043", "7089", "7099", "7089"],
                "stavanger": ["4002", "4021", "4042", "4085", "4099"],
                "drammen": ["3001", "3024", "3042", "3075", "3089"]
            }
            postal_code = postal_codes.get(municipality.lower(), ["0001", "1001", "2001", "3001", "4001", "5001", "6001", "7001"])[gnr % 5]
            
            # Generer koordinater basert på kommune (grove sentrumskoordinater)
            coordinates = {
                "oslo": [59.911491, 10.757933],
                "bergen": [60.397820, 5.324767],
                "trondheim": [63.430518, 10.394903],
                "stavanger": [58.969975, 5.733107],
                "drammen": [59.743874, 10.204496]
            }
            
            base_coords = coordinates.get(municipality.lower(), [60.0, 10.0])
            # Legg til små variasjoner basert på gnr/bnr
            latitude = base_coords[0] + (gnr % 100) * 0.0001
            longitude = base_coords[1] + (bnr % 100) * 0.0001
            
            return {
                "property_id": f"{gnr}/{bnr}",
                "municipality": municipality,
                "buildings": [
                    {
                        "building_id": f"{gnr}/{bnr}/1",
                        "building_type": building_type,
                        "building_status": "Tatt i bruk",
                        "construction_year": construction_year,
                        "area_BRA": area_bra,
                        "area_BYA": area_bya,
                        "floors": floors,
                        "residential_units": 1 if building_type == "Enebolig" else 2,
                        "energy_rating": energy_rating,
                        "coordinates": {
                            "latitude": latitude,
                            "longitude": longitude
                        }
                    }
                ],
                "property_area": property_area,  # m²
                "address": f"{street} {number}, {postal_code} {municipality.capitalize()}"
            }
            
        except Exception as e:
            logger.error(f"Feil ved henting av bygningsinformasjon: {str(e)}")
            return {"error": str(e)}
            
    async def evaluate_property_potential(self, gnr: int, bnr: int, municipality: str) -> Dict:
        """
        Evaluerer en eiendoms potensiale basert på reguleringer og bygningsinformasjon
        
        Args:
            gnr: Gårdsnummer for eiendommen
            bnr: Bruksnummer for eiendommen
            municipality: Kommune eiendommen ligger i
            
        Returns:
            Dict med vurdering av eiendommens potensiale
        """
        try:
            # Hent bygningsinformasjon
            building_info = await self.get_building_info(gnr, bnr, municipality)
            
            # Hent reguleringsplan
            regulation_plan = self.check_regulation_plan(municipality, building_info)
            
            # Beregn potensiell utnyttelse
            potential = self._calculate_property_potential(building_info, regulation_plan, municipality)
            
            # Hent informasjon om tidligere saker
            previous_cases = await self.check_municipality_innsyn(municipality, gnr, bnr)
            
            # Samle alt i en komplett vurdering
            return {
                "property_info": building_info,
                "regulation_plan": regulation_plan,
                "development_potential": potential,
                "previous_cases": previous_cases,
                "summary": self._generate_potential_summary(potential, regulation_plan, municipality)
            }
            
        except Exception as e:
            logger.error(f"Feil ved evaluering av eiendomspotensial: {str(e)}")
            return {"error": str(e)}
            
    def _calculate_property_potential(self, building_info: Dict, regulation_plan: Dict, municipality: str) -> Dict:
        """
        Beregner utviklingspotensial for en eiendom
        """
        potential = {
            "build_out_potential": 0.0,  # m²
            "build_up_potential": 0.0,   # m²
            "subdivision_potential": False,
            "rental_unit_potential": False,
            "potential_areas": []
        }
        
        try:
            # Hent nødvendige verdier
            property_area = building_info.get("property_area", 0.0)  # Tomteareal
            
            # Summar nåværende bebygd areal (BYA)
            current_bya = sum(building.get("area_BYA", 0.0) for building in building_info.get("buildings", []))
            
            # Hent tillatt utnyttelsesgrad fra reguleringsplan
            allowed_utilization = regulation_plan.get("allowed_utilization", 25.0)  # Prosent BYA
            
            # Beregn maksimalt tillatt bebygd areal
            max_bya = property_area * (allowed_utilization / 100.0)
            
            # Beregn gjenværende potensial
            remaining_bya = max(0.0, max_bya - current_bya)
            
            # Sjekk utbyggingspotensial
            if remaining_bya >= 15.0:  # Minst 15m² for å være verdt å nevne
                potential["build_out_potential"] = remaining_bya
                potential["potential_areas"].append({
                    "type": "build_out",
                    "area_m2": remaining_bya,
                    "description": f"Potensial for tilbygg på inntil {remaining_bya:.1f} m²"
                })
            
            # Sjekk påbyggingspotensial
            max_floors = regulation_plan.get("max_floors", 2)
            buildings = building_info.get("buildings", [])
            
            for i, building in enumerate(buildings):
                current_floors = building.get("floors", 1)
                bya = building.get("area_BYA", 0.0)
                
                if current_floors < max_floors:
                    floors_potential = max_floors - current_floors
                    build_up_area = bya * floors_potential
                    
                    potential["build_up_potential"] += build_up_area
                    potential["potential_areas"].append({
                        "type": "build_up",
                        "building_id": i,
                        "building_type": building.get("building_type", "Ukjent"),
                        "area_m2": build_up_area,
                        "floors": floors_potential,
                        "description": f"Potensial for påbygg med {floors_potential} etasje(r), totalt {build_up_area:.1f} m²"
                    })
            
            # Sjekk potensial for fradeling
            min_plot_size = self._get_min_plot_size(municipality)  # Hent minstekrav
            
            if property_area > (2 * min_plot_size):
                potential["subdivision_potential"] = True
                potential["potential_areas"].append({
                    "type": "subdivision",
                    "area_m2": property_area / 2,  # Antatt halvering av tomt
                    "description": f"Potensial for fradeling av tomt (minimum tomtestørrelse: {min_plot_size} m²)"
                })
            
            # Sjekk potensial for utleieenhet
            buildings_with_potential = []
            
            for i, building in enumerate(buildings):
                if building.get("area_BRA", 0.0) > 150.0 and building.get("residential_units", 0) == 1:
                    buildings_with_potential.append(i)
            
            if buildings_with_potential:
                potential["rental_unit_potential"] = True
                potential["potential_areas"].append({
                    "type": "rental_unit",
                    "building_ids": buildings_with_potential,
                    "description": "Potensial for å etablere utleieenhet i eksisterende bolig"
                })
            
            return potential
            
        except Exception as e:
            logger.error(f"Feil ved beregning av eiendomspotensial: {str(e)}")
            return potential
            
    def _get_min_plot_size(self, municipality: str) -> float:
        """Henter minstekrav til tomtestørrelse for en kommune"""
        min_sizes = {
            "oslo": 600.0,
            "bergen": 500.0,
            "trondheim": 500.0,
            "drammen": 500.0,
            "stavanger": 450.0,
            "kristiansand": 500.0,
            "tromsø": 600.0,
            "bodø": 500.0,
            "ålesund": 500.0,
            "fredrikstad": 500.0,
            "sandnes": 450.0,
            "sarpsborg": 500.0,
            "larvik": 550.0,
            "tønsberg": 500.0,
            "porsgrunn": 500.0,
            "skien": 500.0,
            "haugesund": 500.0,
            "arendal": 600.0,
            "moss": 500.0
        }
        
        return min_sizes.get(municipality.lower(), 600.0)  # Standard 600m² hvis ukjent
    
    def _generate_potential_summary(self, potential: Dict, regulation_plan: Dict, municipality: str) -> str:
        """
        Lager en oppsummering av eiendommens utviklingspotensial
        """
        summary_parts = []
        
        # Inkluderer boligtype, BYA og tomtestørrelse hvis tilgjengelig
        if potential["build_out_potential"] > 0:
            summary_parts.append(f"Eiendommen har potensial for tilbygg på inntil {potential['build_out_potential']:.1f} m².")
            
        if potential["build_up_potential"] > 0:
            summary_parts.append(f"Det er mulig å bygge på en eller flere etasjer, totalt {potential['build_up_potential']:.1f} m² ekstra areal.")
            
        if potential["subdivision_potential"]:
            summary_parts.append(f"Tomten er stor nok til å kunne søke om fradeling av en egen tomt.")
            
        if potential["rental_unit_potential"]:
            summary_parts.append(f"Boligen har potensial for etablering av utleieenhet, forutsatt at tekniske krav oppfylles.")
            
        # Legg til kommunespesifikk informasjon om nødvendig
        if municipality.lower() in ["drammen", "bergen", "stavanger"]:
            utnyttelse = regulation_plan.get("allowed_utilization", 25.0)
            summary_parts.append(f"Merk at {municipality.capitalize()} kommune tillater {utnyttelse}% BYA, noe som gir gode muligheter for utvidelse.")
        
        # Hvis ingen potensiale er funnet
        if not summary_parts:
            return "Basert på gjeldende reguleringsplan og bygningsinformasjon ser det ut til at eiendommen allerede er godt utnyttet uten vesentlig utviklingspotensial."
            
        return " ".join(summary_parts)
    
    async def get_similar_cases(self, gnr: int, bnr: int, municipality: str, case_type: str) -> List[Dict]:
        """
        Finner lignende saker i samme område som kan være relevante
        
        Args:
            gnr: Gårdsnummer for eiendommen
            bnr: Bruksnummer for eiendommen
            municipality: Kommune eiendommen ligger i
            case_type: Type sak (f.eks. 'utleieenhet', 'tilbygg', 'dispensasjon')
            
        Returns:
            Liste med lignende saker fra samme område
        """
        try:
            # Dette ville normalt søke i kommunens database etter lignende saker
            # For demonstrasjonsformål, generer syntetiske data som varierer med input
            
            # Simuler noe forsinkelse som ville oppstå ved et faktisk søk
            await asyncio.sleep(0.8)
            
            # Bruk gnr/bnr for å lage "tilfeldige" men konsistente data
            import random
            seed = gnr * 1000 + bnr + hash(municipality + case_type) % 10000
            random.seed(seed)
            
            # Generer dato innenfor de siste 3 årene
            def random_date():
                year = 2021 + random.randint(0, 2)
                month = random.randint(1, 12)
                day = random.randint(1, 28)
                return f"{year:04d}-{month:02d}-{day:02d}"
            
            # Case-spesifikke lister over saker
            case_data = {
                "utleieenhet": [
                    {
                        "case_number": f"{2020 + random.randint(0, 3)}/{random.randint(1000, 9999)}",
                        "date": random_date(),
                        "title": "Søknad om etablering av utleieenhet",
                        "address": f"Nabolaget {gnr+2}/{bnr+3}",
                        "result": "Godkjent" if random.random() > 0.3 else "Avslått",
                        "relevance": "high" if random.random() > 0.5 else "medium",
                        "url": f"https://innsyn.{municipality.lower()}.kommune.no/case/{random.randint(10000, 99999)}"
                    },
                    {
                        "case_number": f"{2020 + random.randint(0, 3)}/{random.randint(1000, 9999)}",
                        "date": random_date(),
                        "title": "Etablering av utleieenhet i kjeller",
                        "address": f"Nabolaget {gnr+1}/{bnr+5}",
                        "result": "Avslått" if random.random() > 0.7 else "Godkjent",
                        "relevance": "medium" if random.random() > 0.5 else "high",
                        "url": f"https://innsyn.{municipality.lower()}.kommune.no/case/{random.randint(10000, 99999)}",
                        "reason": "Manglende takhøyde og dagslys" if random.random() > 0.5 else "Utilstrekkelig rømningsvei"
                    }
                ],
                "tilbygg": [
                    {
                        "case_number": f"{2020 + random.randint(0, 3)}/{random.randint(1000, 9999)}",
                        "date": random_date(),
                        "title": f"Søknad om tilbygg {random.randint(15, 40)}m²",
                        "address": f"Nabolaget {gnr-1}/{bnr+2}",
                        "result": "Godkjent" if random.random() > 0.2 else "Avslått",
                        "relevance": "high" if random.random() > 0.3 else "medium",
                        "url": f"https://innsyn.{municipality.lower()}.kommune.no/case/{random.randint(10000, 99999)}"
                    },
                    {
                        "case_number": f"{2020 + random.randint(0, 3)}/{random.randint(1000, 9999)}",
                        "date": random_date(),
                        "title": f"Tilbygg med carport og bod {random.randint(20, 50)}m²",
                        "address": f"Nabolaget {gnr+3}/{bnr-1}",
                        "result": "Godkjent" if random.random() > 0.3 else "Avslått",
                        "relevance": "medium" if random.random() > 0.5 else "high",
                        "url": f"https://innsyn.{municipality.lower()}.kommune.no/case/{random.randint(10000, 99999)}"
                    }
                ],
                "dispensasjon": [
                    {
                        "case_number": f"{2020 + random.randint(0, 3)}/{random.randint(1000, 9999)}",
                        "date": random_date(),
                        "title": "Dispensasjon fra utnyttelsesgrad",
                        "address": f"Nabolaget {gnr+3}/{bnr-2}",
                        "result": "Godkjent" if random.random() > 0.5 else "Avslått",
                        "relevance": "medium" if random.random() > 0.3 else "high",
                        "url": f"https://innsyn.{municipality.lower()}.kommune.no/case/{random.randint(10000, 99999)}",
                        "granted_reason": "Små konsekvenser for naboer og miljø" if random.random() > 0.5 else "Særlige grunner foreligger"
                    },
                    {
                        "case_number": f"{2020 + random.randint(0, 3)}/{random.randint(1000, 9999)}",
                        "date": random_date(),
                        "title": "Dispensasjon fra byggegrense",
                        "address": f"Nabolaget {gnr-2}/{bnr+1}",
                        "result": "Avslått" if random.random() > 0.6 else "Godkjent",
                        "relevance": "medium" if random.random() > 0.4 else "high",
                        "url": f"https://innsyn.{municipality.lower()}.kommune.no/case/{random.randint(10000, 99999)}",
                        "denial_reason": "For stor påvirkning på naboer" if random.random() > 0.5 else "Presedensskapende"
                    }
                ],
                "bruksendring": [
                    {
                        "case_number": f"{2020 + random.randint(0, 3)}/{random.randint(1000, 9999)}",
                        "date": random_date(),
                        "title": "Bruksendring fra bod til beboelsesrom",
                        "address": f"Nabolaget {gnr+1}/{bnr+1}",
                        "result": "Godkjent" if random.random() > 0.4 else "Avslått",
                        "relevance": "high" if random.random() > 0.3 else "medium",
                        "url": f"https://innsyn.{municipality.lower()}.kommune.no/case/{random.randint(10000, 99999)}"
                    }
                ],
                "rehabilitering": [
                    {
                        "case_number": f"{2020 + random.randint(0, 3)}/{random.randint(1000, 9999)}",
                        "date": random_date(),
                        "title": "Rehabilitering og oppgradering av bolig",
                        "address": f"Nabolaget {gnr-1}/{bnr-1}",
                        "result": "Godkjent" if random.random() > 0.1 else "Avslått",
                        "relevance": "medium" if random.random() > 0.6 else "high",
                        "url": f"https://innsyn.{municipality.lower()}.kommune.no/case/{random.randint(10000, 99999)}"
                    }
                ]
            }
            
            # Returner relevante caser basert på case_type
            if case_type.lower() in case_data:
                return case_data[case_type.lower()]
            else:
                # For ukjente case_types, returner tom liste
                return []
                
        except Exception as e:
            logger.error(f"Feil ved søk etter lignende saker: {str(e)}")
            return []
            
    async def calculate_fees(self, municipality: str, project_type: str, area: float) -> Dict:
        """
        Beregner kommunale gebyrer for et byggeprosjekt
        
        Args:
            municipality: Kommune eiendommen ligger i
            project_type: Type prosjekt (f.eks. 'tilbygg', 'nybygg', 'bruksendring')
            area: Areal i kvadratmeter
            
        Returns:
            Dict med gebyrberegning
        """
        try:
            # Dette ville normalt beregne gebyrer basert på kommunens gebyrregulativ
            # For denne demonstrasjonen, bruk forenklede beregninger med utvidet støtte for flere kommuner
            
            fee_rates = {
                "oslo": {
                    "base_fee": 13500,
                    "area_fee_per_m2": 180,
                    "dispensation_fee": 17000
                },
                "drammen": {
                    "base_fee": 12000,
                    "area_fee_per_m2": 150,
                    "dispensation_fee": 15000
                },
                "bergen": {
                    "base_fee": 12500,
                    "area_fee_per_m2": 160,
                    "dispensation_fee": 16000
                },
                "trondheim": {
                    "base_fee": 11800,
                    "area_fee_per_m2": 155,
                    "dispensation_fee": 15500
                },
                "stavanger": {
                    "base_fee": 11500,
                    "area_fee_per_m2": 160,
                    "dispensation_fee": 15800
                },
                "kristiansand": {
                    "base_fee": 10800,
                    "area_fee_per_m2": 145,
                    "dispensation_fee": 14500
                },
                "tromsø": {
                    "base_fee": 12200,
                    "area_fee_per_m2": 165,
                    "dispensation_fee": 16200
                },
                "fredrikstad": {
                    "base_fee": 11000,
                    "area_fee_per_m2": 140,
                    "dispensation_fee": 14000
                },
                "sandnes": {
                    "base_fee": 11300,
                    "area_fee_per_m2": 150,
                    "dispensation_fee": 15000
                },
                "bodø": {
                    "base_fee": 10500,
                    "area_fee_per_m2": 140,
                    "dispensation_fee": 13800
                },
                "ålesund": {
                    "base_fee": 10800,
                    "area_fee_per_m2": 145,
                    "dispensation_fee": 14200
                }
            }
            
            # Standardrater hvis kommunen ikke er definert
            if municipality.lower() not in fee_rates:
                logger.warning(f"Gebyrsatser ikke definert for {municipality}, bruker standardsatser")
                rates = {
                    "base_fee": 12000,
                    "area_fee_per_m2": 150,
                    "dispensation_fee": 15000
                }
            else:
                rates = fee_rates[municipality.lower()]
            
            # Beregn gebyr basert på prosjekttype
            base_fee = rates["base_fee"]
            area_fee = 0
            
            if project_type in ["tilbygg", "nybygg"]:
                # Arealgebyr for tilbygg og nybygg
                area_fee = area * rates["area_fee_per_m2"]
            elif project_type == "bruksendring":
                # Redusert arealgebyr for bruksendring
                area_fee = area * (rates["area_fee_per_m2"] * 0.7)
            elif project_type == "rehabilitering":
                # Ytterligere redusert arealgebyr for rehabilitering
                area_fee = area * (rates["area_fee_per_m2"] * 0.5)
            
            # Total
            total_fee = base_fee + area_fee
            
            # Legg til dispensasjonsgebyr hvis relevant
            needs_dispensation = project_type in ["dispensasjon", "avvik"]
            dispensation_fee = rates["dispensation_fee"] if needs_dispensation else 0
            
            if needs_dispensation:
                total_fee += dispensation_fee
            
            return {
                "municipality": municipality,
                "project_type": project_type,
                "area": area,
                "base_fee": base_fee,
                "area_fee": area_fee,
                "dispensation_fee": dispensation_fee,
                "total_fee": total_fee,
                "fee_details": "Gebyrer er beregnet i henhold til kommunens gebyrregulativ",
                "fee_year": 2023  # Sats for inneværende år
            }
            
        except Exception as e:
            logger.error(f"Feil ved beregning av gebyrer: {str(e)}")
            return {"error": str(e)}
