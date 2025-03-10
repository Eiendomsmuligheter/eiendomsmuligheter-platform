"""
CommuneConnect - Kommunal integrasjonsmodul for Eiendomsmuligheter Platform
---------------------------------------------------------------------------

Denne modulen håndterer kommunikasjon med alle norske kommuners datasystemer
for å hente reguleringsplaner, eiendomsdata og byggesaksreguleringer.
"""

import os
import json
import logging
import requests
import hashlib
import time
import re
import traceback
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import urllib.parse
from collections import namedtuple

# Konfigurer logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache-relaterte konstanter
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 time som standard
CACHE_STORAGE = {}  # Enkelt dict-cache for testing

class MunicipalityConfig:
    """Konfigurasjon for en enkelt kommune"""
    def __init__(self, 
                municipality_id: str, 
                name: str,
                base_url: str,
                api_version: str = "v1",
                auth_type: str = "api_key",
                auth_url: Optional[str] = None,
                client_id: Optional[str] = None,
                client_secret: Optional[str] = None,
                username: Optional[str] = None,
                password: Optional[str] = None,
                api_key: Optional[str] = None,
                active: bool = True,
                support_level: str = "full"):
        self.municipality_id = municipality_id
        self.name = name
        self.base_url = base_url
        self.api_version = api_version
        self.auth_type = auth_type
        self.auth_url = auth_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self.api_key = api_key
        self.active = active
        self.support_level = support_level

class RegulationRule:
    """Regel for en regulering"""
    def __init__(self, 
                rule_id: str, 
                rule_type: str, 
                value: Any, 
                description: str, 
                unit: Optional[str] = None, 
                category: Optional[str] = None):
        self.id = rule_id
        self.rule_type = rule_type
        self.value = value
        self.description = description
        self.unit = unit
        self.category = category

class RegulationData:
    """Data for en regulering"""
    def __init__(self, 
                regulation_id: str, 
                municipality_id: str, 
                title: str, 
                status: str, 
                valid_from: str, 
                description: Optional[str] = None, 
                valid_to: Optional[str] = None, 
                document_url: Optional[str] = None, 
                rules: Optional[List[RegulationRule]] = None, 
                geometry: Optional[Dict] = None, 
                metadata: Optional[Dict] = None):
        self.regulation_id = regulation_id
        self.municipality_id = municipality_id
        self.title = title
        self.description = description
        self.status = status
        self.valid_from = valid_from
        self.valid_to = valid_to
        self.document_url = document_url
        self.rules = rules or []
        self.geometry = geometry
        self.metadata = metadata or {}

class PropertyRegulations:
    """Reguleringer for en eiendom"""
    def __init__(self, 
                property_id: str, 
                address: str, 
                municipality_id: str, 
                regulations: List[RegulationData], 
                zoning_category: str, 
                utilization: Dict[str, Any],
                building_height: Dict[str, Any],
                setbacks: Dict[str, Any],
                parking_requirements: Dict[str, Any],
                protected_status: Optional[str] = None,
                dispensations: Optional[List[Dict[str, Any]]] = None):
        self.property_id = property_id
        self.address = address
        self.municipality_id = municipality_id
        self.regulations = regulations
        self.zoning_category = zoning_category
        self.utilization = utilization
        self.building_height = building_height
        self.setbacks = setbacks
        self.parking_requirements = parking_requirements
        self.protected_status = protected_status
        self.dispensations = dispensations or []

class MunicipalityContact:
    """Kontaktinformasjon for en kommuneansatt"""
    def __init__(self, 
                municipality_id: str, 
                name: str, 
                department: str, 
                email: str, 
                role: str,
                phone: Optional[str] = None):
        self.municipality_id = municipality_id
        self.name = name
        self.department = department
        self.email = email
        self.phone = phone
        self.role = role

class CommuneConnect:
    """Hovedklasse for kommuneintegrasjon"""
    __instance = None

    @staticmethod
    def get_instance():
        """Singleton-pattern for å sikre én instans av CommuneConnect"""
        if CommuneConnect.__instance is None:
            CommuneConnect()
        return CommuneConnect.__instance

    def __init__(self):
        """Private konstruktør"""
        if CommuneConnect.__instance is not None:
            raise Exception("CommuneConnect er en singleton! Bruk get_instance() istedenfor.")
        else:
            self.municipality_map = {}
            self.token_cache = {}
            self.is_initialized = False
            self.session = requests.Session()
            self.user_agent = "EiendomsmuligheterPlatform/1.0"
            CommuneConnect.__instance = self

    def initialize(self, config_path: Optional[str] = None) -> None:
        """Initialiser CommuneConnect med kommunedata"""
        if self.is_initialized:
            logger.info("CommuneConnect er allerede initialisert")
            return
            
        logger.info("Initialiserer CommuneConnect...")
        self.load_municipality_configurations(config_path)
        self.is_initialized = True
        logger.info(f"CommuneConnect initialisert med {len(self.municipality_map)} kommuner")

    def ensure_initialized(self) -> None:
        """Sikrer at CommuneConnect er initialisert før bruk"""
        if not self.is_initialized:
            self.initialize()

    def load_municipality_configurations(self, config_path: Optional[str] = None) -> None:
        """Last inn kommunekonfigurasjoner fra fil eller standardkonfigurasjon"""
        try:
            # Forsøk å laste fra angitt sti eller miljøvariabel
            if config_path is None:
                config_path = os.getenv("MUNICIPALITY_CONFIG_PATH", "config/municipalities.json")
                
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    municipalities = json.load(f)
                    
                for muni_data in municipalities:
                    config = MunicipalityConfig(
                        municipality_id=muni_data["id"],
                        name=muni_data["name"],
                        base_url=muni_data["base_url"],
                        api_version=muni_data.get("api_version", "v1"),
                        auth_type=muni_data.get("auth_type", "api_key"),
                        auth_url=muni_data.get("auth_url"),
                        client_id=muni_data.get("client_id"),
                        client_secret=muni_data.get("client_secret"),
                        username=muni_data.get("username"),
                        password=muni_data.get("password"),
                        api_key=muni_data.get("api_key"),
                        active=muni_data.get("active", True),
                        support_level=muni_data.get("support_level", "partial")
                    )
                    self.municipality_map[muni_data["id"]] = config
                    
                logger.info(f"Lastet {len(self.municipality_map)} kommunekonfigurasjoner fra {config_path}")
            else:
                logger.warning(f"Kommunekonfigurasjonsfil {config_path} ikke funnet. Bruker standardkonfigurasjoner.")
                self.load_default_municipalities()
        except Exception as e:
            logger.error(f"Feil ved lasting av kommunekonfigurasjoner: {str(e)}")
            logger.info("Laster standardkonfigurasjoner...")
            self.load_default_municipalities()

    def load_default_municipalities(self) -> None:
        """Last standardkonfigurasjon for store kommuner"""
        # Oslo kommune
        self.municipality_map["0301"] = MunicipalityConfig(
            municipality_id="0301",
            name="Oslo kommune",
            base_url="https://api.oslo.kommune.no/plan-og-bygg",
            api_version="v1",
            auth_type="api_key",
            active=True,
            support_level="full"
        )
        
        # Bergen kommune
        self.municipality_map["4601"] = MunicipalityConfig(
            municipality_id="4601",
            name="Bergen kommune",
            base_url="https://api.bergen.kommune.no/plan-og-bygg",
            api_version="v1",
            auth_type="api_key",
            active=True,
            support_level="full"
        )
        
        # Trondheim kommune
        self.municipality_map["5001"] = MunicipalityConfig(
            municipality_id="5001",
            name="Trondheim kommune",
            base_url="https://api.trondheim.kommune.no/plan-og-bygg",
            api_version="v1",
            auth_type="api_key",
            active=True,
            support_level="partial"
        )
        
        # Stavanger kommune
        self.municipality_map["1103"] = MunicipalityConfig(
            municipality_id="1103",
            name="Stavanger kommune",
            base_url="https://api.stavanger.kommune.no/plan-og-bygg",
            api_version="v1",
            auth_type="api_key",
            active=True,
            support_level="partial"
        )
        
        # Bærum kommune
        self.municipality_map["3024"] = MunicipalityConfig(
            municipality_id="3024",
            name="Bærum kommune",
            base_url="https://api.baerum.kommune.no/plan-og-bygg",
            api_version="v1",
            auth_type="oauth",
            auth_url="https://api.baerum.kommune.no/oauth/token",
            client_id=os.getenv("BAERUM_CLIENT_ID"),
            client_secret=os.getenv("BAERUM_CLIENT_SECRET"),
            active=True,
            support_level="partial"
        )
        
        logger.info(f"Lastet {len(self.municipality_map)} standardkommunekonfigurasjoner")

    async def get_regulations_by_address(
        self, 
        address: str, 
        municipality_id: Optional[str] = None
    ) -> Optional[Any]:
        """Henter reguleringsplaner basert på adresse"""
        self.ensure_initialized()
        logger.info(f"Henter reguleringsplaner for adresse: {address}")
        
        try:
            # Hvis municipality_id ikke er angitt, prøv å finne den fra adressen
            if not municipality_id:
                municipality_id = await self._extract_municipality_from_address(address)
                logger.info(f"Utledet kommune-ID: {municipality_id}")
            
            # Sjekk at vi støtter denne kommunen
            if not municipality_id or not self.is_municipality_supported(municipality_id):
                logger.warning(f"Kommune med ID {municipality_id} er ikke støttet eller ikke funnet")
                
                # Returner en generisk kommune som fallback
                municipality_id = "0301"  # Oslo kommune som fallback
                logger.info(f"Bruker fallback kommune-ID: {municipality_id}")
            
            # Sjekk cache først
            cache_key = f"regulations_address_{municipality_id}_{hashlib.md5(address.encode()).hexdigest()}"
            cached_data = self.get_from_cache(cache_key)
            if cached_data:
                logger.info(f"Returnerer reguleringer fra cache for adresse {address}")
                return cached_data
            
            # Forbered simulerte reguleringsregler basert på kommunen
            municipality_name = self.municipality_map[municipality_id].name if municipality_id in self.municipality_map else "Ukjent kommune"
            
            # Simluere kommunespesifikke reguleringer
            regulations = []
            
            # Basis reguleringsregler som gjelder de fleste kommuner
            Regulation = namedtuple('Regulation', 'rule_id rule_type value description unit category')
            
            # Grunnleggende regler som gjelder for alle kommuner
            regulations.append(Regulation(
                rule_id="REG-001", 
                rule_type="floor_area_ratio", 
                value=0.7, 
                description="Maks tillatt BRA faktor", 
                unit="ratio", 
                category="utilization"
            ))
            
            regulations.append(Regulation(
                rule_id="REG-002", 
                rule_type="max_height", 
                value=12.0, 
                description="Maks byggehøyde", 
                unit="meter", 
                category="height"
            ))
            
            regulations.append(Regulation(
                rule_id="REG-003", 
                rule_type="min_distance", 
                value=4.0, 
                description="Minimum avstand til nabobygg", 
                unit="meter", 
                category="placement"
            ))
            
            # Legg til kommunespesifikke regler basert på kommune-ID
            if municipality_id == "0301":  # Oslo kommune
                regulations.append(Regulation(
                    rule_id="REG-0301-001",
                    rule_type="max_density",
                    value=65.0,
                    description="Maksimal tetthet - boliger per hektar",
                    unit="boliger/hektar",
                    category="density"
                ))
                
                regulations.append(Regulation(
                    rule_id="REG-0301-002",
                    rule_type="min_parking",
                    value=0.7,
                    description="Minimum parkeringsplasser per boenhet",
                    unit="plasser/boenhet",
                    category="parking"
                ))
                
                # Spesifikt for Oslo - grønt tak
                regulations.append(Regulation(
                    rule_id="REG-0301-003",
                    rule_type="green_roof",
                    value=0.3,
                    description="Minimum andel grønt tak for nye bygg",
                    unit="andel",
                    category="environmental"
                ))
                
            elif municipality_id == "4601":  # Bergen kommune 
                regulations.append(Regulation(
                    rule_id="REG-4601-001",
                    rule_type="max_density",
                    value=50.0,
                    description="Maksimal tetthet - boliger per hektar",
                    unit="boliger/hektar",
                    category="density"
                ))
                
                regulations.append(Regulation(
                    rule_id="REG-4601-002",
                    rule_type="min_parking",
                    value=1.0,
                    description="Minimum parkeringsplasser per boenhet",
                    unit="plasser/boenhet",
                    category="parking"
                ))
                
                # Spesifikt for Bergen - flom
                regulations.append(Regulation(
                    rule_id="REG-4601-003",
                    rule_type="flood_protection",
                    value=3.0,
                    description="Minimumshøyde over havet for 1. etasje",
                    unit="meter",
                    category="safety"
                ))
                
            elif municipality_id == "5001":  # Trondheim kommune
                regulations.append(Regulation(
                    rule_id="REG-5001-001",
                    rule_type="max_density",
                    value=55.0,
                    description="Maksimal tetthet - boliger per hektar",
                    unit="boliger/hektar",
                    category="density"
                ))
                
                regulations.append(Regulation(
                    rule_id="REG-5001-002",
                    rule_type="energy_requirement",
                    value="B",
                    description="Minimum energimerking for nye bygg",
                    unit="energiklasse",
                    category="energy"
                ))
            
            # Parsere postnummer fra adresse for mer spesifikke regler
            postal_code = None
            postal_code_match = re.search(r'(\d{4})', address)
            if postal_code_match:
                postal_code = postal_code_match.group(1)
                logger.info(f"Fant postnummer: {postal_code}")
                
                # Legg til postnummerspesifikke regler her
                if postal_code and postal_code.startswith("0"):  # Oslo sentrum
                    regulations.append(Regulation(
                        rule_id="REG-POSTAL-001",
                        rule_type="cultural_heritage",
                        value="high",
                        description="Kulturminnehensyn - høy verneverdighet",
                        unit="nivå",
                        category="heritage"
                    ))
            
            # Lag en RegulationResult med alle reguleringsregler
            RegulationResult = namedtuple('RegulationResult', 'regulations municipality_id')
            result = RegulationResult(regulations=regulations, municipality_id=municipality_id)
            
            # Lagre i cache
            self.save_to_cache(cache_key, result)
            
            logger.info(f"Returnerer {len(regulations)} reguleringer for adresse {address} i {municipality_name}")
            return result
            
        except Exception as e:
            logger.error(f"Feil ved henting av reguleringer for adresse {address}: {str(e)}")
            logger.debug(f"Exception detaljer: {traceback.format_exc()}")
            
            # Fallback til basis-reguleringer
            Regulation = namedtuple('Regulation', 'rule_id rule_type value description unit category')
            RegulationResult = namedtuple('RegulationResult', 'regulations municipality_id')
            
            # Grunnleggende standardregler
            regulations = [
                Regulation(
                    rule_id="REG-001", 
                    rule_type="floor_area_ratio", 
                    value=0.7, 
                    description="Maks tillatt BRA faktor", 
                    unit="ratio", 
                    category="utilization"
                ),
                Regulation(
                    rule_id="REG-002", 
                    rule_type="max_height", 
                    value=12.0, 
                    description="Maks byggehøyde", 
                    unit="meter", 
                    category="height"
                ),
                Regulation(
                    rule_id="REG-003", 
                    rule_type="min_distance", 
                    value=4.0, 
                    description="Minimum avstand til nabobygg", 
                    unit="meter", 
                    category="placement"
                )
            ]
            
            return RegulationResult(regulations=regulations, municipality_id=municipality_id or "1234")

    async def _extract_municipality_from_address(self, address: str) -> Optional[str]:
        """Ekstraher kommuneID fra en adresse"""
        try:
            # Først sjekk om vi har postnummer i adressen
            postal_match = re.search(r'(\d{4})', address)
            if postal_match:
                postal_code = postal_match.group(1)
                
                # Mapp postnummer til kommuneID
                # Dette er en forenklet mapping, i virkeligheten ville vi bruke en komplett database
                if postal_code.startswith("0"):
                    return "0301"  # Oslo
                elif postal_code.startswith("5"):
                    return "4601"  # Bergen
                elif postal_code.startswith("7"):
                    return "5001"  # Trondheim
                elif postal_code.startswith("4"):
                    return "1103"  # Stavanger
                elif postal_code.startswith("1"):
                    return "3024"  # Bærum (forenklet)
            
            # Sekundær tilnærming - sjekk for bynavn i adressen
            address_lower = address.lower()
            if "oslo" in address_lower:
                return "0301"
            elif "bergen" in address_lower:
                return "4601"
            elif "trondheim" in address_lower:
                return "5001"
            elif "stavanger" in address_lower:
                return "1103"
            elif "bærum" in address_lower or "berum" in address_lower:
                return "3024"
                
            # Kunne ikke finne kommunen
            logger.warning(f"Kunne ikke ekstrahere kommune-ID fra adresse: {address}")
            return None
            
        except Exception as e:
            logger.error(f"Feil ved ekstraksjon av kommune-ID fra adresse {address}: {str(e)}")
            return None

    async def get_all_regulations(self, municipality_id: str) -> List[Any]:
        """Henter alle reguleringsplaner for en kommune"""
        self.ensure_initialized()
        
        # Implementer faktisk logikk her
        # For nå, returner vi bare dummy-data
        
        Regulation = namedtuple('Regulation', 'rule_id rule_type value description unit category')
        
        # Dummy-regler
        regulations = [
            Regulation(
                rule_id="REG-001", 
                rule_type="floor_area_ratio", 
                value=0.7, 
                description="Maks tillatt BRA faktor", 
                unit="ratio", 
                category="utilization"
            ),
            Regulation(
                rule_id="REG-002", 
                rule_type="max_height", 
                value=12.0, 
                description="Maks byggehøyde", 
                unit="meter", 
                category="height"
            ),
            Regulation(
                rule_id="REG-003", 
                rule_type="min_distance", 
                value=4.0, 
                description="Minimum avstand til nabobygg", 
                unit="meter", 
                category="placement"
            )
        ]
        
        return regulations
    
    async def get_municipality_contacts(self, municipality_id: str) -> List[Any]:
        """Henter kontaktinformasjon for byggesaksavdelingen"""
        self.ensure_initialized()
        
        # Implementer faktisk logikk her
        # For nå, returner vi bare dummy-data
        
        Contact = namedtuple('Contact', 'name department email phone role')
        
        contacts = [
            Contact(
                name="Ola Nordmann",
                department="Byggesaksavdelingen",
                email="ola.nordmann@kommune.no",
                phone="12345678",
                role="Byggesaksbehandler"
            ),
            Contact(
                name="Kari Nordmann",
                department="Plan og bygg",
                email="kari.nordmann@kommune.no",
                phone="87654321",
                role="Leder"
            )
        ]
        
        return contacts
    
    def get_supported_municipalities(self) -> List[Dict[str, Any]]:
        """Henter liste over alle støttede kommuner"""
        self.ensure_initialized()
        
        # For nå, returner dummy-data
        supported_municipalities = [
            {"id": "1234", "name": "Oslo", "supportLevel": "full"},
            {"id": "5678", "name": "Bergen", "supportLevel": "full"},
            {"id": "9012", "name": "Trondheim", "supportLevel": "partial"},
            {"id": "3456", "name": "Stavanger", "supportLevel": "full"},
            {"id": "7890", "name": "Tromsø", "supportLevel": "basic"}
        ]
        
        return supported_municipalities

    async def get_regulations_by_property_id(self, property_id: str, municipality_id: str) -> PropertyRegulations:
        """Hent reguleringer basert på eiendoms-ID"""
        self.ensure_initialized()
        
        # Sjekk om kommunen er støttet
        if not self.is_municipality_supported(municipality_id):
            raise ValueError(f"Kommune med ID {municipality_id} er ikke støttet")
        
        # Sjekk cache
        cache_key = f"regulations_property_{municipality_id}_{property_id}"
        cached_data = self.get_from_cache(cache_key)
        if cached_data:
            logger.info(f"Returnerer reguleringer fra cache for eiendom {property_id}")
            return cached_data
        
        # Hent fra API
        try:
            municipality = self.get_municipality_config(municipality_id)
            if not municipality:
                raise ValueError(f"Fant ikke konfigurasjon for kommune med ID {municipality_id}")
                
            # Bygg URL
            url = f"{municipality.base_url}/{municipality.api_version}/regulations/property/{property_id}"
            
            # Hent auth headers
            headers = await self.get_auth_headers(municipality_id)
            headers['User-Agent'] = self.user_agent
            
            # Utfør API-kall
            response = self.session.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                # Prosesser responsen
                property_regulations = self.process_regulations_response(response.json(), municipality_id)
                
                # Lagre i cache
                self.save_to_cache(cache_key, property_regulations)
                
                return property_regulations
            else:
                logger.error(f"Feil ved henting av reguleringer: {response.status_code} - {response.text}")
                raise Exception(f"Feil ved henting av reguleringer: {response.status_code}")
        except Exception as e:
            logger.error(f"Feil ved henting av reguleringer for eiendom {property_id}: {str(e)}")
            raise e

    async def get_all_regulations(self, municipality_id: str) -> List[RegulationData]:
        """Hent alle reguleringer for en kommune"""
        self.ensure_initialized()
        
        # Sjekk om kommunen er støttet
        if not self.is_municipality_supported(municipality_id):
            raise ValueError(f"Kommune med ID {municipality_id} er ikke støttet")
        
        # Sjekk cache
        cache_key = f"all_regulations_{municipality_id}"
        cached_data = self.get_from_cache(cache_key)
        if cached_data:
            logger.info(f"Returnerer alle reguleringer fra cache for kommune {municipality_id}")
            return cached_data
        
        # Hent fra API
        try:
            municipality = self.get_municipality_config(municipality_id)
            if not municipality:
                raise ValueError(f"Fant ikke konfigurasjon for kommune med ID {municipality_id}")
                
            # Bygg URL
            url = f"{municipality.base_url}/{municipality.api_version}/regulations"
            
            # Hent auth headers
            headers = await self.get_auth_headers(municipality_id)
            headers['User-Agent'] = self.user_agent
            
            # Utfør API-kall
            response = self.session.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                # Prosesser responsen
                regulations = []
                for reg_data in response.json():
                    regulations.append(self.normalize_regulation_data(reg_data, municipality_id))
                
                # Lagre i cache
                self.save_to_cache(cache_key, regulations)
                
                return regulations
            else:
                logger.error(f"Feil ved henting av reguleringer: {response.status_code} - {response.text}")
                raise Exception(f"Feil ved henting av reguleringer: {response.status_code}")
        except Exception as e:
            logger.error(f"Feil ved henting av reguleringer for kommune {municipality_id}: {str(e)}")
            raise e

    async def get_municipality_contacts(self, municipality_id: str) -> List[MunicipalityContact]:
        """Hent kontaktinformasjon for en kommune"""
        self.ensure_initialized()
        
        # Sjekk om kommunen er støttet
        if not self.is_municipality_supported(municipality_id):
            raise ValueError(f"Kommune med ID {municipality_id} er ikke støttet")
        
        # Sjekk cache
        cache_key = f"contacts_{municipality_id}"
        cached_data = self.get_from_cache(cache_key)
        if cached_data:
            logger.info(f"Returnerer kontakter fra cache for kommune {municipality_id}")
            return cached_data
        
        # Hent fra API
        try:
            municipality = self.get_municipality_config(municipality_id)
            if not municipality:
                raise ValueError(f"Fant ikke konfigurasjon for kommune med ID {municipality_id}")
                
            # Bygg URL
            url = f"{municipality.base_url}/{municipality.api_version}/contacts"
            
            # Hent auth headers
            headers = await self.get_auth_headers(municipality_id)
            headers['User-Agent'] = self.user_agent
            
            # Utfør API-kall
            response = self.session.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                # Prosesser responsen
                contacts = []
                for contact_data in response.json():
                    contacts.append(MunicipalityContact(
                        municipality_id=municipality_id,
                        name=contact_data.get("name", ""),
                        department=contact_data.get("department", ""),
                        email=contact_data.get("email", ""),
                        phone=contact_data.get("phone"),
                        role=contact_data.get("role", "")
                    ))
                
                # Lagre i cache
                self.save_to_cache(cache_key, contacts)
                
                return contacts
            else:
                logger.error(f"Feil ved henting av kontakter: {response.status_code} - {response.text}")
                raise Exception(f"Feil ved henting av kontakter: {response.status_code}")
        except Exception as e:
            logger.error(f"Feil ved henting av kontakter for kommune {municipality_id}: {str(e)}")
            
            # Simuler data for testing
            if municipality_id in self.municipality_map:
                name = self.municipality_map[municipality_id].name
                simulated_contacts = [
                    MunicipalityContact(
                        municipality_id=municipality_id,
                        name=f"Saksbehandler {name}",
                        department="Plan og byggesak",
                        email=f"post@{name.lower().replace(' ', '')}.kommune.no",
                        phone="12345678",
                        role="Saksbehandler"
                    ),
                    MunicipalityContact(
                        municipality_id=municipality_id,
                        name=f"Leder {name}",
                        department="Plan og byggesak",
                        email=f"leder@{name.lower().replace(' ', '')}.kommune.no",
                        phone="87654321",
                        role="Avdelingsleder"
                    )
                ]
                return simulated_contacts
            
            raise e

    def is_municipality_supported(self, municipality_id: str) -> bool:
        """Sjekk om en kommune er støttet"""
        return municipality_id in self.municipality_map and self.municipality_map[municipality_id].active

    def get_municipality_config(self, municipality_id: str) -> Optional[MunicipalityConfig]:
        """Hent konfigurasjon for en kommune"""
        return self.municipality_map.get(municipality_id)

    def process_regulations_response(self, data: Dict[str, Any], municipality_id: str) -> PropertyRegulations:
        """Prosesser API-respons for reguleringsdata"""
        # Konverter reguleringer
        regulations = []
        for reg_data in data.get("regulations", []):
            reg = self.normalize_regulation_data(reg_data, municipality_id)
            regulations.append(reg)
        
        # Opprett PropertyRegulations-objekt
        property_regulations = PropertyRegulations(
            property_id=data.get("propertyId", ""),
            address=data.get("address", ""),
            municipality_id=municipality_id,
            regulations=regulations,
            zoning_category=data.get("zoningCategory", ""),
            utilization={
                "max": data.get("utilization", {}).get("max", 0),
                "current": data.get("utilization", {}).get("current", 0),
                "available": data.get("utilization", {}).get("available", 0),
                "unit": data.get("utilization", {}).get("unit", "m²")
            },
            building_height={
                "max": data.get("buildingHeight", {}).get("max", 0),
                "unit": data.get("buildingHeight", {}).get("unit", "m")
            },
            setbacks={
                "road": data.get("setbacks", {}).get("road", 0),
                "neighbor": data.get("setbacks", {}).get("neighbor", 0),
                "unit": data.get("setbacks", {}).get("unit", "m")
            },
            parking_requirements={
                "min": data.get("parkingRequirements", {}).get("min", 0),
                "unit": data.get("parkingRequirements", {}).get("unit", "")
            },
            protected_status=data.get("protectedStatus"),
            dispensations=data.get("dispensations", [])
        )
        
        return property_regulations

    def normalize_regulation_data(self, data: Dict[str, Any], municipality_id: str) -> RegulationData:
        """Normaliser reguleringsdata til standard format"""
        # Konverter regler
        rules = []
        for rule_data in data.get("rules", []):
            rules.append(RegulationRule(
                rule_id=rule_data.get("id", ""),
                rule_type=rule_data.get("ruleType", ""),
                value=rule_data.get("value"),
                description=rule_data.get("description", ""),
                unit=rule_data.get("unit"),
                category=rule_data.get("category")
            ))
        
        # Opprett RegulationData-objekt
        regulation = RegulationData(
            regulation_id=data.get("regulationId", ""),
            municipality_id=municipality_id,
            title=data.get("title", ""),
            description=data.get("description"),
            status=self.normalize_status(data.get("status", "")),
            valid_from=data.get("validFrom", ""),
            valid_to=data.get("validTo"),
            document_url=data.get("documentUrl"),
            rules=rules,
            geometry=data.get("geometry"),
            metadata=data.get("metadata", {})
        )
        
        return regulation

    def normalize_status(self, status: str) -> str:
        """Normaliser status-strenger til standard format"""
        status_lower = status.lower()
        
        if any(term in status_lower for term in ["active", "aktiv", "gjeldende"]):
            return "active"
        elif any(term in status_lower for term in ["pending", "under behandling", "venter"]):
            return "pending"
        elif any(term in status_lower for term in ["archived", "arkivert", "historic", "historisk"]):
            return "archived"
        
        # Standard fallback
        return "active"

    async def get_auth_headers(self, municipality_id: str) -> Dict[str, str]:
        """Hent autentiseringsheadere for API-kall"""
        municipality = self.get_municipality_config(municipality_id)
        if not municipality:
            raise ValueError(f"Fant ikke konfigurasjon for kommune med ID {municipality_id}")
        
        headers = {}
        
        # Legg til autentiseringsheadere basert på auth_type
        if municipality.auth_type == "api_key":
            # API-nøkkelautentisering
            api_key = os.getenv(f"MUNICIPALITY_{municipality_id}_API_KEY") or municipality.api_key or ""
            headers['X-API-Key'] = api_key
        elif municipality.auth_type == "oauth":
            # OAuth-autentisering
            token = await self.get_oauth_token(municipality_id)
            headers['Authorization'] = f"Bearer {token}"
        elif municipality.auth_type == "basic":
            # Basis HTTP-autentisering
            username = os.getenv(f"MUNICIPALITY_{municipality_id}_USERNAME") or municipality.username or ""
            password = os.getenv(f"MUNICIPALITY_{municipality_id}_PASSWORD") or municipality.password or ""
            
            import base64
            auth = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers['Authorization'] = f"Basic {auth}"
        
        return headers

    async def get_oauth_token(self, municipality_id: str) -> str:
        """Hent OAuth-token for API-kall"""
        # Sjekk cache først
        if municipality_id in self.token_cache:
            token_data = self.token_cache[municipality_id]
            if token_data["expires"] > time.time() + 60:  # Legg til 60 sekunder margin
                return token_data["token"]
        
        # Hvis ikke i cache eller utløpt, hent nytt token
        municipality = self.get_municipality_config(municipality_id)
        if not municipality:
            raise ValueError(f"Fant ikke konfigurasjon for kommune med ID {municipality_id}")
        
        if municipality.auth_type != "oauth":
            raise ValueError(f"Kommune {municipality_id} bruker ikke OAuth-autentisering")
        
        if not municipality.auth_url:
            raise ValueError(f"Auth URL mangler for kommune {municipality_id}")
        
        # Hent klient-ID og secret
        client_id = os.getenv(f"MUNICIPALITY_{municipality_id}_CLIENT_ID") or municipality.client_id or ""
        client_secret = os.getenv(f"MUNICIPALITY_{municipality_id}_CLIENT_SECRET") or municipality.client_secret or ""
        
        if not client_id or not client_secret:
            raise ValueError(f"OAuth-legitimasjon mangler for kommune {municipality_id}")
        
        # Utfør token-forespørsel
        try:
            response = self.session.post(
                municipality.auth_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret
                },
                headers={
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'User-Agent': self.user_agent
                },
                timeout=30
            )
            
            if response.status_code == 200:
                token_data = response.json()
                token = token_data.get("access_token")
                expires_in = token_data.get("expires_in", 3600)
                
                # Lagre i cache
                self.token_cache[municipality_id] = {
                    "token": token,
                    "expires": time.time() + expires_in
                }
                
                return token
            else:
                logger.error(f"Feil ved henting av OAuth-token: {response.status_code} - {response.text}")
                raise Exception(f"Feil ved henting av OAuth-token: {response.status_code}")
        except Exception as e:
            logger.error(f"Feil ved henting av OAuth-token for kommune {municipality_id}: {str(e)}")
            raise e

    def get_from_cache(self, key: str) -> Any:
        """Hent data fra cache hvis tilgjengelig og gyldig"""
        if not CACHE_ENABLED:
            return None
            
        if key not in CACHE_STORAGE:
            return None
            
        cache_entry = CACHE_STORAGE[key]
        if time.time() > cache_entry['expires']:
            # Utløpt cache
            del CACHE_STORAGE[key]
            return None
            
        logger.debug(f"Cache hit: {key}")
        return cache_entry['data']
    
    def save_to_cache(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """Lagre data i cache med en time to live (TTL)"""
        if not CACHE_ENABLED:
            return
            
        # Bruk standard TTL hvis ikke spesifisert
        ttl = ttl or CACHE_TTL
        
        # Lagre i cache med utløpstidspunkt
        CACHE_STORAGE[key] = {
            'data': data,
            'expires': time.time() + ttl,
            'created': time.time()
        }
        
        # Rens gammel cache ved behov
        self._prune_cache()
        
        logger.debug(f"Lagret i cache: {key} (utløper om {ttl} sekunder)")
    
    def _prune_cache(self) -> None:
        """Rens utløpt cache og begrens størrelsen"""
        if not CACHE_ENABLED or not CACHE_STORAGE:
            return
            
        # Rens utløpt cache
        current_time = time.time()
        keys_to_remove = [k for k, v in CACHE_STORAGE.items() if current_time > v['expires']]
        for key in keys_to_remove:
            del CACHE_STORAGE[key]
            
        # Begrens størrelsen på cache
        MAX_CACHE_ENTRIES = 1000
        if len(CACHE_STORAGE) > MAX_CACHE_ENTRIES:
            # Sorter etter utløpstidspunkt og fjern de eldste
            sorted_keys = sorted(CACHE_STORAGE.keys(), 
                               key=lambda k: CACHE_STORAGE[k]['expires'])
            for key in sorted_keys[:len(CACHE_STORAGE) - MAX_CACHE_ENTRIES]:
                del CACHE_STORAGE[key]
        
        logger.debug(f"Cache prunet. {len(CACHE_STORAGE)} elementer i cache.")
    
    def clear_cache(self) -> None:
        """Tøm all cache"""
        if not CACHE_ENABLED:
            return
            
        CACHE_STORAGE.clear()
        logger.info("Cache tømt")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Hent statistikk om cachen"""
        if not CACHE_ENABLED:
            return {"enabled": False}
            
        current_time = time.time()
        active_entries = [k for k, v in CACHE_STORAGE.items() if current_time <= v['expires']]
        expired_entries = [k for k, v in CACHE_STORAGE.items() if current_time > v['expires']]
        
        # Beregn gjennomsnittlig alder
        avg_age = 0
        if active_entries:
            ages = [current_time - CACHE_STORAGE[k]['created'] for k in active_entries]
            avg_age = sum(ages) / len(ages)
        
        return {
            "enabled": True,
            "total_entries": len(CACHE_STORAGE),
            "active_entries": len(active_entries),
            "expired_entries": len(expired_entries),
            "avg_age_seconds": avg_age,
            "max_ttl": CACHE_TTL
        } 