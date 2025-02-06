import aiohttp
import asyncio
from typing import Dict, Optional, List
import logging
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class MunicipalityService:
    """
    Service for kommunikasjon med kommunale systemer.
    Håndterer:
    - Søk i byggesaksarkiv
    - Henting av reguleringsplaner
    - Sjekk av byggtekniske forskrifter
    - Automatisk utfylling av byggesøknader
    """
    
    def __init__(self):
        self.session = None
        self.municipalities = self._load_municipality_configs()
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    def _load_municipality_configs(self) -> Dict:
        """Last inn konfigurasjon for hver kommune"""
        # TODO: Last fra database eller config fil
        return {
            "drammen": {
                "innsyn_url": "https://innsyn2020.drammen.kommune.no/application",
                "advanced_search_url": "https://innsyn2020.drammen.kommune.no/postjournal-v2/fb851964-3185-43eb-81ba-9ac75226dfa8",
                "map_service_url": "https://kart.drammen.kommune.no",
                "regulation_api": "https://api.drammen.kommune.no/regulation",
            }
        }
        
    async def get_property_records(self,
                                 municipality: str,
                                 gnr: int,
                                 bnr: int,
                                 from_year: int = 2020) -> List[Dict]:
        """
        Hent alle saker knyttet til en eiendom fra kommunens arkiv
        """
        try:
            config = self.municipalities.get(municipality.lower())
            if not config:
                raise ValueError(f"Ukjent kommune: {municipality}")
                
            # Konstruer søkeforespørsel
            search_params = {
                "gnr": gnr,
                "bnr": bnr,
                "fromYear": from_year,
                "recordType": "byggesak"
            }
            
            async with self.session.post(
                config["advanced_search_url"],
                json=search_params
            ) as response:
                if response.status != 200:
                    raise Exception(f"Feil ved søk i arkiv: {await response.text()}")
                    
                data = await response.json()
                return self._parse_property_records(data)
                
        except Exception as e:
            logger.error(f"Feil ved henting av eiendomsarkiv: {str(e)}")
            raise
            
    async def get_zoning_plan(self,
                             municipality: str,
                             coordinates: Dict[str, float]) -> Dict:
        """
        Hent gjeldende reguleringsplan for en eiendom
        """
        try:
            config = self.municipalities.get(municipality.lower())
            if not config:
                raise ValueError(f"Ukjent kommune: {municipality}")
                
            # Hent reguleringsplan basert på koordinater
            async with self.session.get(
                f"{config['regulation_api']}/zoning",
                params={
                    "lat": coordinates["lat"],
                    "lon": coordinates["lon"]
                }
            ) as response:
                if response.status != 200:
                    raise Exception(f"Feil ved henting av reguleringsplan: {await response.text()}")
                    
                plan_data = await response.json()
                return self._parse_zoning_plan(plan_data)
                
        except Exception as e:
            logger.error(f"Feil ved henting av reguleringsplan: {str(e)}")
            raise
            
    async def get_building_regulations(self,
                                     municipality: str,
                                     property_id: str) -> Dict:
        """
        Hent gjeldende byggtekniske forskrifter for en eiendom
        """
        try:
            # Determiner hvilken TEK som gjelder (10 eller 17)
            tek_version = await self._determine_tek_version(
                municipality,
                property_id
            )
            
            # Hent detaljerte forskrifter
            regulations = await self._fetch_building_regulations(
                municipality,
                tek_version
            )
            
            return {
                "tek_version": tek_version,
                "regulations": regulations,
                "requirements": await self._get_specific_requirements(
                    municipality,
                    property_id,
                    tek_version
                )
            }
            
        except Exception as e:
            logger.error(f"Feil ved henting av byggtekniske forskrifter: {str(e)}")
            raise
            
    async def generate_building_application(self,
                                          municipality: str,
                                          property_data: Dict,
                                          project_details: Dict) -> Dict:
        """
        Generer ferdig utfylt byggesøknad basert på prosjektdetaljer
        """
        try:
            # Hent riktige søknadsskjemaer
            forms = await self._get_application_forms(
                municipality,
                project_details["application_type"]
            )
            
            # Fyll ut skjemaene automatisk
            filled_forms = await self._fill_application_forms(
                forms,
                property_data,
                project_details
            )
            
            # Valider at alle påkrevde felt er fylt ut
            validation_result = self._validate_applications(filled_forms)
            
            return {
                "forms": filled_forms,
                "validation": validation_result,
                "submission_ready": validation_result["is_valid"],
                "missing_info": validation_result.get("missing_fields", [])
            }
            
        except Exception as e:
            logger.error(f"Feil ved generering av byggesøknad: {str(e)}")
            raise
            
    def _parse_property_records(self, data: Dict) -> List[Dict]:
        """Parse og strukturer eiendomsarkiv data"""
        records = []
        for record in data.get("records", []):
            records.append({
                "id": record.get("id"),
                "date": record.get("date"),
                "title": record.get("title"),
                "case_type": record.get("caseType"),
                "status": record.get("status"),
                "documents": record.get("documents", []),
                "decisions": record.get("decisions", [])
            })
        return records
        
    def _parse_zoning_plan(self, data: Dict) -> Dict:
        """Parse og strukturer reguleringsplan data"""
        return {
            "plan_id": data.get("planId"),
            "name": data.get("name"),
            "type": data.get("planType"),
            "status": data.get("status"),
            "valid_from": data.get("validFrom"),
            "regulations": data.get("regulations", {}),
            "restrictions": data.get("restrictions", {}),
            "allowed_usage": data.get("allowedUsage", {}),
            "max_utilization": data.get("maxUtilization"),
            "building_heights": data.get("buildingHeights", {}),
            "special_considerations": data.get("specialConsiderations", [])
        }
        
    async def _determine_tek_version(self,
                                   municipality: str,
                                   property_id: str) -> str:
        """Determiner hvilken TEK-versjon som gjelder"""
        # TODO: Implementer logikk for å bestemme TEK-versjon
        return "TEK17"  # Default til nyeste versjon
        
    async def _fetch_building_regulations(self,
                                        municipality: str,
                                        tek_version: str) -> Dict:
        """Hent detaljerte byggtekniske forskrifter"""
        # TODO: Implementer faktisk henting av forskrifter
        return {}
        
    async def _get_specific_requirements(self,
                                       municipality: str,
                                       property_id: str,
                                       tek_version: str) -> Dict:
        """Hent spesifikke krav for eiendommen"""
        # TODO: Implementer henting av spesifikke krav
        return {}