import logging
from typing import List, Dict, Any, Optional
import aiohttp
import json
from bs4 import BeautifulSoup
import re

class MunicipalityService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache = {}

    async def get_regulations(
        self,
        municipality_code: str
    ) -> List[Dict[str, Any]]:
        """Get regulations for a municipality"""
        try:
            # Check cache first
            cache_key = f"regulations_{municipality_code}"
            if cache_key in self.cache:
                return self.cache[cache_key]

            # For Drammen specifically
            if municipality_code == "3005":
                regulations = await self._get_drammen_regulations()
            else:
                regulations = await self._get_generic_regulations(municipality_code)

            # Cache the results
            self.cache[cache_key] = regulations
            return regulations
        except Exception as e:
            self.logger.error(f"Error getting regulations: {str(e)}")
            raise

    async def _get_drammen_regulations(self) -> List[Dict[str, Any]]:
        """Get regulations specifically for Drammen municipality"""
        try:
            # Scrape Drammen's building regulations website
            async with aiohttp.ClientSession() as session:
                url = "https://www.drammen.kommune.no/tjenester/byggesak/slik-soker-du/"
                async with session.get(url) as response:
                    if response.status == 200:
                        text = await response.text()
                        soup = BeautifulSoup(text, 'html.parser')
                        
                        regulations = []
                        
                        # Process main content
                        content = soup.find('main')
                        if content:
                            for section in content.find_all(['h2', 'h3', 'p']):
                                if section.name in ['h2', 'h3']:
                                    title = section.text.strip()
                                    if title:
                                        regulations.append({
                                            "title": title,
                                            "description": "",
                                            "type": "section_header"
                                        })
                                else:
                                    text = section.text.strip()
                                    if text and regulations:
                                        regulations[-1]["description"] += text + "\n"

            # Add specific building requirements
            regulations.extend([
                {
                    "title": "BYA-beregning",
                    "description": "Bebygd areal (BYA) er det arealet som bygningen opptar av terrenget",
                    "type": "calculation",
                    "formula": "%-BYA = (Bebygd areal / Tomteareal) x 100"
                },
                {
                    "title": "Byggegrenser",
                    "description": "Minimum avstand til nabogrense er 4 meter hvis ikke annet er spesifisert i reguleringsplan",
                    "type": "requirement"
                },
                {
                    "title": "Høydebestemmelser",
                    "description": "Maksimal gesimshøyde og mønehøyde er definert i gjeldende reguleringsplan",
                    "type": "requirement"
                },
                {
                    "title": "Utleiedel krav",
                    "description": """
                        - Minimum takhøyde: 2.2 meter
                        - Separate inngang
                        - Tilfredsstillende lysforhold
                        - Brannskiller mot hovedboenhet
                        - Ventilasjon iht. TEK17
                        - Våtrom må tilfredsstille tekniske krav
                    """.strip(),
                    "type": "requirement"
                }
            ])

            return regulations
        except Exception as e:
            self.logger.error(f"Error getting Drammen regulations: {str(e)}")
            raise

    async def _get_generic_regulations(
        self,
        municipality_code: str
    ) -> List[Dict[str, Any]]:
        """Get regulations for other municipalities"""
        # This would integrate with other municipality systems
        # For now, return basic regulations
        return [
            {
                "title": "Generelle byggeregler",
                "description": "Standard byggeregler som gjelder i kommunen",
                "type": "general"
            },
            {
                "title": "Avstandskrav",
                "description": "4 meter til nabogrense",
                "type": "requirement"
            }
        ]

    async def get_property_history(
        self,
        property_id: str
    ) -> List[Dict[str, Any]]:
        """Get property history from municipal archives"""
        try:
            municipality_code = property_id[:4]
            
            if municipality_code == "3005":  # Drammen
                return await self._get_drammen_property_history(property_id)
            else:
                return await self._get_generic_property_history(property_id)
        except Exception as e:
            self.logger.error(f"Error getting property history: {str(e)}")
            raise

    async def _get_drammen_property_history(
        self,
        property_id: str
    ) -> List[Dict[str, Any]]:
        """Get property history specifically from Drammen municipality"""
        try:
            # Parse property ID to get gnr/bnr
            match = re.match(r"3005-(\d+)-(\d+)", property_id)
            if not match:
                raise ValueError("Invalid property ID format")
            
            gnr, bnr = match.groups()
            
            async with aiohttp.ClientSession() as session:
                url = (
                    "https://innsyn2020.drammen.kommune.no/postjournal-v2/"
                    f"fb851964-3185-43eb-81ba-9ac75226dfa8?gnr={gnr}&bnr={bnr}"
                )
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_drammen_cases(data)
                    else:
                        raise ValueError(
                            f"Failed to get property history: {response.status}"
                        )
        except Exception as e:
            self.logger.error(
                f"Error getting Drammen property history: {str(e)}"
            )
            raise

    def _parse_drammen_cases(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse case data from Drammen's API"""
        cases = []
        for case in data.get("cases", []):
            cases.append({
                "case_number": case.get("caseNumber"),
                "title": case.get("title"),
                "status": case.get("status"),
                "date": case.get("date"),
                "documents": [
                    {
                        "title": doc.get("title"),
                        "date": doc.get("date"),
                        "type": doc.get("type"),
                        "url": doc.get("url")
                    }
                    for doc in case.get("documents", [])
                ]
            })
        return cases

    async def _get_generic_property_history(
        self,
        property_id: str
    ) -> List[Dict[str, Any]]:
        """Get property history for other municipalities"""
        # This would integrate with other municipality systems
        # For now, return empty list
        return []

    async def check_zoning_restrictions(
        self,
        municipality_code: str,
        property_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check zoning restrictions for a property"""
        try:
            if municipality_code == "3005":  # Drammen
                return await self._check_drammen_zoning(property_info)
            else:
                return await self._check_generic_zoning(
                    municipality_code,
                    property_info
                )
        except Exception as e:
            self.logger.error(f"Error checking zoning: {str(e)}")
            raise

    async def _check_drammen_zoning(
        self,
        property_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check zoning restrictions specifically for Drammen"""
        # This would integrate with Drammen's zoning system
        # For now, return mock data
        return {
            "zone_type": "residential",
            "max_bya": 30,
            "max_height": 9.0,
            "min_plot_size": 600,
            "restrictions": [
                "Maksimalt to boenheter per eiendom",
                "Minimum 18m² uteoppholdsareal per boenhet"
            ]
        }

    async def _check_generic_zoning(
        self,
        municipality_code: str,
        property_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check zoning restrictions for other municipalities"""
        # This would integrate with other municipality systems
        # For now, return basic restrictions
        return {
            "zone_type": "residential",
            "max_bya": 25,
            "max_height": 8.0,
            "restrictions": [
                "Standard boligformål"
            ]
        }