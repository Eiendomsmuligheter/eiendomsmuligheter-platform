from typing import Dict, List, Optional
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import json
import re

class MunicipalityService:
    def __init__(self):
        self.base_url = "https://innsyn2020.drammen.kommune.no"
        self.session = None
        self.regulations_cache = {}

    async def _init_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def get_regulations(
        self,
        municipality: str,
        gnr: int,
        bnr: int
    ) -> Dict:
        """
        Hent reguleringsbestemmelser for en eiendom
        """
        await self._init_session()
        
        # Generer cache-nøkkel
        cache_key = f"{municipality}_{gnr}_{bnr}"
        
        # Sjekk cache først
        if cache_key in self.regulations_cache:
            return self.regulations_cache[cache_key]

        # Hent data fra kommunen
        regulations = await self._fetch_regulations(municipality, gnr, bnr)
        
        # Lagre i cache
        self.regulations_cache[cache_key] = regulations
        
        return regulations

    async def _fetch_regulations(
        self,
        municipality: str,
        gnr: int,
        bnr: int
    ) -> Dict:
        """
        Hent reguleringsdata fra kommunens API
        """
        # Konstruer søkestreng
        search_url = f"{self.base_url}/postjournal-v2/search"
        params = {
            "gnr": gnr,
            "bnr": bnr,
            "year_from": "2020",
            "municipality": municipality
        }

        try:
            async with self.session.get(search_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._parse_regulation_data(data)
                else:
                    raise Exception(f"Failed to fetch regulations: {response.status}")
        except Exception as e:
            print(f"Error fetching regulations: {str(e)}")
            return {}

    async def _parse_regulation_data(self, raw_data: Dict) -> Dict:
        """
        Parse og strukturer reguleringsdata
        """
        regulations = {
            "zoning": "",
            "utilization_rate": 0.0,
            "height_restrictions": {},
            "special_regulations": [],
            "building_lines": {},
            "parking_requirements": {},
            "cases": []
        }

        try:
            # Parser reguleringsinformasjon
            if "reguleringsplan" in raw_data:
                plan_data = raw_data["reguleringsplan"]
                regulations["zoning"] = plan_data.get("formål", "")
                regulations["utilization_rate"] = self._parse_utilization_rate(
                    plan_data.get("utnyttelse", "")
                )
                regulations["height_restrictions"] = self._parse_height_restrictions(
                    plan_data.get("høyder", {})
                )

            # Parser byggesaker
            if "byggesaker" in raw_data:
                for case in raw_data["byggesaker"]:
                    parsed_case = self._parse_building_case(case)
                    if parsed_case:
                        regulations["cases"].append(parsed_case)

        except Exception as e:
            print(f"Error parsing regulation data: {str(e)}")

        return regulations

    def _parse_utilization_rate(self, utilization_str: str) -> float:
        """
        Parser utnyttelsesgrad fra string til float
        """
        try:
            # Fjern % og konverter til float
            match = re.search(r"(\d+(?:\.\d+)?)", utilization_str)
            if match:
                return float(match.group(1))
        except Exception:
            pass
        return 0.0

    def _parse_height_restrictions(self, height_data: Dict) -> Dict:
        """
        Parser høyderestriksjoner
        """
        restrictions = {
            "max_height": 0.0,
            "max_stories": 0,
            "specific_restrictions": []
        }

        try:
            if isinstance(height_data, dict):
                restrictions["max_height"] = float(height_data.get("maks_høyde", 0))
                restrictions["max_stories"] = int(height_data.get("maks_etasjer", 0))
                
                if "spesielle_krav" in height_data:
                    restrictions["specific_restrictions"] = height_data["spesielle_krav"]

        except Exception as e:
            print(f"Error parsing height restrictions: {str(e)}")

        return restrictions

    def _parse_building_case(self, case_data: Dict) -> Optional[Dict]:
        """
        Parser byggesaksinformasjon
        """
        try:
            return {
                "case_number": case_data.get("saksnummer", ""),
                "case_type": case_data.get("sakstype", ""),
                "status": case_data.get("status", ""),
                "decision": case_data.get("vedtak", ""),
                "date": case_data.get("dato", ""),
                "documents": self._parse_case_documents(case_data.get("dokumenter", []))
            }
        except Exception:
            return None

    def _parse_case_documents(self, documents: List) -> List[Dict]:
        """
        Parser dokumentliste fra byggesak
        """
        parsed_docs = []
        
        for doc in documents:
            try:
                parsed_docs.append({
                    "title": doc.get("tittel", ""),
                    "type": doc.get("dokumenttype", ""),
                    "date": doc.get("dato", ""),
                    "url": doc.get("url", "")
                })
            except Exception:
                continue

        return parsed_docs

    async def get_municipality_requirements(self, municipality: str) -> Dict:
        """
        Hent generelle krav for kommunen
        """
        requirements = {
            "parking": self._get_parking_requirements(municipality),
            "outdoor_area": self._get_outdoor_area_requirements(municipality),
            "waste": self._get_waste_management_requirements(municipality)
        }
        return requirements

    def _get_parking_requirements(self, municipality: str) -> Dict:
        """
        Hent parkeringskrav for kommunen
        """
        # Dette vil variere mellom kommuner
        if municipality.lower() == "drammen":
            return {
                "residential": {
                    "car": {
                        "min_spaces_per_unit": 1,
                        "visitor_spaces": 0.2
                    },
                    "bicycle": {
                        "min_spaces_per_unit": 2,
                        "visitor_spaces": 0.5
                    }
                }
            }
        return {}

    def _get_outdoor_area_requirements(self, municipality: str) -> Dict:
        """
        Hent krav til utearealer
        """
        if municipality.lower() == "drammen":
            return {
                "residential": {
                    "min_total_area_per_unit": 30,
                    "min_private_area": 5,
                    "min_common_area": 25,
                    "quality_requirements": [
                        "Minimum 50% skal ha direkte sollys kl. 15:00 ved vårjevndøgn",
                        "Støynivå skal ikke overstige 55 dB",
                        "Universell utforming kreves"
                    ]
                }
            }
        return {}

    def _get_waste_management_requirements(self, municipality: str) -> Dict:
        """
        Hent krav til avfallshåndtering
        """
        if municipality.lower() == "drammen":
            return {
                "residential": {
                    "sorting_requirements": [
                        "Restavfall",
                        "Matavfall",
                        "Papp/papir",
                        "Plast",
                        "Glass/metall"
                    ],
                    "accessibility": "Maksimalt 100 meter fra hovedinngang",
                    "technical_requirements": [
                        "Nedgravde containere ved 20 boenheter eller mer",
                        "Tilgjengelig for renovasjonsbil"
                    ]
                }
            }
        return {}

    async def close(self):
        """
        Lukk aiohttp session
        """
        if self.session:
            await self.session.close()
            self.session = None