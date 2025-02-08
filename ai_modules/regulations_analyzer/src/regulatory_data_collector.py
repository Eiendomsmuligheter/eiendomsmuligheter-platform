"""
Regulatory Data Collector
------------------------
Modul for automatisk innhenting og oppdatering av forskrifter og krav fra
relevante kilder som DiBK, Standard Norge, Byggforsk, etc.
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
from datetime import datetime
import json
import asyncio
import aiohttp
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RegulatorySource:
    name: str
    base_url: str
    update_frequency: str
    last_updated: datetime
    data_type: str

class RegulatoryDataCollector:
    def __init__(self):
        self.sources = {
            "dibk": RegulatorySource(
                name="Direktoratet for byggkvalitet",
                base_url="https://www.dibk.no",
                update_frequency="daily",
                last_updated=datetime.now(),
                data_type="building_regulations"
            ),
            "standard_norge": RegulatorySource(
                name="Standard Norge",
                base_url="https://www.standard.no",
                update_frequency="weekly",
                last_updated=datetime.now(),
                data_type="standards"
            ),
            "byggforsk": RegulatorySource(
                name="Byggforsk",
                base_url="https://www.byggforsk.no",
                update_frequency="weekly",
                last_updated=datetime.now(),
                data_type="technical_guidelines"
            ),
            "lovdata": RegulatorySource(
                name="Lovdata",
                base_url="https://lovdata.no",
                update_frequency="daily",
                last_updated=datetime.now(),
                data_type="laws_regulations"
            )
        }
        self.cache_dir = Path("data/regulatory_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def update_all_regulations(self) -> Dict[str, Any]:
        """Oppdater alle forskrifter og krav fra alle kilder"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for source_id, source in self.sources.items():
                tasks.append(self.update_source(session, source_id))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return self._process_update_results(results)

    async def update_source(self, session: aiohttp.ClientSession, source_id: str) -> Dict[str, Any]:
        """Oppdater data fra en spesifikk kilde"""
        source = self.sources[source_id]
        try:
            if source_id == "dibk":
                return await self._update_dibk(session)
            elif source_id == "standard_norge":
                return await self._update_standard_norge(session)
            elif source_id == "byggforsk":
                return await self._update_byggforsk(session)
            elif source_id == "lovdata":
                return await self._update_lovdata(session)
            else:
                raise ValueError(f"Ukjent kilde: {source_id}")

        except Exception as e:
            logger.error(f"Feil ved oppdatering av {source.name}: {str(e)}")
            return {"error": str(e), "source": source_id}

    async def _update_dibk(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Oppdater byggeforskrifter fra DiBK"""
        regulations = {
            "tek17": await self._get_tek17_regulations(session),
            "sak10": await self._get_sak10_regulations(session),
            "guidelines": await self._get_dibk_guidelines(session)
        }
        
        self._cache_data("dibk", regulations)
        return regulations

    async def _update_standard_norge(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Oppdater relevante standarder fra Standard Norge"""
        standards = {
            "building": await self._get_building_standards(session),
            "safety": await self._get_safety_standards(session),
            "energy": await self._get_energy_standards(session)
        }
        
        self._cache_data("standard_norge", standards)
        return standards

    async def _update_byggforsk(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Oppdater tekniske anvisninger fra Byggforsk"""
        guidelines = {
            "construction": await self._get_construction_guidelines(session),
            "renovation": await self._get_renovation_guidelines(session),
            "rental": await self._get_rental_guidelines(session)
        }
        
        self._cache_data("byggforsk", guidelines)
        return guidelines

    async def _update_lovdata(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Oppdater lover og forskrifter fra Lovdata"""
        regulations = {
            "building_act": await self._get_building_act(session),
            "planning_act": await self._get_planning_act(session),
            "local_regulations": await self._get_local_regulations(session)
        }
        
        self._cache_data("lovdata", regulations)
        return regulations

    def _cache_data(self, source_id: str, data: Dict[str, Any]) -> None:
        """Lagre data i lokalt cache"""
        cache_file = self.cache_dir / f"{source_id}_cache.json"
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump({
                "last_updated": datetime.now().isoformat(),
                "data": data
            }, f, ensure_ascii=False, indent=2)

    def get_cached_data(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Hent cached data for en kilde"""
        cache_file = self.cache_dir / f"{source_id}_cache.json"
        if cache_file.exists():
            with cache_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        return None

    async def get_rental_requirements(self) -> Dict[str, Any]:
        """Hent alle krav relatert til utleie"""
        requirements = {
            "building": await self._get_building_requirements(),
            "fire_safety": await self._get_fire_safety_requirements(),
            "ventilation": await self._get_ventilation_requirements(),
            "tax": await self._get_tax_requirements()
        }
        return requirements

    async def _get_building_requirements(self) -> Dict[str, Any]:
        """Hent byggtekniske krav"""
        return {
            "ceiling_height": {
                "minimum": 2.4,
                "source": "TEK17 § 12-7",
                "description": "Minimum takhøyde i oppholdsrom"
            },
            "window_area": {
                "minimum_ratio": 0.10,
                "source": "TEK17 § 13-7",
                "description": "Minimum glassareal i forhold til gulvareal"
            },
            "room_sizes": {
                "bedroom": {
                    "minimum": 7.0,
                    "source": "TEK17 § 12-7",
                    "description": "Minimum areal for soverom"
                },
                "living_room": {
                    "minimum": 15.0,
                    "source": "TEK17 § 12-7",
                    "description": "Anbefalt minimum for stue"
                }
            }
        }

    async def _get_fire_safety_requirements(self) -> Dict[str, Any]:
        """Hent brannsikkerhetskrav"""
        return {
            "fire_resistance": {
                "walls": "EI 30",
                "ceilings": "EI 30",
                "source": "TEK17 § 11-8"
            },
            "escape_routes": {
                "maximum_distance": 30,
                "minimum_width": 0.9,
                "source": "TEK17 § 11-14"
            },
            "fire_alarms": {
                "required": True,
                "interconnected": True,
                "source": "TEK17 § 11-12"
            }
        }

    async def _get_ventilation_requirements(self) -> Dict[str, Any]:
        """Hent ventilasjonskrav"""
        return {
            "air_exchange": {
                "minimum": 0.5,
                "unit": "luftvekslinger per time",
                "source": "TEK17 § 13-1"
            },
            "fresh_air": {
                "living_areas": {
                    "minimum": 1.2,
                    "unit": "m³ per time per m² gulvareal",
                    "source": "TEK17 § 13-1"
                },
                "bedrooms": {
                    "minimum": 26,
                    "unit": "m³ per time per person",
                    "source": "TEK17 § 13-1"
                }
            }
        }

    async def _get_tax_requirements(self) -> Dict[str, Any]:
        """Hent skattekrav for utleie"""
        return {
            "rental_income": {
                "declaration_required": True,
                "tax_rate": 0.22,
                "source": "Skatteloven § 7-2"
            },
            "deductions": {
                "maintenance": True,
                "insurance": True,
                "municipal_fees": True,
                "source": "Skatteloven § 7-2"
            }
        }