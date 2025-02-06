"""
MunicipalityAPIClient - Klient for kommunikasjon med kommunale systemer
"""
from typing import Dict, List, Optional
import aiohttp
import asyncio
from dataclasses import dataclass

@dataclass
class MunicipalityPlan:
    plan_id: str
    plan_name: str
    plan_type: str
    status: str
    valid_from: str
    documents: List[str]
    regulations: Dict

class MunicipalityAPIClient:
    def __init__(self):
        self.session = None
        self.base_urls = {
            "drammen": {
                "innsyn": "https://innsyn2020.drammen.kommune.no/",
                "kart": "https://kart.drammen.kommune.no/",
                "byggesak": "https://www.drammen.kommune.no/tjenester/byggesak/"
            }
        }

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_property_cases(self, 
                               municipality: str,
                               gnr: int, 
                               bnr: int,
                               from_year: int = 2020) -> List[Dict]:
        """
        Henter alle saker knyttet til en eiendom
        """
        if municipality.lower() == "drammen":
            return await self._get_drammen_cases(gnr, bnr, from_year)
        # TODO: Implementer støtte for flere kommuner
        raise NotImplementedError(f"Støtte for {municipality} er ikke implementert ennå")

    async def get_zoning_plan(self,
                            municipality: str,
                            gnr: int,
                            bnr: int) -> MunicipalityPlan:
        """
        Henter gjeldende reguleringsplan for eiendommen
        """
        if municipality.lower() == "drammen":
            return await self._get_drammen_zoning_plan(gnr, bnr)
        raise NotImplementedError(f"Støtte for {municipality} er ikke implementert ennå")

    async def get_municipal_plan(self,
                               municipality: str,
                               gnr: int,
                               bnr: int) -> MunicipalityPlan:
        """
        Henter gjeldende kommuneplan for eiendommen
        """
        if municipality.lower() == "drammen":
            return await self._get_drammen_municipal_plan(gnr, bnr)
        raise NotImplementedError(f"Støtte for {municipality} er ikke implementert ennå")

    async def get_property_drawings(self,
                                  municipality: str,
                                  gnr: int,
                                  bnr: int) -> List[Dict]:
        """
        Henter godkjente tegninger for eiendommen
        """
        if municipality.lower() == "drammen":
            return await self._get_drammen_drawings(gnr, bnr)
        raise NotImplementedError(f"Støtte for {municipality} er ikke implementert ennå")

    async def _get_drammen_cases(self,
                                gnr: int,
                                bnr: int,
                                from_year: int) -> List[Dict]:
        """
        Henter saker fra Drammen kommune
        """
        # TODO: Implementer integrasjon mot Drammen kommune
        url = f"{self.base_urls['drammen']['innsyn']}application"
        # TODO: Implementer søkelogikk
        pass

    async def _get_drammen_zoning_plan(self,
                                      gnr: int,
                                      bnr: int) -> MunicipalityPlan:
        """
        Henter reguleringsplan fra Drammen kommune
        """
        # TODO: Implementer integrasjon mot Drammen kommune sitt kartsystem
        pass

    async def _get_drammen_municipal_plan(self,
                                        gnr: int,
                                        bnr: int) -> MunicipalityPlan:
        """
        Henter kommuneplan fra Drammen kommune
        """
        # TODO: Implementer integrasjon mot Drammen kommune sitt kartsystem
        pass

    async def _get_drammen_drawings(self,
                                  gnr: int,
                                  bnr: int) -> List[Dict]:
        """
        Henter tegninger fra Drammen kommune sitt arkiv
        """
        # TODO: Implementer integrasjon mot Drammen kommune sitt saksarkiv
        pass