import logging
from typing import Dict, Any, List
import aiohttp
import json
import os
from bs4 import BeautifulSoup
from fastapi import HTTPException

class MunicipalityService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://innsyn2020.drammen.kommune.no"
        self.api_key = os.getenv("MUNICIPALITY_API_KEY")
        
    async def get_building_history(self, gnr: str, bnr: str) -> List[Dict[str, Any]]:
        """
        Hent byggesakshistorikk fra kommunens arkiv
        """
        try:
            # URL for avansert søk med GNR/BNR
            search_url = f"{self.base_url}/postjournal-v2/fb851964-3185-43eb-81ba-9ac75226dfa8"
            
            async with aiohttp.ClientSession() as session:
                # Utfør søk i byggesaksarkivet
                async with session.post(search_url, json={
                    "gnr": gnr,
                    "bnr": bnr
                }) as response:
                    if response.status != 200:
                        raise HTTPException(status_code=response.status, 
                                         detail="Feil ved henting av byggesakshistorikk")
                    
                    data = await response.json()
                    return self._parse_building_history(data)
        
        except Exception as e:
            self.logger.error(f"Feil ved henting av byggesakshistorikk: {str(e)}")
            raise HTTPException(status_code=500, 
                             detail="Kunne ikke hente byggesakshistorikk")
    
    async def get_zoning_regulations(self, gnr: str, bnr: str) -> Dict[str, Any]:
        """
        Hent reguleringsplan og kommunale bestemmelser
        """
        try:
            regulations = await self._fetch_zoning_regulations(gnr, bnr)
            municipal_plan = await self._fetch_municipal_plan(gnr, bnr)
            
            return {
                "regulations": regulations,
                "municipal_plan": municipal_plan,
                "combined_rules": self._combine_regulations(regulations, municipal_plan)
            }
        
        except Exception as e:
            self.logger.error(f"Feil ved henting av reguleringsbestemmelser: {str(e)}")
            raise HTTPException(status_code=500, 
                             detail="Kunne ikke hente reguleringsbestemmelser")
    
    async def check_development_restrictions(self, gnr: str, bnr: str) -> Dict[str, Any]:
        """
        Sjekk utviklingsrestriksjoner for eiendommen
        """
        try:
            # Hent alle relevante bestemmelser
            regulations = await self.get_zoning_regulations(gnr, bnr)
            
            # Analyser restriksjoner
            restrictions = {
                "max_height": self._extract_max_height(regulations),
                "coverage_rate": self._extract_coverage_rate(regulations),
                "minimum_plot_size": self._extract_minimum_plot_size(regulations),
                "parking_requirements": self._extract_parking_requirements(regulations),
                "special_considerations": self._extract_special_considerations(regulations)
            }
            
            return restrictions
        
        except Exception as e:
            self.logger.error(f"Feil ved sjekk av utviklingsrestriksjoner: {str(e)}")
            raise HTTPException(status_code=500, 
                             detail="Kunne ikke sjekke utviklingsrestriksjoner")
    
    async def _fetch_zoning_regulations(self, gnr: str, bnr: str) -> Dict[str, Any]:
        """
        Hent detaljert reguleringsplan
        """
        # Implementer henting av reguleringsplan
        pass
    
    async def _fetch_municipal_plan(self, gnr: str, bnr: str) -> Dict[str, Any]:
        """
        Hent kommuneplan
        """
        # Implementer henting av kommuneplan
        pass
    
    def _combine_regulations(self, regulations: Dict[str, Any], 
                           municipal_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Kombiner reguleringsplan og kommuneplan
        """
        # Implementer kombinering av bestemmelser
        pass
    
    def _parse_building_history(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse byggesakshistorikk fra rådata
        """
        history = []
        for item in data.get("items", []):
            history.append({
                "case_number": item.get("case_number"),
                "date": item.get("date"),
                "title": item.get("title"),
                "status": item.get("status"),
                "documents": item.get("documents", [])
            })
        return history
    
    def _extract_max_height(self, regulations: Dict[str, Any]) -> float:
        """
        Ekstraher maksimal byggehøyde fra bestemmelser
        """
        # Implementer ekstraksjon av maksimal høyde
        pass
    
    def _extract_coverage_rate(self, regulations: Dict[str, Any]) -> float:
        """
        Ekstraher maksimal utnyttelsesgrad
        """
        # Implementer ekstraksjon av utnyttelsesgrad
        pass
    
    def _extract_minimum_plot_size(self, regulations: Dict[str, Any]) -> float:
        """
        Ekstraher minimum tomtestørrelse
        """
        # Implementer ekstraksjon av minimum tomtestørrelse
        pass
    
    def _extract_parking_requirements(self, regulations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ekstraher parkeringskrav
        """
        # Implementer ekstraksjon av parkeringskrav
        pass
    
    def _extract_special_considerations(self, regulations: Dict[str, Any]) -> List[str]:
        """
        Ekstraher spesielle hensyn og bestemmelser
        """
        # Implementer ekstraksjon av spesielle hensyn
        pass

    async def check_tek_requirements(self, construction_year: int) -> str:
        """
        Bestem hvilken TEK som gjelder for bygningen
        """
        if construction_year >= 2017:
            return "TEK17"
        elif construction_year >= 2010:
            return "TEK10"
        else:
            return "Eldre TEK"