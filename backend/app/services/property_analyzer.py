from typing import Dict, List, Optional
import cv2
import numpy as np
from PIL import Image
import pytesseract
import requests
from bs4 import BeautifulSoup
from ..models.property_analysis import PropertyAnalysis
from sqlalchemy.orm import Session
import os
import json

class PropertyAnalyzer:
    def __init__(self, db: Session):
        self.db = db
        self.municipality_cache = {}
        
    async def analyze_property(
        self,
        image_path: Optional[str] = None,
        address: Optional[str] = None,
        municipality_code: Optional[str] = None
    ) -> Dict:
        """
        Analyserer en eiendom basert på bilde og/eller adresse
        """
        results = {
            "property_info": {},
            "development_potential": {},
            "regulations": {},
            "room_analysis": {},
            "measurements": {},
            "recommendations": []
        }
        
        if image_path:
            # Analyser plantegning
            floor_plan_results = await self._analyze_floor_plan(image_path)
            results.update(floor_plan_results)
        
        if address:
            # Hent eiendomsinformasjon
            property_info = await self._fetch_property_info(address)
            results["property_info"] = property_info
            
            # Hent reguleringsdata
            if municipality_code:
                regulations = await self._fetch_regulations(
                    municipality_code,
                    property_info.get("gnr"),
                    property_info.get("bnr")
                )
                results["regulations"] = regulations
        
        # Analyser utviklingspotensial
        results["development_potential"] = await self._analyze_development_potential(
            results["property_info"],
            results["regulations"],
            results.get("room_analysis", {})
        )
        
        return results
    
    async def _analyze_floor_plan(self, image_path: str) -> Dict:
        """
        Analyserer en plantegning ved hjelp av computer vision
        """
        # Last inn bildet
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Konverter til gråtoner for bedre analyse
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Finn konturer (vegger, rom, etc.)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Analyser rom
        rooms = self._analyze_rooms(contours, image.shape)
        
        # OCR for å finne mål og tekst
        measurements = self._extract_measurements(image_rgb)
        
        return {
            "room_analysis": rooms,
            "measurements": measurements
        }
    
    def _analyze_rooms(self, contours: List, image_shape: tuple) -> Dict:
        """
        Analyserer rom basert på konturer
        """
        rooms = []
        min_room_area = 1000  # Minimum areal for å regne som rom
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_room_area:
                # Finn rommets form
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                
                # Beregn areal i kvadratmeter (antar skala)
                area_m2 = area / 100  # Justér etter faktisk skala
                
                # Klassifiser romtype basert på form og størrelse
                room_type = self._classify_room_type(area_m2, len(approx))
                
                rooms.append({
                    "type": room_type,
                    "area_m2": round(area_m2, 2),
                    "corners": len(approx)
                })
        
        return {
            "total_rooms": len(rooms),
            "rooms": rooms,
            "total_area": sum(r["area_m2"] for r in rooms)
        }
    
    def _classify_room_type(self, area: float, corners: int) -> str:
        """
        Klassifiserer romtype basert på areal og form
        """
        if area < 5:
            return "WC"
        elif area < 8:
            return "Bad"
        elif area < 12:
            return "Soverom"
        elif area < 20:
            return "Kjøkken"
        elif area < 30:
            return "Stue"
        else:
            return "Oppholdsrom"
    
    def _extract_measurements(self, image: np.ndarray) -> Dict:
        """
        Ekstraherer mål fra tegningen ved hjelp av OCR
        """
        # Konverter numpy array til PIL Image
        pil_image = Image.fromarray(image)
        
        # Utfør OCR
        text = pytesseract.image_to_string(pil_image)
        
        # Finn mål i teksten
        measurements = {
            "width": [],
            "length": [],
            "height": []
        }
        
        # Parse tekst for å finne mål
        import re
        
        # Finn mål på formen "X,XX m" eller "X.XX m"
        pattern = r'(\d+[.,]\d+)\s*m'
        found_measurements = re.findall(pattern, text)
        
        for measure in found_measurements:
            # Konverter til float og erstatt komma med punktum
            value = float(measure.replace(',', '.'))
            
            # Kategoriser målet basert på størrelse
            if value < 3:
                measurements["height"].append(value)
            elif value < 10:
                measurements["width"].append(value)
            else:
                measurements["length"].append(value)
        
        return measurements
    
    async def _fetch_property_info(self, address: str) -> Dict:
        """
        Henter eiendomsinformasjon fra offentlige kilder
        """
        # Eksempel-implementasjon - erstatt med faktisk API-kall
        try:
            # Kartverket API-kall
            api_url = f"https://ws.geonorge.no/adresser/v1/sok?sok={address}"
            response = requests.get(api_url)
            data = response.json()
            
            if data.get("adresser"):
                address_data = data["adresser"][0]
                return {
                    "address": address_data.get("adressetekst"),
                    "municipality": address_data.get("kommunenavn"),
                    "municipality_code": address_data.get("kommunenummer"),
                    "gnr": address_data.get("gardsnummer"),
                    "bnr": address_data.get("bruksnummer"),
                    "coordinates": {
                        "lat": address_data.get("representasjonspunkt", {}).get("lat"),
                        "lon": address_data.get("representasjonspunkt", {}).get("lon")
                    }
                }
            return {}
            
        except Exception as e:
            print(f"Feil ved henting av eiendomsinfo: {str(e)}")
            return {}
    
    async def _fetch_regulations(
        self,
        municipality_code: str,
        gnr: str,
        bnr: str
    ) -> Dict:
        """
        Henter reguleringsdata fra kommunen
        """
        # Cache-sjekk
        cache_key = f"{municipality_code}_{gnr}_{bnr}"
        if cache_key in self.municipality_cache:
            return self.municipality_cache[cache_key]
        
        # Eksempel for Drammen kommune
        if municipality_code == "3005":
            url = f"https://innsyn2020.drammen.kommune.no/postjournal-v2/fb851964-3185-43eb-81ba-9ac75226dfa8"
            
            try:
                # Hent reguleringsdata
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                regulations = {
                    "zoning_plan": self._extract_zoning_info(soup),
                    "building_restrictions": self._extract_restrictions(soup),
                    "max_utilization": self._extract_utilization(soup),
                    "special_conditions": self._extract_conditions(soup)
                }
                
                # Cache resultatet
                self.municipality_cache[cache_key] = regulations
                return regulations
                
            except Exception as e:
                print(f"Feil ved henting av reguleringsdata: {str(e)}")
                return {}
        
        return {}
    
    def _extract_zoning_info(self, soup: BeautifulSoup) -> Dict:
        """
        Ekstraherer reguleringsplaninformasjon fra HTML
        """
        return {
            "plan_type": "Kommuneplan",  # Standard verdi - må tilpasses
            "purpose": "Bolig",  # Standard verdi - må tilpasses
            "plan_id": "Unknown",
            "valid_from": "Unknown"
        }
    
    def _extract_restrictions(self, soup: BeautifulSoup) -> Dict:
        """
        Ekstraherer byggerestriksjoner fra HTML
        """
        return {
            "max_height": "8m",  # Standard verdi - må tilpasses
            "min_distance_boundary": "4m",
            "parking_requirements": "2 per unit"
        }
    
    def _extract_utilization(self, soup: BeautifulSoup) -> Dict:
        """
        Ekstraherer utnyttelsesgrad fra HTML
        """
        return {
            "max_bya": "30%",  # Standard verdi - må tilpasses
            "max_bra": "60%",
            "max_units": "2"
        }
    
    def _extract_conditions(self, soup: BeautifulSoup) -> List[str]:
        """
        Ekstraherer spesielle vilkår fra HTML
        """
        return [
            "Verneverdig bebyggelse",
            "Krav om utomhusplan"
        ]
    
    async def _analyze_development_potential(
        self,
        property_info: Dict,
        regulations: Dict,
        room_analysis: Dict
    ) -> Dict:
        """
        Analyserer utviklingspotensial basert på all tilgjengelig informasjon
        """
        potential = {
            "possible_developments": [],
            "constraints": [],
            "recommendations": []
        }
        
        # Sjekk mulighet for utleieenhet
        if self._can_add_rental_unit(regulations, room_analysis):
            potential["possible_developments"].append({
                "type": "rental_unit",
                "description": "Mulighet for utleieenhet",
                "requirements": [
                    "Minimum 40m² BRA",
                    "Egen inngang",
                    "Oppfyller tekniske krav"
                ]
            })
        
        # Sjekk mulighet for påbygg
        if self._can_add_extension(regulations, property_info):
            potential["possible_developments"].append({
                "type": "extension",
                "description": "Mulighet for påbygg",
                "potential_area": "40m²"
            })
        
        # Sjekk mulighet for garasje/carport
        if self._can_add_garage(regulations, property_info):
            potential["possible_developments"].append({
                "type": "garage",
                "description": "Mulighet for garasje",
                "max_size": "50m²"
            })
        
        # Legg til begrensninger
        if regulations.get("special_conditions"):
            potential["constraints"].extend(regulations["special_conditions"])
        
        # Legg til anbefalinger
        potential["recommendations"] = self._generate_recommendations(
            potential["possible_developments"],
            regulations
        )
        
        return potential
    
    def _can_add_rental_unit(self, regulations: Dict, room_analysis: Dict) -> bool:
        """
        Sjekker om det er mulig å legge til utleieenhet
        """
        max_bya = self._parse_percentage(regulations.get("max_bya", "0%"))
        current_area = room_analysis.get("total_area", 0)
        
        # Forenklet sjekk - må utvides med flere kriterier
        return current_area >= 120 and max_bya >= 30
    
    def _can_add_extension(self, regulations: Dict, property_info: Dict) -> bool:
        """
        Sjekker om det er mulig å bygge på
        """
        # Forenklet implementasjon - må utvides
        return True
    
    def _can_add_garage(self, regulations: Dict, property_info: Dict) -> bool:
        """
        Sjekker om det er mulig å bygge garasje
        """
        # Forenklet implementasjon - må utvides
        return True
    
    def _parse_percentage(self, percentage_str: str) -> float:
        """
        Konverterer prosentstreng til float
        """
        try:
            return float(percentage_str.strip('%'))
        except:
            return 0.0
    
    def _generate_recommendations(
        self,
        possibilities: List[Dict],
        regulations: Dict
    ) -> List[str]:
        """
        Genererer anbefalinger basert på muligheter og reguleringer
        """
        recommendations = []
        
        for possibility in possibilities:
            if possibility["type"] == "rental_unit":
                recommendations.append(
                    "Anbefaler å vurdere utleieenhet i underetasjen"
                )
            elif possibility["type"] == "extension":
                recommendations.append(
                    "Vurder påbygg for å øke boarealet"
                )
            elif possibility["type"] == "garage":
                recommendations.append(
                    "Garasje vil øke eiendommens verdi"
                )
        
        return recommendations