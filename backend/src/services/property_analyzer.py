import os
import uuid
import logging
from typing import Optional, Dict, Any, List
from fastapi import UploadFile
import numpy as np
import cv2
from PIL import Image
import pytesseract
import json
import aiohttp
import asyncio
from datetime import datetime

class PropertyAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.uploads_dir = "uploads"
        os.makedirs(self.uploads_dir, exist_ok=True)

    async def process_upload(self, file: UploadFile) -> str:
        """Process uploaded file and return a file ID"""
        try:
            file_id = str(uuid.uuid4())
            file_path = os.path.join(self.uploads_dir, file_id)
            
            # Save uploaded file
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            return file_id
        except Exception as e:
            self.logger.error(f"Error processing upload: {str(e)}")
            raise

    async def analyze(
        self,
        address: Optional[str] = None,
        file_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze property based on address or uploaded file"""
        try:
            if address:
                return await self._analyze_from_address(address)
            elif file_id:
                return await self._analyze_from_file(file_id)
            else:
                raise ValueError("Either address or file_id must be provided")
        except Exception as e:
            self.logger.error(f"Error in analysis: {str(e)}")
            raise

    async def _analyze_from_address(self, address: str) -> Dict[str, Any]:
        """Analyze property from address"""
        try:
            # Geocode address
            coordinates = await self._geocode_address(address)
            
            # Get municipality code from coordinates
            municipality_code = await self._get_municipality_code(coordinates)
            
            # Get property details from municipality
            property_details = await self._get_property_details(
                municipality_code,
                coordinates
            )
            
            return {
                "address": address,
                "coordinates": coordinates,
                "municipality_code": municipality_code,
                **property_details
            }
        except Exception as e:
            self.logger.error(f"Error analyzing from address: {str(e)}")
            raise

    async def _analyze_from_file(self, file_id: str) -> Dict[str, Any]:
        """Analyze property from uploaded file"""
        try:
            file_path = os.path.join(self.uploads_dir, file_id)
            
            # Extract text from image using OCR
            text = await self._extract_text_from_image(file_path)
            
            # Analyze floor plan
            floor_plan_data = await self._analyze_floor_plan(file_path)
            
            # Extract address if available
            address = await self._extract_address_from_text(text)
            
            if address:
                # Get additional data from address
                address_data = await self._analyze_from_address(address)
                return {
                    **address_data,
                    "floor_plan": floor_plan_data
                }
            else:
                return {
                    "floor_plan": floor_plan_data,
                    "extracted_text": text
                }
        except Exception as e:
            self.logger.error(f"Error analyzing from file: {str(e)}")
            raise

    async def _geocode_address(self, address: str) -> Dict[str, float]:
        """Geocode address to coordinates"""
        # Using Kartverket's geocoding service
        async with aiohttp.ClientSession() as session:
            url = f"https://ws.geonorge.no/adresser/v1/sok?sok={address}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("adresser"):
                        addr = data["adresser"][0]
                        return {
                            "lat": addr["representasjonspunkt"]["lat"],
                            "lon": addr["representasjonspunkt"]["lon"]
                        }
                raise ValueError(f"Could not geocode address: {address}")

    async def _get_municipality_code(self, coordinates: Dict[str, float]) -> str:
        """Get municipality code from coordinates"""
        async with aiohttp.ClientSession() as session:
            url = (f"https://ws.geonorge.no/kommuneinfo/v1/punkt?"
                  f"lat={coordinates['lat']}&lon={coordinates['lon']}")
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("kommunenummer")
                raise ValueError("Could not determine municipality code")

    async def _get_property_details(
        self,
        municipality_code: str,
        coordinates: Dict[str, float]
    ) -> Dict[str, Any]:
        """Get property details from municipality"""
        # This would typically integrate with municipality's API
        # For now, return mock data
        return {
            "plot_size": 500.0,
            "build_year": 1985,
            "bra": 150.0,
            "property_type": "residential",
            "building_type": "detached",
            "floors": 2,
            "has_basement": True,
            "has_attic": True,
            "current_usage": "residential"
        }

    async def _extract_text_from_image(self, file_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image, lang='nor')
            return text
        except Exception as e:
            self.logger.error(f"Error in OCR: {str(e)}")
            return ""

    async def _analyze_floor_plan(self, file_path: str) -> Dict[str, Any]:
        """Analyze floor plan image"""
        try:
            # Load image
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("Could not load image")

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(
                edges,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Analyze room layout
            rooms = self._analyze_rooms(contours)
            
            return {
                "rooms": rooms,
                "total_area": sum(room["area"] for room in rooms),
                "room_count": len(rooms)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing floor plan: {str(e)}")
            return {}

    def _analyze_rooms(self, contours) -> List[Dict[str, Any]]:
        """Analyze rooms from contours"""
        rooms = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter out small contours
                x, y, w, h = cv2.boundingRect(contour)
                rooms.append({
                    "area": area,
                    "dimensions": {"width": w, "height": h},
                    "position": {"x": x, "y": y}
                })
        return rooms

    async def _extract_address_from_text(self, text: str) -> Optional[str]:
        """Extract address from OCR text"""
        # This would need more sophisticated address parsing
        # For now, return None
        return None

    async def analyze_potential(
        self,
        property_info: Dict[str, Any],
        regulations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze development potential"""
        try:
            potential_options = []
            
            # Check basement conversion potential
            if property_info.get("has_basement"):
                potential_options.append({
                    "title": "Utleiedel i kjeller",
                    "description": "Konvertere kjeller til utleiedel",
                    "estimatedCost": 500000,
                    "potentialValue": 800000,
                    "requirements": [
                        "Minimum takhøyde 2.2m",
                        "Separate inngang",
                        "Brannkrav"
                    ]
                })
            
            # Check attic conversion potential
            if property_info.get("has_attic"):
                potential_options.append({
                    "title": "Utvidelse av loft",
                    "description": "Konvertere loft til boareal",
                    "estimatedCost": 600000,
                    "potentialValue": 1000000,
                    "requirements": [
                        "Minimum takhøyde",
                        "Rømningsveier",
                        "Isolasjonskrav"
                    ]
                })
            
            # Check plot division potential
            if property_info.get("plot_size", 0) > 800:
                potential_options.append({
                    "title": "Tomtedeling",
                    "description": "Dele tomten for nybygg",
                    "estimatedCost": 100000,
                    "potentialValue": 2000000,
                    "requirements": [
                        "Minimum tomtestørrelse",
                        "Reguleringsplan",
                        "Vei og VA-tilkobling"
                    ]
                })

            return {
                "options": potential_options,
                "totalPotentialValue": sum(
                    opt["potentialValue"] for opt in potential_options
                )
            }
        except Exception as e:
            self.logger.error(f"Error analyzing potential: {str(e)}")
            raise

    async def analyze_energy(
        self,
        property_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze energy efficiency"""
        try:
            # Mock energy calculation
            base_consumption = property_info.get("bra", 0) * 200  # kWh/year
            
            # Adjust for building age
            year = property_info.get("build_year", 2000)
            age_factor = max(0.5, min(1.5, (2025 - year) / 50))
            
            consumption = base_consumption * age_factor
            
            # Determine energy rating
            rating = self._calculate_energy_rating(consumption, property_info["bra"])
            
            return {
                "rating": rating,
                "consumption": consumption,
                "enovaSupport": [
                    {
                        "title": "Varmepumpe",
                        "description": "Installasjon av luft-til-luft varmepumpe",
                        "amount": 25000,
                        "requirements": ["Godkjent installatør"]
                    },
                    {
                        "title": "Etterisolering",
                        "description": "Etterisolering av tak og vegger",
                        "amount": 40000,
                        "requirements": ["Minimum isolasjonstykkelse"]
                    }
                ]
            }
        except Exception as e:
            self.logger.error(f"Error analyzing energy: {str(e)}")
            raise

    def _calculate_energy_rating(
        self,
        consumption: float,
        area: float
    ) -> str:
        """Calculate energy rating based on consumption and area"""
        consumption_per_m2 = consumption / area
        if consumption_per_m2 < 95:
            return "A"
        elif consumption_per_m2 < 120:
            return "B"
        elif consumption_per_m2 < 145:
            return "C"
        elif consumption_per_m2 < 175:
            return "D"
        elif consumption_per_m2 < 205:
            return "E"
        elif consumption_per_m2 < 250:
            return "F"
        else:
            return "G"