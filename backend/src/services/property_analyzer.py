import os
import re
import uuid
import logging
import urllib.parse
from typing import Optional, Dict, Any, List, Tuple
from fastapi import UploadFile, HTTPException
import numpy as np
import cv2
from PIL import Image
import pytesseract
import json
import aiohttp
import asyncio
from datetime import datetime
from pathlib import Path
import fitz  # PyMuPDF for PDF processing
import ezdxf  # for CAD file processing
from ..models.analysis import AnalysisSession, AnalysisResult
from ..models.property import PropertyDetails, Room, BuildingRegulations

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
        files: Optional[List[UploadFile]] = None,
        payment_intent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze property based on address and/or uploaded files
        Args:
            address: Property address
            files: List of uploaded files (images, PDFs, CAD files)
            payment_intent_id: Stripe payment intent ID
        Returns:
            Complete analysis including property details, potential, and recommendations
        """
        try:
            analysis_id = str(uuid.uuid4())
            self.logger.info(f"Starting analysis {analysis_id}")
            
            # Start analysis session
            session = await self._create_analysis_session(analysis_id, payment_intent_id)
            
            results = {
                "analysis_id": analysis_id,
                "status": "processing",
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            if address:
                # Analyze from address
                address_results = await self._analyze_from_address(address)
                results.update(address_results)
            
            if files:
                # Process all uploaded files
                file_results = await asyncio.gather(
                    *[self._process_file(file) for file in files]
                )
                
                # Combine results from all files
                combined_results = await self._combine_file_results(file_results)
                results.update(combined_results)
            
            # Validate results
            if not results.get("property_details"):
                raise ValueError("Could not determine property details")
            
            # Get municipality regulations
            regulations = await self._get_municipality_regulations(
                results["property_details"]["municipality_code"]
            )
            
            # Analyze development potential
            potential = await self.analyze_potential(
                results["property_details"],
                regulations
            )
            results["development_potential"] = potential
            
            # Analyze energy efficiency
            energy = await self.analyze_energy(results["property_details"])
            results["energy_analysis"] = energy
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(results)
            results["recommendations"] = recommendations
            
            # Update status
            results["status"] = "completed"
            await self._update_analysis_session(analysis_id, results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in analysis {analysis_id}: {str(e)}")
            if analysis_id:
                await self._update_analysis_session(
                    analysis_id,
                    {"status": "failed", "error": str(e)}
                )
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
        """
        Analyze rooms from contours with advanced room recognition
        Returns list of rooms with area, dimensions, and suggested usage
        """
        rooms = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter out small contours
                x, y, w, h = cv2.boundingRect(contour)
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                
                # Analyze room shape
                shape_factor = (4 * np.pi * area) / (perimeter ** 2)
                rectangularity = area / (w * h)
                
                # Determine likely room type based on size and shape
                room_type = self._determine_room_type(area, shape_factor, rectangularity)
                
                # Calculate minimum ceiling height based on room type
                min_ceiling_height = self._get_min_ceiling_height(room_type)
                
                rooms.append({
                    "area": area,
                    "dimensions": {
                        "width": w,
                        "height": h,
                        "min_ceiling_height": min_ceiling_height
                    },
                    "position": {"x": x, "y": y},
                    "shape_analysis": {
                        "corners": len(approx),
                        "shape_factor": shape_factor,
                        "rectangularity": rectangularity
                    },
                    "suggested_usage": room_type,
                    "requirements": self._get_room_requirements(room_type)
                })
        
        # Analyze room relationships
        rooms = self._analyze_room_relationships(rooms)
        return rooms
    
    def _determine_room_type(
        self,
        area: float,
        shape_factor: float,
        rectangularity: float
    ) -> str:
        """Determine room type based on characteristics"""
        if area < 5000:  # Less than 5m²
            if rectangularity > 0.85:
                return "bathroom"
            else:
                return "storage"
        elif area < 10000:  # 5-10m²
            if shape_factor > 0.8:
                return "bedroom"
            else:
                return "kitchen"
        elif area < 20000:  # 10-20m²
            if rectangularity > 0.9:
                return "living_room"
            else:
                return "dining_room"
        else:  # Larger than 20m²
            return "multi_purpose"
    
    def _get_min_ceiling_height(self, room_type: str) -> float:
        """Get minimum ceiling height based on room type"""
        heights = {
            "living_room": 2.4,
            "bedroom": 2.2,
            "kitchen": 2.2,
            "bathroom": 2.2,
            "storage": 2.0,
            "multi_purpose": 2.4
        }
        return heights.get(room_type, 2.2)
    
    def _get_room_requirements(self, room_type: str) -> List[str]:
        """Get building code requirements for room type"""
        requirements = {
            "living_room": [
                "Minimum ett vindu med lysareal minst 10% av gulvareal",
                "Direkte tilgang til rømningsvei",
                "God ventilasjon"
            ],
            "bedroom": [
                "Minimum ett vindu med lysareal minst 10% av gulvareal",
                "Rømningsvei via vindu eller dør",
                "Minimum 7m² for enkeltrom"
            ],
            "kitchen": [
                "Mekanisk avtrekk",
                "Tilgang til vann og avløp",
                "Minimum 30cm benkeplate på hver side av komfyr"
            ],
            "bathroom": [
                "Vanntett membran på gulv og vegger",
                "Mekanisk ventilasjon",
                "Sluk i gulv"
            ],
            "storage": [
                "Ventilasjon",
                "Minimum 1.5m² dersom det er hovedbod"
            ],
            "multi_purpose": [
                "Minimum ett vindu med lysareal minst 10% av gulvareal",
                "God ventilasjon",
                "Rømningsvei"
            ]
        }
        return requirements.get(room_type, [])

    def _analyze_room_relationships(self, rooms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze spatial relationships between rooms"""
        for i, room in enumerate(rooms):
            adjacent_rooms = []
            room_center = (
                room["position"]["x"] + room["dimensions"]["width"] / 2,
                room["position"]["y"] + room["dimensions"]["height"] / 2
            )
            
            for j, other in enumerate(rooms):
                if i != j:
                    other_center = (
                        other["position"]["x"] + other["dimensions"]["width"] / 2,
                        other["position"]["y"] + other["dimensions"]["height"] / 2
                    )
                    
                    # Check if rooms are adjacent
                    distance = np.sqrt(
                        (room_center[0] - other_center[0]) ** 2 +
                        (room_center[1] - other_center[1]) ** 2
                    )
                    
                    if distance < (room["dimensions"]["width"] + other["dimensions"]["width"]) / 2:
                        adjacent_rooms.append({
                            "room_type": other["suggested_usage"],
                            "direction": self._get_direction(room_center, other_center)
                        })
            
            room["adjacent_rooms"] = adjacent_rooms
            
            # Add compatibility analysis
            room["compatibility_issues"] = self._check_room_compatibility(
                room["suggested_usage"],
                adjacent_rooms
            )
        
        return rooms
    
    def _get_direction(self, from_point: tuple, to_point: tuple) -> str:
        """Determine relative direction between two points"""
        dx = to_point[0] - from_point[0]
        dy = to_point[1] - from_point[1]
        
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        if -45 <= angle <= 45:
            return "east"
        elif 45 < angle <= 135:
            return "north"
        elif angle > 135 or angle < -135:
            return "west"
        else:
            return "south"
    
    def _check_room_compatibility(
        self,
        room_type: str,
        adjacent_rooms: List[Dict[str, str]]
    ) -> List[str]:
        """Check for potential issues with room arrangements"""
        issues = []
        
        # Define incompatible arrangements
        incompatible = {
            "kitchen": ["bathroom"],
            "bedroom": ["kitchen"],
            "living_room": ["bathroom"],
        }
        
        # Check for incompatible adjacent rooms
        if room_type in incompatible:
            for adj in adjacent_rooms:
                if adj["room_type"] in incompatible[room_type]:
                    issues.append(
                        f"Uheldig plassering: {room_type} bør ikke ligge inntil {adj['room_type']}"
                    )
        
        return issues

    async def _process_file(self, file: UploadFile) -> Dict[str, Any]:
        """Process uploaded file with comprehensive analysis"""
        try:
            # Save file temporarily
            temp_path = f"temp_{uuid.uuid4()}"
            with open(temp_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            results = {}
            
            # Determine file type
            file_type = file.content_type or self._guess_file_type(temp_path)
            
            if file_type.startswith("image/"):
                # Process image
                results["image_analysis"] = await self._analyze_image(temp_path)
            elif file_type == "application/pdf":
                # Process PDF
                results["pdf_analysis"] = await self._analyze_pdf(temp_path)
            elif file_type in ["application/dxf", "application/acad"]:
                # Process CAD file
                results["cad_analysis"] = await self._analyze_cad(temp_path)
            
            # Clean up
            os.remove(temp_path)
            
            return results
        except Exception as e:
            self.logger.error(f"Error processing file: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def _guess_file_type(self, file_path: str) -> str:
        """Guess file type from content"""
        with open(file_path, "rb") as f:
            header = f.read(1024)
            
        # Check for PDF
        if header.startswith(b"%PDF"):
            return "application/pdf"
        
        # Check for DXF
        if b"SECTION" in header and b"HEADER" in header:
            return "application/dxf"
        
        # Default to generic binary
        return "application/octet-stream"
    
    async def _create_analysis_session(
        self,
        analysis_id: str,
        payment_intent_id: Optional[str]
    ) -> Dict[str, Any]:
        """Create new analysis session"""
        session = {
            "analysis_id": analysis_id,
            "payment_intent_id": payment_intent_id,
            "status": "started",
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat()
        }
        
        # Store session in database
        # This would typically use a database, but for now we'll use a file
        session_file = os.path.join(self.uploads_dir, f"{analysis_id}.json")
        with open(session_file, "w") as f:
            json.dump(session, f)
        
        return session
    
    async def _update_analysis_session(
        self,
        analysis_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """Update analysis session with new data"""
        session_file = os.path.join(self.uploads_dir, f"{analysis_id}.json")
        
        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                session = json.load(f)
            
            session.update(updates)
            session["last_updated"] = datetime.utcnow().isoformat()
            
            with open(session_file, "w") as f:
                json.dump(session, f)
    
    async def _extract_address_from_text(self, text: str) -> Optional[str]:
        """Extract address from OCR text using advanced pattern matching"""
        try:
            # Common address patterns in Norwegian
            patterns = [
                r'\b\d{1,5}\s+[A-ZÆØÅa-zæøå\s]+\d{4}\s+[A-ZÆØÅa-zæøå\s]+',  # Standard address
                r'\b[A-ZÆØÅa-zæøå\s]+\s+\d{1,5}[A-Z]?\s+\d{4}\s+[A-ZÆØÅa-zæøå\s]+',  # Street first
                r'\b[A-ZÆØÅa-zæøå\s]+\d{1,5}[A-Z]?,?\s*\d{4}\s+[A-ZÆØÅa-zæøå\s]+'  # Compact form
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    # Validate found address
                    address = match.group(0).strip()
                    if await self._validate_address_format(address):
                        return address
            
            return None
        except Exception as e:
            self.logger.error(f"Error extracting address: {str(e)}")
            return None
    
    async def _validate_address_format(self, address: str) -> bool:
        """Validate address format and check if it exists"""
        try:
            # Check basic format
            if not re.match(r'.*\d{4}.*', address):  # Must contain 4-digit postal code
                return False
            
            # Validate against Kartverket
            async with aiohttp.ClientSession() as session:
                url = f"https://ws.geonorge.no/adresser/v1/sok?sok={urllib.parse.quote(address)}"
                async with session.get(url) as response:
                    if response.status != 200:
                        return False
                    
                    data = await response.json()
                    return len(data.get("adresser", [])) > 0
                    
        except Exception as e:
            self.logger.error(f"Error validating address: {str(e)}")
            return False

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