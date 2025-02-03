from typing import Dict, List, Any, Optional, Union
import logging
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import pytesseract
import requests
from io import BytesIO
from bs4 import BeautifulSoup

class AIService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._load_models()

    def _load_models(self):
        """Last inn alle nødvendige AI-modeller"""
        try:
            # Last inn modeller (i en ekte implementasjon ville disse vært faktiske modeller)
            self.building_detector = self._load_building_detection_model()
            self.floor_plan_analyzer = self._load_floor_plan_model()
            self.text_recognizer = self._load_text_recognition_model()
        except Exception as e:
            self.logger.error(f"Error loading AI models: {str(e)}")
            raise

    def _load_building_detection_model(self):
        """Last inn bygningsdeteksjonsmodell"""
        # Dette ville normalt laste en faktisk TensorFlow/PyTorch modell
        return None

    def _load_floor_plan_model(self):
        """Last inn plantegningsanalysemodell"""
        # Dette ville normalt laste en faktisk TensorFlow/PyTorch modell
        return None

    def _load_text_recognition_model(self):
        """Last inn OCR-modell"""
        # Vi bruker Tesseract OCR
        return pytesseract

    async def analyze_building(self, image_data: bytes) -> Dict[str, Any]:
        """Analyser bygning fra bilde"""
        try:
            # Konverter bytes til bilde
            image = Image.open(BytesIO(image_data))
            np_image = np.array(image)

            # Kjør bygningsdeteksjon
            building_info = await self._detect_building(np_image)

            # Analyser fasade
            facade_analysis = await self._analyze_facade(np_image)

            # Kombiner resultater
            return {
                **building_info,
                **facade_analysis,
                "image_quality": await self._check_image_quality(np_image)
            }

        except Exception as e:
            self.logger.error(f"Error analyzing building: {str(e)}")
            raise

    async def analyze_building_from_link(self, url: str) -> Dict[str, Any]:
        """Analyser bygning fra nettside (f.eks. Finn.no)"""
        try:
            # Last ned siden
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Ekstraher informasjon
            images = await self._extract_images(soup)
            property_info = await self._extract_property_info(soup)

            # Analyser første bilde
            if images:
                image_response = requests.get(images[0])
                building_analysis = await self.analyze_building(image_response.content)
            else:
                building_analysis = {}

            return {
                **property_info,
                **building_analysis,
                "source_url": url,
                "additional_images": images[1:] if images else []
            }

        except Exception as e:
            self.logger.error(f"Error analyzing building from link: {str(e)}")
            raise

    async def analyze_floor_plan(
        self,
        image_data: Union[bytes, str]
    ) -> Dict[str, Any]:
        """Analyser plantegning"""
        try:
            # Last bilde
            if isinstance(image_data, str):  # URL
                response = requests.get(image_data)
                image_data = response.content
            
            image = Image.open(BytesIO(image_data))
            np_image = np.array(image)

            # Konverter til gråtone
            gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)

            # Gjenkjenn tekst (mål, romnavn, etc.)
            text = self.text_recognizer.image_to_string(gray, lang='nor')

            # Analyser rominndeling
            rooms = await self._analyze_rooms(gray)

            # Mål og dimensjoner
            measurements = await self._extract_measurements(gray, text)

            return {
                "rooms": rooms,
                "measurements": measurements,
                "text_content": text,
                "quality_score": await self._check_image_quality(np_image)
            }

        except Exception as e:
            self.logger.error(f"Error analyzing floor plan: {str(e)}")
            raise

    async def _detect_building(self, image: np.ndarray) -> Dict[str, Any]:
        """Detekter og analyser bygning i bilde"""
        # Dette ville normalt bruke en trent modell
        # Her returnerer vi mock-data
        return {
            "building_type": "enebolig",
            "stories": 2,
            "has_basement": True,
            "has_garage": True,
            "facade_material": "wood",
            "roof_type": "saddle",
            "estimated_age": "1980-1990",
            "condition": "good"
        }

    async def _analyze_facade(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyser fasaden for detaljer"""
        # Mock-implementasjon
        return {
            "window_count": 8,
            "door_count": 2,
            "facade_color": "white",
            "architectural_style": "traditional",
            "maintenance_needs": [
                "paint_refresh",
                "window_frames"
            ]
        }

    async def _check_image_quality(self, image: np.ndarray) -> float:
        """Sjekk bildekvalitet"""
        # Beregn basis bildekvalitetsmetrikker
        blur = cv2.Laplacian(image, cv2.CV_64F).var()
        brightness = np.mean(image)
        contrast = np.std(image)

        # Normaliser og kombiner metrikker
        quality_score = (
            min(1.0, blur / 500) * 0.4 +
            min(1.0, brightness / 128) * 0.3 +
            min(1.0, contrast / 64) * 0.3
        )

        return quality_score

    async def _analyze_rooms(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Analyser rominndeling i plantegning"""
        # Mock-implementasjon
        return [
            {
                "type": "living_room",
                "area": 25.5,
                "windows": 2,
                "doors": 1
            },
            {
                "type": "kitchen",
                "area": 15.0,
                "windows": 1,
                "doors": 1
            },
            {
                "type": "bedroom",
                "area": 12.0,
                "windows": 1,
                "doors": 1
            }
        ]

    async def _extract_measurements(
        self,
        image: np.ndarray,
        text: str
    ) -> Dict[str, Any]:
        """Ekstraher målangivelser fra plantegning"""
        # Mock-implementasjon
        return {
            "total_area": 120.5,
            "width": 10.5,
            "length": 11.5,
            "ceiling_height": 2.4,
            "room_measurements": {
                "living_room": {"width": 5.0, "length": 5.1},
                "kitchen": {"width": 3.0, "length": 5.0},
                "bedroom": {"width": 3.0, "length": 4.0}
            }
        }

    async def _extract_images(self, soup: BeautifulSoup) -> List[str]:
        """Ekstraher bilder fra nettside"""
        images = []
        for img in soup.find_all('img'):
            src = img.get('src')
            if src and any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png']):
                images.append(src)
        return images

    async def _extract_property_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Ekstraher eiendomsinformasjon fra nettside"""
        # Mock-implementasjon for Finn.no
        return {
            "address": self._find_text(soup, "address"),
            "price": self._find_text(soup, "price"),
            "size": self._find_text(soup, "size"),
            "build_year": self._find_text(soup, "year"),
            "property_type": self._find_text(soup, "type")
        }

    def _find_text(self, soup: BeautifulSoup, field: str) -> str:
        """Hjelpefunksjon for å finne tekst i HTML"""
        # Dette ville normalt inneholde spesifikk logikk for hver støttet nettside
        return "Not implemented"

    async def analyze_development_potential(
        self,
        building_info: Dict[str, Any],
        zoning_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyser utviklingspotensial basert på AI-analyse og reguleringer"""
        try:
            potential = {
                "basement_apartment": await self._analyze_basement_potential(
                    building_info
                ),
                "extension": await self._analyze_extension_potential(
                    building_info,
                    zoning_info
                ),
                "lot_division": await self._analyze_lot_division_potential(
                    building_info,
                    zoning_info
                ),
                "facade_improvements": await self._analyze_facade_improvements(
                    building_info
                )
            }

            # Beregn score for hvert potensial
            for key in potential:
                if potential[key]["possible"]:
                    potential[key]["score"] = await self._calculate_potential_score(
                        potential[key]
                    )

            return potential

        except Exception as e:
            self.logger.error(f"Error analyzing development potential: {str(e)}")
            raise

    async def _analyze_basement_potential(
        self,
        building_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyser potensial for kjellerleilighet"""
        if not building_info.get("has_basement"):
            return {"possible": False, "reason": "No basement"}

        return {
            "possible": True,
            "estimated_size": building_info.get("basement_area", 60),
            "requirements": [
                "Minimum takhøyde 2.2m",
                "Separate inngang",
                "Brannskille mot hovedetasje"
            ],
            "challenges": [],
            "cost_estimate": 500000
        }

    async def _analyze_extension_potential(
        self,
        building_info: Dict[str, Any],
        zoning_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyser potensial for tilbygg"""
        current_bya = building_info.get("bya", 0)
        lot_size = building_info.get("lot_size", 0)
        max_bya = (zoning_info.get("max_bya", 30) / 100) * lot_size

        if current_bya >= max_bya:
            return {
                "possible": False,
                "reason": "Maximum BYA reached"
            }

        return {
            "possible": True,
            "max_size": max_bya - current_bya,
            "recommended_location": "south",
            "type": "living_area",
            "cost_estimate": 25000 * (max_bya - current_bya)
        }

    async def _analyze_lot_division_potential(
        self,
        building_info: Dict[str, Any],
        zoning_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyser potensial for tomtedeling"""
        lot_size = building_info.get("lot_size", 0)
        min_lot_size = zoning_info.get("min_lot_size", 600)

        if lot_size < min_lot_size * 2:
            return {
                "possible": False,
                "reason": "Lot too small for division"
            }

        return {
            "possible": True,
            "potential_lots": lot_size // min_lot_size,
            "min_lot_size": min_lot_size,
            "suggested_division": "lengthwise",
            "estimated_value_increase": 2000000
        }

    async def _analyze_facade_improvements(
        self,
        building_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyser potensial for fasadeforbedringer"""
        return {
            "possible": True,
            "recommended_improvements": [
                {
                    "type": "windows",
                    "description": "Upgrade to energy efficient windows",
                    "cost": 150000
                },
                {
                    "type": "insulation",
                    "description": "Add exterior insulation",
                    "cost": 200000
                }
            ],
            "estimated_energy_saving": 30,
            "aesthetic_improvement": "high"
        }

    async def _calculate_potential_score(
        self,
        potential: Dict[str, Any]
    ) -> float:
        """Beregn en score for utviklingspotensialet"""
        # Mock-implementasjon
        # I en ekte implementasjon ville dette brukt en mer sofistikert modell
        score = 0.0
        
        if "cost_estimate" in potential:
            roi = potential.get("estimated_value_increase", 0) / potential["cost_estimate"]
            score += min(roi, 3) / 3  # Max 1.0 for ROI

        if "challenges" in potential:
            score -= len(potential["challenges"]) * 0.1

        if "requirements" in potential:
            score -= len(potential["requirements"]) * 0.05

        return max(0, min(1, score))