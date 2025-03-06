import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum

# Sett opp logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisObjective(Enum):
    """Enum for analyseobjektiver basert på kundens behov"""
    RENTAL_INCOME = "rental_income"  # Maksimere leieinntekter
    PROPERTY_VALUE = "property_value"  # Øke eiendomsverdi
    ENERGY_EFFICIENCY = "energy_efficiency"  # Forbedre energieffektivitet
    LIVING_QUALITY = "living_quality"  # Forbedre bokvalitet
    MINIMAL_COST = "minimal_cost"  # Minimale kostnader

@dataclass
class MunicipalityRegulations:
    """Klasse for kommunale reguleringer"""
    min_ceiling_height: float  # Minimum takhøyde
    min_room_size: float  # Minimum romstørrelse
    min_window_area_ratio: float  # Minimum vindusareal ift. gulvareal
    min_apartment_size: float  # Minimum leilighetsstørrelse
    fire_safety_requirements: List[str]  # Brannsikkerhetskrav
    sound_isolation_requirements: bool  # Krav om lydisolasjon
    separate_entrance_required: bool  # Krav om separat inngang
    parking_requirements: Dict[str, int]  # Parkeringskrav

class FloorPlanAnalyzer:
    def __init__(self, regulations_path: Optional[str] = None):
        """
        Initialiserer FloorPlanAnalyzer med nødvendige modeller og data
        
        Args:
            regulations_path: Sti til JSON-fil med kommunale reguleringer
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.transform = self._get_transforms()
        self.regulations = self._load_regulations(regulations_path)
        self.room_type_map = {
            'bedroom': 'Soverom',
            'living_room': 'Stue',
            'kitchen': 'Kjøkken',
            'bathroom': 'Bad',
            'hallway': 'Gang',
            'storage': 'Bod',
            'basement': 'Kjeller',
            'attic': 'Loft'
        }
        # Lastepixel-til-meter skala (vil bli kalibrert fra målestokk på plantegning)
        self.scale_factor = 0.05  # Standard: 1 pixel = 5cm
        
        # Lastinn TEK17 krav
        self.building_code = self._load_building_code()

    def _load_model(self) -> nn.Module:
        """
        Laster inn forhåndstrent modell for plantegningsanalyse
        """
        try:
            model = torch.hub.load('ultralytics/yolov5', 'custom', 'models/floor_plan_detector.pt')
            model.to(self.device)
            logger.info("Modell lastet inn vellykket")
            return model
        except Exception as e:
            logger.error(f"Feil ved lasting av modell: {str(e)}")
            # Fallback til en enkel modell hvis hovedmodellen ikke lastes
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            model.to(self.device)
            return model

    def _get_transforms(self) -> transforms.Compose:
        """
        Definerer bildetransformasjoner
        """
        return transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def _load_regulations(self, regulations_path: Optional[str]) -> Dict[str, MunicipalityRegulations]:
        """
        Laster inn kommunale reguleringer fra JSON-fil
        """
        if regulations_path:
            try:
                with open(regulations_path, 'r') as f:
                    regulations_data = json.load(f)
                
                regulations = {}
                for municipality, data in regulations_data.items():
                    regulations[municipality] = MunicipalityRegulations(
                        min_ceiling_height=data.get('min_ceiling_height', 2.4),
                        min_room_size=data.get('min_room_size', 6.0),
                        min_window_area_ratio=data.get('min_window_area_ratio', 0.1),
                        min_apartment_size=data.get('min_apartment_size', 25.0),
                        fire_safety_requirements=data.get('fire_safety_requirements', []),
                        sound_isolation_requirements=data.get('sound_isolation_requirements', False),
                        separate_entrance_required=data.get('separate_entrance_required', False),
                        parking_requirements=data.get('parking_requirements', {})
                    )
                return regulations
            except Exception as e:
                logger.error(f"Feil ved lasting av reguleringer: {str(e)}")
        
        # Standard reguleringer hvis ingen fil er angitt
        default_regs = MunicipalityRegulations(
            min_ceiling_height=2.4,
            min_room_size=6.0,
            min_window_area_ratio=0.1,
            min_apartment_size=25.0,
            fire_safety_requirements=["Røykvarsler", "Brannslukningsapparat"],
            sound_isolation_requirements=True,
            separate_entrance_required=True,
            parking_requirements={"bolig": 1, "utleieenhet": 1}
        )
        return {"default": default_regs}
    
    def _load_building_code(self) -> Dict[str, Any]:
        """
        Laster inn byggetekniske forskrifter (TEK17)
        """
        # I en full implementasjon ville dette lastes fra en database eller fil
        return {
            "TEK17": {
                "min_ceiling_height": 2.4,
                "min_window_area": 0.07,  # 7% av BRA
                "fire_safety": {
                    "separate_fire_cell": True,
                    "smoke_detector_requirements": "Alle rom og felles areal",
                    "escape_routes": "Minimum 2 uavhengige rømningsveier"
                },
                "ventilation": {
                    "kitchen": 36,  # m³/h
                    "bathroom": 54,  # m³/h
                    "living_areas": 1.2  # Luftskifte per time
                },
                "sound_insulation": {
                    "walls_between_units": 55,  # dB
                    "floors_between_units": 53  # dB
                },
                "energy_requirements": {
                    "u_value_walls": 0.22,
                    "u_value_roof": 0.18,
                    "u_value_floor": 0.18,
                    "u_value_windows": 1.2
                }
            }
        }

    async def analyze(self, 
                      image_path: str, 
                      municipality: str = "default", 
                      objective: AnalysisObjective = AnalysisObjective.RENTAL_INCOME,
                      property_data: Optional[Dict] = None) -> Dict:
        """
        Analyserer plantegning og returnerer detaljert analyse
        
        Args:
            image_path: Sti til plantegningsbilde
            municipality: Kommune for å hente riktige reguleringer
            objective: Kundens mål med analysen
            property_data: Ytterligere eiendomsdata
            
        Returns:
            Dict: Komplett analyse med rom, mål, muligheter og anbefalinger
        """
        try:
            # Last inn og preprosesser bilde
            image = Image.open(image_path)
            self._calibrate_scale(image, property_data)
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Detekter rom og funksjoner
            with torch.no_grad():
                detections = self.model(image_tensor)

            # Prosesser deteksjoner
            rooms = self._process_room_detections(detections)
            walls = self._detect_walls(image)
            measurements = self._calculate_measurements(rooms, walls)
            doors_windows = self._detect_doors_windows(image)
            
            # Finn strukturelle elementer (bærevegger, søyler)
            structural_elements = self._detect_structural_elements(image, walls)
            
            # Identifiser eksisterende leiligheter/enheter
            existing_units = self._identify_existing_units(rooms, doors_windows)
            
            # Gjør dybdeanalyse basert på kundens mål
            opportunities = self._analyze_opportunities(
                rooms, walls, measurements, doors_windows, 
                structural_elements, existing_units, 
                municipality, objective, property_data
            )
            
            # Finn utfordringer og begrensninger
            constraints = self._identify_constraints(
                rooms, walls, measurements, structural_elements,
                municipality, property_data
            )
            
            # Generer anbefalinger og løsninger
            recommendations = self._generate_recommendations(
                rooms, measurements, opportunities, constraints,
                objective, municipality
            )
            
            # Estimer kostnader og ROI
            economics = self._calculate_economics(recommendations, objective)
            
            # Generer optimalisert plantegning
            optimized_layout = self._generate_optimized_layout(
                rooms, walls, measurements, opportunities, recommendations
            )

            return {
                "rooms": rooms,
                "walls": walls,
                "measurements": measurements,
                "doors_windows": doors_windows,
                "structural_elements": structural_elements,
                "existing_units": existing_units,
                "opportunities": opportunities,
                "constraints": constraints,
                "recommendations": recommendations,
                "economics": economics,
                "optimized_layout": optimized_layout,
                "analysis_summary": self._generate_analysis_summary(locals())
            }

        except Exception as e:
            logger.error(f"Feil ved analysering av plantegning: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def _calibrate_scale(self, image: Image.Image, property_data: Optional[Dict]) -> None:
        """
        Kalibrerer pixel-til-meter skala basert på målestokk eller kjente mål
        """
        if property_data and "known_dimensions" in property_data:
            # Hvis vi har kjente dimensjoner, bruk dem for å kalibrere
            known_width_meters = property_data["known_dimensions"].get("width", 0)
            if known_width_meters > 0:
                image_width_pixels = image.width
                self.scale_factor = known_width_meters / image_width_pixels
                logger.info(f"Skala kalibrert: 1 pixel = {self.scale_factor*100:.2f} cm")
        else:
            # Prøv å detektere målestokk på plantegningen
            # (Avansert implementasjon ville bruke tekstgjenkjenning for å finne målestokk)
            # For nå, bruk standardverdien
            self.scale_factor = 0.05
            logger.info(f"Bruker standard skala: 1 pixel = {self.scale_factor*100:.2f} cm")

    def _process_room_detections(self, detections) -> List[Dict]:
        """
        Prosesserer romdeteksjoner fra modellutdata
        """
        rooms = []
        for det in detections.pred[0]:
            if det[4] > 0.5:  # Confidence threshold
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                room_type = self.model.names[int(cls)]
                
                # Konverter rom_type til norsk
                room_type_no = self.room_type_map.get(room_type.lower(), room_type)
                
                area_m2 = self._calculate_room_area(x1, y1, x2, y2)
                
                room = {
                    "type": room_type,
                    "type_no": room_type_no,
                    "confidence": float(conf),
                    "coordinates": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    },
                    "area_m2": area_m2,
                    "width_m": abs(x2 - x1) * self.scale_factor,
                    "length_m": abs(y2 - y1) * self.scale_factor,
                    "meets_regulations": self._check_room_regulations(room_type, area_m2)
                }
                rooms.append(room)
        
        # Sorter rom etter størrelse (største først)
        rooms.sort(key=lambda r: r["area_m2"], reverse=True)
        
        return rooms
    
    def _check_room_regulations(self, room_type: str, area_m2: float) -> bool:
        """
        Sjekker om rommet oppfyller forskriftene for minimumsstørrelse
        """
        min_sizes = {
            "bedroom": 7.0,  # Minimum for soverom
            "living_room": 15.0,  # Minimum for stue
            "kitchen": 6.0,  # Minimum for kjøkken
            "bathroom": 4.0,  # Minimum for bad
            "hallway": 3.0,  # Minimum for gang
            "storage": 2.0,  # Minimum for bod
            "combined_kitchen_living": 20.0  # Minimum for kombinert kjøkken/stue
        }
        
        required_min = min_sizes.get(room_type.lower(), 0.0)
        return area_m2 >= required_min

    def _detect_walls(self, image: Image.Image) -> List[Dict]:
        """
        Detekterer vegger i plantegningen
        """
        # Konverter til numpy array
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Kantdeteksjon
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough-transform for å detektere linjer
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        
        walls = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length_m = self._calculate_wall_length(x1, y1, x2, y2)
                
                # Beregn veggretning (horisontal/vertikal)
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                orientation = "horizontal" if (angle < 45 or angle > 135) else "vertical"
                
                walls.append({
                    "coordinates": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    },
                    "length_m": length_m,
                    "orientation": orientation,
                    "is_load_bearing": False  # Vil bli oppdatert i _detect_structural_elements
                })
        
        return walls

    def _detect_structural_elements(self, image: Image.Image, walls: List[Dict]) -> Dict:
        """
        Detekterer bærekonstruksjoner (bærevegger, søyler, etc.)
        """
        # Dette er en forenklet implementasjon
        # En mer avansert implementasjon ville bruke maskinfjerelæring for å identifisere
        # tykkere vegger, søyler, og andre strukturelle elementer
        
        structural_elements = {
            "load_bearing_walls": [],
            "columns": [],
            "beams": []
        }
        
        # Antakelse: Lengre vegger er mer sannsynlig å være bærevegger
        long_walls = [w for w in walls if w["length_m"] > 3.0]
        
        # Finn vegger som er parallelle med yttervegger
        # (Dette er en forenklet antakelse)
        if long_walls:
            # Sorter etter lengde (lengste først)
            long_walls.sort(key=lambda w: w["length_m"], reverse=True)
            
            # De 20% lengste veggene antas å være bærevegger
            num_load_bearing = max(1, len(long_walls) // 5)
            for i in range(num_load_bearing):
                wall = long_walls[i]
                wall["is_load_bearing"] = True
                structural_elements["load_bearing_walls"].append(wall)
        
        # Detekter søyler basert på dimensjoner
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Forenklet søyledeteksjon
        # En full implementasjon ville bruke mer sofistikerte bildegjenkjenningsmetoder
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=5, maxRadius=20
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                structural_elements["columns"].append({
                    "center": {"x": int(x), "y": int(y)},
                    "radius_pixels": int(r),
                    "radius_m": r * self.scale_factor
                })
        
        return structural_elements

    def _calculate_measurements(self,
        rooms: List[Dict],
        walls: List[Dict]
    ) -> Dict:
        """
        Beregner mål og dimensjoner
        """
        # Totalt areal
        total_area = sum(room["area_m2"] for room in rooms)
        
        # Beregn romfordeling
        room_distribution = {}
        for room in rooms:
            room_type = room["type"]
            if room_type not in room_distribution:
                room_distribution[room_type] = {"count": 0, "total_area": 0.0}
            
            room_distribution[room_type]["count"] += 1
            room_distribution[room_type]["total_area"] += room["area_m2"]
        
        # Finn ytterveggenes lengde
        exterior_wall_length = self._estimate_exterior_wall_length(walls)
        
        measurements = {
            "total_area_m2": total_area,
            "room_dimensions": self._calculate_room_dimensions(rooms),
            "wall_lengths_m": [wall["length_m"] for wall in walls],
            "total_wall_length_m": sum(wall["length_m"] for wall in walls),
            "estimated_exterior_wall_length_m": exterior_wall_length,
            "room_distribution": room_distribution,
            "area_efficiency": self._calculate_area_efficiency(rooms, walls)
        }
        
        return measurements

    def _estimate_exterior_wall_length(self, walls: List[Dict]) -> float:
        """
        Estimerer lengden på yttervegger basert på vegganalyse
        """
        # Dette er en forenklet estimering
        # En mer nøyaktig metode ville identifisere ytterkanten av plantegningen
        
        # Sorter vegger etter lengde
        sorted_walls = sorted(walls, key=lambda w: w["length_m"], reverse=True)
        
        # De lengste veggene er mest sannsynlig yttervegger
        exterior_walls = sorted_walls[:4]  # Antar at de 4 lengste er yttervegger
        
        return sum(wall["length_m"] for wall in exterior_walls)
    
    def _calculate_area_efficiency(self, rooms: List[Dict], walls: List[Dict]) -> float:
        """
        Beregner arealutnyttelse
        """
        total_room_area = sum(room["area_m2"] for room in rooms)
        
        # Estimer total areal inkludert vegger
        # (Dette er en forenkling - en mer nøyaktig metode ville beregne faktisk bygningsavtrykk)
        estimated_wall_area = sum(wall["length_m"] * 0.1 for wall in walls)  # Antar 10 cm veggtykkelse
        
        total_estimated_area = total_room_area + estimated_wall_area
        
        return (total_room_area / total_estimated_area) * 100 if total_estimated_area > 0 else 0

    def _detect_doors_windows(self, image: Image.Image) -> Dict:
        """
        Detekterer dører og vinduer i plantegningen
        """
        # Konverter til numpy array
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # I en full implementasjon ville vi ha brukt en spesialisert ML-modell for dette
        # For nå, bruker vi en forenklet tilnærming med templatematching
        
        doors = []
        windows = []
        
        # Implementer en mer sofistikert deteksjon ved å bruke ML-modeller
        # for spesifikk dør- og vindusgjenkjenning
        
        # Forenkling: Bruk Haar Cascade Classifier
        try:
            door_cascade = cv2.CascadeClassifier('models/door_cascade.xml')
            door_detections = door_cascade.detectMultiScale(gray, 1.1, 3)
            
            for (x, y, w, h) in door_detections:
                doors.append({
                    "coordinates": {
                        "x": int(x),
                        "y": int(y)
                    },
                    "width_m": w * self.scale_factor,
                    "height_m": h * self.scale_factor,
                    "type": self._determine_door_type(w, h)
                })
                
            window_cascade = cv2.CascadeClassifier('models/window_cascade.xml')
            window_detections = window_cascade.detectMultiScale(gray, 1.1, 3)
            
            for (x, y, w, h) in window_detections:
                windows.append({
                    "coordinates": {
                        "x": int(x),
                        "y": int(y)
                    },
                    "width_m": w * self.scale_factor,
                    "height_m": h * self.scale_factor,
                    "area_m2": (w * h) * (self.scale_factor ** 2)
                })
        except:
            # Fallback til en dummyimplementasjon hvis cascade-filene ikke finnes
            logger.warning("Kunne ikke laste cascade-klassifikatorer, bruker dummy-deteksjon")
            
            # Dummy-implementasjon: Anta noen dører og vinduer for demo
            # I en faktisk implementasjon ville disse detekteres fra bildet
            doors = [
                {
                    "coordinates": {"x": 100, "y": 150},
                    "width_m": 0.9,
                    "height_m": 2.1,
                    "type": "internal"
                },
                {
                    "coordinates": {"x": 300, "y": 50},
                    "width_m": 1.0,
                    "height_m": 2.1,
                    "type": "external"
                }
            ]
            
            windows = [
                {
                    "coordinates": {"x": 150, "y": 50},
                    "width_m": 1.2,
                    "height_m": 1.2,
                    "area_m2": 1.44
                },
                {
                    "coordinates": {"x": 400, "y": 150},
                    "width_m": 1.0,
                    "height_m": 1.2,
                    "area_m2": 1.2
                }
            ]
        
        return {
            "doors": doors,
            "windows": windows,
            "door_count": len(doors),
            "window_count": len(windows)
        }
    
    def _determine_door_type(self, width_pixels: int, height_pixels: int) -> str:
        """
        Bestemmer dørtypen basert på dimensjoner
        """
        width_m = width_pixels * self.scale_factor
        
        if width_m >= 0.9:
            return "external"  # Ytterdør
        else:
            return "internal"  # Innerdør

    def _calculate_room_area(self,
        x1: float,
        y1: float,
        x2: float,
        y2: float
    ) -> float:
        """
        Beregner romareal i kvadratmeter
        """
        width = abs(x2 - x1) * self.scale_factor
        height = abs(y2 - y1) * self.scale_factor
        return width * height

    def _calculate_wall_length(self,
        x1: float,
        y1: float,
        x2: float,
        y2: float
    ) -> float:
        """
        Beregner vegglengde i meter
        """
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * self.scale_factor

    def _calculate_room_dimensions(self, rooms: List[Dict]) -> List[Dict]:
        """
        Beregner dimensjoner for hvert rom
        """
        dimensions = []
        for room in rooms:
            coords = room["coordinates"]
            width = abs(coords["x2"] - coords["x1"]) * self.scale_factor
            length = abs(coords["y2"] - coords["y1"]) * self.scale_factor
            
            dimensions.append({
                "room_type": room["type"],
                "room_type_no": room["type_no"],
                "width_m": width,
                "length_m": length,
                "area_m2": width * length,
                "ratio": max(width, length) / min(width, length) if min(width, length) > 0 else 0
            })
        
        return dimensions
    
    def _identify_existing_units(self, rooms: List[Dict], doors_windows: Dict) -> List[Dict]:
        """
        Identifiserer eksisterende leiligheter/enheter basert på romorganisering
        """
        units = []
        
        # Gruppering av rom i leiligheter basert på nærhet og tilgang
        # Dette er en forenklet implementasjon
        
        # I en faktisk implementasjon ville vi bruke grafanalyse for å gruppere rom
        # basert på dørforbindelser og nærhet
        
        # Finn hovedenheten (største bad+kjøkken+stue kombinasjon)
        main_unit_rooms = []
        remaining_rooms = rooms.copy()
        
        # Finn bad
        bathrooms = [r for r in remaining_rooms if r["type"].lower() == "bathroom"]
        if bathrooms:
            main_bathroom = max(bathrooms, key=lambda r: r["area_m2"])
            main_unit_rooms.append(main_bathroom)
            remaining_rooms.remove(main_bathroom)
        
        # Finn kjøkken
        kitchens = [r for r in remaining_rooms if r["type"].lower() == "kitchen"]
        if kitchens:
            main_kitchen = max(kitchens, key=lambda r: r["area_m2"])
            main_unit_rooms.append(main_kitchen)
            remaining_rooms.remove(main_kitchen)
        
        # Finn stue
        living_rooms = [r for r in remaining_rooms if r["type"].lower() == "living_room"]
        if living_rooms:
            main_living = max(living_rooms, key=lambda r: r["area_m2"])
            main_unit_rooms.append(main_living)
            remaining_rooms.remove(main_living)
        
        # Legg til hovedsoverom
        bedrooms = [r for r in remaining_rooms if r["type"].lower() == "bedroom"]
        if bedrooms:
            main_bedroom = max(bedrooms, key=lambda r: r["area_m2"])
            main_unit_rooms.append(main_bedroom)
            remaining_rooms.remove(main_bedroom)
        
        # Hvis vi har funnet minst 3 rom, anta at dette er hovedenheten
        if len(main_unit_rooms) >= 3:
            total_area = sum(r["area_m2"] for r in main_unit_rooms)
            units.append({
                "type": "main",
                "rooms": main_unit_rooms,
                "total_area_m2": total_area,
                "room_count": len(main_unit_rooms)
            })
        
        # Hvis det er gjenværende rom som inneholder bad og kjøkken,
        # kan de utgjøre en sekundærenhet
        secondary_unit_rooms = []
        
        # Finn sekundær-bad
        bathrooms = [r for r in remaining_rooms if r["type"].lower() == "bathroom"]
        if bathrooms:
            secondary_bathroom = max(bathrooms, key=lambda r: r["area_m2"])
            secondary_unit_rooms.append(secondary_bathroom)
            remaining_rooms.remove(secondary_bathroom)
        
        # Finn sekundær-kjøkken
        kitchens = [r for r in remaining_rooms if r["type"].lower() == "kitchen"]
        if kitchens:
            secondary_kitchen = max(kitchens, key=lambda r: r["area_m2"])
            secondary_unit_rooms.append(secondary_kitchen)
            remaining_rooms.remove(secondary_kitchen)
        
        # Hvis vi har funnet både bad og kjøkken, anta at dette er en sekundærenhet
            if len(secondary_unit_rooms) >= 2:
                # Finn et rom til som kan være stue/soverom
                living_or_bedrooms = [r for r in remaining_rooms if r["type"].lower() in ["bedroom", "living_room"]]
                if living_or_bedrooms:
                    secondary_room = max(living_or_bedrooms, key=lambda r: r["area_m2"])
                    secondary_unit_rooms.append(secondary_room)
                    remaining_rooms.remove(secondary_room)
                
                total_area = sum(r["area_m2"] for r in secondary_unit_rooms)
                units.append({
                    "type": "secondary",
                    "rooms": secondary_unit_rooms,
                    "total_area_m2": total_area,
                    "room_count": len(secondary_unit_rooms)
                })
        
        # Alle andre rom tilhører enten hovedenheten eller er uklassifiserte
        if remaining_rooms:
            for unit in units:
                if unit["type"] == "main":
                    unit["rooms"].extend(remaining_rooms)
                    unit["total_area_m2"] += sum(r["area_m2"] for r in remaining_rooms)
                    unit["room_count"] += len(remaining_rooms)
                    break
            else:
                # Hvis ingen hovedenhet ble funnet, opprett en
                total_area = sum(r["area_m2"] for r in remaining_rooms)
                units.append({
                    "type": "main",
                    "rooms": remaining_rooms,
                    "total_area_m2": total_area,
                    "room_count": len(remaining_rooms)
                })
        
        return units

    def _analyze_opportunities(self,
                             rooms: List[Dict],
                             walls: List[Dict],
                             measurements: Dict,
                             doors_windows: Dict,
                             structural_elements: Dict,
                             existing_units: List[Dict],
                             municipality: str,
                             objective: AnalysisObjective,
                             property_data: Optional[Dict]) -> Dict:
        """
        Identifiserer muligheter for forbedringer og utvikling basert på kundens mål
        """
        opportunities = {
            "rental_units": [],
            "room_improvements": [],
            "layout_changes": [],
            "extensions": []
        }
        
        # Hent relevante reguleringer
        regulations = self.regulations.get(municipality, self.regulations["default"])
        
        # Sjekk muligheter for utleieenheter
        opportunities["rental_units"] = self._find_rental_opportunities(
            rooms, walls, measurements, existing_units, regulations
        )
        
        # Sjekk muligheter for romforbedringer
        opportunities["room_improvements"] = self._find_room_improvement_opportunities(
            rooms, doors_windows, regulations, objective
        )
        
        # Sjekk muligheter for planløsningsendringer
        opportunities["layout_changes"] = self._find_layout_change_opportunities(
            rooms, walls, structural_elements, regulations, objective
        )
        
        # Sjekk muligheter for tilbygg/påbygg
        if property_data and "property_area" in property_data:
            opportunities["extensions"] = self._find_extension_opportunities(
                measurements, property_data, regulations, objective
            )
        
        # Prioriter muligheter basert på kundens mål
        opportunities["priorities"] = self._prioritize_opportunities(
            opportunities, objective, regulations
        )
        
        return opportunities
    
    def _find_rental_opportunities(self,
                                rooms: List[Dict],
                                walls: List[Dict],
                                measurements: Dict,
                                existing_units: List[Dict],
                                regulations: MunicipalityRegulations) -> List[Dict]:
        """
        Identifiserer muligheter for utleieenheter
        """
        opportunities = []
        
        # Sjekk om det allerede er en sekundærenhet
        has_secondary_unit = any(unit["type"] == "secondary" for unit in existing_units)
        
        # Sjekk mulighet for å konvertere kjeller til utleieenhet
        basement_rooms = [r for r in rooms if r["type"].lower() == "basement"]
        if basement_rooms:
            basement_area = sum(room["area_m2"] for room in basement_rooms)
            
            if basement_area >= regulations.min_apartment_size:
                # Sjekk om kjeller har nødvendige rom eller plass til å lage dem
                has_bathroom = any(r["type"].lower() == "bathroom" for r in basement_rooms)
                has_kitchen = any(r["type"].lower() == "kitchen" for r in basement_rooms)
                
                bathroom_needed = not has_bathroom
                kitchen_needed = not has_kitchen
                
                # Beregn estimerte kostnader og leieverdier
                renovation_cost = basement_area * 10000  # Antar 10,000 kr per m² for full renovering
                if bathroom_needed:
                    renovation_cost += 150000  # Legge til bad
                if kitchen_needed:
                    renovation_cost += 100000  # Legge til kjøkken
                
                # Anslå leievirkdi (forenklet - ville ideelt sett bruke markedsdata)
                monthly_rent = basement_area * 200  # Antar 200 kr per m² månedlig
                annual_income = monthly_rent * 12
                roi_years = renovation_cost / annual_income if annual_income > 0 else float('inf')
                
                opportunities.append({
                    "type": "basement_conversion",
                    "description": "Konverter kjeller til utleieenhet",
                    "area_m2": basement_area,
                    "rooms_available": [r["type"] for r in basement_rooms],
                    "rooms_needed": (["bathroom"] if bathroom_needed else []) + 
                                   (["kitchen"] if kitchen_needed else []),
                    "estimated_cost": renovation_cost,
                    "estimated_monthly_rent": monthly_rent,
                    "estimated_roi_years": roi_years,
                    "requires_building_permit": True,
                    "feasibility": "high" if basement_area >= 50 else "medium"
                })
        
        # Sjekk mulighet for å konvertere loft til utleieenhet
        attic_rooms = [r for r in rooms if r["type"].lower() == "attic"]
        if attic_rooms:
            attic_area = sum(room["area_m2"] for room in attic_rooms)
            
            if attic_area >= regulations.min_apartment_size:
                # Lignende logikk som for kjeller
                # ...
                
                opportunities.append({
                    "type": "attic_conversion",
                    "description": "Konverter loft til utleieenhet",
                    "area_m2": attic_area,
                    # Øvrige detaljer...
                    "feasibility": "medium"  # Kan være vanskeligere enn kjeller pga. takhøyde
                })
        
        # Sjekk mulighet for å dele eksisterende bolig i to enheter
        if not has_secondary_unit and existing_units:
            main_unit = next((u for u in existing_units if u["type"] == "main"), None)
            if main_unit and main_unit["total_area_m2"] >= regulations.min_apartment_size * 2:
                # Stor nok til å dele i to
                division_cost = main_unit["total_area_m2"] * 8000  # Antar 8,000 kr per m² for omfattende ombygging
                
                # Beregn verdi for utleie
                new_unit_size = main_unit["total_area_m2"] / 3  # Antar 1/3 av arealet blir ny enhet
                monthly_rent = new_unit_size * 220  # Antar 220 kr per m² månedlig
                annual_income = monthly_rent * 12
                roi_years = division_cost / annual_income if annual_income > 0 else float('inf')
                
                opportunities.append({
                    "type": "house_division",
                    "description": "Del boligen i to separate enheter",
                    "main_unit_current_area_m2": main_unit["total_area_m2"],
                    "new_unit_estimated_area_m2": new_unit_size,
                    "estimated_cost": division_cost,
                    "estimated_monthly_rent": monthly_rent,
                    "estimated_roi_years": roi_years,
                    "requires_building_permit": True,
                    "feasibility": "medium"
                })
        
        return opportunities
    
    def _find_room_improvement_opportunities(self,
                                          rooms: List[Dict],
                                          doors_windows: Dict,
                                          regulations: MunicipalityRegulations,
                                          objective: AnalysisObjective) -> List[Dict]:
        """
        Identifiserer muligheter for romforbedringer
        """
        opportunities = []
        
        # Sjekk vindusareal ift. romareal
        for room in rooms:
            room_area = room["area_m2"]
            
            # Finn vinduer som sannsynligvis tilhører dette rommet
            room_windows = []
            for window in doors_windows["windows"]:
                wx, wy = window["coordinates"]["x"], window["coordinates"]["y"]
                if (room["coordinates"]["x1"] <= wx <= room["coordinates"]["x2"] and
                    room["coordinates"]["y1"] <= wy <= room["coordinates"]["y2"]):
                    room_windows.append(window)
            
            total_window_area = sum(w.get("area_m2", 0) for w in room_windows)
            window_ratio = total_window_area / room_area if room_area > 0 else 0
            
            # Sjekk mot forskriftskrav
            if window_ratio < regulations.min_window_area_ratio:
                opportunities.append({
                    "type": "increase_windows",
                    "room": room["type"],
                    "description": f"Øk vindusareal i {room['type_no']}",
                    "current_window_area_m2": total_window_area,
                    "required_window_area_m2": room_area * regulations.min_window_area_ratio,
                    "additional_area_needed_m2": (room_area * regulations.min_window_area_ratio) - total_window_area,
                    "estimated_cost": ((room_area * regulations.min_window_area_ratio) - total_window_area) * 15000,  # 15,000 kr per m² nytt vindu
                    "benefit": "improved_lighting" if objective == AnalysisObjective.LIVING_QUALITY else "code_compliance"
                })
        
        # Sjekk romstørrelser mot minimumskrav
        for room in rooms:
            if room["area_m2"] < self._get_minimum_room_size(room["type"]):
                opportunities.append({
                    "type": "increase_room_size",
                    "room": room["type"],
                    "description": f"Øk størrelse på {room['type_no']}",
                    "current_size_m2": room["area_m2"],
                    "recommended_size_m2": self._get_minimum_room_size(room["type"]),
                    "additional_area_needed_m2": self._get_minimum_room_size(room["type"]) - room["area_m2"],
                    "estimated_cost": (self._get_minimum_room_size(room["type"]) - room["area_m2"]) * 20000,  # 20,000 kr per m² utvidelse
                    "feasibility": "medium"  # Avhenger av planløsning
                })
        
        return opportunities
    
    def _find_layout_change_opportunities(self,
                                        rooms: List[Dict],
                                        walls: List[Dict],
                                        structural_elements: Dict,
                                        regulations: MunicipalityRegulations,
                                        objective: AnalysisObjective) -> List[Dict]:
        """
        Identifiserer muligheter for planløsningsendringer
        """
        opportunities = []
        
        # Identifiser små, ineffektive rom som kan slås sammen
        small_rooms = [r for r in rooms if r["area_m2"] < 10 and r["type"].lower() not in ["bathroom", "hallway"]]
        for i, room1 in enumerate(small_rooms):
            for room2 in small_rooms[i+1:]:
                # Sjekk om rommene er ved siden av hverandre (forenklet sjekk)
                adjacent = self._are_rooms_adjacent(room1, room2)
                
                if adjacent:
                    # Sjekk om veggen mellom dem er bærende
                    wall_between = self._find_wall_between_rooms(room1, room2, walls)
                    is_load_bearing = False
                    
                    if wall_between:
                        is_load_bearing = any(
                            wall["coordinates"] == wall_between["coordinates"] 
                            for wall in structural_elements["load_bearing_walls"]
                        )
                    
                    combined_area = room1["area_m2"] + room2["area_m2"]
                    base_cost = 50000  # Grunnkostnad for å fjerne vegg
                    
                    if is_load_bearing:
                        base_cost = 150000  # Mye dyrere hvis bærende
                    
                    opportunities.append({
                        "type": "combine_rooms",
                        "description": f"Slå sammen {room1['type_no']} og {room2['type_no']}",
                        "rooms": [room1["type"], room2["type"]],
                        "combined_area_m2": combined_area,
                        "is_load_bearing_wall": is_load_bearing,
                        "estimated_cost": base_cost,
                        "benefit": "improved_space_utilization",
                        "feasibility": "low" if is_load_bearing else "high"
                    })
        
        # Identifiser muligheter for åpen planløsning (spesielt kjøkken/stue)
        kitchen = next((r for r in rooms if r["type"].lower() == "kitchen"), None)
        living_room = next((r for r in rooms if r["type"].lower() == "living_room"), None)
        
        if kitchen and living_room:
            adjacent = self._are_rooms_adjacent(kitchen, living_room)
            
            if adjacent:
                wall_between = self._find_wall_between_rooms(kitchen, living_room, walls)
                is_load_bearing = False
                
                if wall_between:
                    is_load_bearing = any(
                        wall["coordinates"] == wall_between["coordinates"] 
                        for wall in structural_elements["load_bearing_walls"]
                    )
                
                combined_area = kitchen["area_m2"] + living_room["area_m2"]
                base_cost = 80000  # Grunnkostnad for å fjerne vegg og oppgradere kjøkken
                
                if is_load_bearing:
                    base_cost = 200000  # Mye dyrere hvis bærende
                
                opportunities.append({
                    "type": "open_plan",
                    "description": "Lag åpen kjøkken/stue-løsning",
                    "combined_area_m2": combined_area,
                    "is_load_bearing_wall": is_load_bearing,
                    "estimated_cost": base_cost,
                    "benefit": "modernization",
                    "value_increase_estimate": combined_area * 5000,  # Antar 5,000 kr per m² verdiøkning
                    "feasibility": "medium" if is_load_bearing else "high"
                })
        
        return opportunities
    
    def _are_rooms_adjacent(self, room1: Dict, room2: Dict) -> bool:
        """
        Sjekker om to rom er ved siden av hverandre
        """
        # Forenklet sjekk - i virkeligheten ville dette vært mer komplisert
        r1 = room1["coordinates"]
        r2 = room2["coordinates"]
        
        # Sjekk om rommene deler en kant
        shares_x = (r1["x1"] <= r2["x2"] and r2["x1"] <= r1["x2"])
        shares_y = (r1["y1"] <= r2["y2"] and r2["y1"] <= r1["y2"])
        
        # Rommene er ved siden av hverandre hvis de deler enten x- eller y-koordinater
        # men ikke begge (da ville de overlappe)
        return (shares_x and (r1["y2"] == r2["y1"] or r1["y1"] == r2["y2"])) or \
               (shares_y and (r1["x2"] == r2["x1"] or r1["x1"] == r2["x2"]))
    
    def _find_wall_between_rooms(self, room1: Dict, room2: Dict, walls: List[Dict]) -> Optional[Dict]:
        """
        Finner veggen mellom to rom
        """
        # Forenklet implementasjon - i virkeligheten ville dette vært mer komplisert
        # og tatt hensyn til veggens nøyaktige posisjon
        
        r1 = room1["coordinates"]
        r2 = room2["coordinates"]
        
        for wall in walls:
            w = wall["coordinates"]
            
            # Sjekk om veggen ligger mellom rommene
            # Dette er en veldig forenklet sjekk
            is_between = False
            
            if r1["x2"] == r2["x1"] or r1["x1"] == r2["x2"]:  # Rom ved siden av hverandre horisontalt
                is_between = (w["x1"] == w["x2"] == r1["x2"] == r2["x1"]) or \
                             (w["x1"] == w["x2"] == r1["x1"] == r2["x2"])
            elif r1["y2"] == r2["y1"] or r1["y1"] == r2["y2"]:  # Rom ved siden av hverandre vertikalt
                is_between = (w["y1"] == w["y2"] == r1["y2"] == r2["y1"]) or \
                             (w["y1"] == w["y2"] == r1["y1"] == r2["y2"])
            
            if is_between:
                return wall
        
        return None
    
    def _find_extension_opportunities(self,
                                    measurements: Dict,
                                    property_data: Dict,
                                    regulations: MunicipalityRegulations,
                                    objective: AnalysisObjective) -> List[Dict]:
        """
        Identifiserer muligheter for tilbygg/påbygg
        """
        opportunities = []
        
        # Hent relevante data
        current_building_area = measurements["total_area_m2"]
        property_area = property_data.get("property_area", 0)
        
        # Beregn tillatt utnyttelsesgrad
        # Dette ville normalt komme fra kommuneplanen
        allowed_utilization = property_data.get("allowed_utilization", 0.35)  # Standard 35% BYA
        
        max_building_area = property_area * allowed_utilization
        remaining_building_area = max_building_area - current_building_area
        
        if remaining_building_area > 20:  # Minst 20 m² tilgjengelig for å være praktisk
            # Mulighet for tilbygg
            opportunities.append({
                "type": "extension",
                "description": "Bygg tilbygg for å øke boligareal",
                "available_area_m2": remaining_building_area,
                "estimated_cost": remaining_building_area * 30000,  # 30,000 kr per m² for tilbygg
                "estimated_value_increase": remaining_building_area * 40000,  # 40,000 kr per m² verdiøkning
                "requires_building_permit": True,
                "feasibility": "high" if remaining_building_area > 40 else "medium"
            })
        
        # Sjekk om det er mulig å bygge på en ekstra etasje
        if property_data.get("stories", 1) < 2:
            opportunities.append({
                "type": "additional_story",
                "description": "Bygg på en ekstra etasje",
                "potential_new_area_m2": measurements["total_area_m2"] / property_data.get("stories", 1),
                "estimated_cost": (measurements["total_area_m2"] / property_data.get("stories", 1)) * 35000,  # 35,000 kr per m² for påbygg
                "estimated_value_increase": (measurements["total_area_m2"] / property_data.get("stories", 1)) * 45000,  # 45,000 kr per m² verdiøkning
                "requires_building_permit": True,
                "feasibility": "medium"  # Mer komplisert enn tilbygg
            })
        
        return opportunities
    
    def _prioritize_opportunities(self,
                               opportunities: Dict,
                               objective: AnalysisObjective,
                               regulations: MunicipalityRegulations) -> List[str]:
        """
        Prioriterer muligheter basert på kundens mål
        """
        all_opportunities = []
        
        # Samle alle muligheter i én liste med kategorier
        for category, items in opportunities.items():
            if category != "priorities":  # Unngå å inkludere priorities-nøkkelen selv
                for item in items:
                    all_opportunities.append({
                        "category": category,
                        "item": item
                    })
        
        # Definer scoringsfunksjon basert på mål
        def score_opportunity(opp):
            item = opp["item"]
            category = opp["category"]
            
            # Basiscore for å sikre at alle muligheter får en score
            base_score = 0
            
            # Legg til kategori-spesifikk scoring
            if category == "rental_units":
                roi_years = item.get("estimated_roi_years", float('inf'))
                feasibility = {"high": 3, "medium": 2, "low": 1}.get(item.get("feasibility", "low"), 0)
                
                if objective == AnalysisObjective.RENTAL_INCOME:
                    # For leieinntekt er ROI viktigst
                    base_score = 100 - min(roi_years * 10, 90)  # Lavere ROI-tid gir høyere score
                    base_score += feasibility * 10
                elif objective == AnalysisObjective.PROPERTY_VALUE:
                    # For eiendomsverdi er potensial for verdiøkning viktigst
                    base_score = 50 + feasibility * 10
                elif objective == AnalysisObjective.MINIMAL_COST:
                    # For minimale kostnader er lav investering viktigst
                    if item.get("estimated_cost", 0) < 200000:
                        base_score = 80
                    else:
                        base_score = 30
            
            elif category == "room_improvements":
                if objective == AnalysisObjective.LIVING_QUALITY:
                    # For bokvalitet er romforbedringer viktige
                    base_score = 80
                elif objective == AnalysisObjective.PROPERTY_VALUE:
                    # For eiendomsverdi er de også relevante
                    base_score = 60
                else:
                    base_score = 30
            
            elif category == "layout_changes":
                value_increase = item.get("value_increase_estimate", 0)
                cost = item.get("estimated_cost", 1)
                roi = value_increase / cost if cost > 0 else 0
                
                if objective == AnalysisObjective.PROPERTY_VALUE:
                    # For eiendomsverdi er verdiøkning viktig
                    base_score = 70 + min(roi * 30, 30)
                elif objective == AnalysisObjective.LIVING_QUALITY:
                    # For bokvalitet kan planløsningsendringer være viktige
                    base_score = 60
                else:
                    base_score = 40
            
            elif category == "extensions":
                value_increase = item.get("estimated_value_increase", 0)
                cost = item.get("estimated_cost", 1)
                roi = value_increase / cost if cost > 0 else 0
                
                if objective == AnalysisObjective.PROPERTY_VALUE:
                    # For eiendomsverdi er tilbygg ofte et godt valg
                    base_score = 80 + min(roi * 20, 20)
                elif objective == AnalysisObjective.RENTAL_INCOME:
                    # For leieinntekt kan utvidelse gi mer utleiepotensial
                    base_score = 70
                elif objective == AnalysisObjective.MINIMAL_COST:
                    # For minimale kostnader er tilbygg ofte for dyrt
                    base_score = 20
                else:
                    base_score = 50
            
            return base_score
        
        # Score og sorter alle muligheter
        scored_opportunities = [
            (opp, score_opportunity(opp)) for opp in all_opportunities
        ]
        scored_opportunities.sort(key=lambda x: x[1], reverse=True)
        
        # Returner prioritert liste med identifikatorer
        priorities = []
        for opp, score in scored_opportunities:
            item = opp["item"]
            category = opp["category"]
            
            # Lag en unik identifikator for hver mulighet
            identifier = f"{category}:{item['type']}:{item.get('description', 'unknown')}"
            priorities.append(identifier)
        
        return priorities
    
    def _identify_constraints(self,
                            rooms: List[Dict],
                            walls: List[Dict],
                            measurements: Dict,
                            structural_elements: Dict,
                            municipality: str,
                            property_data: Optional[Dict]) -> List[Dict]:
        """
        Identifiserer begrensninger og utfordringer
        """
        constraints = []
        
        # Sjekk for bærende vegger som begrenser muligheter
        load_bearing_walls = structural_elements["load_bearing_walls"]
        if load_bearing_walls:
            constraints.append({
                "type": "structural",
                "description": "Bærende vegger begrenser planløsningsendringer",
                "elements": [w["coordinates"] for w in load_bearing_walls],
                "impact": "high",
                "mitigation": "Strukturelle endringer krever fagkyndig vurdering og kan være kostbart"
            })
        
        # Sjekk romstørrelser mot forskriftskrav
        small_rooms = [r for r in rooms if r["area_m2"] < self._get_minimum_room_size(r["type"])]
        if small_rooms:
            constraints.append({
                "type": "regulatory",
                "description": "Rom som ikke oppfyller minimumsstørrelser",
                "rooms": [r["type"] for r in small_rooms],
                "impact": "medium",
                "mitigation": "Romstørrelser må økes for å oppfylle forskriftskrav"
            })
        
        # Sjekk takhøyde hvis data finnes
        if property_data and "ceiling_height" in property_data:
            ceiling_height = property_data["ceiling_height"]
            regulations = self.regulations.get(municipality, self.regulations["default"])
            
            if ceiling_height < regulations.min_ceiling_height:
                constraints.append({
                    "type": "regulatory",
                    "description": "Takhøyde under forskriftskrav",
                    "current_height": ceiling_height,
                    "required_height": regulations.min_ceiling_height,
                    "impact": "high",
                    "mitigation": "Økt takhøyde krever omfattende renovering"
                })
        
        # Sjekk brannkrav for utleieenheter
        if any(o["type"] in ["rental_units", "basement_conversion", "attic_conversion"] 
               for category in ["rental_units", "room_improvements", "layout_changes"]
               for o in category):
            constraints.append({
                "type": "fire_safety",
                "description": "Brannsikkerhetskrav for utleieenheter",
                "requirements": self.regulations.get(municipality, self.regulations["default"]).fire_safety_requirements,
                "impact": "medium",
                "mitigation": "Installering av brannsikkerhetsutstyr og evt. brannskiller"
            })
        
        return constraints

    def _generate_recommendations(self,
                                rooms: List[Dict],
                                measurements: Dict,
                                opportunities: Dict,
                                constraints: List[Dict],
                                objective: AnalysisObjective,
                                municipality: str) -> List[Dict]:
        """
        Genererer anbefalinger basert på analysen
        """
        recommendations = []
        
        # Hent prioriterte muligheter
        priorities = opportunities["priorities"]
        all_opportunities = {
            "rental_units": opportunities["rental_units"],
            "room_improvements": opportunities["room_improvements"],
            "layout_changes": opportunities["layout_changes"],
            "extensions": opportunities["extensions"]
        }
        
        # Flatt ut alle muligheter for enklere søk
        flat_opportunities = []
        for category, items in all_opportunities.items():
            for item in items:
                flat_opportunities.append({
                    "category": category,
                    "item": item,
                    "identifier": f"{category}:{item['type']}:{item.get('description', 'unknown')}"
                })
        
        # Lag anbefalinger basert på prioriterte muligheter
        for priority in priorities[:3]:  # Topp 3 prioriteter
            matching_opp = next((o for o in flat_opportunities if o["identifier"] == priority), None)
            
            if matching_opp:
                category = matching_opp["category"]
                item = matching_opp["item"]
                
                # Sjekk eventuelle begrensninger
                relevant_constraints = [c for c in constraints if self._is_constraint_relevant(c, category, item)]
                
                recommendation = {
                    "title": item["description"],
                    "category": category,
                    "details": item,
                    "relevant_constraints": relevant_constraints,
                    "steps": self._generate_implementation_steps(category, item, municipality),
                    "estimated_cost": item.get("estimated_cost", 0),
                    "estimated_benefit": self._calculate_benefit(item, objective),
                    "roi": self._calculate_roi(item, objective),
                    "timeline": self._estimate_timeline(category, item)
                }
                
                recommendations.append(recommendation)
        
        # Legg til generelle anbefalinger basert på byggteknisk forskrift
        if not any(r["category"] == "regulatory_compliance" for r in recommendations):
            # Sjekk om det er noen åpenbare forskriftskrav som ikke er oppfylt
            regulations = self.regulations.get(municipality, self.regulations["default"])
            compliance_issues = []
            
            # Sjekk romstørrelser
            small_rooms = [r for r in rooms if r["area_m2"] < self._get_minimum_room_size(r["type"])]
            if small_rooms:
                compliance_issues.append({
                    "issue": "Romstørrelser under minimumskrav",
                    "rooms": [r["type"] for r in small_rooms],
                    "estimated_cost": sum((self._get_minimum_room_size(r["type"]) - r["area_m2"]) * 15000 for r in small_rooms)
                })
            
            if compliance_issues:
                recommendations.append({
                    "title": "Oppgrader bolig til å oppfylle gjeldende byggtekniske forskrifter",
                    "category": "regulatory_compliance",
                    "details": {
                        "type": "compliance_upgrade",
                        "issues": compliance_issues
                    },
                    "relevant_constraints": [],
                    "steps": ["Kontakt fagperson for vurdering", "Søk om byggetillatelse", "Gjennomfør nødvendige oppgraderinger"],
                    "estimated_cost": sum(issue["estimated_cost"] for issue in compliance_issues),
                    "estimated_benefit": "Sikre lovlig bruk og unngå potensielle problemer ved salg",
                    "roi": "Medium",
                    "timeline": "3-6 måneder"
                })
        
        return recommendations
    
    def _is_constraint_relevant(self, constraint: Dict, category: str, item: Dict) -> bool:
        """
        Sjekker om en begrensning er relevant for en gitt mulighet
        """
        if constraint["type"] == "structural" and category in ["layout_changes", "room_improvements"]:
            return True
        
        if constraint["type"] == "regulatory" and category in ["rental_units", "extensions"]:
            return True
        
        if constraint["type"] == "fire_safety" and category == "rental_units":
            return True
        
        return False
    
    def _generate_implementation_steps(self, category: str, item: Dict, municipality: str) -> List[str]:
        """
        Genererer trinnvise implementeringsinstruksjoner
        """
        if category == "rental_units":
            return [
                "Kontakt kommunen for veiledning om utleie",
                "Engasjer arkitekt for detaljprosjektering",
                "Søk om byggetillatelse",
                "Engasjer håndverkere for ombygging",
                "Installer nødvendig brannsikringsutstyr",
                f"Oppgrader rom i henhold til TEK17-krav: {', '.join(item.get('rooms_needed', []))}",
                "Søk om ferdigattest",
                "Registrer utleiedelen i Matrikkelen"
            ]
        
        elif category == "room_improvements":
            steps = [
                "Kontakt fagperson for detaljert vurdering",
                "Få utarbeidet tekniske tegninger"
            ]
            
            if item["type"] == "increase_windows":
                steps.extend([
                    "Sjekk om fasadeendring krever byggesøknad",
                    "Kontakt leverandør for nye vinduer",
                    "Engasjer snekker for installasjon"
                ])
            
            elif item["type"] == "increase_room_size":
                steps.extend([
                    "Søk om byggetillatelse",
                    "Engasjer entreprenør for arbeidet",
                    "Avklar om det påvirker bærende konstruksjon"
                ])
            
            return steps
        
        elif category == "layout_changes":
            steps = [
                "Kontakt arkitekt for ny planløsning",
                "Få utarbeidet tekniske tegninger",
                "Søk om byggetillatelse"
            ]
            
            if item.get("is_load_bearing_wall", False):
                steps.extend([
                    "Engasjer byggingeniør for vurdering av bærekonstruksjon",
                    "Installer nødvendig avlastningsbjelke",
                    "Kontakt entreprenør med erfaring fra bærevegger"
                ])
            else:
                steps.append("Engasjer entreprenør for ombygging")
            
            return steps
        
        elif category == "extensions":
            return [
                "Kontakt arkitekt for prosjektering av tilbygg",
                "Utarbeid situasjonsplan og tegninger",
                "Sjekk reguleringsplan for tomten",
                "Søk om byggetillatelse",
                "Innhent tilbud fra entreprenører",
                "Søk om igangsettingstillatelse",
                "Bygg tilbygg i henhold til TEK17",
                "Søk om ferdigattest"
            ]
        
        return ["Kontakt fagperson for vurdering", "Lag detaljert plan", "Gjennomfør arbeidet"]
    
    def _calculate_benefit(self, item: Dict, objective: AnalysisObjective) -> str:
        """
        Beregner og formulerer fordelen ved en anbefaling
        """
        if objective == AnalysisObjective.RENTAL_INCOME:
            if "estimated_monthly_rent" in item:
                return f"Økt månedlig leieinntekt på ca. kr {item['estimated_monthly_rent']:.0f}"
            else:
                return "Økt potensial for leieinntekter"
        
        elif objective == AnalysisObjective.PROPERTY_VALUE:
            if "estimated_value_increase" in item:
                return f"Estimert verdiøkning på kr {item['estimated_value_increase']:.0f}"
            elif "value_increase_estimate" in item:
                return f"Estimert verdiøkning på kr {item['value_increase_estimate']:.0f}"
            else:
                return "Økt boligverdi og attraktivitet"
        
        elif objective == AnalysisObjective.LIVING_QUALITY:
            if "benefit" in item and item["benefit"] == "improved_lighting":
                return "Forbedret lysforhold og bokvalitet"
            elif "benefit" in item and item["benefit"] == "improved_space_utilization":
                return "Mer funksjonell og praktisk planløsning"
            elif "benefit" in item and item["benefit"] == "modernization":
                return "Modernisert bolig med tidsmessig planløsning"
            else:
                return "Forbedret bokvalitet og funksjonalitet"
        
        elif objective == AnalysisObjective.ENERGY_EFFICIENCY:
            return "Forbedret energieffektivitet og reduserte oppvarmingskostnader"
        
        elif objective == AnalysisObjective.MINIMAL_COST:
            if "estimated_cost" in item:
                return f"Kostnadseffektiv forbedring med god effekt ift. investering"
            else:
                return "Kostnadseffektivt tiltak med god avkastning"
        
        return "Forbedret boligkvalitet og verdi"
    
    def _calculate_roi(self, item: Dict, objective: AnalysisObjective) -> str:
        """
        Beregner og kategoriserer avkastning på investering
        """
        if "estimated_roi_years" in item:
            years = item["estimated_roi_years"]
            
            if years < 5:
                return "Høy (< 5 år)"
            elif years < 10:
                return "Medium (5-10 år)"
            else:
                return "Lav (> 10 år)"
        
        elif "estimated_value_increase" in item and "estimated_cost" in item:
            ratio = item["estimated_value_increase"] / item["estimated_cost"] if item["estimated_cost"] > 0 else 0
            
            if ratio > 1.5:
                return "Høy (> 150%)"
            elif ratio > 1.0:
                return "Medium (100-150%)"
            else:
                return "Lav (< 100%)"
        
        # Lignende logikk for andre typer verdivurderinger
        return "Medium"
    
    def _estimate_timeline(self, category: str, item: Dict) -> str:
        """
        Estimerer tidslinje for implementering
        """
        if category == "rental_units":
            return "3-6 måneder"
        
        elif category == "room_improvements":
            if item["type"] == "increase_windows":
                return "2-4 uker"
            elif item["type"] == "increase_room_size":
                return "1-3 måneder"
        
        elif category == "layout_changes":
            if item.get("is_load_bearing_wall", False):
                return "2-4 måneder"
            else:
                return "1-2 måneder"
        
        elif category == "extensions":
            if item["type"] == "extension":
                return "4-8 måneder"
            elif item["type"] == "additional_story":
                return "6-12 måneder"
        
        return "2-4 måneder"
    
    def _calculate_economics(self, recommendations: List[Dict], objective: AnalysisObjective) -> Dict:
        """
        Beregner økonomisk oversikt for anbefalinger
        """
        total_cost = sum(r["estimated_cost"] for r in recommendations if "estimated_cost" in r)
        
        income_potential = 0
        value_increase = 0
        
        for r in recommendations:
            if "details" in r:
                details = r["details"]
                
                if "estimated_monthly_rent" in details:
                    income_potential += details["estimated_monthly_rent"] * 12  # Årlig inntekt
                
                if "estimated_value_increase" in details:
                    value_increase += details["estimated_value_increase"]
                elif "value_increase_estimate" in details:
                    value_increase += details["value_increase_estimate"]
        
        # Beregn ROI basert på kundens mål
        if objective == AnalysisObjective.RENTAL_INCOME and income_potential > 0:
            roi_years = total_cost / income_potential if income_potential > 0 else float('inf')
            roi_percentage = (income_potential / total_cost) * 100 if total_cost > 0 else 0
        else:
            roi_years = total_cost / value_increase if value_increase > 0 else float('inf')
            roi_percentage = (value_increase / total_cost) * 100 if total_cost > 0 else 0
        
        return {
            "total_cost": total_cost,
            "annual_income_potential": income_potential,
            "estimated_value_increase": value_increase,
            "roi_years": roi_years,
            "roi_percentage": roi_percentage,
            "financing_options": [
                {
                    "type": "loan",
                    "description": "Boliglån med sikkerhet i eiendommen",
                    "typical_interest": "3-4%",
                    "monthly_payment": (total_cost * 0.035) / 12 if total_cost > 0 else 0  # Enkel beregning med 3.5% rente over 20 år
                },
                {
                    "type": "equity",
                    "description": "Egenkapital",
                    "advantage": "Ingen rentekostnader"
                }
            ]
        }
    
    def _generate_optimized_layout(self,
                                 rooms: List[Dict],
                                 walls: List[Dict],
                                 measurements: Dict,
                                 opportunities: Dict,
                                 recommendations: List[Dict]) -> Dict:
        """
        Genererer optimalisert plantegning basert på anbefalinger
        """
        # Dette er en placeholder for en faktisk implementasjon
        # I virkeligheten ville dette generere en ny plantegning
        
        # Samle endringer fra anbefalinger
        changes = []
        for recommendation in recommendations:
            if "details" in recommendation:
                details = recommendation["details"]
                
                if recommendation["category"] == "layout_changes" and details["type"] == "combine_rooms":
                    changes.append({
                        "type": "remove_wall",
                        "description": f"Fjern vegg mellom {details.get('rooms', ['rom', 'rom'])[0]} og {details.get('rooms', ['rom', 'rom'])[1]}"
                    })
                
                elif recommendation["category"] == "layout_changes" and details["type"] == "open_plan":
                    changes.append({
                        "type": "remove_wall",
                        "description": "Fjern vegg mellom kjøkken og stue"
                    })
                
                elif recommendation["category"] == "room_improvements" and details["type"] == "increase_windows":
                    changes.append({
                        "type": "add_window",
                        "description": f"Legg til vindu i {details.get('room', 'rom')}"
                    })
                
                elif recommendation["category"] == "rental_units":
                    if details["type"] == "basement_conversion":
                        changes.append({
                            "type": "convert_space",
                            "description": "Konverter kjeller til utleieenhet"
                        })
                    elif details["type"] == "attic_conversion":
                        changes.append({
                            "type": "convert_space",
                            "description": "Konverter loft til utleieenhet"
                        })
        
        return {
            "changes": changes,
            "description": "Optimalisert plantegning med implementerte anbefalinger",
            "diagram_data": {
                "width": 800,
                "height": 600,
                "elements": []  # Her ville faktiske tegningsdata ligge
            }
        }
    
    def _generate_analysis_summary(self, analysis_data: Dict) -> Dict:
        """
        Genererer oppsummering av analysen
        """
        rooms = analysis_data["rooms"]
        measurements = analysis_data["measurements"]
        opportunities = analysis_data["opportunities"]
        recommendations = analysis_data["recommendations"]
        economics = analysis_data["economics"]
        
        return {
            "property_summary": {
                "total_area_m2": measurements["total_area_m2"],
                "room_count": len(rooms),
                "layout_efficiency": measurements.get("area_efficiency", 0)
            },
            "opportunity_summary": {
                "rental_potential": len(opportunities["rental_units"]) > 0,
                "improvement_count": len(opportunities["room_improvements"]),
                "layout_change_options": len(opportunities["layout_changes"]),
                "extension_options": len(opportunities["extensions"])
            },
            "recommendation_summary": {
                "count": len(recommendations),
                "top_recommendation": recommendations[0]["title"] if recommendations else "Ingen anbefalinger",
                "total_cost": economics["total_cost"],
                "roi": economics["roi_percentage"]
            }
        }
    
    def _get_minimum_room_size(self, room_type: str) -> float:
        """
        Henter anbefalt minimumsstørrelse for forskjellige romtyper
        """
        minimum_sizes = {
            "bedroom": 7.0,
            "living_room": 15.0,
            "kitchen": 6.0,
            "bathroom": 4.0,
            "hallway": 3.0,
            "storage": 2.0,
            "attic": 15.0,
            "basement": 15.0
        }
        
        return minimum_sizes.get(room_type.lower(), 4.0)  # Standard minimum 4 m²
