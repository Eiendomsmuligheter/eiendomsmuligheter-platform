import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class FloorPlanAnalyzer:
    def __init__(self):
        self.model = self._load_model()
        self.room_classifier = self._load_room_classifier()
        
    def _load_model(self) -> tf.keras.Model:
        """Last inn eller initialiser modell for plananalyse"""
        try:
            model_path = Path("models/floor_plan_detector.h5")
            if model_path.exists():
                return tf.keras.models.load_model(str(model_path))
            else:
                # Hvis modellen ikke eksisterer, opprett en ny
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(None, None, 3)),
                    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
                    tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
                    tf.keras.layers.UpSampling2D(),
                    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
                    tf.keras.layers.UpSampling2D(),
                    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
                    tf.keras.layers.Conv2D(4, 1, activation='softmax')  # 4 klasser: vegg, dør, vindu, rom
                ])
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                return model
        except Exception as e:
            logger.error(f"Feil ved lasting av modell: {str(e)}")
            raise
        
    def _load_room_classifier(self) -> tf.keras.Model:
        """Last inn modell for romklassifisering"""
        try:
            model_path = Path("models/room_classifier.h5")
            if model_path.exists():
                return tf.keras.models.load_model(str(model_path))
            else:
                # Opprett ny modell for romklassifisering
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(224, 224, 3)),
                    tf.keras.layers.Conv2D(32, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(64, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(128, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(7, activation='softmax')  # 7 romtyper: stue, kjøkken, soverom, bad, gang, bod, annet
                ])
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                return model
        except Exception as e:
            logger.error(f"Feil ved lasting av romklassifiseringsmodell: {str(e)}")
            raise
        
    def analyze_floor_plan(self, image_path: str) -> Dict[str, Any]:
        """Analyser plantegning og returner detaljert informasjon"""
        try:
            # Last og preprosesser bilde
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Kunne ikke laste bilde")
                
            # Konverter til gråtoner
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Finn rom og vegger
            rooms = self._detect_rooms(gray)
            measurements = self._calculate_measurements(rooms)
            
            return {
                "rooms": len(rooms),
                "total_area": measurements["total_area"],
                "room_details": measurements["room_details"],
                "suggested_improvements": self._suggest_improvements(measurements)
            }
            
        except Exception as e:
            logger.error(f"Feil under analyse av plantegning: {str(e)}")
            return {"error": str(e)}
            
    def _detect_rooms(self, image: np.ndarray) -> List[np.ndarray]:
        """Detekter rom i plantegningen"""
        try:
            # Preprosessering
            processed = cv2.GaussianBlur(image, (5, 5), 0)
            edges = cv2.Canny(processed, 50, 150)
            
            # Finn konturer
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrer og prosesser rom
            rooms = []
            min_room_area = 1000  # Minimum areal for å regne det som et rom
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_room_area:
                    # Approximer polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Sjekk om det er et rektangulært rom
                    if len(approx) >= 4:
                        # Finn minimalt omsluttende rektangel
                        rect = cv2.minAreaRect(contour)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        
                        # Lagre rommet
                        room_mask = np.zeros_like(image)
                        cv2.drawContours(room_mask, [box], 0, (255, 255, 255), -1)
                        rooms.append(room_mask)
            
            return rooms
            
        except Exception as e:
            logger.error(f"Feil ved romdeteksjon: {str(e)}")
            return []
        
    def _calculate_measurements(self, rooms: List[np.ndarray]) -> Dict[str, Any]:
        """Beregn mål og arealer for hvert rom"""
        try:
            # Konstanter for konvertering fra piksler til meter
            PIXELS_PER_METER = 50  # Dette må kalibreres basert på input
            
            measurements = {
                "total_area": 0,
                "room_details": []
            }
            
            for i, room in enumerate(rooms):
                # Finn konturer for rommet
                contours, _ = cv2.findContours(room, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                    
                contour = contours[0]
                
                # Beregn areal
                area_pixels = cv2.contourArea(contour)
                area_meters = area_pixels / (PIXELS_PER_METER ** 2)
                
                # Finn minimalt omsluttende rektangel
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Beregn lengde og bredde
                width = rect[1][0] / PIXELS_PER_METER
                length = rect[1][1] / PIXELS_PER_METER
                
                # Beregn omkrets
                perimeter = cv2.arcLength(contour, True) / PIXELS_PER_METER
                
                # Klassifiser romtype basert på størrelse og form
                room_type = self._classify_room_type(area_meters, width, length)
                
                # Sjekk krav til takhøyde og dagslys
                meets_requirements = self._check_room_requirements(area_meters, room_type)
                
                room_details = {
                    "id": i + 1,
                    "type": room_type,
                    "area": round(area_meters, 2),
                    "width": round(width, 2),
                    "length": round(length, 2),
                    "perimeter": round(perimeter, 2),
                    "meets_requirements": meets_requirements,
                    "suggested_improvements": self._suggest_room_improvements(
                        area_meters, width, length, room_type, meets_requirements
                    )
                }
                
                measurements["room_details"].append(room_details)
                measurements["total_area"] += area_meters
            
            measurements["total_area"] = round(measurements["total_area"], 2)
            
            return measurements
            
        except Exception as e:
            logger.error(f"Feil ved beregning av mål: {str(e)}")
            return {
                "total_area": 0,
                "room_details": []
            }
        
    def _suggest_improvements(self, measurements: Dict[str, Any]) -> List[str]:
        """Foreslå forbedringer basert på analysen"""
        try:
            suggestions = []
            
            # Gjennomgå hvert rom
            for room in measurements.get("room_details", []):
                room_type = room["type"]
                area = room["area"]
                width = room["width"]
                length = room["length"]
                meets_requirements = room["meets_requirements"]
                
                # Sjekk minimumskrav basert på romtype
                if room_type == "soverom":
                    if area < 7:
                        suggestions.append(f"Rom {room['id']} (soverom): Øk arealet til minimum 7 m² for å oppfylle krav til soverom")
                    if min(width, length) < 2:
                        suggestions.append(f"Rom {room['id']} (soverom): Øk minste bredde til minimum 2 meter")
                
                elif room_type == "stue":
                    if area < 15:
                        suggestions.append(f"Rom {room['id']} (stue): Vurder å øke arealet til minimum 15 m² for bedre beboelighet")
                
                elif room_type == "kjøkken":
                    if area < 6:
                        suggestions.append(f"Rom {room['id']} (kjøkken): Øk arealet til minimum 6 m² for funksjonelt kjøkken")
                
                elif room_type == "bad":
                    if area < 4:
                        suggestions.append(f"Rom {room['id']} (bad): Øk arealet til minimum 4 m² for å oppfylle krav til bad")
                    if min(width, length) < 1.5:
                        suggestions.append(f"Rom {room['id']} (bad): Øk minste bredde til minimum 1.5 meter for tilgjengelighet")
                
                # Generelle forbedringer
                if not meets_requirements.get("takhøyde", True):
                    suggestions.append(f"Rom {room['id']} ({room_type}): Sørg for minimum takhøyde på 2.4 meter")
                
                if not meets_requirements.get("dagslys", True):
                    suggestions.append(f"Rom {room['id']} ({room_type}): Øk vindusareal til minimum 10% av gulvareal")
                
                if not meets_requirements.get("ventilasjon", True):
                    suggestions.append(f"Rom {room['id']} ({room_type}): Installer tilstrekkelig ventilasjon")
            
            # Generelle anbefalinger for hele planløsningen
            total_area = measurements.get("total_area", 0)
            if total_area < 40:
                suggestions.append("Vurder muligheten for å utvide totalarealet for bedre beboelighet")
            
            # Sjekk rømningsveier
            if len(measurements.get("room_details", [])) > 1:
                suggestions.append("Sørg for at alle rom har tilgang til rømningsvei i henhold til TEK17")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Feil ved generering av forbedringsforslag: {str(e)}")
            return ["Kunne ikke generere forbedringsforslag på grunn av en feil"]
        
    def visualize_analysis(self, image_path: str, analysis_results: Dict[str, Any]) -> np.ndarray:
        """Generer visuell representasjon av analysen"""
        try:
            # Last original bilde
            original = cv2.imread(image_path)
            if original is None:
                raise ValueError("Kunne ikke laste bilde for visualisering")
            
            # Opprett kopi for visualisering
            visualization = original.copy()
            
            # Fargekoder for forskjellige romtyper
            colors = {
                "soverom": (255, 0, 0),    # Rød
                "stue": (0, 255, 0),       # Grønn
                "kjøkken": (0, 0, 255),    # Blå
                "bad": (255, 255, 0),      # Cyan
                "gang": (255, 0, 255),     # Magenta
                "bod": (128, 128, 128),    # Grå
                "annet": (0, 128, 255)     # Orange
            }
            
            # Tegn rom og legg til tekst
            for room in analysis_results.get("room_details", []):
                # Hent romdata
                room_id = room["id"]
                room_type = room["type"]
                area = room["area"]
                meets_reqs = room["meets_requirements"]
                
                # Finn romkonturer (dette må tilpasses din spesifikke implementasjon)
                if "contour" in room:
                    contour = np.array(room["contour"])
                    
                    # Velg farge basert på om rommet oppfyller krav
                    color = colors.get(room_type, colors["annet"])
                    if not meets_reqs:
                        # Gjør fargen mørkere for rom som ikke oppfyller krav
                        color = tuple(int(c * 0.6) for c in color)
                    
                    # Tegn romkontur
                    cv2.drawContours(visualization, [contour], -1, color, 2)
                    
                    # Finn sentrum av rommet for tekstplassering
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Legg til tekst med rominformasjon
                        text = f"{room_type} ({area:.1f}m²)"
                        cv2.putText(
                            visualization, text, (cx-40, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                        )
            
            # Legg til total informasjon
            total_area = analysis_results.get("total_area", 0)
            cv2.putText(
                visualization,
                f"Total: {total_area:.1f}m²",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            # Legg til fargekodeforklaring
            legend_y = 60
            for room_type, color in colors.items():
                cv2.putText(
                    visualization,
                    room_type,
                    (10, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
                legend_y += 20
            
            return visualization
            
        except Exception as e:
            logger.error(f"Feil ved visualisering: {str(e)}")
            return None
            
    def _classify_room_type(self, area: float, width: float, length: float) -> str:
        """Klassifiser romtype basert på størrelse og form"""
        try:
            # Enkel klassifisering basert på areal og proporsjoner
            if area < 4:
                return "bod"
            elif area < 6:
                if min(width, length) < 1.5:
                    return "gang"
                else:
                    return "bad"
            elif area < 8:
                return "soverom"
            elif area < 12:
                if width/length > 1.5 or length/width > 1.5:
                    return "gang"
                else:
                    return "kjøkken"
            else:
                return "stue"
        except Exception as e:
            logger.error(f"Feil ved romklassifisering: {str(e)}")
            return "annet"
            
    def _check_room_requirements(self, area: float, room_type: str) -> Dict[str, bool]:
        """Sjekk om rommet oppfyller krav til takhøyde, dagslys etc."""
        try:
            requirements = {
                "takhøyde": True,  # Standard 2.4m (må implementeres med faktisk måling)
                "dagslys": True,   # 10% av gulvareal (må implementeres med vindusdeteksjon)
                "ventilasjon": True # Må implementeres med ventilasjonsdeteksjon
            }
            
            # Spesifikke krav basert på romtype
            if room_type == "soverom":
                requirements["dagslys"] = area >= 7  # Minimum 7m² for soverom
            elif room_type == "bad":
                requirements["ventilasjon"] = area >= 4  # Minimum 4m² for bad
            elif room_type == "kjøkken":
                requirements["ventilasjon"] = area >= 6  # Minimum 6m² for kjøkken
                
            return requirements
            
        except Exception as e:
            logger.error(f"Feil ved sjekk av romkrav: {str(e)}")
            return {
                "takhøyde": False,
                "dagslys": False,
                "ventilasjon": False
            }
            
    def _suggest_room_improvements(self, area: float, width: float, length: float,
                                room_type: str, meets_requirements: Dict[str, bool]) -> List[str]:
        """Foreslå forbedringer for et spesifikt rom"""
        try:
            suggestions = []
            
            # Sjekk minimumskrav
            if room_type == "soverom" and area < 7:
                suggestions.append("Øk arealet til minimum 7m²")
            elif room_type == "bad" and area < 4:
                suggestions.append("Øk arealet til minimum 4m²")
            elif room_type == "kjøkken" and area < 6:
                suggestions.append("Øk arealet til minimum 6m²")
                
            # Sjekk andre krav
            if not meets_requirements.get("takhøyde", True):
                suggestions.append("Øk takhøyden til minimum 2.4m")
            if not meets_requirements.get("dagslys", True):
                suggestions.append("Installer større vinduer for bedre dagslys")
            if not meets_requirements.get("ventilasjon", True):
                suggestions.append("Forbedre ventilasjonen")
                
            return suggestions
            
        except Exception as e:
            logger.error(f"Feil ved generering av romforbedringer: {str(e)}")
            return ["Kunne ikke generere forbedringsforslag for rommet"]