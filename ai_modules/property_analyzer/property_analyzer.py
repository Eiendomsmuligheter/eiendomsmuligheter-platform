import os
import sys
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import cv2
import tensorflow as tf
import torch
from PIL import Image
import logging
import yaml
import json
import requests
import asyncio
from datetime import datetime
import re
import time
from pathlib import Path
import traceback
from bs4 import BeautifulSoup
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification

# Sett opp logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PropertyAnalyzer:
    """
    Hovedklasse for analyse av eiendommer basert på bilder, adresse eller lenker.
    Integrerer alle analysemoduler og håndterer koordinering mellom dem.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialiserer PropertyAnalyzer med konfigurasjon og nødvendige modeller
        
        Args:
            config_path: Sti til konfigurasjonsfil
        """
        self.config = self._load_config(config_path)
        self.models = self._initialize_models()
        self.processors = self._initialize_processors()
        self.cache = {}  # Enkel hurtigbuffer for å unngå unødvendige API-kall
        self.model_paths = self._setup_model_paths()
        
        # Importere andre moduler etter behov for å unngå sirkelvise avhengigheter
        self._load_modules()
        
        logger.info("PropertyAnalyzer initialisert")
        
    def _load_config(self, config_path: str) -> Dict:
        """Laster konfigurasjon fra YAML-fil"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    config = yaml.safe_load(file)
                logger.info(f"Konfigurasjon lastet fra {config_path}")
                return config
            else:
                logger.warning(f"Konfigurasjonsfil {config_path} ikke funnet. Bruker standardverdier.")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Feil ved lasting av konfigurasjon: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Returnerer standardkonfigurasjon hvis konfigurasjonsfil ikke finnes"""
        return {
            "api_keys": {
                "google_maps": os.environ.get("GOOGLE_MAPS_API_KEY", ""),
                "finn_api": os.environ.get("FINN_API_KEY", ""),
                "municipality_api": os.environ.get("MUNICIPALITY_API_KEY", "")
            },
            "models": {
                "image_analyzer": "models/image_analyzer",
                "floor_plan_analyzer": "models/floor_plan_analyzer",
                "text_analyzer": "models/text_analyzer",
                "document_analyzer": "models/document_analyzer"
            },
            "paths": {
                "temp_dir": "temp",
                "output_dir": "output",
                "models_dir": "models",
                "data_dir": "data"
            },
            "municipality_data": {
                "drammen": {
                    "api_url": "https://innsyn2020.drammen.kommune.no/api",
                    "map_url": "https://kart.drammen.kommune.no"
                }
            },
            "enova": {
                "api_url": "https://api.enova.no",
                "support_rates": {
                    "heat_pump": 10000,
                    "insulation": 500,  # per m²
                    "windows": 1000     # per window
                }
            },
            "use_gpu": torch.cuda.is_available(),
            "processing": {
                "max_image_size": 1920,
                "image_quality": 90,
                "use_caching": True,
                "parallel_processing": True
            }
        }
    
    def _setup_model_paths(self) -> Dict[str, str]:
        """Setter opp stier til modellfilene"""
        base_dir = self.config.get("paths", {}).get("models_dir", "models")
        
        return {
            "image_analyzer": os.path.join(base_dir, "image_analyzer"),
            "floor_plan_analyzer": os.path.join(base_dir, "floor_plan_analyzer"),
            "document_analyzer": os.path.join(base_dir, "document_analyzer"),
            "region_segmentation": os.path.join(base_dir, "region_segmentation"),
            "3d_model_generator": os.path.join(base_dir, "3d_model_generator")
        }
    
    def _initialize_models(self) -> Dict:
        """Initialiserer alle nødvendige ML-modeller"""
        models = {}
        try:
            # Kun initiere modeller som trengs nå for å spare minne og oppstartstid
            # Andre modeller lastes ved behov
            if "layoutlm" in self.config.get("models", {}).get("document_analyzer", ""):
                processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
                model = LayoutLMv3ForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base")
                models["document_analyzer"] = {
                    "processor": processor,
                    "model": model
                }
            
            logger.info("Modeller initialisert")
            return models
        except Exception as e:
            logger.error(f"Feil ved initialisering av modeller: {str(e)}")
            return {}
    
    def _initialize_processors(self) -> Dict:
        """Initialiserer prosesseringsmoduler"""
        processors = {}
        
        # Disse vil bli konkrete instanser av de andre klassene i prosjektet
        # For nå, returnerer vi en tom dict som vil fylles ved behov
        
        return processors
    
    def _load_modules(self):
        """Laster inn andre moduler og klasser ved behov"""
        try:
            # Import these here to avoid circular imports
            from image_analyzer import ImageAnalyzer
            from floor_plan_analyzer import FloorPlanAnalyzer
            from price_estimator import PriceEstimator
            from energy_analyzer import EnergyAnalyzer
            
            # Initialize processors with specific instances
            self.processors["image_analyzer"] = ImageAnalyzer()
            self.processors["floor_plan_analyzer"] = FloorPlanAnalyzer()
            self.processors["price_estimator"] = PriceEstimator()
            self.processors["energy_analyzer"] = EnergyAnalyzer()
            
            logger.info("Alle moduler lastet inn")
        except ImportError as e:
            logger.warning(f"Kunne ikke importere alle moduler: {str(e)}. Noen funksjoner vil være utilgjengelige.")
        except Exception as e:
            logger.error(f"Feil ved lasting av moduler: {str(e)}")
    
    async def analyze_property(self, 
                        image_path: Optional[str] = None,
                        address: Optional[str] = None,
                        url: Optional[str] = None,
                        floor_plan_path: Optional[str] = None,
                        documents: Optional[List[str]] = None,
                        client_preferences: Optional[Dict] = None) -> Dict:
        """
        Hovedmetode for analyse av eiendom. Kan ta imot bilde, adresse eller URL.
        
        Args:
            image_path: Sti til bilde av eiendommen (eller liste med bilder)
            address: Eiendomsadresse
            url: URL til eiendomsannonse (f.eks. Finn.no)
            floor_plan_path: Sti til plantegning
            documents: Liste med stier til relevante dokumenter
            client_preferences: Kundeprefereranser for analysen
            
        Returns:
            Dict med komplett analyserapport inkludert:
            - Bygningsdetaljer
            - Reguleringsinfo
            - Utviklingspotensial
            - 3D-modell
            - Energianalyse
            - Kostnadskalkyle
        """
        start_time = time.time()
        
        try:
            # Konverter enkelt bilde til liste
            if isinstance(image_path, str):
                image_paths = [image_path]
            else:
                image_paths = image_path if image_path else []
            
            # Sett opp analyse-ID
            analysis_id = f"analysis_{int(time.time())}"
            logger.info(f"Starter eiendomsanalyse {analysis_id}")
            
            # Samle inn all grunnleggende informasjon (parallellisert)
            property_data = await self._gather_property_info(
                image_paths, address, url, floor_plan_path, documents, analysis_id
            )
            
            # Analysere bygningsstruktur og potensial
            structure_analysis = await self._analyze_building_structure(property_data)
            
            # Sjekke kommunale regler og forskrifter
            regulation_info = await self._check_regulations(property_data)
            
            # Analysere utviklingspotensial (inkl. kundepreferanser)
            development_potential = await self._analyze_development_potential(
                structure_analysis, 
                regulation_info,
                client_preferences
            )
            
            # Generere 3D-modell med NVIDIA Omniverse
            model_3d = await self._generate_3d_model(property_data, structure_analysis)
            
            # Utføre energianalyse
            energy_analysis = await self._perform_energy_analysis(property_data)
            
            # Generere kostnadsestimater
            cost_estimation = await self._estimate_costs(development_potential)
            
            # Beregn analyse-tid
            analysis_time = time.time() - start_time
            
            # Samle alle resultater
            results = {
                "analysis_id": analysis_id,
                "timestamp": datetime.now().isoformat(),
                "analysis_time_seconds": analysis_time,
                "property_info": property_data,
                "structure_analysis": structure_analysis,
                "regulations": regulation_info,
                "development_potential": development_potential,
                "3d_model": model_3d,
                "energy_analysis": energy_analysis,
                "cost_estimation": cost_estimation,
                "recommendations": await self._generate_recommendations(
                    property_data,
                    structure_analysis,
                    regulation_info,
                    development_potential,
                    energy_analysis,
                    cost_estimation,
                    client_preferences
                )
            }
            
            # Lagre resultater til disk
            self._save_analysis_results(results, analysis_id)
            
            logger.info(f"Eiendomsanalyse {analysis_id} fullført på {analysis_time:.2f} sekunder")
            return results
            
        except Exception as e:
            logger.error(f"Feil i eiendomsanalyse: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "error_details": traceback.format_exc(),
                "analysis_time_seconds": time.time() - start_time
            }
    
    def _save_analysis_results(self, results: Dict, analysis_id: str):
        """Lagrer analyseresultater til disk"""
        output_dir = self.config.get("paths", {}).get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Lagre hovedresultater
        output_path = os.path.join(output_dir, f"{analysis_id}_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        # Lagre 3D-modell separat (hvis den eksisterer)
        if "3d_model" in results and "model_data" in results["3d_model"]:
            model_path = os.path.join(output_dir, f"{analysis_id}_3d_model.glb")
            with open(model_path, 'wb') as f:
                f.write(results["3d_model"]["model_data"])
            
            # Erstatt binærdata med filsti i hovedresultatene
            results["3d_model"]["model_data"] = model_path
            
        logger.info(f"Analyseresultater lagret til {output_path}")
            
    async def _gather_property_info(self, 
                                 image_paths: List[str], 
                                 address: Optional[str], 
                                 url: Optional[str],
                                 floor_plan_path: Optional[str],
                                 documents: Optional[List[str]],
                                 analysis_id: str) -> Dict:
        """Samler inn all tilgjengelig informasjon om eiendommen"""
        property_info = {
            "source_data": {
                "has_images": bool(image_paths),
                "has_address": bool(address),
                "has_url": bool(url),
                "has_floor_plan": bool(floor_plan_path),
                "has_documents": bool(documents)
            }
        }
        
        # Kjøre oppgaver parallelt for å spare tid
        tasks = []
        
        # Oppgave 1: Analysere bilder hvis tilgjengelig
        if image_paths:
            tasks.append(self._analyze_images(image_paths))
        
        # Oppgave 2: Hente adresseinformasjon hvis tilgjengelig
        if address:
            tasks.append(self._fetch_address_info(address))
        
        # Oppgave 3: Skrape eiendomsannonse hvis tilgjengelig
        if url:
            tasks.append(self._scrape_property_listing(url))
        
        # Oppgave 4: Analysere plantegning hvis tilgjengelig
        if floor_plan_path:
            tasks.append(self._analyze_floor_plan(floor_plan_path))
        
        # Oppgave 5: Analysere dokumenter hvis tilgjengelig
        if documents:
            tasks.append(self._analyze_documents(documents))
        
        # Vent på at alle oppgaver skal fullføres
        results = await asyncio.gather(*tasks)
        
        # Slå sammen resultater
        for result in results:
            property_info.update(result)
        
        # Legge til unik ID
        property_info["analysis_id"] = analysis_id
        
        # Tilføy tidspunkt for analyse
        property_info["analysis_timestamp"] = datetime.now().isoformat()
        
        return property_info
    
    async def _analyze_images(self, image_paths: List[str]) -> Dict:
        """Analyserer bilder av eiendommen"""
        logger.info(f"Analyserer {len(image_paths)} bilder")
        
        try:
            # Bruk ImageAnalyzer-modulen for detaljert bildeanalyse
            if "image_analyzer" in self.processors:
                # Konvertere til absolutte stier
                abs_paths = [os.path.abspath(path) for path in image_paths if os.path.exists(path)]
                
                if not abs_paths:
                    logger.warning("Ingen gyldige bildestier funnet")
                    return {"images_analysis": {"error": "Ingen gyldige bilder funnet"}}
                
                # Kjør bildeanalyse
                results = await self.processors["image_analyzer"].analyze(abs_paths)
                return {"images_analysis": results}
            else:
                logger.warning("ImageAnalyzer ikke tilgjengelig")
                return {"images_analysis": self._simple_image_analysis(image_paths)}
        except Exception as e:
            logger.error(f"Feil ved bildeanalyse: {str(e)}")
            return {"images_analysis": {"error": str(e)}}
    
    def _simple_image_analysis(self, image_paths: List[str]) -> Dict:
        """Enkel bildeanalyse når ImageAnalyzer ikke er tilgjengelig"""
        result = {
            "image_count": len(image_paths),
            "images": []
        }
        
        for path in image_paths:
            if os.path.exists(path):
                try:
                    img = Image.open(path)
                    width, height = img.size
                    
                    result["images"].append({
                        "path": path,
                        "width": width,
                        "height": height,
                        "format": img.format,
                        "size_bytes": os.path.getsize(path)
                    })
                except Exception as e:
                    result["images"].append({
                        "path": path,
                        "error": str(e)
                    })
            else:
                result["images"].append({
                    "path": path,
                    "error": "File not found"
                })
        
        return result
    
    async def _fetch_address_info(self, address: str) -> Dict:
        """Henter informasjon basert på eiendomsadresse"""
        logger.info(f"Henter informasjon for adresse: {address}")
        
        # Rens adressestreng
        clean_address = self._clean_address(address)
        
        # Hent Google Maps API-nøkkel
        api_key = self.config.get("api_keys", {}).get("google_maps", "")
        
        if not api_key:
            logger.warning("Google Maps API-nøkkel mangler")
            return {
                "address_info": {
                    "input_address": address,
                    "cleaned_address": clean_address,
                    "error": "API-nøkkel mangler"
                }
            }
        
        try:
            # Gjør en geocoding-forespørsel
            geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={clean_address}&key={api_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(geocode_url) as response:
                    geocode_data = await response.json()
            
            if geocode_data.get("status") != "OK":
                logger.warning(f"Geocoding-feil: {geocode_data.get('status')}")
                return {
                    "address_info": {
                        "input_address": address,
                        "cleaned_address": clean_address,
                        "error": f"Geocoding-feil: {geocode_data.get('status')}"
                    }
                }
            
            # Hent første resultat
            result = geocode_data["results"][0]
            
            # Hent koordinater
            location = result["geometry"]["location"]
            lat, lng = location["lat"], location["lng"]
            
            # Hent adressekomponenter
            address_components = result["address_components"]
            
            # Bygg adresseinformasjon
            address_info = {
                "input_address": address,
                "formatted_address": result["formatted_address"],
                "coordinates": {
                    "latitude": lat,
                    "longitude": lng
                },
                "components": {}
            }
            
            # Filtrer adressekomponenter
            for component in address_components:
                types = component["types"]
                if "street_number" in types:
                    address_info["components"]["street_number"] = component["long_name"]
                elif "route" in types:
                    address_info["components"]["street"] = component["long_name"]
                elif "locality" in types:
                    address_info["components"]["city"] = component["long_name"]
                elif "administrative_area_level_1" in types:
                    address_info["components"]["county"] = component["long_name"]
                elif "postal_code" in types:
                    address_info["components"]["postal_code"] = component["long_name"]
                elif "country" in types:
                    address_info["components"]["country"] = component["long_name"]
            
            # Hent kommune (municipality)
            address_info["components"]["municipality"] = self._get_municipality_from_coordinates(lat, lng)
            
            # Hent eiendomsinformasjon (gnr/bnr)
            property_id = await self._get_property_id(address_info)
            if property_id:
                address_info["property_id"] = property_id
            
            return {"address_info": address_info}
            
        except Exception as e:
            logger.error(f"Feil ved henting av adresseinformasjon: {str(e)}")
            return {
                "address_info": {
                    "input_address": address,
                    "cleaned_address": clean_address,
                    "error": str(e)
                }
            }
    
    def _clean_address(self, address: str) -> str:
        """Renser adressestreng"""
        # Fjern ekstra mellomrom
        address = re.sub(r'\s+', ' ', address).strip()
        
        # Legg til Norge hvis det mangler
        if not re.search(r'norge|norway', address.lower()):
            address += ", Norge"
        
        return address
    
    def _get_municipality_from_coordinates(self, lat: float, lng: float) -> str:
        """Henter kommunen basert på koordinater"""
        # I en reell implementasjon ville dette bruke Kartverkets API eller lignende
        # For nå, bruk en forenklet tilnærming
        
        # Hent predefinerte kommunegrenser fra config
        municipality_boundaries = self.config.get("municipality_boundaries", {})
        
        # Sjekk om koordinatene er innenfor grensene
        for municipality, bounds in municipality_boundaries.items():
            if (bounds["lat_min"] <= lat <= bounds["lat_max"] and
                bounds["lng_min"] <= lng <= bounds["lng_max"]):
                return municipality
        
        # Fallback: basert på storbyer
        # Oslo
        if 59.8 <= lat <= 60.0 and 10.6 <= lng <= 10.9:
            return "oslo"
        # Bergen
        elif 60.3 <= lat <= 60.45 and 5.2 <= lng <= 5.4:
            return "bergen"
        # Trondheim
        elif 63.4 <= lat <= 63.5 and 10.3 <= lng <= 10.5:
            return "trondheim"
        # Drammen
        elif 59.7 <= lat <= 59.8 and 10.15 <= lng <= 10.25:
            return "drammen"
        
        # Hvis ingen match, returner en generisk verdi
        return "unknown"
    
    async def _get_property_id(self, address_info: Dict) -> Optional[Dict]:
        """Henter eiendomsinformasjon (gnr/bnr) fra adresse"""
        try:
            municipality = address_info.get("components", {}).get("municipality", "")
            street = address_info.get("components", {}).get("street", "")
            street_number = address_info.get("components", {}).get("street_number", "")
            
            if not (municipality and street and street_number):
                logger.warning("Ufullstendig adresseinformasjon for eiendomsID-oppslag")
                return None
            
            # Bruk kommune-API for å hente eiendomsinformasjon
            # Dette er en forenklet implementasjon
            api_url = self.config.get("municipality_data", {}).get(municipality, {}).get("api_url", "")
            
            if not api_url:
                logger.warning(f"Mangler API-URL for kommune: {municipality}")
                return None
            
            # I en reell implementasjon ville dette bruke faktisk API
            # For nå, returner dummy-data
            return {
                "municipality_code": "3005",
                "gnr": 10,
                "bnr": 25,
                "fnr": None,
                "snr": None
            }
            
        except Exception as e:
            logger.error(f"Feil ved henting av eiendomsID: {str(e)}")
            return None
    
    async def _scrape_property_listing(self, url: str) -> Dict:
        """Henter informasjon fra eiendomsannonse"""
        logger.info(f"Henter informasjon fra annonse: {url}")
        
        try:
            # Sjekk om URL-en er støttet
            if "finn.no" in url.lower():
                return await self._scrape_finn_listing(url)
            else:
                logger.warning(f"Ikke-støttet annonse-URL: {url}")
                return {"listing_info": {"error": "Ikke-støttet annonse-URL"}}
        except Exception as e:
            logger.error(f"Feil ved skraping av annonse: {str(e)}")
            return {"listing_info": {"error": str(e)}}
    
    async def _scrape_finn_listing(self, url: str) -> Dict:
        """Henter informasjon fra Finn.no-annonse"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.warning(f"Feil ved henting av Finn-annonse: {response.status}")
                        return {"listing_info": {"error": f"HTTP-feil: {response.status}"}}
                    
                    html = await response.text()
            
            # Parse HTML med BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Hent grunnleggende informasjon
            title = soup.find('h1', class_='u-t2').text.strip() if soup.find('h1', class_='u-t2') else "Ukjent tittel"
            
            # Hent prisinfo
            price_elem = soup.find('span', class_='u-t1')
            price = price_elem.text.strip() if price_elem else "Ukjent pris"
            
            # Hent nøkkelinformasjon
            key_info = {}
            key_info_table = soup.find('dl', class_='definition-list')
            
            if key_info_table:
                terms = key_info_table.find_all('dt')
                details = key_info_table.find_all('dd')
                
                for i in range(min(len(terms), len(details))):
                    key = terms[i].text.strip()
                    value = details[i].text.strip()
                    key_info[key] = value
            
            # Hent beskrivelse
            description = ""
            desc_elem = soup.find('div', class_='markdown')
            if desc_elem:
                description = desc_elem.text.strip()
            
            # Hent bildekoblinger
            images = []
            image_elements = soup.find_all('img', class_='image__img')
            for img in image_elements:
                if 'src' in img.attrs:
                    images.append(img['src'])
            
            return {
                "listing_info": {
                    "title": title,
                    "url": url,
                    "price": price,
                    "key_info": key_info,
                    "description": description,
                    "image_urls": images
                }
            }
            
        except Exception as e:
            logger.error(f"Feil ved parsering av Finn-annonse: {str(e)}")
            return {"listing_info": {"error": str(e), "url": url}}
    
    async def _analyze_floor_plan(self, floor_plan_path: str) -> Dict:
        """Analyserer plantegning"""
        logger.info(f"Analyserer plantegning: {floor_plan_path}")
        
        try:
            # Sjekk om filen eksisterer
            if not os.path.exists(floor_plan_path):
                logger.warning(f"Plantegning ikke funnet: {floor_plan_path}")
                return {"floor_plan_analysis": {"error": "Fil ikke funnet"}}
            
            # Bruk FloorPlanAnalyzer-modulen hvis tilgjengelig
            if "floor_plan_analyzer" in self.processors:
                abs_path = os.path.abspath(floor_plan_path)
                results = await self.processors["floor_plan_analyzer"].analyze(abs_path)
                return {"floor_plan_analysis": results}
            else:
                logger.warning("FloorPlanAnalyzer ikke tilgjengelig")
                return {"floor_plan_analysis": self._simple_floor_plan_analysis(floor_plan_path)}
        except Exception as e:
            logger.error(f"Feil ved analyse av plantegning: {str(e)}")
            return {"floor_plan_analysis": {"error": str(e)}}
    
    def _simple_floor_plan_analysis(self, floor_plan_path: str) -> Dict:
        """Enkel plantegningsanalyse når FloorPlanAnalyzer ikke er tilgjengelig"""
        try:
            img = Image.open(floor_plan_path)
            width, height = img.size
            
            # Enkel analyse basert på dimensjoner og farger
            return {
                "path": floor_plan_path,
                "dimensions": {
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height if height > 0 else 0
                },
                "estimated_area": (width / 100) * (height / 100),  # Very rough estimate
                "is_color": img.mode == "RGB",
                "is_hand_drawn": self._estimate_if_hand_drawn(img)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _estimate_if_hand_drawn(self, img: Image.Image) -> bool:
        """Estimerer om en plantegning er håndtegnet"""
        # Konverter til gråtoner
        gray_img = img.convert("L")
        np_img = np.array(gray_img)
        
        # Beregn kantdeteksjon
        edges = cv2.Canny(np_img, 100, 200)
        
        # Beregn ikke-rette kanter
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return False
        
        non_straight_lines = 0
        total_lines = len(lines)
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Sjekk om linjen er skrå (ikke horisontal eller vertikal)
            if not (angle < 5 or abs(angle - 90) < 5 or abs(angle - 180) < 5):
                non_straight_lines += 1
        
        # Hvis mer enn 30% av linjene er skrå, er det sannsynligvis håndtegnet
        return (non_straight_lines / total_lines) > 0.3 if total_lines > 0 else False
    
    async def _analyze_documents(self, document_paths: List[str]) -> Dict:
        """Analyserer relevante dokumenter for eiendommen"""
        logger.info(f"Analyserer {len(document_paths)} dokumenter")
        
        results = {"documents_analysis": {"documents": []}}
        
        for doc_path in document_paths:
            if not os.path.exists(doc_path):
                results["documents_analysis"]["documents"].append({
                    "path": doc_path,
                    "error": "Fil ikke funnet"
                })
                continue
            
            try:
                # Bestem filtype
                file_ext = os.path.splitext(doc_path)[1].lower()
                
                if file_ext in ['.pdf', '.PDF']:
                    doc_result = await self._analyze_pdf(doc_path)
                elif file_ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                    doc_result = await self._analyze_document_image(doc_path)
                elif file_ext in ['.doc', '.docx']:
                    doc_result = await self._analyze_word_document(doc_path)
                else:
                    doc_result = {
                        "path": doc_path,
                        "type": "unknown",
                        "error": "Ikke-støttet filtype"
                    }
                
                results["documents_analysis"]["documents"].append(doc_result)
                
            except Exception as e:
                results["documents_analysis"]["documents"].append({
                    "path": doc_path,
                    "error": str(e)
                })
        
        return results
    
    async def _analyze_pdf(self, pdf_path: str) -> Dict:
        """Analyserer PDF-dokument"""
        # I en reell implementasjon ville dette bruke en PDF-parser
        # For nå, returnerer vi dummy-data
        return {
            "path": pdf_path,
            "type": "pdf",
            "page_count": 5,
            "document_type": self._estimate_document_type(pdf_path)
        }
    
    async def _analyze_document_image(self, image_path: str) -> Dict:
        """Analyserer dokumentbilde med OCR"""
        # I en reell implementasjon ville dette bruke OCR
        # For nå, returnerer vi dummy-data
        return {
            "path": image_path,
            "type": "image",
            "document_type": self._estimate_document_type(image_path)
        }
    
    async def _analyze_word_document(self, doc_path: str) -> Dict:
        """Analyserer Word-dokument"""
        # I en reell implementasjon ville dette bruke en docx-parser
        # For nå, returnerer vi dummy-data
        return {
            "path": doc_path,
            "type": "word",
            "document_type": self._estimate_document_type(doc_path)
        }
    
    def _estimate_document_type(self, doc_path: str) -> str:
        """Estimerer dokumenttype basert på filnavn"""
        filename = os.path.basename(doc_path).lower()
        
        if any(term in filename for term in ["tegning", "plan", "drawing"]):
            return "floor_plan"
        elif any(term in filename for term in ["byggesak", "application"]):
            return "building_application"
        elif any(term in filename for term in ["regulering", "zoning"]):
            return "zoning_plan"
        elif any(term in filename for term in ["energi", "energy"]):
            return "energy_certificate"
        else:
            return "unknown"
    
    async def _analyze_building_structure(self, property_data: Dict) -> Dict:
        """Analyserer bygningsstruktur og identifiserer muligheter"""
        logger.info("Analyserer bygningsstruktur")
        
        # Samle alle relevante data fra ulike kilder
        floor_plan_data = property_data.get("floor_plan_analysis", {})
        images_data = property_data.get("images_analysis", {})
        address_data = property_data.get("address_info", {})
        listing_data = property_data.get("listing_info", {})
        
        # Hent areal fra annonsen hvis tilgjengelig
        area_m2 = self._extract_area_from_listing(listing_data)
        
        try:
            # Analyser etasjer
            floors = await self._analyze_floors(property_data)
            
            # Analyser kjeller
            basement = await self._analyze_basement(property_data)
            
            # Analyser loft
            attic = await self._analyze_attic(property_data)
            
            # Hent mål
            measurements = self._get_measurements(property_data)
            
            # Identifiser konstruksjonstype
            construction_type = self._identify_construction_type(property_data)
            
            return {
                "floors": floors,
                "basement": basement,
                "attic": attic,
                "measurements": measurements,
                "construction_type": construction_type,
                "total_area_m2": area_m2,
                "room_count": self._extract_room_count(listing_data, floor_plan_data)
            }
        
        except Exception as e:
            logger.error(f"Feil ved bygningsstrukturanalyse: {str(e)}")
            return {
                "error": str(e),
                "floors": [],
                "basement": {"exists": False},
                "attic": {"exists": False},
                "measurements": {},
                "construction_type": "unknown"
            }
    
    def _extract_area_from_listing(self, listing_data: Dict) -> Optional[float]:
        """Henter arealinfo fra annonsedata"""
        key_info = listing_data.get("key_info", {})
        
        # Prøv å finne areal i ulike nøkkelord
        for key in ["Bruksareal", "P-rom", "Bra", "Primærrom"]:
            if key in key_info:
                value = key_info[key]
                # Ekstraher tall fra streng
                match = re.search(r'(\d+[.,]?\d*)', value)
                if match:
                    try:
                        # Konverter til float, bytt komma med punktum
                        return float(match.group(1).replace(',', '.'))
                    except ValueError:
                        pass
        
        return None
    
    def _extract_room_count(self, listing_data: Dict, floor_plan_data: Dict) -> Optional[int]:
        """Henter antall rom fra annonse eller plantegning"""
        # Prøv fra annonsen først
        key_info = listing_data.get("key_info", {})
        
        for key in ["Rom", "Antall rom", "Soverom"]:
            if key in key_info:
                value = key_info[key]
                match = re.search(r'(\d+)', value)
                if match:
                    try:
                        return int(match.group(1))
                    except ValueError:
                        pass
        
        # Prøv fra plantegning
        rooms = floor_plan_data.get("rooms", [])
        if rooms:
            return len(rooms)
        
        return None
    
    async def _analyze_floors(self, property_data: Dict) -> List[Dict]:
        """Analyserer etasjer i bygningen"""
        # Hent relevant data fra ulike kilder
        floor_plan_data = property_data.get("floor_plan_analysis", {})
        listing_data = property_data.get("listing_info", {})
        
        floors = []
        
        # Hent fra plantegning hvis tilgjengelig
        if "rooms" in floor_plan_data:
            # Grupper rom etter etasje
            rooms_by_floor = {}
            
            for room in floor_plan_data.get("rooms", []):
                floor_num = room.get("floor", 1)
                if floor_num not in rooms_by_floor:
                    rooms_by_floor[floor_num] = []
                rooms_by_floor[floor_num].append(room)
            
            # Lag etasjeobjekter
            for floor_num, rooms in rooms_by_floor.items():
                floor_area = sum(room.get("area_m2", 0) for room in rooms)
                floor_obj = {
                    "floor_number": floor_num,
                    "rooms": rooms,
                    "area_m2": floor_area,
                    "room_count": len(rooms)
                }
                floors.append(floor_obj)
        else:
            # Hvis plantegning ikke er tilgjengelig, lag en generisk etasje
            # basert på annonseinfo
            area_m2 = self._extract_area_from_listing(listing_data)
            room_count = self._extract_room_count(listing_data, floor_plan_data)
            
            floor_obj = {
                "floor_number": 1,
                "area_m2": area_m2,
                "room_count": room_count,
                "rooms": []
            }
            floors.append(floor_obj)
        
        # Sorter etasjer etter nummer
        floors.sort(key=lambda f: f["floor_number"])
        
        return floors
    
    async def _analyze_basement(self, property_data: Dict) -> Dict:
        """Analyserer kjeller hvis det finnes"""
        # Hent relevant data
        floor_plan_data = property_data.get("floor_plan_analysis", {})
        listing_data = property_data.get("listing_info", {})
        
        # Sjekk om kjeller er nevnt i annonsen
        has_basement = False
        basement_info = {}
        
        # Sjekk nøkkelinfo
        key_info = listing_data.get("key_info", {})
        for key, value in key_info.items():
            if "kjeller" in key.lower():
                has_basement = True
                basement_info["mentioned_in_listing"] = value
        
        # Sjekk beskrivelse
        description = listing_data.get("description", "")
        if "kjeller" in description.lower():
            has_basement = True
            basement_info["mentioned_in_description"] = True
        
        # Sjekk plantegning
        rooms = floor_plan_data.get("rooms", [])
        basement_rooms = [r for r in rooms if r.get("floor", 0) < 1]
        
        if basement_rooms:
            has_basement = True
            basement_area = sum(room.get("area_m2", 0) for room in basement_rooms)
            basement_info["rooms"] = basement_rooms
            basement_info["area_m2"] = basement_area
            basement_info["room_count"] = len(basement_rooms)
        
        result = {
            "exists": has_basement,
            "info": basement_info if has_basement else {}
        }
        
        # Vurder kjelleren for utleie hvis den eksisterer
        if has_basement:
            result["rental_potential"] = self._assess_basement_rental_potential(basement_info)
        
        return result
    
    def _assess_basement_rental_potential(self, basement_info: Dict) -> Dict:
        """Vurderer potensialet for utleie av kjeller"""
        potential = {
            "suitable_for_rental": False,
            "confidence": 0.0,
            "required_modifications": []
        }
        
        # Sjekk areal
        area = basement_info.get("area_m2", 0)
        if area >= 30:
            potential["suitable_for_rental"] = True
            potential["confidence"] = min(1.0, area / 50)  # Høyere areal, høyere tillit
        else:
            potential["required_modifications"].append("Utilstrekkelig areal for utleie")
            return potential
        
        # Sjekk romtyper
        rooms = basement_info.get("rooms", [])
        has_bathroom = any(r.get("type", "").lower() in ["bathroom", "bad"] for r in rooms)
        has_kitchen = any(r.get("type", "").lower() in ["kitchen", "kjøkken"] for r in rooms)
        
        if not has_bathroom:
            potential["required_modifications"].append("Mangler bad")
            potential["confidence"] -= 0.3
        
        if not has_kitchen:
            potential["required_modifications"].append("Mangler kjøkken")
            potential["confidence"] -= 0.2
        
        # Juster endelig tillit
        potential["confidence"] = max(0.0, min(1.0, potential["confidence"]))
        
        # Oppdater egnethet basert på tillit
        potential["suitable_for_rental"] = potential["confidence"] >= 0.5
        
        return potential
    
    async def _analyze_attic(self, property_data: Dict) -> Dict:
        """Analyserer loft hvis det finnes"""
        # Lignende implementasjon som for kjeller
        floor_plan_data = property_data.get("floor_plan_analysis", {})
        listing_data = property_data.get("listing_info", {})
        
        # Sjekk om loft er nevnt i annonsen
        has_attic = False
        attic_info = {}
        
        # Sjekk nøkkelinfo
        key_info = listing_data.get("key_info", {})
        for key, value in key_info.items():
            if any(term in key.lower() for term in ["loft", "attic"]):
                has_attic = True
                attic_info["mentioned_in_listing"] = value
        
        # Sjekk beskrivelse
        description = listing_data.get("description", "")
        if any(term in description.lower() for term in ["loft", "attic"]):
            has_attic = True
            attic_info["mentioned_in_description"] = True
        
        # Sjekk plantegning - anta at øverste etasje over første kan være loft
        rooms = floor_plan_data.get("rooms", [])
        floor_numbers = [r.get("floor", 1) for r in rooms]
        max_floor = max(floor_numbers) if floor_numbers else 1
        
        attic_rooms = [r for r in rooms if r.get("floor", 0) == max_floor and max_floor > 1]
        
        if attic_rooms:
            has_attic = True
            attic_area = sum(room.get("area_m2", 0) for room in attic_rooms)
            attic_info["rooms"] = attic_rooms
            attic_info["area_m2"] = attic_area
            attic_info["room_count"] = len(attic_rooms)
        
        result = {
            "exists": has_attic,
            "info": attic_info if has_attic else {}
        }
        
        # Vurder loftet for utleie hvis det eksisterer
        if has_attic:
            result["conversion_potential"] = self._assess_attic_conversion_potential(attic_info)
        
        return result
    
    def _assess_attic_conversion_potential(self, attic_info: Dict) -> Dict:
        """Vurderer potensialet for konvertering av loft"""
        potential = {
            "suitable_for_conversion": False,
            "confidence": 0.0,
            "required_modifications": []
        }
        
        # Sjekk areal
        area = attic_info.get("area_m2", 0)
        if area >= 15:
            potential["suitable_for_conversion"] = True
            potential["confidence"] = min(1.0, area / 40)  # Høyere areal, høyere tillit
        else:
            potential["required_modifications"].append("Utilstrekkelig areal for konvertering")
            return potential
        
        # Sjekk romtyper - hvis det allerede er innredet
        rooms = attic_info.get("rooms", [])
        is_already_converted = any(r.get("type", "").lower() in ["bedroom", "soverom", "living_room", "stue"] for r in rooms)
        
        if is_already_converted:
            potential["suitable_for_conversion"] = True
            potential["confidence"] = 0.9
            potential["already_converted"] = True
        else:
            potential["required_modifications"].append("Behov for full innredning av loft")
            potential["confidence"] -= 0.2
        
        # Juster endelig tillit
        potential["confidence"] = max(0.0, min(1.0, potential["confidence"]))
        
        # Oppdater egnethet basert på tillit
        potential["suitable_for_conversion"] = potential["confidence"] >= 0.5
        
        return potential
    
    def _get_measurements(self, property_data: Dict) -> Dict:
        """Henter ut mål og dimensjoner"""
        measurements = {}
        
        # Hent areal
        area_m2 = self._extract_area_from_listing(property_data.get("listing_info", {}))
        if area_m2:
            measurements["total_area_m2"] = area_m2
        
        # Hent data fra plantegning hvis tilgjengelig
        floor_plan_data = property_data.get("floor_plan_analysis", {})
        
        if "measurements" in floor_plan_data:
            measurements.update(floor_plan_data["measurements"])
        
        # Beregn ekstra nyttige dimensjoner
        if "total_area_m2" in measurements:
            # Estimert fasadeareal (grovt anslag)
            floor_count = max(1, len(property_data.get("structure_analysis", {}).get("floors", [1])))
            floor_height = 2.4  # Standard etasjehøyde
            
            # Estimere grunnflate (antar kvadratisk form hvis ikke annet er spesifisert)
            if "width" in measurements and "length" in measurements:
                footprint = measurements["width"] * measurements["length"]
            else:
                footprint = measurements["total_area_m2"] / floor_count
                measurements["estimated_width"] = np.sqrt(footprint)
                measurements["estimated_length"] = np.sqrt(footprint)
            
            # Beregn estimert fasadeareal
            perimeter = 2 * (measurements.get("estimated_width", 0) + measurements.get("estimated_length", 0))
            facade_area = perimeter * floor_height * floor_count
            measurements["estimated_facade_area_m2"] = facade_area
        
        return measurements
    
    def _identify_construction_type(self, property_data: Dict) -> Dict:
        """Identifiserer konstruksjonstype basert på tilgjengelig informasjon"""
        construction_type = {
            "primary_material": "unknown",
            "construction_year": None,
            "style": "unknown",
            "confidence": 0.0
        }
        
        # Sjekk om byggeår er angitt i annonsen
        listing_data = property_data.get("listing_info", {})
        key_info = listing_data.get("key_info", {})
        
        for key in ["Byggeår", "Bygget", "Byggår"]:
            if key in key_info:
                match = re.search(r'(\d{4})', key_info[key])
                if match:
                    construction_type["construction_year"] = int(match.group(1))
                    break
        
        # Sjekk materialtype fra bildeanalyse
        images_analysis = property_data.get("images_analysis", {})
        exterior_analysis = images_analysis.get("exterior_analysis", {})
        materials = exterior_analysis.get("materials", {})
        
        if materials:
            detected_materials = materials.get("detected_materials", [])
            if detected_materials:
                # Sorter etter dekning og ta det viktigste materialet
                sorted_materials = sorted(detected_materials, key=lambda m: m.get("coverage", 0), reverse=True)
                primary_material = sorted_materials[0]
                construction_type["primary_material"] = primary_material.get("type", "unknown")
                construction_type["confidence"] = primary_material.get("confidence", 0.5)
        
        # Sjekk stil fra bildeanalyse
        style_info = exterior_analysis.get("style", {})
        if style_info:
            construction_type["style"] = style_info.get("detected_style", "unknown")
        
        # Estimer konstruksjonstype basert på byggeår hvis materialet er ukjent
        if construction_type["primary_material"] == "unknown" and construction_type["construction_year"]:
            year = construction_type["construction_year"]
            
            if year < 1950:
                construction_type["primary_material"] = "wood"
                construction_type["confidence"] = 0.7
            elif 1950 <= year < 1980:
                construction_type["primary_material"] = "concrete"
                construction_type["confidence"] = 0.6
            else:
                construction_type["primary_material"] = "mixed"
                construction_type["confidence"] = 0.5
        
        return construction_type
    
    async def _check_regulations(self, property_data: Dict) -> Dict:
        """Sjekker gjeldende reguleringsplan og byggtekniske forskrifter"""
        logger.info("Sjekker reguleringer og forskrifter")
        
        try:
            # Hent kommune
            address_info = property_data.get("address_info", {})
            municipality = self._get_municipality(address_info)
            
            # Hent reguleringsplan
            zoning_plan = await self._fetch_zoning_plan(municipality, property_data)
            
            # Hent byggtekniske forskrifter
            building_regulations = await self._get_building_regulations(municipality)
            
            # Sjekk restriksjoner
            restrictions = await self._check_restrictions(municipality, property_data)
            
            # Hent krav
            requirements = await self._get_requirements(municipality, property_data)
            
            return {
                "municipality": municipality,
                "zoning_plan": zoning_plan,
                "building_regulations": building_regulations,
                "restrictions": restrictions,
                "requirements": requirements,
                "allowed_development": self._determine_allowed_development(zoning_plan, restrictions)
            }
        
        except Exception as e:
            logger.error(f"Feil ved sjekk av reguleringer: {str(e)}")
            return {
                "error": str(e),
                "municipality": "unknown",
                "zoning_plan": {},
                "building_regulations": {},
                "restrictions": [],
                "requirements": {}
            }
    
    def _get_municipality(self, address_info: Dict) -> str:
        """Henter kommunenavn fra adresseinformasjon"""
        # Prøv å hente fra adresseinformasjon
        components = address_info.get("components", {})
        if "municipality" in components:
            return components["municipality"]
        
        # Alternativ: Prøv å hente fra by
        city = components.get("city", "").lower()
        
        # Enkel mapping fra by til kommune
        city_to_municipality = {
            "oslo": "oslo",
            "bergen": "bergen",
            "trondheim": "trondheim",
            "stavanger": "stavanger",
            "drammen": "drammen",
            "tromsø": "tromsø",
            "kristiansand": "kristiansand"
        }
        
        return city_to_municipality.get(city, "unknown")
    
    async def _fetch_zoning_plan(self, municipality: str, property_data: Dict) -> Dict:
        """Henter reguleringsplan for eiendommen"""
        logger.info(f"Henter reguleringsplan for {municipality}")
        
        try:
            # Sjekk om vi har eiendomsID (gnr/bnr)
            property_id = property_data.get("address_info", {}).get("property_id")
            
            if not property_id:
                logger.warning("Mangler eiendomsID for reguleringsplan-oppslag")
                return {"error": "Mangler eiendomsID"}
            
            # Hent relevante API-endepunkter fra konfigurasjon
            api_url = self.config.get("municipality_data", {}).get(municipality, {}).get("api_url")
            
            if not api_url:
                logger.warning(f"Mangler API-URL for kommune: {municipality}")
                return self._generate_dummy_zoning_plan(property_id, municipality)
            
            # I en reell implementasjon ville dette gjøre et API-kall
            # For nå, returner dummy-data
            return self._generate_dummy_zoning_plan(property_id, municipality)
            
        except Exception as e:
            logger.error(f"Feil ved henting av reguleringsplan: {str(e)}")
            return {"error": str(e)}
    
    def _generate_dummy_zoning_plan(self, property_id: Dict, municipality: str) -> Dict:
        """Genererer dummy-data for reguleringsplan"""
        # Dette er kun for demo-formål
        zoning_types = {
            "oslo": "boligformål",
            "bergen": "boligformål",
            "drammen": "boligformål",
            "trondheim": "boligformål",
            "default": "boligformål"
        }
        
        utilization_rates = {
            "oslo": 24,
            "bergen": 30,
            "drammen": 35,
            "trondheim": 30,
            "default": 25
        }
        
        return {
            "plan_id": f"R{property_id.get('gnr', 0)}-{property_id.get('bnr', 0)}",
            "plan_name": f"Reguleringsplan for {municipality.capitalize()}",
            "zoning_type": zoning_types.get(municipality, zoning_types["default"]),
            "maximum_utilization": utilization_rates.get(municipality, utilization_rates["default"]),
            "maximum_height": 8.0,
            "maximum_floors": 2
        }
    
    async def _get_building_regulations(self, municipality: str) -> Dict:
        """Henter gjeldende byggtekniske forskrifter"""
        logger.info(f"Henter byggforskrifter for {municipality}")
        
        # For nå, returner nasjonal byggteknisk forskrift
        # I en reell implementasjon ville dette sjekke kommunespesifikke forskrifter
        
        return {
            "code_version": "TEK17",
            "municipality_specific_requirements": self._get_municipality_specific_requirements(municipality),
            "key_requirements": {
                "minimum_ceiling_height": 2.4,
                "minimum_room_area": 7.0,
                "minimum_window_area_ratio": 0.06,
                "accessibility_requirements": True,
                "fire_safety_requirements": True,
                "energy_requirements": {
                    "u_value_walls": 0.22,
                    "u_value_roof": 0.18,
                    "u_value_floor": 0.18,
                    "u_value_windows": 1.2
                }
            }
        }
    
    def _get_municipality_specific_requirements(self, municipality: str) -> Dict:
        """Henter kommunespesifikke krav"""
        # Dummy-data for demonstrasjon
        requirements = {
            "oslo": {
                "parking_requirement": 0.9,  # per unit
                "minimum_outdoor_area": 20,  # per unit
                "energy_requirement": "higher"
            },
            "bergen": {
                "parking_requirement": 1.0,
                "minimum_outdoor_area": 25,
                "energy_requirement": "standard"
            },
            "drammen": {
                "parking_requirement": 1.0,
                "minimum_outdoor_area": 30,
                "energy_requirement": "standard"
            },
            "default": {
                "parking_requirement": 1.0,
                "minimum_outdoor_area": 25,
                "energy_requirement": "standard"
            }
        }
        
        return requirements.get(municipality, requirements["default"])
    
    async def _check_restrictions(self, municipality: str, property_data: Dict) -> List[Dict]:
        """Sjekker eventuelle restriksjoner på eiendommen"""
        logger.info(f"Sjekker restriksjoner for {municipality}")
        
        restrictions = []
        
        # Sjekk om eiendommen er i Byantikvaren
        if municipality == "oslo":
            property_id = property_data.get("address_info", {}).get("property_id")
            if property_id:
                # Dummy-sjekk - i virkeligheten ville dette være et API-kall
                if property_id.get("gnr", 0) < 50:
                    restrictions.append({
                        "type": "cultural_heritage",
                        "description": "Eiendommen er registrert hos Byantikvaren",
                        "impact": "moderate",
                        "details": "Fasadeendringer må godkjennes av Byantikvaren"
                    })
        
        # Sjekk avstand til vei
        address_info = property_data.get("address_info", {})
        coordinates = address_info.get("coordinates", {})
        
        if coordinates:
            # Dummy-sjekk - i virkeligheten ville dette sjekke faktisk avstand
            restrictions.append({
                "type": "road_distance",
                "description": "Byggegrense mot vei",
                "impact": "moderate",
                "details": "Minimum 4 meter fra veikant"
            })
        
        return restrictions
    
    async def _get_requirements(self, municipality: str, property_data: Dict) -> Dict:
        """Henter krav for utvikling av eiendommen"""
        logger.info(f"Henter krav for {municipality}")
        
        # Basiskrav fra byggteknisk forskrift
        requirements = {
            "baseline": "TEK17",
            "renovation": {
                "permit_required": True,
                "documentation_required": ["tegninger", "søknad", "ansvarsrett"]
            },
            "conversion": {
                "permit_required": True,
                "documentation_required": ["tegninger", "søknad", "ansvarsrett", "brannprosjektering"]
            },
            "expansion": {
                "permit_required": True,
                "documentation_required": ["tegninger", "søknad", "ansvarsrett", "nabovarsel"]
            },
            "rental_unit": {
                "permit_required": True,
                "requirements": [
                    "egen inngang",
                    "brannskille",
                    "lydisolering",
                    "rømningsveier"
                ]
            }
        }
        
        # Legg til eventuell kommunespesifikk info
        if municipality in ["oslo", "bergen", "drammen"]:
            requirements["municipality_specific"] = {
                "municipality": municipality,
                "local_requirements": [
                    "Nabovarsel til alle naboer",
                    "Situasjonsplan med høydeangivelser",
                    "Fasadetegninger med høydeangivelser"
                ]
            }
        
        return requirements
    
    def _determine_allowed_development(self, zoning_plan: Dict, restrictions: List[Dict]) -> Dict:
        """Bestemmer hva som er tillatt utvikling basert på reguleringsplan og restriksjoner"""
        allowed = {
            "can_convert_basement": True,
            "can_convert_attic": True,
            "can_expand": True,
            "can_build_additional_unit": True,
            "limitations": []
        }
        
        # Sjekk restriksjoner
        for restriction in restrictions:
            if restriction["type"] == "cultural_heritage":
                allowed["can_expand"] = False
                allowed["limitations"].append("Kulturminnerestriksjon: Kan ikke gjøre vesentlige endringer på eksteriør")
            
            if restriction["type"] == "road_distance" and restriction["impact"] in ["high", "severe"]:
                allowed["can_expand"] = False
                allowed["limitations"].append("Veiavstand: Kan ikke bygge nærmere veien")
        
        # Sjekk utnyttelsesgrad
        max_util = zoning_plan.get("maximum_utilization", 0)
        if max_util < 25:
            allowed["can_expand"] = False
            allowed["limitations"].append(f"Lav maksimal utnyttelsesgrad: {max_util}%")
        
        return allowed
    
    async def _analyze_development_potential(self, 
                                     structure: Dict, 
                                     regulations: Dict,
                                     client_preferences: Optional[Dict] = None) -> Dict:
        """Analyserer utviklingspotensial basert på struktur og reguleringer"""
        logger.info("Analyserer utviklingspotensial")
        
        try:
            # Hent kundepreferanser eller bruk standardverdier
            if client_preferences is None:
                client_preferences = {
                    "priority": "rental_income",  # rental_income, value_increase, minimal_cost
                    "budget": "medium",           # low, medium, high
                    "timeframe": "medium"         # short, medium, long
                }
            
            # Sjekk utleiepotensial
            rental_options = await self._analyze_rental_potential(structure, regulations, client_preferences)
            
            # Sjekk utvidelsesmuligheter
            expansion_options = await self._find_expansion_options(structure, regulations, client_preferences)
            
            # Sjekk potensial for deling av eiendom
            division_options = await self._analyze_division_potential(structure, regulations, client_preferences)
            
            # Identifiser renoveringsbehov
            renovation_needs = await self._identify_renovation_needs(structure, client_preferences)
            
            # Beregn optimalt scenario basert på kundepreferanser
            optimal_scenario = self._determine_optimal_scenario(
                rental_options, 
                expansion_options, 
                division_options, 
                renovation_needs,
                client_preferences
            )
            
            return {
                "rental_units": rental_options,
                "expansion_possibilities": expansion_options,
                "property_division": division_options,
                "renovation_needs": renovation_needs,
                "optimal_scenario": optimal_scenario,
                "client_preferences": client_preferences
            }
            
        except Exception as e:
            logger.error(f"Feil ved analyse av utviklingspotensial: {str(e)}")
            return {
                "error": str(e),
                "rental_units": [],
                "expansion_possibilities": [],
                "property_division": {"feasible": False},
                "renovation_needs": []
            }
    
    async def _analyze_rental_potential(self, 
                                    structure: Dict, 
                                    regulations: Dict,
                                    client_preferences: Dict) -> List[Dict]:
        """Analyserer potensial for utleieenheter"""
        logger.info("Analyserer utleiepotensial")
        
        rental_options = []
        
        # Sjekk om regulering tillater utleie
        allowed_development = regulations.get("allowed_development", {})
        if not allowed_development.get("can_build_additional_unit", True):
            logger.info("Reguleringer tillater ikke ytterligere utleieenheter")
            return rental_options
        
        # Sjekk kjellerpotensial
        basement = structure.get("basement", {})
        if basement.get("exists", False) and allowed_development.get("can_convert_basement", True):
            basement_info = basement.get("info", {})
            basement_area = basement_info.get("area_m2", 0)
            
            if basement_area >= 25:  # Minimum for utleieenhet
                rental_options.append({
                    "type": "basement_conversion",
                    "description": "Konverter kjeller til utleieenhet",
                    "area_m2": basement_area,
                    "estimated_cost": self._estimate_conversion_cost(basement_info, "basement"),
                    "estimated_monthly_rent": self._estimate_monthly_rent(basement_area),
                    "roi_years": self._calculate_roi_years(
                        self._estimate_conversion_cost(basement_info, "basement"),
                        self._estimate_monthly_rent(basement_area)
                    ),
                    "requirements": [
                        "Separat inngang",
                        "Bad og kjøkken",
                        "Ventilasjon iht. TEK17",
                        "Brannskille mot hovedbolig"
                    ],
                    "challenges": self._identify_basement_challenges(basement_info)
                })
        
        # Sjekk loftspotensial
        attic = structure.get("attic", {})
        if attic.get("exists", False) and allowed_development.get("can_convert_attic", True):
            attic_info = attic.get("info", {})
            attic_area = attic_info.get("area_m2", 0)
            
            if attic_area >= 25:  # Minimum for utleieenhet
                rental_options.append({
                    "type": "attic_conversion",
                    "description": "Konverter loft til utleieenhet",
                    "area_m2": attic_area,
                    "estimated_cost": self._estimate_conversion_cost(attic_info, "attic"),
                    "estimated_monthly_rent": self._estimate_monthly_rent(attic_area),
                    "roi_years": self._calculate_roi_years(
                        self._estimate_conversion_cost(attic_info, "attic"),
                        self._estimate_monthly_rent(attic_area)
                    ),
                    "requirements": [
                        "Separat inngang",
                        "Bad og kjøkken",
                        "Tilstrekkelig takhøyde",
                        "Brannskille mot hovedbolig"
                    ],
                    "challenges": self._identify_attic_challenges(attic_info)
                })
        
        # Sjekk hovedetasje-deling
        floors = structure.get("floors", [])
        for floor in floors:
            if floor.get("floor_number", 0) == 1:  # Hovedetasje
                area = floor.get("area_m2", 0)
                
                if area >= 80:  # Stort nok til å dele
                    half_area = area / 2
                    rental_options.append({
                        "type": "main_floor_division",
                        "description": "Del hovedetasje i to separate enheter",
                        "area_m2": half_area,
                        "estimated_cost": self._estimate_conversion_cost({"area_m2": half_area}, "main_floor_division"),
                        "estimated_monthly_rent": self._estimate_monthly_rent(half_area),
                        "roi_years": self._calculate_roi_years(
                            self._estimate_conversion_cost({"area_m2": half_area}, "main_floor_division"),
                            self._estimate_monthly_rent(half_area)
                        ),
                        "requirements": [
                            "Separat inngang",
                            "Brannskille mellom enhetene",
                            "Egen strømmåler",
                            "Lydisolering"
                        ],
                        "challenges": [
                            "Krever omfattende ombygging",
                            "Potensielt høye kostnader"
                        ]
                    })
        
        # Sorter basert på kundepreferanser
        if client_preferences.get("priority") == "rental_income":
            rental_options.sort(key=lambda x: x.get("estimated_monthly_rent", 0), reverse=True)
        elif client_preferences.get("priority") == "minimal_cost":
            rental_options.sort(key=lambda x: x.get("estimated_cost", float('inf')))
        else:  # ROI som standard
            rental_options.sort(key=lambda x: x.get("roi_years", float('inf')))
        
        return rental_options
    
    def _identify_basement_challenges(self, basement_info: Dict) -> List[str]:
        """Identifiserer utfordringer med kjellerkonvertering"""
        challenges = []
        
        rooms = basement_info.get("rooms", [])
        
        # Sjekk for bad
        if not any(r.get("type", "").lower() in ["bathroom", "bad"] for r in rooms):
            challenges.append("Mangler bad - må installeres")
        
        # Sjekk for kjøkken
        if not any(r.get("type", "").lower() in ["kitchen", "kjøkken"] for r in rooms):
            challenges.append("Mangler kjøkken - må installeres")
        
        # Sjekk for inngang
        if "separate_entrance" not in basement_info or not basement_info["separate_entrance"]:
            challenges.append("Trenger separat inngang")
        
        # Sjekk for vinduer
        has_windows = any("window" in r or "vindu" in r.lower() for r in basement_info.keys())
        if not has_windows:
            challenges.append("Kan mangle tilstrekkelige vinduer/dagslys")
        
        # Sjekk for takhøyde
        ceiling_height = basement_info.get("ceiling_height", 0)
        if ceiling_height and ceiling_height < 2.2:
            challenges.append(f"Lav takhøyde ({ceiling_height} m) - minimum er 2.2m")
        
        # Legg til standard utfordringer
        challenges.extend([
            "Behov for fuktsikring",
            "Behov for tilstrekkelig isolasjon"
        ])
        
        return challenges
    
    def _identify_attic_challenges(self, attic_info: Dict) -> List[str]:
        """Identifiserer utfordringer med loftskonvertering"""
        challenges = []
        
        # Sjekk for takhøyde
        ceiling_height = attic_info.get("ceiling_height", 0)
        if ceiling_height and ceiling_height < 2.2:
            challenges.append(f"Lav takhøyde ({ceiling_height} m) - minimum er 2.2m")
        elif not ceiling_height:
            challenges.append("Ukjent takhøyde - må måles")
        
        # Sjekk for trapp
        if "has_stairs" not in attic_info or not attic_info["has_stairs"]:
            challenges.append("Kan trenge ny trappetilgang")
        
        # Sjekk for isolasjon
        challenges.append("Behov for isolasjon i tak og vegger")
        
        # Legg til standard utfordringer
        challenges.extend([
            "Kan være behov for takvinduer for dagslys",
            "Behov for brannsikring"
        ])
        
        return challenges
    
    def _estimate_conversion_cost(self, space_info: Dict, conversion_type: str) -> float:
        """Estimerer kostnad for konvertering til utleieenhet"""
        area = space_info.get("area_m2", 0)
        
        # Basiskostnad per kvadratmeter
        if conversion_type == "basement":
            base_cost_per_m2 = 15000  # NOK/m²
        elif conversion_type == "attic":
            base_cost_per_m2 = 18000  # NOK/m²
        elif conversion_type == "main_floor_division":
            base_cost_per_m2 = 12000  # NOK/m²
        else:
            base_cost_per_m2 = 15000  # Standard
        
        # Basiskostnad
        base_cost = area * base_cost_per_m2
        
        # Justeringer basert på romtype
        rooms = space_info.get("rooms", [])
        
        # Legg til kostnad for bad hvis det mangler
        if not any(r.get("type", "").lower() in ["bathroom", "bad"] for r in rooms):
            base_cost += 150000  # Kostnad for nytt bad
        
        # Legg til kostnad for kjøkken hvis det mangler
        if not any(r.get("type", "").lower() in ["kitchen", "kjøkken"] for r in rooms):
            base_cost += 100000  # Kostnad for nytt kjøkken
        
        # Legg til kostnad for separat inngang hvis det mangler
        if "separate_entrance" not in space_info or not space_info["separate_entrance"]:
            base_cost += 50000  # Kostnad for ny inngang
        
        # Legg til 20% for uforutsette kostnader
        total_cost = base_cost * 1.2
        
        return total_cost
    
    def _estimate_monthly_rent(self, area_m2: float) -> float:
        """Estimerer månedlig leieinntekt basert på areal"""
        # Enkel modell: 200 NOK per m² per måned
        # Dette ville i virkeligheten avhenge av beliggenhet, standard, etc.
        return area_m2 * 200
    
    def _calculate_roi_years(self, cost: float, monthly_rent: float) -> float:
        """Beregner tilbakebetalingstid i år"""
        if monthly_rent <= 0:
            return float('inf')
            
        annual_rent = monthly_rent * 12
        return cost / annual_rent
    
    async def _find_expansion_options(self, 
                                   structure: Dict, 
                                   regulations: Dict,
                                   client_preferences: Dict) -> List[Dict]:
        """Identifiserer muligheter for utvidelse"""
        logger.info("Analyserer utvidelsesmuligheter")
        
        expansion_options = []
        
        # Sjekk om regulering tillater utvidelse
        allowed_development = regulations.get("allowed_development", {})
        if not allowed_development.get("can_expand", True):
            logger.info("Reguleringer tillater ikke utvidelse")
            return expansion_options
        
        # Hent tomtestørrelse hvis tilgjengelig
        measurements = structure.get("measurements", {})
        total_area = measurements.get("total_area_m2", 0)
        
        # Hvis tomtestørrelse ikke er tilgjengelig, kan vi ikke beregne utnyttelsesgrad
        property_area = 500  # Standard antagelse for tomtestørrelse
        
        # Hent tillatt utnyttelsesgrad
        zoning_plan = regulations.get("zoning_plan", {})
        max_utilization = zoning_plan.get("maximum_utilization", 25)  # Standard 25% hvis ikke spesifisert
        
        # Hent maksimal byggehøyde
        max_height = zoning_plan.get("maximum_height", 8.0)  # Standard 8m hvis ikke spesifisert
        max_floors = zoning_plan.get("maximum_floors", 2)    # Standard 2 etasjer hvis ikke spesifisert
        
        # Beregn gjenværende utbyggingspotensial
        max_building_area = property_area * (max_utilization / 100)
        remaining_building_area = max(0, max_building_area - total_area)
        
        # Sjekk påbygg (ekstra etasje)
        floor_count = len(structure.get("floors", []))
        if floor_count < max_floors:
            # Finn arealet for øverste etasje
            first_floor_area = 0
            for floor in structure.get("floors", []):
                if floor.get("floor_number", 0) == 1:
                    first_floor_area = floor.get("area_m2", 0)
                    break
            
            if first_floor_area > 0:
                expansion_options.append({
                    "type": "additional_floor",
                    "description": "Bygge på en ekstra etasje",
                    "area_m2": first_floor_area,
                    "estimated_cost": first_floor_area * 25000,  # 25,000 NOK per m²
                    "value_increase": first_floor_area * 45000,  # 45,000 NOK per m² i verdiøkning
                    "roi_percentage": (first_floor_area * 45000) / (first_floor_area * 25000) * 100,
                    "requirements": [
                        "Byggesøknad",
                        "Konstruksjonsvurdering",
                        "Arkitekttegninger"
                    ],
                    "challenges": [
                        "Kostbart prosjekt",
                        "Behov for midlertidig flytting under byggeperiode",
                        "Strukturell vurdering nødvendig"
                    ]
                })
        
        # Sjekk tilbygg (utvidelse til siden)
        if remaining_building_area >= 20:  # Minimum 20m² for at det skal være verdt det
            expansion_options.append({
                "type": "extension",
                "description": "Bygge tilbygg",
                "area_m2": min(remaining_building_area, 40),  # Begrens til maks 40m² for et realistisk tilbygg
                "estimated_cost": min(remaining_building_area, 40) * 30000,  # 30,000 NOK per m²
                "value_increase": min(remaining_building_area, 40) * 40000,  # 40,000 NOK per m² i verdiøkning
                "roi_percentage": 40000 / 30000 * 100,  # 33.3% ROI
                "requirements": [
                    "Byggesøknad",
                    "Nabovarsel",
                    "Arkitekttegninger",
                    "Situasjonsplan"
                ],
                "challenges": [
                    "Avhengig av tomtens utforming",
                    "Kan påvirke utomhusareal",
                    "Potensielle naboinnsigelser"
                ]
            })
        
        # Sjekk takterrasse
        attic = structure.get("attic", {})
        if attic.get("exists", False):
            expansion_options.append({
                "type": "roof_terrace",
                "description": "Bygge takterrasse",
                "area_m2": 20,  # Anslått størrelse
                "estimated_cost": 300000,  # Fast pris
                "value_increase": 500000,  # Anslått verdiøkning
                "roi_percentage": 500000 / 300000 * 100,  # 66.7% ROI
                "requirements": [
                    "Byggesøknad",
                    "Konstruksjonsvurdering",
                    "Arkitekttegninger"
                ],
                "challenges": [
                    "Værutsatt",
                    "Kan ha begrensninger i reguleringsplan"
                ]
            })
        
        # Sorter basert på kundepreferanser
        if client_preferences.get("priority") == "value_increase":
            expansion_options.sort(key=lambda x: x.get("value_increase", 0), reverse=True)
        elif client_preferences.get("priority") == "minimal_cost":
            expansion_options.sort(key=lambda x: x.get("estimated_cost", float('inf')))
        else:  # ROI som standard
            expansion_options.sort(key=lambda x: x.get("roi_percentage", 0), reverse=True)
        
        return expansion_options
    
    async def _analyze_division_potential(self, 
                                       structure: Dict, 
                                       regulations: Dict,
                                       client_preferences: Dict) -> Dict:
        """Analyserer potensial for deling av eiendom"""
        logger.info("Analyserer potensial for tomtedeling")
        
        # Standard resultat hvis deling ikke er mulig
        division_result = {
            "feasible": False,
            "reason": "Ukjent tomtestørrelse"
        }
        
        # Sjekk etter nødvendig informasjon
        if not structure or "measurements" not in structure:
            return division_result
        
        # Sjekk tomtestørrelse
        # Dette ville normalt komme fra matrikkeldata eller lignende
        property_area = 500  # Standard antagelse for tomtestørrelse
        
        # Sjekk reguleringsplan
        zoning_plan = regulations.get("zoning_plan", {})
        min_plot_size = zoning_plan.get("minimum_plot_size", 300)  # Standard 300m² hvis ikke spesifisert
        
        # Sjekk om tomten er stor nok for deling
        if property_area < (min_plot_size * 2):
            return {
                "feasible": False,
                "reason": f"Tomten er for liten for deling. Minimum krav er {min_plot_size*2}m², mens tomten er {property_area}m²"
            }
        
        # Sjekk bygningens plassering på tomten
        # Dette ville normalt komme fra kartdata eller lignende
        building_position_ok = True  # Antagelse for nå
        
        if not building_position_ok:
            return {
                "feasible": False,
                "reason": "Bygningen er plassert slik at deling ikke er mulig"
            }
        
        # Beregn ny tomtestørrelse og verdier
        new_plot_size = property_area / 2
        current_property_value = 5000000  # Antagelse for nå
        divided_property_value = new_plot_size * 10000  # 10,000 NOK per m²
        new_plot_value = new_plot_size * 8000  # 8,000 NOK per m²
        
        # Total verdi etter deling
        total_value_after_division = divided_property_value + new_plot_value
        value_increase = total_value_after_division - current_property_value
        
        # Kostnader ved deling
        division_costs = {
            "application_fee": 30000,     # Søknadsgebyr
            "surveying": 25000,           # Oppmåling
            "legal_fees": 50000,          # Advokatkostnader
            "infrastructure": 200000,     # Infrastruktur til ny tomt
            "total": 305000               # Total kostnad
        }
        
        # ROI-beregning
        roi_percentage = (value_increase / division_costs["total"]) * 100
        
        return {
            "feasible": True,
            "current_property_area": property_area,
            "new_plot_size": new_plot_size,
            "value_estimate": {
                "current_property_value": current_property_value,
                "divided_property_value": divided_property_value,
                "new_plot_value": new_plot_value,
                "total_value_after_division": total_value_after_division,
                "value_increase": value_increase
            },
            "division_costs": division_costs,
            "roi_percentage": roi_percentage,
            "requirements": [
                "Søknad om deling",
                "Godkjent reguleringsplan",
                "Kartforretning",
                "Infrastruktur til ny tomt"
            ],
            "challenges": [
                "Tidkrevende prosess",
                "Usikker godkjenning",
                "Kostnader ved infrastruktur"
            ]
        }
    
    async def _identify_renovation_needs(self, 
                                     structure: Dict,
                                     client_preferences: Dict) -> List[Dict]:
        """Identifiserer renoveringsbehov"""
        logger.info("Analyserer renoveringsbehov")
        
        renovation_needs = []
        
        # Hent bygningens alder
        construction_type = structure.get("construction_type", {})
        construction_year = construction_type.get("construction_year", 1970)  # Standard antagelse hvis ikke spesifisert
        
        # Beregn bygningens alder
        current_year = datetime.now().year
        building_age = current_year - construction_year
        
        # Legg til standardrenoveringer basert på alder
        if building_age > 30:
            renovation_needs.append({
                "type": "electrical",
                "description": "Oppgradering av elektrisk anlegg",
                "priority": "high",
                "estimated_cost": 100000,
                "value_increase": 150000,
                "roi_percentage": 50.0,
                "reason": "Eldre elektrisk anlegg kan være utdatert og utgjøre sikkerhetsrisiko"
            })
        
        if building_age > 25:
            renovation_needs.append({
                "type": "bathroom",
                "description": "Renovering av bad",
                "priority": "high",
                "estimated_cost": 150000,
                "value_increase": 250000,
                "roi_percentage": 66.7,
                "reason": "Bad har typisk levetid på 25-30 år før fuktproblemer kan oppstå"
            })
        
        if building_age > 20:
            renovation_needs.append({
                "type": "kitchen",
                "description": "Oppgradering av kjøkken",
                "priority": "medium",
                "estimated_cost": 120000,
                "value_increase": 200000,
                "roi_percentage": 66.7,
                "reason": "Moderne kjøkken kan betydelig øke boligens verdi og attraktivitet"
            })
        
        if building_age > 40:
            renovation_needs.append({
                "type": "windows",
                "description": "Utskifting av vinduer",
                "priority": "medium",
                "estimated_cost": 100000,
                "value_increase": 150000,
                "roi_percentage": 50.0,
                "reason": "Eldre vinduer er ofte lite energieffektive og kan ha kort gjenværende levetid"
            })
        
        if building_age > 50:
            renovation_needs.append({
                "type": "roof",
                "description": "Utskifting av tak",
                "priority": "high",
                "estimated_cost": 200000,
                "value_increase": 250000,
                "roi_percentage": 25.0,
                "reason": "Tak har begrenset levetid og er kritisk for bygningens tilstand"
            })
        
        # Legg til energieffektiviseringstiltak
        renovation_needs.append({
            "type": "energy_efficiency",
            "description": "Energieffektivisering (isolasjon, varmepumpe)",
            "priority": "medium",
            "estimated_cost": 150000,
            "value_increase": 200000,
            "roi_percentage": 33.3,
            "reason": "Øker komfort og reduserer energikostnader",
            "enova_support": 50000  # Anslått støtte fra Enova
        })
        
        # Sorter basert på kundepreferanser
        if client_preferences.get("priority") == "value_increase":
            renovation_needs.sort(key=lambda x: x.get("value_increase", 0), reverse=True)
        elif client_preferences.get("priority") == "minimal_cost":
            renovation_needs.sort(key=lambda x: x.get("estimated_cost", float('inf')))
        else:  # ROI som standard
            renovation_needs.sort(key=lambda x: x.get("roi_percentage", 0), reverse=True)
        
        return renovation_needs
    
    def _determine_optimal_scenario(self,
                                  rental_options: List[Dict],
                                  expansion_options: List[Dict],
                                  division_options: Dict,
                                  renovation_needs: List[Dict],
                                  client_preferences: Dict) -> Dict:
        """Bestemmer det optimale scenariet basert på kundepreferanser"""
        logger.info("Bestemmer optimalt scenario")
        
        # Hent kundepreferanser
        priority = client_preferences.get("priority", "rental_income")
        budget = client_preferences.get("budget", "medium")
        timeframe = client_preferences.get("timeframe", "medium")
        
        # Definer budsjettgrenser
        budget_limits = {
            "low": 300000,      # 300k NOK
            "medium": 1000000,  # 1M NOK
            "high": 3000000     # 3M NOK
        }
        budget_limit = budget_limits.get(budget, 1000000)
        
        # Definer tidsrammer
        timeframe_limits = {
            "short": 6,    # 6 måneder
            "medium": 12,  # 12 måneder
            "long": 24     # 24 måneder
        }
        timeframe_months = timeframe_limits.get(timeframe, 12)
        
        # Definer ulike scenarioer
        scenarios = []
        
        # Scenario 1: Utleie
        if rental_options:
            top_rental = rental_options[0]
            rental_cost = top_rental.get("estimated_cost", 0)
            rental_income = top_rental.get("estimated_monthly_rent", 0) * 12  # Årlig
            rental_roi = top_rental.get("roi_years", float('inf'))
            
            if rental_cost <= budget_limit:
                scenarios.append({
                    "name": "Utleie",
                    "description": top_rental.get("description", "Konvertering til utleieenhet"),
                    "primary_focus": "rental_income",
                    "actions": [top_rental],
                    "total_cost": rental_cost,
                    "annual_income": rental_income,
                    "roi_years": rental_roi,
                    "implementation_time_months": 6,
                    "suitable_for_preferences": priority == "rental_income"
                })
        
        # Scenario 2: Utvidelse
        if expansion_options:
            top_expansion = expansion_options[0]
            expansion_cost = top_expansion.get("estimated_cost", 0)
            value_increase = top_expansion.get("value_increase", 0)
            expansion_roi = top_expansion.get("roi_percentage", 0) / 100  # Konvertere til desimal
            
            if expansion_cost <= budget_limit:
                scenarios.append({
                    "name": "Utvidelse",
                    "description": top_expansion.get("description", "Utvidelse av boligarealet"),
                    "primary_focus": "value_increase",
                    "actions": [top_expansion],
                    "total_cost": expansion_cost,
                    "value_increase": value_increase,
                    "roi_percentage": expansion_roi * 100,  # Tilbake til prosent
                    "implementation_time_months": 12,
                    "suitable_for_preferences": priority == "value_increase"
                })
        
        # Scenario 3: Tomtedeling
        if division_options.get("feasible", False):
            division_cost = division_options.get("division_costs", {}).get("total", 0)
            value_increase = division_options.get("value_estimate", {}).get("value_increase", 0)
            division_roi = division_options.get("roi_percentage", 0) / 100  # Konvertere til desimal
            
            if division_cost <= budget_limit:
                scenarios.append({
                    "name": "Tomtedeling",
                    "description": "Dele tomten for å selge deler av eiendommen",
                    "primary_focus": "value_increase",
                    "actions": [division_options],
                    "total_cost": division_cost,
                    "value_increase": value_increase,
                    "roi_percentage": division_roi * 100,  # Tilbake til prosent
                    "implementation_time_months": 18,
                    "suitable_for_preferences": priority == "value_increase" and timeframe == "long"
                })
        
        # Scenario 4: Renovering
        if renovation_needs:
            high_priority_renovations = [r for r in renovation_needs if r.get("priority") == "high"]
            
            if high_priority_renovations:
                renovation_cost = sum(r.get("estimated_cost", 0) for r in high_priority_renovations)
                renovation_value_increase = sum(r.get("value_increase", 0) for r in high_priority_renovations)
                renovation_roi = renovation_value_increase / renovation_cost if renovation_cost > 0 else 0
                
                if renovation_cost <= budget_limit:
                    scenarios.append({
                        "name": "Renovering",
                        "description": "Utføre høyprioriterte renovasjoner",
                        "primary_focus": "maintenance",
                        "actions": high_priority_renovations,
                        "total_cost": renovation_cost,
                        "value_increase": renovation_value_increase,
                        "roi_percentage": renovation_roi * 100,
                        "implementation_time_months": 3,
                        "suitable_for_preferences": priority == "minimal_cost" or budget == "low"
                    })
        
        # Scenario 5: Kombinert (hvis budsjett tillater)
        combined_options = []
        combined_cost = 0
        combined_income = 0
        combined_value_increase = 0
        
        # Legg til høyprioriterte renoveringer
        for renovation in renovation_needs:
            if renovation.get("priority") == "high" and combined_cost + renovation.get("estimated_cost", 0) <= budget_limit:
                combined_options.append(renovation)
                combined_cost += renovation.get("estimated_cost", 0)
                combined_value_increase += renovation.get("value_increase", 0)
        
        # Legg til beste utleiemulighet
        if rental_options and combined_cost + rental_options[0].get("estimated_cost", 0) <= budget_limit:
            combined_options.append(rental_options[0])
            combined_cost += rental_options[0].get("estimated_cost", 0)
            combined_income += rental_options[0].get("estimated_monthly_rent", 0) * 12
            combined_value_increase += rental_options[0].get("estimated_monthly_rent", 0) * 12 * 10  # 10x årlig inntekt som verdiøkning
        
        if combined_options and len(combined_options) > 1:
            combined_roi = (combined_income + (combined_value_increase * 0.1)) / combined_cost  # Vektet ROI
            
            scenarios.append({
                "name": "Kombinert tilnærming",
                "description": "Kombinere nødvendige renoveringer med inntektsmuligheter",
                "primary_focus": "balanced",
                "actions": combined_options,
                "total_cost": combined_cost,
                "annual_income": combined_income,
                "value_increase": combined_value_increase,
                "roi_percentage": combined_roi * 100,
                "implementation_time_months": 9,
                "suitable_for_preferences": True  # En balansert tilnærming passer ofte for de fleste
            })
        
        # Velg optimalt scenario
        if not scenarios:
            return {
                "name": "Ingen tiltak",
                "description": "Ingen egnede scenarier funnet innenfor gitte preferanser",
                "reason": "Budsjett eller tidsramme for restriktiv, eller mangel på muligheter"
            }
        
        # Vurder ulike faktorer for å finne det optimale scenariet
        for scenario in scenarios:
            scenario["score"] = 0
            
            # Høy score hvis det passer kundepreferanser
            if scenario.get("suitable_for_preferences", False):
                scenario["score"] += 30
            
            # Score basert på ROI
            if "roi_percentage" in scenario:
                scenario["score"] += min(scenario["roi_percentage"] / 2, 30)  # Maks 30 poeng for ROI
            elif "roi_years" in scenario:
                roi_years = scenario["roi_years"]
                if roi_years > 0:
                    scenario["score"] += min(30 / roi_years, 30)  # Maks 30 poeng for ROI
            
            # Score basert på implementeringstid
            impl_time = scenario.get("implementation_time_months", 12)
            time_score = max(0, 20 - (abs(impl_time - timeframe_months) / 2))
            scenario["score"] += time_score
            
            # Score basert på totalkostnad i forhold til budsjett
            cost_ratio = scenario.get("total_cost", 0) / budget_limit
            if cost_ratio <= 1.0:
                scenario["score"] += 20 * (1 - cost_ratio/2)  # Høyere score for lavere kostnadsandel
        
        # Sorter scenarioer etter score
        scenarios.sort(key=lambda x: x.get("score", 0), reverse=True)
        optimal = scenarios[0]
        
        # Fjern hjelpefelt
        if "score" in optimal:
            del optimal["score"]
        
        # Legg til alternative scenarier
        alternatives = [s for s in scenarios[1:3]] if len(scenarios) > 1 else []
        for alt in alternatives:
            if "score" in alt:
                del alt["score"]
        
        optimal["alternatives"] = alternatives
        
        return optimal
    
    async def _generate_3d_model(self, property_data: Dict, structure_analysis: Dict) -> Dict:
        """Genererer detaljert 3D-modell med NVIDIA Omniverse"""
        logger.info("Genererer 3D-modell")
        
        try:
            # Sjekk om vi har tilstrekkelig data
            if not property_data or not structure_analysis:
                logger.warning("Utilstrekkelig data for 3D-modellgenerering")
                return {
                    "error": "Utilstrekkelig data for 3D-modellgenerering",
                    "model_url": None
                }
            
            # Hent plantegningsdata hvis tilgjengelig
            floor_plan_data = property_data.get("floor_plan_analysis", {})
            
            # Generer 3D-modell basert på tilgjengelige data
            model_url = await self._create_3d_model(property_data, structure_analysis)
            
            # Generer plantegninger
            floor_plans = self._generate_floor_plans(structure_analysis)
            
            # Generer fasadetegninger
            facade_drawings = self._generate_facade_drawings(structure_analysis)
            
            # Generer situasjonsplan
            site_plan = self._generate_site_plan(property_data)
            
            return {
                "model_url": model_url,
                "floor_plans": floor_plans,
                "facade_drawings": facade_drawings,
                "site_plan": site_plan,
                "viewer_options": {
                    "show_measurements": True,
                    "enable_cutaway_view": True,
                    "enable_daylight_simulation": True
                }
            }
            
        except Exception as e:
            logger.error(f"Feil ved generering av 3D-modell: {str(e)}")
            return {
                "error": str(e),
                "model_url": None
            }
    
    async def _create_3d_model(self, property_data: Dict, structure_analysis: Dict) -> str:
        """Genererer 3D-modell med NVIDIA Omniverse"""
        # I en reell implementasjon ville dette integrere med Omniverse
        # For nå, returner en dummy-URL
        
        return "https://example.com/models/property_3d_model.glb"
    
    def _generate_floor_plans(self, structure_analysis: Dict) -> List[Dict]:
        """Genererer plantegninger for hver etasje"""
        floor_plans = []
        
        # Generer plantegning for hver etasje
        for floor in structure_analysis.get("floors", []):
            floor_plans.append({
                "floor_number": floor.get("floor_number", 0),
                "url": f"https://example.com/floor_plans/floor_{floor.get('floor_number', 0)}.png",
                "area_m2": floor.get("area_m2", 0),
                "room_count": floor.get("room_count", 0)
            })
        
        # Legg til kjeller hvis den eksisterer
        basement = structure_analysis.get("basement", {})
        if basement.get("exists", False):
            floor_plans.append({
                "floor_number": -1,
                "url": "https://example.com/floor_plans/basement.png",
                "area_m2": basement.get("info", {}).get("area_m2", 0),
                "room_count": basement.get("info", {}).get("room_count", 0)
            })
        
        # Legg til loft hvis det eksisterer
        attic = structure_analysis.get("attic", {})
        if attic.get("exists", False):
            floor_plans.append({
                "floor_number": len(structure_analysis.get("floors", [])) + 1,
                "url": "https://example.com/floor_plans/attic.png",
                "area_m2": attic.get("info", {}).get("area_m2", 0),
                "room_count": attic.get("info", {}).get("room_count", 0)
            })
        
        return floor_plans
    
    def _generate_facade_drawings(self, structure_analysis: Dict) -> Dict:
        """Genererer fasadetegninger"""
        return {
            "north": "https://example.com/facades/north.png",
            "south": "https://example.com/facades/south.png",
            "east": "https://example.com/facades/east.png",
            "west": "https://example.com/facades/west.png"
        }
    
    def _generate_site_plan(self, property_data: Dict) -> str:
        """Genererer situasjonsplan"""
        return "https://example.com/site_plan.png"
    
    async def _perform_energy_analysis(self, property_data: Dict) -> Dict:
        """Utfører energianalyse og identifiserer forbedringspotensial"""
        logger.info("Utfører energianalyse")
        
        try:
            # Bruk EnergyAnalyzer-modulen hvis tilgjengelig
            if "energy_analyzer" in self.processors:
                energy_analyzer = self.processors["energy_analyzer"]
                
                # Samle bygningsdata
                building_data = self._prepare_building_data_for_energy_analysis(property_data)
                
                # Samle konstruksjonsdetaljer
                construction_details = self._prepare_construction_details(property_data)
                
                # Kjør energianalyse
                energy_analysis = await energy_analyzer.analyze_energy_performance(
                    building_data,
                    construction_details
                )
                
                return energy_analysis
            else:
                logger.warning("EnergyAnalyzer ikke tilgjengelig")
                return self._simple_energy_analysis(property_data)
                
        except Exception as e:
            logger.error(f"Feil ved energianalyse: {str(e)}")
            return {
                "error": str(e),
                "current_rating": "Unknown",
                "potential_rating": "Unknown"
            }
    
    def _prepare_building_data_for_energy_analysis(self, property_data: Dict) -> Dict:
        """Forbereder bygningsdata for energianalyse"""
        # Hent relevant data
        structure_analysis = property_data.get("structure_analysis", {})
        measurements = structure_analysis.get("measurements", {})
        construction_type = structure_analysis.get("construction_type", {})
        
        # Beregn oppvarmet areal
        heated_area = measurements.get("total_area_m2", 0)
        
        # Beregn volum (antar standard takhøyde hvis ikke spesifisert)
        ceiling_height = 2.4  # Standard
        volume = heated_area * ceiling_height
        
        # Hent byggeår
        construction_year = construction_type.get("construction_year", 1970)
        
        return {
            "heated_area": heated_area,
            "volume": volume,
            "construction_year": construction_year,
            "building_type": "residential",
            "number_of_floors": len(structure_analysis.get("floors", [])),
            "location": property_data.get("address_info", {}).get("components", {}).get("city", "Oslo")
        }
    
    def _prepare_construction_details(self, property_data: Dict) -> Dict:
        """Forbereder konstruksjonsdetaljer for energianalyse"""
        structure_analysis = property_data.get("structure_analysis", {})
        construction_type = structure_analysis.get("construction_type", {})
        
        # Hent primært byggemateriale
        primary_material = construction_type.get("primary_material", "unknown")
        
        # Bestem konstruksjonsdetaljer basert på byggeår og materiale
        construction_year = construction_type.get("construction_year", 1970)
        
        # Definer U-verdier basert på byggeår
        if construction_year < 1970:
            u_values = {
                "walls": 0.8,
                "roof": 0.6,
                "floor": 0.6,
                "windows": 2.6
            }
        elif construction_year < 1985:
            u_values = {
                "walls": 0.4,
                "roof": 0.3,
                "floor": 0.3,
                "windows": 2.1
            }
        elif construction_year < 2000:
            u_values = {
                "walls": 0.3,
                "roof": 0.2,
                "floor": 0.25,
                "windows": 1.6
            }
        elif construction_year < 2010:
            u_values = {
                "walls": 0.22,
                "roof": 0.18,
                "floor": 0.18,
                "windows": 1.2
            }
        else:
            u_values = {
                "walls": 0.18,
                "roof": 0.13,
                "floor": 0.15,
                "windows": 0.8
            }
        
        # Juster basert på materiale
        if primary_material == "wood":
            u_values["walls"] *= 0.9  # Bedre for tre
        elif primary_material == "concrete":
            u_values["walls"] *= 1.1  # Verre for betong
        
        return {
            "walls": {
                "material": primary_material,
                "u_value": u_values["walls"],
                "area": structure_analysis.get("measurements", {}).get("estimated_facade_area_m2", 100)
            },
            "roof": {
                "material": "composite",
                "u_value": u_values["roof"],
                "area": structure_analysis.get("measurements", {}).get("total_area_m2", 100) / len(structure_analysis.get("floors", [1]))
            },
            "floor": {
                "material": "concrete",
                "u_value": u_values["floor"],
                "area": structure_analysis.get("measurements", {}).get("total_area_m2", 100) / len(structure_analysis.get("floors", [1]))
            },
            "windows": {
                "type": "double_glazed" if construction_year >= 1985 else "single_glazed",
                "u_value": u_values["windows"],
                "area": structure_analysis.get("measurements", {}).get("total_area_m2", 100) * 0.15  # Antar 15% av gulvareal
            }
        }
    
    def _simple_energy_analysis(self, property_data: Dict) -> Dict:
        """Enkel energianalyse når EnergyAnalyzer ikke er tilgjengelig"""
        # Hent byggeår
        structure_analysis = property_data.get("structure_analysis", {})
        construction_type = structure_analysis.get("construction_type", {})
        construction_year = construction_type.get("construction_year", 1970)
        
        # Bestem energimerke basert på byggeår
        current_rating = None
        if construction_year < 1970:
            current_rating = "F"
        elif construction_year < 1985:
            current_rating = "E"
        elif construction_year < 2000:
            current_rating = "D"
        elif construction_year < 2010:
            current_rating = "C"
        else:
            current_rating = "B"
        
        # Estimer potensielt energimerke etter oppgradering
        potential_rating = chr(ord(current_rating) - 2) if current_rating > "C" else "B"
        
        # Estimer energibehov
        heated_area = structure_analysis.get("measurements", {}).get("total_area_m2", 100)
        current_energy_demand = None
        
        if current_rating == "F":
            current_energy_demand = heated_area * 280  # kWh/år
        elif current_rating == "E":
            current_energy_demand = heated_area * 220  # kWh/år
        elif current_rating == "D":
            current_energy_demand = heated_area * 170  # kWh/år
        elif current_rating == "C":
            current_energy_demand = heated_area * 120  # kWh/år
        elif current_rating == "B":
            current_energy_demand = heated_area * 90   # kWh/år
        else:
            current_energy_demand = heated_area * 65   # kWh/år
        
        # Estimer potensielt energibehov etter forbedringer
        potential_energy_demand = heated_area * 100  # kWh/år
        energy_saving = max(0, current_energy_demand - potential_energy_demand)
        
        # Estimer kostnadsbesparelse (antatt 1.2 NOK/kWh)
        annual_cost_saving = energy_saving * 1.2
        
        # Identifiser forbedringstiltak
        improvement_measures = []
        
        if construction_year < 2000:
            improvement_measures.append({
                "type": "insulation",
                "description": "Etterisolering av yttervegg",
                "cost": heated_area * 700,  # 700 NOK/m² BRA
                "energy_saving": heated_area * 40,  # kWh/år
                "roi_years": heated_area * 700 / (heated_area * 40 * 1.2)
            })
        
        if construction_year < 2010:
            improvement_measures.append({
                "type": "windows",
                "description": "Utskifting til energieffektive vinduer",
                "cost": heated_area * 0.15 * 5000,  # 5000 NOK/m² vindusareal (15% av BRA)
                "energy_saving": heated_area * 30,  # kWh/år
                "roi_years": (heated_area * 0.15 * 5000) / (heated_area * 30 * 1.2)
            })
        
        improvement_measures.append({
            "type": "heat_pump",
            "description": "Installering av luft-til-luft varmepumpe",
            "cost": 25000,  # NOK
            "energy_saving": heated_area * 50,  # kWh/år
            "roi_years": 25000 / (heated_area * 50 * 1.2)
        })
        
        # Sorter tiltak etter tilbakebetalingstid
        improvement_measures.sort(key=lambda x: x.get("roi_years", float('inf')))
        
        # Beregn Enova-støtte
        enova_support = self._calculate_enova_support(improvement_measures)
        
        return {
            "current_rating": current_rating,
            "potential_rating": potential_rating,
            "current_energy_demand": current_energy_demand,
            "potential_energy_demand": potential_energy_demand,
            "energy_saving": energy_saving,
            "annual_cost_saving": annual_cost_saving,
            "improvement_measures": improvement_measures,
            "enova_support": enova_support
        }
    
    def _calculate_enova_support(self, improvement_measures: List[Dict]) -> Dict:
        """Beregner støtte fra Enova for energitiltak"""
        support_rates = self.config.get("enova", {}).get("support_rates", {})
        
        total_support = 0
        supported_measures = []
        
        for measure in improvement_measures:
            measure_type = measure.get("type")
            support_amount = 0
            
            if measure_type == "heat_pump":
                support_amount = support_rates.get("heat_pump", 10000)
            elif measure_type == "insulation":
                # Basert på areal (antatt 20% av BRA)
                area = measure.get("cost", 0) / 700  # 700 NOK/m²
                insulation_area = area * 0.2
                support_amount = insulation_area * support_rates.get("insulation", 500)
            elif measure_type == "windows":
                # Basert på antall vinduer (antatt 1 vindu per 5 m² fasade)
                window_area = measure.get("cost", 0) / 5000  # 5000 NOK/m²
                window_count = window_area / 1.2  # Antatt 1.2 m² per vindu
                support_amount = window_count * support_rates.get("windows", 1000)
            
            if support_amount > 0:
                supported_measures.append({
                    "type": measure_type,
                    "description": measure.get("description", ""),
                    "support_amount": support_amount
                })
                total_support += support_amount
        
        return {
            "total_support": total_support,
            "supported_measures": supported_measures,
            "application_url": "https://www.enova.no/privat/alle-energitiltak/",
            "requirements": [
                "Tiltakene må være gjennomført av fagfolk",
                "Dokumentasjon på utført arbeid må vedlegges søknaden",
                "Søknad må sendes inn innen 60 dager etter ferdigstillelse"
            ]
        }
    
    async def _estimate_costs(self, development_potential: Dict) -> Dict:
        """Estimerer kostnader for ulike utviklingsmuligheter"""
        logger.info("Estimerer kostnader")
        
        try:
            # Hent all relevant data
            rental_options = development_potential.get("rental_units", [])
            expansion_options = development_potential.get("expansion_possibilities", [])
            renovation_needs = development_potential.get("renovation_needs", [])
            division_options = development_potential.get("property_division", {})
            
            # Beregn renoveringskostnader
            renovation_costs = self._calculate_renovation_costs(renovation_needs)
            
            # Beregn konverteringskostnader
            conversion_costs = self._calculate_conversion_costs(rental_options)
            
            # Beregn utvidelseskostnader
            expansion_costs = self._calculate_expansion_costs(expansion_options)
            
            # Estimer potensielle inntekter
            potential_revenue = self._estimate_revenue(
                rental_options,
                expansion_options,
                division_options
            )
            
            # Beregn ROI
            roi_analysis = self._calculate_roi_metrics(
                renovation_costs, 
                conversion_costs, 
                expansion_costs, 
                potential_revenue
            )
            
            return {
                "renovation_costs": renovation_costs,
                "conversion_costs": conversion_costs,
                "expansion_costs": expansion_costs,
                "potential_revenue": potential_revenue,
                "roi_analysis": roi_analysis,
                "funding_options": self._suggest_funding_options(
                    renovation_costs, 
                    conversion_costs, 
                    expansion_costs
                )
            }
            
        except Exception as e:
            logger.error(f"Feil ved kostnadsestimering: {str(e)}")
            return {
                "error": str(e),
                "renovation_costs": {},
                "conversion_costs": {},
                "expansion_costs": {},
                "potential_revenue": {}
            }
    
    def _calculate_renovation_costs(self, renovation_needs: List[Dict]) -> Dict:
        """Beregner kostnader for renoveringsbehovene"""
        if not renovation_needs:
            return {"total": 0, "items": []}
        
        renovation_items = []
        total_cost = 0
        
        for need in renovation_needs:
            cost = need.get("estimated_cost", 0)
            renovation_items.append({
                "type": need.get("type", "unknown"),
                "description": need.get("description", ""),
                "cost": cost,
                "priority": need.get("priority", "medium")
            })
            total_cost += cost
        
        # Sorter etter prioritet
        priority_order = {"high": 0, "medium": 1, "low": 2}
        renovation_items.sort(key=lambda x: priority_order.get(x.get("priority", "medium"), 1))
        
        return {
            "total": total_cost,
            "items": renovation_items,
            "cost_per_square_meter": total_cost / 100  # Dummy-verdi hvis vi ikke har areal
        }
    
    def _calculate_conversion_costs(self, rental_options: List[Dict]) -> Dict:
        """Beregner kostnader for konvertering til utleieenheter"""
        if not rental_options:
            return {"total": 0, "options": []}
        
        conversion_options = []
        total_cost = 0
        
        for option in rental_options:
            cost = option.get("estimated_cost", 0)
            conversion_options.append({
                "type": option.get("type", "unknown"),
                "description": option.get("description", ""),
                "cost": cost,
                "area_m2": option.get("area_m2", 0),
                "cost_per_square_meter": cost / option.get("area_m2", 1) if option.get("area_m2", 0) > 0 else 0
            })
            total_cost += cost
        
        return {
            "total": total_cost,
            "options": conversion_options
        }
    
    def _calculate_expansion_costs(self, expansion_options: List[Dict]) -> Dict:
        """Beregner kostnader for utvidelse"""
        if not expansion_options:
            return {"total": 0, "options": []}
        
        expansion_items = []
        total_cost = 0
        
        for option in expansion_options:
            cost = option.get("estimated_cost", 0)
            expansion_items.append({
                "type": option.get("type", "unknown"),
                "description": option.get("description", ""),
                "cost": cost,
                "area_m2": option.get("area_m2", 0),
                "cost_per_square_meter": cost / option.get("area_m2", 1) if option.get("area_m2", 0) > 0 else 0
            })
            total_cost += cost
        
        return {
            "total": total_cost,
            "options": expansion_items
        }
    
    def _estimate_revenue(self,
                        rental_options: List[Dict],
                        expansion_options: List[Dict],
                        division_options: Dict) -> Dict:
        """Estimerer potensielle inntekter"""
        # Beregn leieinntekter
        rental_income = {
            "monthly": sum(option.get("estimated_monthly_rent", 0) for option in rental_options),
            "annual": sum(option.get("estimated_monthly_rent", 0) * 12 for option in rental_options),
            "options": [{
                "type": option.get("type", "unknown"),
                "description": option.get("description", ""),
                "monthly_income": option.get("estimated_monthly_rent", 0),
                "annual_income": option.get("estimated_monthly_rent", 0) * 12,
                "roi_years": option.get("roi_years", 0)
            } for option in rental_options]
        }
        
        # Beregn verdiøkning
        value_increase = {
            "total": sum(option.get("value_increase", 0) for option in expansion_options),
            "options": [{
                "type": option.get("type", "unknown"),
                "description": option.get("description", ""),
                "increase_amount": option.get("value_increase", 0),
                "roi_percentage": option.get("roi_percentage", 0)
            } for option in expansion_options]
        }
        
        # Legg til tomtedeling hvis aktuelt
        if division_options.get("feasible", False):
            division_value = division_options.get("value_estimate", {}).get("value_increase", 0)
            value_increase["total"] += division_value
            value_increase["options"].append({
                "type": "property_division",
                "description": "Tomtedeling og salg",
                "increase_amount": division_value,
                "roi_percentage": division_options.get("roi_percentage", 0)
            })
        
        return {
            "rental_income": rental_income,
            "value_increase": value_increase,
            "total_annual_benefit": rental_income["annual"] + (value_increase["total"] * 0.05)  # Antar 5% årlig avkastning på verdiøkning
        }
    
    def _calculate_roi_metrics(self,
                             renovation_costs: Dict,
                             conversion_costs: Dict,
                             expansion_costs: Dict,
                             potential_revenue: Dict) -> Dict:
        """Beregner ROI-metrics for ulike scenarier"""
        # Total kostnad
        total_cost = renovation_costs.get("total", 0) + conversion_costs.get("total", 0) + expansion_costs.get("total", 0)
        
        # Årlig fordel
        annual_benefit = potential_revenue.get("total_annual_benefit", 0)
        
        # ROI beregning
        roi_percentage = (annual_benefit / total_cost) * 100 if total_cost > 0 else 0
        payback_years = total_cost / annual_benefit if annual_benefit > 0 else float('inf')
        
        # Analyse av scenarier
        scenarios = []
        
        # Scenario 1: Kun renovering
        if renovation_costs.get("total", 0) > 0:
            # Antar 5% verdiøkning av eiendommen fra renovering
            renovation_value_increase = renovation_costs.get("total", 0) * 1.2  # 20% verdiøkning
            renovation_annual_benefit = renovation_value_increase * 0.05  # 5% årlig avkastning
            renovation_roi = (renovation_annual_benefit / renovation_costs.get("total", 0)) * 100 if renovation_costs.get("total", 0) > 0 else 0
            
            scenarios.append({
                "name": "Kun renovering",
                "cost": renovation_costs.get("total", 0),
                "annual_benefit": renovation_annual_benefit,
                "roi_percentage": renovation_roi,
                "payback_years": renovation_costs.get("total", 0) / renovation_annual_benefit if renovation_annual_benefit > 0 else float('inf')
            })
        
        # Scenario 2: Kun utleie
        if conversion_costs.get("total", 0) > 0 and potential_revenue.get("rental_income", {}).get("annual", 0) > 0:
            rental_annual_benefit = potential_revenue.get("rental_income", {}).get("annual", 0)
            rental_roi = (rental_annual_benefit / conversion_costs.get("total", 0)) * 100 if conversion_costs.get("total", 0) > 0 else 0
            
            scenarios.append({
                "name": "Kun utleie",
                "cost": conversion_costs.get("total", 0),
                "annual_benefit": rental_annual_benefit,
                "roi_percentage": rental_roi,
                "payback_years": conversion_costs.get("total", 0) / rental_annual_benefit if rental_annual_benefit > 0 else float('inf')
            })
        
        # Scenario 3: Kun utvidelse
        if expansion_costs.get("total", 0) > 0 and potential_revenue.get("value_increase", {}).get("total", 0) > 0:
            expansion_value_increase = potential_revenue.get("value_increase", {}).get("total", 0)
            expansion_annual_benefit = expansion_value_increase * 0.05  # 5% årlig avkastning
            expansion_roi = (expansion_annual_benefit / expansion_costs.get("total", 0)) * 100 if expansion_costs.get("total", 0) > 0 else 0
            
            scenarios.append({
                "name": "Kun utvidelse",
                "cost": expansion_costs.get("total", 0),
                "annual_benefit": expansion_annual_benefit,
                "roi_percentage": expansion_roi,
                "payback_years": expansion_costs.get("total", 0) / expansion_annual_benefit if expansion_annual_benefit > 0 else float('inf')
            })
        
        # Sorter scenarier etter ROI
        scenarios.sort(key=lambda x: x.get("roi_percentage", 0), reverse=True)
        
        return {
            "total_investment": total_cost,
            "annual_benefit": annual_benefit,
            "overall_roi_percentage": roi_percentage,
            "payback_years": payback_years,
            "best_scenario": scenarios[0] if scenarios else None,
            "scenarios": scenarios
        }
    
    def _suggest_funding_options(self,
                               renovation_costs: Dict,
                               conversion_costs: Dict,
                               expansion_costs: Dict) -> List[Dict]:
        """Foreslår finansieringsalternativer basert på kostnader"""
        total_cost = renovation_costs.get("total", 0) + conversion_costs.get("total", 0) + expansion_costs.get("total", 0)
        
        if total_cost == 0:
            return []
        
        funding_options = []
        
        # Alternativ 1: Boliglån
        funding_options.append({
            "type": "mortgage",
            "name": "Boliglån",
            "description": "Utvide eksisterende boliglån eller ta opp nytt lån med sikkerhet i eiendommen",
            "typical_interest_rate": "3.5-4.5%",
            "typical_term_years": 20,
            "requirements": [
                "Tilstrekkelig egenkapital (normalt minst 15%)",
                "God betalingsevne",
                "Sikkerhet i eiendommen"
            ],
            "monthly_payment": self._calculate_monthly_loan_payment(total_cost, 0.04, 20),
            "suitable_for": ["store investeringer", "langsiktige prosjekter"]
        })
        
        # Alternativ 2: Forbrukslån
        if total_cost < 500000:
            funding_options.append({
                "type": "consumer_loan",
                "name": "Forbrukslån",
                "description": "Usikret lån for mindre renoveringer",
                "typical_interest_rate": "8-12%",
                "typical_term_years": 5,
                "requirements": [
                    "God kredittscore",
                    "Stabil inntekt"
                ],
                "monthly_payment": self._calculate_monthly_loan_payment(total_cost, 0.1, 5),
                "suitable_for": ["mindre prosjekter", "kortsiktige investeringer"]
            })
        
        # Alternativ 3: Husbankens grunnlån
        if renovation_costs.get("total", 0) > 0:
            funding_options.append({
                "type": "husbanken",
                "name": "Husbankens grunnlån",
                "description": "Lån til boligforbedring med fokus på energieffektivisering",
                "typical_interest_rate": "2.5-3.5%",
                "typical_term_years": 30,
                "requirements": [
                    "Prosjektet må møte Husbankens kvalitetskrav",
                    "Fokus på energieffektivisering",
                    "Universell utforming"
                ],
                "monthly_payment": self._calculate_monthly_loan_payment(renovation_costs.get("total", 0), 0.03, 30),
                "suitable_for": ["energioppgradering", "tilpasning av bolig", "miljøvennlige løsninger"]
            })
        
        # Alternativ 4: Enova-støtte
        if renovation_costs.get("total", 0) > 0:
            funding_options.append({
                "type": "enova",
                "name": "Enova-støtte",
                "description": "Tilskudd til energieffektivisering",
                "requirements": [
                    "Prosjektet må kvalifisere for Enovas støtteordninger",
                    "Dokumentasjon på utført arbeid"
                ],
                "typical_amount": min(renovation_costs.get("total", 0) * 0.2, 100000),  # Maks 20% eller 100,000 NOK
                "suitable_for": ["energitiltak", "varmepumper", "etterisolering"]
            })
        
        return funding_options
    
    def _calculate_monthly_loan_payment(self, loan_amount: float, annual_interest_rate: float, term_years: int) -> float:
        """Beregner månedlig lånebetaling"""
        if loan_amount <= 0 or annual_interest_rate <= 0 or term_years <= 0:
            return 0
            
        monthly_rate = annual_interest_rate / 12
        num_payments = term_years * 12
        
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1)
        
        return monthly_payment
    
    async def _generate_recommendations(self,
                                     property_data: Dict,
                                     structure_analysis: Dict,
                                     regulation_info: Dict,
                                     development_potential: Dict,
                                     energy_analysis: Dict,
                                     cost_estimation: Dict,
                                     client_preferences: Optional[Dict] = None) -> List[Dict]:
        """Genererer prioriterte anbefalinger basert på all analysert data"""
        logger.info("Genererer anbefalinger")
        
        try:
            # Bruk optimal scenario fra utviklingspotensiale hvis det eksisterer
            optimal_scenario = development_potential.get("optimal_scenario", {})
            
            if optimal_scenario and "actions" in optimal_scenario:
                recommendations = []
                
                for i, action in enumerate(optimal_scenario["actions"]):
                    # Konverter handling til anbefaling
                    recommendation = self._convert_action_to_recommendation(action, i+1)
                    recommendations.append(recommendation)
                
                # Legg til standard anbefalinger
                recommendations.extend(self._generate_standard_recommendations(
                    property_data,
                    structure_analysis,
                    regulation_info,
                    energy_analysis,
                    cost_estimation
                ))
                
                return recommendations[:5]  # Begrens til 5 anbefalinger
            else:
                # Generer anbefalinger fra bunnen av
                return self._generate_recommendations_from_scratch(
                    property_data,
                    structure_analysis,
                    regulation_info,
                    development_potential,
                    energy_analysis,
                    cost_estimation,
                    client_preferences
                )
        
        except Exception as e:
            logger.error(f"Feil ved generering av anbefalinger: {str(e)}")
            return [{
                "title": "Teknisk feil ved anbefalingsgenerering",
                "description": "Det oppstod en feil ved generering av anbefalinger.",
                "cost": None,
                "benefit": None,
                "roi": None,
                "timeline": None,
                "requirements": ["Kontakt kundeservice for assistanse"],
                "next_steps": ["Kontakt kundeservice"]
            }]
    
    def _convert_action_to_recommendation(self, action: Dict, priority: int) -> Dict:
        """Konverterer en handling fra optimalt scenario til en anbefaling"""
        # Standardverdier
        recommendation = {
            "title": action.get("description", "Anbefalt tiltak"),
            "description": f"Prioritet {priority}: " + action.get("description", "Anbefalt tiltak"),
            "cost": action.get("estimated_cost", 0),
            "benefit": None,
            "roi": None,
            "timeline": "3-6 måneder",
            "requirements": [],
            "next_steps": []
        }
        
        # Legg til spesifikk informasjon basert på handlingstype
        action_type = action.get("type", "unknown")
        
        if action_type in ["basement_conversion", "attic_conversion", "main_floor_division"]:
            # Utleietiltak
            recommendation["benefit"] = f"Månedlig leieinntekt på kr {action.get('estimated_monthly_rent', 0):.0f}"
            recommendation["roi"] = f"Tilbakebetalingstid: {action.get('roi_years', 0):.1f} år"
            recommendation["timeline"] = "6-12 måneder"
            recommendation["requirements"] = action.get("requirements", [])
            recommendation["next_steps"] = [
                "Kontakt arkitekt for tegninger",
                "Søk kommunen om byggetillatelse",
                "Innhent tilbud fra håndverkere"
            ]
        
        elif action_type in ["additional_floor", "extension", "roof_terrace"]:
            # Utvidelsestiltak
            recommendation["benefit"] = f"Verdiøkning på ca. kr {action.get('value_increase', 0):.0f}"
            recommendation["roi"] = f"ROI: {action.get('roi_percentage', 0):.1f}%"
            recommendation["timeline"] = "6-18 måneder"
            recommendation["requirements"] = action.get("requirements", [])
            recommendation["next_steps"] = [
                "Kontakt arkitekt for tegninger",
                "Søk kommunen om byggetillatelse",
                "Innhent tilbud fra entreprenører"
            ]
        
        elif action_type in ["electrical", "bathroom", "kitchen", "windows", "roof", "energy_efficiency"]:
            # Renoveringstiltak
            recommendation["benefit"] = f"Verdiøkning på ca. kr {action.get('value_increase', 0):.0f}"
            recommendation["roi"] = f"ROI: {action.get('roi_percentage', 0):.1f}%"
            recommendation["timeline"] = "2-3 måneder"
            recommendation["next_steps"] = [
                "Innhent tilbud fra håndverkere",
                "Prioriter arbeidet i henhold til anbefalt rekkefølge",
                "Sjekk muligheter for Enova-støtte"
            ]
        
        return recommendation
    
    def _generate_standard_recommendations(self,
                                        property_data: Dict,
                                        structure_analysis: Dict,
                                        regulation_info: Dict,
                                        energy_analysis: Dict,
                                        cost_estimation: Dict) -> List[Dict]:
        """Genererer standardanbefalinger basert på data"""
        recommendations = []
        
        # Energieffektiviseringsanbefaling
        if energy_analysis:
            if energy_analysis.get("current_rating", "") in ["E", "F", "G"] and energy_analysis.get("improvement_measures", []):
                top_energy_measure = energy_analysis["improvement_measures"][0]
                
                recommendations.append({
                    "title": "Energieffektivisering",
                    "description": f"Oppgrader energiklassifiseringen fra {energy_analysis.get('current_rating', '')} til {energy_analysis.get('potential_rating', '')}",
                    "cost": top_energy_measure.get("cost", 0),
                    "benefit": f"Årlig besparelse på ca. kr {energy_analysis.get('annual_cost_saving', 0):.0f}",
                    "roi": f"Tilbakebetalingstid: {top_energy_measure.get('roi_years', 0):.1f} år",
                    "timeline": "2-4 måneder",
                    "requirements": ["Profesjonell installasjon for Enova-støtte"],
                    "next_steps": [
                        "Innhent tilbud fra leverandør",
                        "Søk Enova-støtte",
                        "Planlegg installasjon"
                    ]
                })
        
        # ROI-beregninger
        roi_analysis = cost_estimation.get("roi_analysis", {})
        if roi_analysis.get("best_scenario"):
            best_scenario = roi_analysis["best_scenario"]
            
            recommendations.append({
                "title": best_scenario.get("name", "Beste økonomiske scenario"),
                "description": f"Dette scenarioet gir best avkastning på investeringen med {best_scenario.get('roi_percentage', 0):.1f}% ROI",
                "cost": best_scenario.get("cost", 0),
                "benefit": f"Årlig fordel på ca. kr {best_scenario.get('annual_benefit', 0):.0f}",
                "roi": f"Tilbakebetalingstid: {best_scenario.get('payback_years', 0):.1f} år",
                "timeline": "6-12 måneder",
                "requirements": [],
                "next_steps": [
                    "Utarbeid detaljert prosjektplan",
                    "Innhent flere pristilbud",
                    "Vurder finansieringsalternativer"
                ]
            })
        
        # Finansieringsanbefaling
        funding_options = cost_estimation.get("funding_options", [])
        if funding_options:
            best_funding = min(funding_options, key=lambda x: x.get("monthly_payment", float('inf')) if x.get("type") != "enova" else float('inf'))
            
            if best_funding.get("type") != "enova":
                recommendations.append({
                    "title": f"Finansiering via {best_funding.get('name', 'lån')}",
                    "description": f"Optimal finansieringsløsning: {best_funding.get('description', '')}",
                    "cost": f"Månedlig betaling: kr {best_funding.get('monthly_payment', 0):.0f}",
                    "benefit": "Realisere verdiskapende prosjekter",
                    "roi": None,
                    "timeline": "1-2 måneder",
                    "requirements": best_funding.get("requirements", []),
                    "next_steps": [
                        "Kontakt bank for lånetilbud",
                        "Sammenlign betingelser fra flere banker",
                        "Samle nødvendig dokumentasjon"
                    ]
                })
        
        return recommendations
    
    def _generate_recommendations_from_scratch(self,
                                            property_data: Dict,
                                            structure_analysis: Dict,
                                            regulation_info: Dict,
                                            development_potential: Dict,
                                            energy_analysis: Dict,
                                            cost_estimation: Dict,
                                            client_preferences: Optional[Dict] = None) -> List[Dict]:
        """Genererer anbefalinger fra bunnen av hvis optimalt scenario mangler"""
        # Dette er en fallback-metode hvis utviklingspotensiale mangler optimalt scenario
        recommendations = []
        
        # Sjekk etter utleiemuligheter
        rental_options = development_potential.get("rental_units", [])
        if rental_options:
            top_rental = rental_options[0]
            recommendations.append({
                "title": top_rental.get("description", "Utleieenhet"),
                "description": f"Utleiepotensial med månedlig inntekt på kr {top_rental.get('estimated_monthly_rent', 0):.0f}",
                "cost": top_rental.get("estimated_cost", 0),
                "benefit": f"Årlig leieinntekt på kr {top_rental.get('estimated_monthly_rent', 0) * 12:.0f}",
                "roi": f"Tilbakebetalingstid: {top_rental.get('roi_years', 0):.1f} år",
                "timeline": "6-12 måneder",
                "requirements": top_rental.get("requirements", []),
                "next_steps": [
                    "Kontakt arkitekt for tegninger",
                    "Søk kommunen om byggetillatelse",
                    "Innhent tilbud fra håndverkere"
                ]
            })
        
        # Sjekk etter utvidelsesmuligheter
        expansion_options = development_potential.get("expansion_possibilities", [])
        if expansion_options:
            top_expansion = expansion_options[0]
            recommendations.append({
                "title": top_expansion.get("description", "Utvidelse"),
                "description": f"Utvide boligareal med {top_expansion.get('area_m2', 0):.1f} m²",
                "cost": top_expansion.get("estimated_cost", 0),
                "benefit": f"Verdiøkning på ca. kr {top_expansion.get('value_increase', 0):.0f}",
                "roi": f"ROI: {top_expansion.get('roi_percentage', 0):.1f}%",
                "timeline": "6-18 måneder",
                "requirements": top_expansion.get("requirements", []),
                "next_steps": [
                    "Kontakt arkitekt for tegninger",
                    "Søk kommunen om byggetillatelse",
                    "Innhent tilbud fra entreprenører"
                ]
            })
        
        # Sjekk etter renoveringsbehov
        renovation_needs = development_potential.get("renovation_needs", [])
        if renovation_needs:
            high_priority = [r for r in renovation_needs if r.get("priority") == "high"]
            
            if high_priority:
                top_renovation = high_priority[0]
                recommendations.append({
                    "title": top_renovation.get("description", "Renovering"),
                    "description": f"Høyprioritert renovering: {top_renovation.get('description', '')}",
                    "cost": top_renovation.get("estimated_cost", 0),
                    "benefit": f"Verdiøkning på ca. kr {top_renovation.get('value_increase', 0):.0f}",
                    "roi": f"ROI: {top_renovation.get('roi_percentage', 0):.1f}%",
                    "timeline": "2-4 måneder",
                    "requirements": [],
                    "next_steps": [
                        "Innhent tilbud fra håndverkere",
                        "Planlegg arbeidet i riktig rekkefølge",
                        "Sjekk muligheter for finansiering"
                    ]
                })
        
        # Sjekk etter energieffektiviseringsmuligheter
        if energy_analysis and energy_analysis.get("improvement_measures", []):
            top_energy_measure = energy_analysis["improvement_measures"][0]
            
            recommendations.append({
                "title": "Energieffektivisering",
                "description": top_energy_measure.get("description", "Energitiltak"),
                "cost": top_energy_measure.get("cost", 0),
                "benefit": f"Årlig besparelse på ca. kr {energy_analysis.get('annual_cost_saving', 0):.0f}",
                "roi": f"Tilbakebetalingstid: {top_energy_measure.get('roi_years', 0):.1f} år",
                "timeline": "2-4 måneder",
                "requirements": ["Profesjonell installasjon for Enova-støtte"],
                "next_steps": [
                    "Innhent tilbud fra leverandør",
                    "Søk Enova-støtte",
                    "Planlegg installasjon"
                ]
            })
        
        # Legg til generell anbefaling hvis listen er tom
        if not recommendations:
            recommendations.append({
                "title": "Vedlikehold av eiendommen",
                "description": "Jevnlig vedlikehold anbefales for å opprettholde eiendommens verdi",
                "cost": None,
                "benefit": "Langsiktig verdibevaring",
                "roi": None,
                "timeline": "Løpende",
                "requirements": [],
                "next_steps": [
                    "Sett opp en vedlikeholdsplan",
                    "Utfør regelmessig inspeksjon av tak, vinduer og fasade",
                    "Sett av midler til årlig vedlikehold"
                ]
            })
        
        return recommendations[:5]  # Begrens til 5 anbefalinger
    
    def _log_error(self, message: str):
        """Logger feilmelding"""
        logger.error(message)
