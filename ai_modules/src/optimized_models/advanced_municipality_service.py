from typing import Dict, Any, List, Optional, Tuple, Union
import aiohttp
import asyncio
import json
import logging
from datetime import datetime, timedelta
import redis
from bs4 import BeautifulSoup
import re
from dataclasses import dataclass, field
import hashlib
from aiohttp import ClientTimeout
from concurrent.futures import ThreadPoolExecutor
import os
from functools import lru_cache
import urllib.parse
import time
import pyproj
from shapely.geometry import Polygon, Point
from shapely.ops import transform
import numpy as np

@dataclass
class CacheConfig:
    """Konfigurasjon for intelligent caching med variabel TTL basert på datatype"""
    redis_url: str = os.environ.get("REDIS_URL", "redis://localhost")
    default_ttl: int = 3600  # 1 time
    regulations_ttl: int = 86400 * 7  # 7 dager (reguleringsplaner endres sjelden)
    property_ttl: int = 43200  # 12 timer
    zoning_ttl: int = 86400 * 14  # 14 dager
    search_results_ttl: int = 900  # 15 minutter
    max_parallel_requests: int = 20
    cache_compression: bool = True
    cache_namespace: str = "eiendomsplattform"
    fallback_to_memory_cache: bool = True  # Fallback hvis Redis ikke er tilgjengelig
    memory_cache_size: int = 1000  # Maks antall elementer i minnecache

@dataclass
class ApiConfig:
    """Konfigurasjon for eksterne API-tjenester"""
    kartverket_url: str = "https://ws.geonorge.no/eiendom/v1"
    kartverket_token: str = os.environ.get("KARTVERKET_TOKEN", "")
    dibk_url: str = "https://dibk-api.no/v1"
    dibk_token: str = os.environ.get("DIBK_TOKEN", "")
    kommune_api_urls: Dict[str, str] = field(default_factory=lambda: {
        "3005": "https://api.drammen.kommune.no/eiendom/v1",  # Drammen
        "3024": "https://api.baerum.kommune.no/eiendom/v1",   # Bærum
        "0301": "https://api.oslo.kommune.no/eiendom/v1",     # Oslo
        "5001": "https://api.trondheim.kommune.no/eiendom/v1" # Trondheim
    })
    retries: int = 3
    timeout: int = 30
    user_agent: str = "EiendomsPlattform/1.0"

class MemoryCache:
    """Enkel minnecache som brukes hvis Redis ikke er tilgjengelig"""
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.expiry_times = {}
        
    def get(self, key: str) -> Optional[str]:
        """Henter verdi fra cache hvis den finnes og ikke er utløpt"""
        if key not in self.cache:
            return None
            
        # Sjekk om verdien er utløpt
        if key in self.expiry_times and self.expiry_times[key] < time.time():
            del self.cache[key]
            del self.expiry_times[key]
            return None
            
        return self.cache[key]
        
    def setex(self, key: str, ttl: int, value: str) -> None:
        """Setter verdi i cache med utløpstid"""
        # Fjern eldste element hvis cachen er full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.expiry_times.keys(), key=lambda k: self.expiry_times[k])
            del self.cache[oldest_key]
            del self.expiry_times[oldest_key]
            
        self.cache[key] = value
        self.expiry_times[key] = time.time() + ttl
        
    def delete(self, key: str) -> None:
        """Fjerner en nøkkel fra cachen"""
        if key in self.cache:
            del self.cache[key]
        if key in self.expiry_times:
            del self.expiry_times[key]

class AdvancedMunicipalityService:
    """
    Avansert tjeneste for kommune- og eiendomsanalyse.
    
    Denne tjenesten kommuniserer med kommunale API-er for å innhente 
    eiendomsinformasjon, reguleringsplaner, byggetillatelser og annen
    relevant informasjon for eiendomsutvikling. Tjenesten bruker intelligent
    caching for å minimere API-kall og redusere belastningen.
    """
    def __init__(
        self, 
        cache_config: Optional[CacheConfig] = None,
        api_config: Optional[ApiConfig] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.cache_config = cache_config or CacheConfig()
        self.api_config = api_config or ApiConfig()
        
        # Initialiserer Redis-klient
        try:
            self.redis_client = redis.Redis.from_url(
                self.cache_config.redis_url,
                decode_responses=True
            )
            self.logger.info("Redis-tilkobling opprettet")
            # Test tilkoblingen
            self.redis_client.ping()
            self.use_redis = True
        except Exception as e:
            self.logger.warning(f"Kunne ikke koble til Redis: {str(e)}. Fallback til minnecache.")
            self.memory_cache = MemoryCache(self.cache_config.memory_cache_size)
            self.use_redis = False
        
        # Oppretter HTTP-sesjon med konfigurerbare parametere
        self.session_pool = aiohttp.ClientSession(
            timeout=ClientTimeout(total=self.api_config.timeout),
            connector=aiohttp.TCPConnector(
                limit=self.cache_config.max_parallel_requests,
                ssl=False,
                ttl_dns_cache=300  # Cache DNS-oppslag i 5 minutter
            ),
            headers={
                "User-Agent": self.api_config.user_agent,
                "Accept": "application/json"
            }
        )
        
        # ThreadPoolExecutor for CPU-intensive oppgaver
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Lokale koordinatsystemer for norske kommuner (EPSG-koder)
        self.municipality_crs = {
            "3005": "EPSG:5972",  # Drammen
            "0301": "EPSG:5973",  # Oslo
            "5001": "EPSG:5975",  # Trondheim
            "4601": "EPSG:5974"   # Bergen
        }
        
        # Initialiserer projeksjonstransformasjoner for koordinatkonvertering
        self.wgs84 = pyproj.CRS("EPSG:4326")  # Standard GPS-koordinater
        self.projections = {}
        for code, epsg in self.municipality_crs.items():
            try:
                crs = pyproj.CRS(epsg)
                self.projections[code] = pyproj.Transformer.from_crs(
                    self.wgs84, crs, always_xy=True
                )
            except Exception as e:
                self.logger.error(f"Kunne ikke initialisere projeksjon for {code}: {str(e)}")
        
    async def __aenter__(self):
        """Støtte for async with-uttrykk"""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Lukker ressurser ved exit fra async with-blokk"""
        await self.close()
        
    async def close(self):
        """Lukker alle åpne ressurser"""
        if hasattr(self, 'session_pool'):
            await self.session_pool.close()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

    async def get_property_details(
        self,
        address: Optional[str] = None,
        municipality_code: Optional[str] = None,
        gnr: Optional[int] = None,
        bnr: Optional[int] = None,
        coordinates: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Henter detaljert informasjon om eiendom basert på adresse, gnr/bnr eller koordinater.
        
        Args:
            address: Eiendommens adresse
            municipality_code: Kommunenummer
            gnr: Gårdsnummer
            bnr: Bruksnummer
            coordinates: Koordinater (breddegrad, lengdegrad)
            
        Returns:
            Dict med komplett eiendomsinformasjon
        """
        # Valider inndata
        if not any([address, (gnr is not None and bnr is not None), coordinates]):
            raise ValueError("Minst én av adresse, gnr/bnr eller koordinater må angis")
        
        # Generer cache-nøkkel basert på tilgjengelig informasjon
        cache_key_parts = []
        if address:
            cache_key_parts.append(f"address:{address}")
        if municipality_code:
            cache_key_parts.append(f"municipality:{municipality_code}")
        if gnr is not None and bnr is not None:
            cache_key_parts.append(f"gnr:{gnr}:bnr:{bnr}")
        if coordinates:
            lat, lon = coordinates
            cache_key_parts.append(f"coord:{lat:.6f},{lon:.6f}")
        
        cache_key = f"property:{hashlib.md5(':'.join(cache_key_parts).encode()).hexdigest()}"
        
        # Sjekk cache først
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            self.logger.info(f"Hentet eiendomsdata fra cache: {cache_key}")
            return cached_data

        # Konverter adresse til gnr/bnr hvis nødvendig
        if gnr is None or bnr is None:
            if address:
                property_id = await self._address_to_property_id(address, municipality_code)
                if property_id:
                    gnr, bnr = property_id.get('gnr'), property_id.get('bnr')
                    municipality_code = property_id.get('municipality_code', municipality_code)
            elif coordinates:
                property_id = await self._coordinates_to_property_id(coordinates, municipality_code)
                if property_id:
                    gnr, bnr = property_id.get('gnr'), property_id.get('bnr')
                    municipality_code = property_id.get('municipality_code', municipality_code)
        
        if not municipality_code:
            raise ValueError("Kunne ikke bestemme kommunekode")
            
        if gnr is None or bnr is None:
            raise ValueError("Kunne ikke finne gårds- og bruksnummer")

        # Parallell innhenting av data
        tasks = [
            self._get_property_base_info(gnr, bnr, municipality_code),
            self._get_property_regulations(gnr, bnr, municipality_code),
            self._get_property_history(gnr, bnr, municipality_code),
            self._get_zoning_details(gnr, bnr, municipality_code),
            self._get_building_permits(gnr, bnr, municipality_code),
            self._get_property_boundaries(gnr, bnr, municipality_code),
            self._get_nearby_services(gnr, bnr, municipality_code),
            self._get_property_tax_info(gnr, bnr, municipality_code)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Kombiner resultatene
        property_data = {
            'property_id': {
                'gnr': gnr,
                'bnr': bnr,
                'municipality_code': municipality_code
            },
            'base_info': results[0] if not isinstance(results[0], Exception) else {},
            'regulations': results[1] if not isinstance(results[1], Exception) else [],
            'history': results[2] if not isinstance(results[2], Exception) else [],
            'zoning': results[3] if not isinstance(results[3], Exception) else {},
            'permits': results[4] if not isinstance(results[4], Exception) else [],
            'boundaries': results[5] if not isinstance(results[5], Exception) else {},
            'nearby_services': results[6] if not isinstance(results[6], Exception) else {},
            'tax_info': results[7] if not isinstance(results[7], Exception) else {}
        }
        
        # Logg feil
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Feil ved innhenting av data (task {i}): {str(result)}")
        
        # Cache resultatet
        await self._cache_data(
            cache_key,
            property_data,
            self.cache_config.property_ttl
        )
        
        return property_data
        
    async def _address_to_property_id(
        self,
        address: str,
        municipality_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """Konverterer adresse til gårds- og bruksnummer"""
        cache_key = f"address_lookup:{address}:{municipality_code or 'unknown'}"
        
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        # Bygg API-URL
        url = f"{self.api_config.kartverket_url}/sok"
        params = {'adresse': address}
        if municipality_code:
            params['kommunenummer'] = municipality_code
        
        # Utfør API-kall med gjentakelser ved feil
        for attempt in range(self.api_config.retries):
            try:
                async with self.session_pool.get(
                    url, 
                    params=params,
                    headers=self._get_auth_headers('kartverket')
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Prosesser resultatet
                        result = self._extract_property_id_from_response(data)
                        
                        # Cache resultatet hvis det ble funnet
                        if result:
                            await self._cache_data(
                                cache_key,
                                result,
                                self.cache_config.default_ttl
                            )
                            return result
                    elif response.status == 429:  # Rate limiting
                        wait_time = int(response.headers.get('Retry-After', attempt + 1))
                        self.logger.warning(f"Rate limiting fra API, venter {wait_time} sekunder")
                        await asyncio.sleep(wait_time)
                    else:
                        self.logger.error(f"API-feil: {response.status}, {await response.text()}")
                        await asyncio.sleep(attempt + 1)
            except aiohttp.ClientError as e:
                self.logger.error(f"Nettverksfeil: {str(e)}")
                await asyncio.sleep(attempt + 1)
                
        return {}
        
    def _extract_property_id_from_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Henter ut eiendoms-ID fra API-respons"""
        if not data.get('adresser'):
            return {}
            
        # Finn første treff
        for address in data['adresser']:
            if 'matrikkel' in address:
                matrikkel = address['matrikkel']
                return {
                    'gnr': int(matrikkel.get('gardsnummer', 0)),
                    'bnr': int(matrikkel.get('bruksnummer', 0)),
                    'fnr': int(matrikkel.get('festenummer', 0)) if matrikkel.get('festenummer') else None,
                    'snr': int(matrikkel.get('seksjonsnummer', 0)) if matrikkel.get('seksjonsnummer') else None,
                    'municipality_code': matrikkel.get('kommunenummer')
                }
                
        return {}
        
    async def _coordinates_to_property_id(
        self,
        coordinates: Tuple[float, float],
        municipality_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """Konverterer koordinater til gårds- og bruksnummer"""
        lat, lon = coordinates
        cache_key = f"coord_lookup:{lat:.6f},{lon:.6f}:{municipality_code or 'unknown'}"
        
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        # Bygg API-URL
        url = f"{self.api_config.kartverket_url}/punkt"
        params = {'nord': lat, 'ost': lon}
        if municipality_code:
            params['kommunenummer'] = municipality_code
            
        # Utfør API-kall med gjentakelser ved feil
        for attempt in range(self.api_config.retries):
            try:
                async with self.session_pool.get(
                    url, 
                    params=params,
                    headers=self._get_auth_headers('kartverket')
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Prosesser resultatet
                        result = self._extract_property_id_from_response(data)
                        
                        # Cache resultatet hvis det ble funnet
                        if result:
                            await self._cache_data(
                                cache_key,
                                result,
                                self.cache_config.default_ttl
                            )
                            return result
                    elif response.status == 429:  # Rate limiting
                        wait_time = int(response.headers.get('Retry-After', attempt + 1))
                        self.logger.warning(f"Rate limiting fra API, venter {wait_time} sekunder")
                        await asyncio.sleep(wait_time)
                    else:
                        self.logger.error(f"API-feil: {response.status}, {await response.text()}")
                        await asyncio.sleep(attempt + 1)
            except aiohttp.ClientError as e:
                self.logger.error(f"Nettverksfeil: {str(e)}")
                await asyncio.sleep(attempt + 1)
                
        return {}

    async def analyze_development_potential(
        self,
        property_data: Dict[str, Any],
        objective: str = "rental_unit",
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyserer utviklingspotensial basert på kommunale regler og kundens mål.
        
        Args:
            property_data: Eiendomsdata fra get_property_details
            objective: Kundens hovedmål ('rental_unit', 'value_increase', 'energy_efficiency', etc.)
            constraints: Eventuelle begrensninger (budsjett, tidslinje, etc.)
            
        Returns:
            Dict med detaljert analyse av utviklingspotensial
        """
        # Standardiserer mål
        objective = objective.lower()
        if constraints is None:
            constraints = {}
            
        # Hent eiendomsidentifikasjon
        property_id = property_data.get('property_id', {})
        if not property_id:
            raise ValueError("Eiendomsidentifikasjon mangler")
            
        gnr = property_id.get('gnr')
        bnr = property_id.get('bnr')
        municipality_code = property_id.get('municipality_code')
        
        if not all([gnr, bnr, municipality_code]):
            raise ValueError("Ufullstendig eiendomsidentifikasjon")
            
        # Genererer cache-nøkkel
        cache_key = f"potential:{municipality_code}:{gnr}:{bnr}:{objective}:{hashlib.md5(json.dumps(constraints, sort_keys=True).encode()).hexdigest()}"
        
        # Sjekk cache først
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            self.logger.info(f"Hentet utviklingspotensial fra cache: {cache_key}")
            return cached_data
            
        # Parallell analyse av ulike aspekter
        tasks = [
            self._analyze_zoning_potential(property_data, objective, constraints),
            self._analyze_building_restrictions(property_data, objective, constraints),
            self._analyze_historical_precedents(property_data, objective, constraints),
            self._analyze_infrastructure_requirements(property_data, objective, constraints),
            self._analyze_floor_plan_potential(property_data, objective, constraints),
            self._analyze_economic_potential(property_data, objective, constraints),
            self._analyze_environmental_factors(property_data, objective, constraints),
            self._analyze_nearby_property_values(property_data, objective, constraints)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Behandle resultater og håndter feil
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Feil ved analyse (task {i}): {str(result)}")
                processed_results.append({})
            else:
                processed_results.append(result)
                
        # Utpakk resultater for enklere tilgang
        zoning_potential, restrictions, precedents, infrastructure, floor_plan, economics, environmental, property_values = processed_results
        
        # Kombiner resultater i én komplett analyserapport
        analysis = {
            'zoning_analysis': zoning_potential,
            'building_restrictions': restrictions,
            'historical_precedents': precedents,
            'infrastructure_requirements': infrastructure,
            'floor_plan_analysis': floor_plan,
            'economic_analysis': economics,
            'environmental_analysis': environmental,
            'property_values': property_values,
            'recommended_approaches': self._generate_development_recommendations(
                processed_results,
                objective,
                constraints
            ),
            'estimated_timeline': self._estimate_development_timeline(
                processed_results,
                objective,
                constraints
            ),
            'risk_assessment': self._assess_development_risks(
                processed_results,
                objective,
                constraints
            ),
            'optimized_solutions': self._generate_optimized_solutions(
                processed_results,
                objective,
                constraints
            )
        }
        
        # Cache resultatet
        await self._cache_data(
            cache_key,
            analysis,
            self.cache_config.default_ttl
        )
        
        return analysis
        
    async def _analyze_zoning_potential(
        self,
        property_data: Dict[str, Any],
        objective: str,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyserer potensial basert på regulering"""
        zoning = property_data.get('zoning', {})
        base_info = property_data.get('base_info', {})
        
        # Beregn utnyttelsesgrad
        max_bya_percentage = zoning.get('max_bya_percentage', 0)
        max_bya = 0
        current_bya = base_info.get('current_bya', 0)
        plot_size = base_info.get('plot_size', 0)
        
        if max_bya_percentage > 0 and plot_size > 0:
            max_bya = plot_size * max_bya_percentage / 100
        
        # Beregn høydebegrensninger
        max_height = zoning.get('max_height', 0)
        max_floors = zoning.get('max_floors', 0)
        current_height = base_info.get('height', 0)
        current_floors = base_info.get('floors', 0)
        
        # Analyser muligheter
        potential_uses = await self._get_potential_uses(
            property_data.get('property_id', {}).get('municipality_code', ''),
            zoning.get('zone_type', ''),
            objective
        )
        
        # Beregn utbyggingspotensial
        potential_expansion = max_bya - current_bya if max_bya > 0 else 0
        
        # Analyser potensial for ulike typer utvikling
        development_types = []
        
        # Vurder tilbyggspotensial
        if potential_expansion > 15:  # Minst 15 m² for å være meningsfullt
            development_types.append({
                'type': 'extension',
                'potential_area': potential_expansion,
                'constraints': [],
                'requirements': []
            })
            
        # Vurder påbyggspotensial (ekstra etasje)
        if max_floors > current_floors or (max_height > 0 and current_height + 3 <= max_height):
            floor_area = base_info.get('footprint', 0)
            development_types.append({
                'type': 'additional_floor',
                'potential_area': floor_area,
                'constraints': [],
                'requirements': []
            })
            
        # Vurder utleieenhetspotensial
        if objective == 'rental_unit':
            development_types.append({
                'type': 'rental_unit',
                'potential': self._evaluate_rental_unit_potential(property_data),
                'constraints': [],
                'requirements': []
            })
            
        # Vurder tomtedelingspotensial
        subdivision_potential = self._evaluate_subdivision_potential(property_data)
        if subdivision_potential.get('feasible', False):
            development_types.append({
                'type': 'subdivision',
                'potential': subdivision_potential,
                'constraints': [],
                'requirements': []
            })
        
        return {
            'unutilized_bya': potential_expansion,
            'height_potential': (max_height - current_height) if max_height > 0 else None,
            'floor_potential': (max_floors - current_floors) if max_floors > 0 else None,
            'potential_uses': potential_uses,
            'zoning_restrictions': zoning.get('restrictions', []),
            'development_types': development_types,
            'development_scenarios': await self._generate_development_scenarios(
                property_data,
                objective,
                constraints
            )
        }
        
    def _evaluate_rental_unit_potential(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluerer potensial for utleieenhet"""
        base_info = property_data.get('base_info', {})
        buildings = base_info.get('buildings', [])
        
        # Ingen bygninger = ingen utleieenhet
        if not buildings:
            return {
                'feasible': False,
                'reason': 'Ingen bygninger funnet'
            }
            
        # Sjekk om hovedbygget har kjeller
        main_building = buildings[0] if buildings else {}
        has_basement = main_building.get('has_basement', False)
        basement_area = main_building.get('basement_area', 0)
        
        # Sjekk om hovedbygget har loft
        has_attic = main_building.get('has_attic', False)
        attic_area = main_building.get('attic_area', 0)
        
        # Sjekk total boligareal
        total_area = main_building.get('area_BRA', 0)
        
        # Vurder ulike muligheter
        options = []
        
        # Kjeller som utleieenhet?
        if has_basement and basement_area >= 25:  # Minimum 25 m² for utleieenhet
            options.append({
                'type': 'basement_conversion',
                'area': basement_area,
                'feasibility': 'high' if basement_area >= 40 else 'medium',
                'estimated_cost': basement_area * 15000,  # Omtrent 15 000 kr per m²
                'estimated_monthly_rent': basement_area * 200  # Omtrent 200 kr per m² per måned
            })
            
        # Loft som utleieenhet?
        if has_attic and attic_area >= 25:
            options.append({
                'type': 'attic_conversion',
                'area': attic_area,
                'feasibility': 'medium',
                'estimated_cost': attic_area * 18000,  # Omtrent 18 000 kr per m²
                'estimated_monthly_rent': attic_area * 200
            })
            
        # Del av hovedetasje som utleieenhet?
        if total_area >= 120:  # Tilstrekkelig stort for deling
            section_area = total_area / 3  # Anta ca. 1/3 av arealet
            options.append({
                'type': 'main_floor_division',
                'area': section_area,
                'feasibility': 'medium',
                'estimated_cost': section_area * 12000,  # Omtrent 12 000 kr per m²
                'estimated_monthly_rent': section_area * 220  # Litt høyere leiepris for hovedetasje
            })
            
        # Garasje/uthus-konvertering?
        for i, building in enumerate(buildings[1:], 1):  # Skip hovedbygget
            if building.get('building_type', '').lower() in ['garage', 'outbuilding', 'annex']:
                building_area = building.get('area_BRA', 0)
                if building_area >= 20:
                    options.append({
                        'type': 'outbuilding_conversion',
                        'building_index': i,
                        'area': building_area,
                        'feasibility': 'medium',
                        'estimated_cost': building_area * 20000,  # Omtrent 20 000 kr per m²
                        'estimated_monthly_rent': building_area * 180
                    })
        
        # Returner en samlet vurdering
        return {
            'feasible': len(options) > 0,
            'options': options,
            'best_option': self._find_best_rental_option(options) if options else None
        }
        
    def _find_best_rental_option(self, options: List[Dict[str, Any]]) -> Dict[str, Any]:
       """Finner beste utleiealternativ basert på ROI"""
       if not options:
           return {}
           
       # Beregn ROI for hvert alternativ
       for option in options:
           monthly_rent = option.get('estimated_monthly_rent', 0)
           annual_rent = monthly_rent * 12
           cost = option.get('estimated_cost', 1)  # Unngå divisjon med null
           
           # Beregn ROI og tilbakebetalingstid
           roi_percentage = (annual_rent / cost) * 100 if cost > 0 else 0
           payback_years = cost / annual_rent if annual_rent > 0 else float('inf')
           
           option['roi_percentage'] = roi_percentage
           option['payback_years'] = payback_years
           
       # Sorter alternativer etter ROI (høyest først)
       sorted_options = sorted(options, key=lambda x: x.get('roi_percentage', 0), reverse=True)
       
       return sorted_options[0] if sorted_options else {}
       
   def _evaluate_subdivision_potential(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
       """Evaluerer potensial for tomtedeling"""
       base_info = property_data.get('base_info', {})
       zoning = property_data.get('zoning', {})
       
       plot_size = base_info.get('plot_size', 0)
       municipality_code = property_data.get('property_id', {}).get('municipality_code', '')
       
       # Beregn minimumskrav basert på kommune
       min_plot_size = self._get_minimum_plot_size(municipality_code)
       
       # Sjekk om tomten er stor nok til å deles
       if plot_size < 2 * min_plot_size:
           return {
               'feasible': False,
               'reason': f'Tomten er for liten for deling (minimum {2 * min_plot_size} m²)'
           }
           
       # Sjekk byggenes plassering
       buildings = base_info.get('buildings', [])
       if not buildings:
           return {
               'feasible': False,
               'reason': 'Ingen bygninger funnet på tomten'
           }
           
       # Sjekk om bygningene er plassert slik at tomten kan deles
       # Dette er en forenklet vurdering - en reell vurdering trenger geometri-analyse
       
       # For denne forenklede versjonen, anta at tomten kan deles hvis den er stor nok
       # og bygningene ikke opptar mesteparten av arealet
       
       # Beregn totalt bebygd areal (BYA)
       total_bya = sum(building.get('footprint', 0) for building in buildings)
       
       # Hvis BYA er mer enn 30% av den ene halvdelen, blir deling vanskelig
       if total_bya > (plot_size / 2) * 0.3:
           return {
               'feasible': False,
               'reason': 'Bygningene opptar for stor del av tomten for effektiv deling'
           }
           
       # Estimert verdi av eksisterende tomt
       current_value = base_info.get('estimated_value', 0)
       
       # Estimert verdi etter deling
       # Mindre tomter har typisk høyere verdi per m²
       value_factor = 1.2  # 20% høyere verdi per m² for mindre tomter
       new_plot_size = plot_size / 2
       new_plot_value = (current_value / plot_size) * new_plot_size * value_factor
       
       # Kostnader ved deling
       division_costs = {
           'application_fee': 30000,
           'surveying': 25000,
           'infrastructure': 150000,
           'legal_fees': 50000,
           'total': 255000
       }
       
       # Beregn ROI
       roi = ((2 * new_plot_value) - current_value - division_costs['total']) / division_costs['total']
       
       return {
           'feasible': True,
           'current_plot_size': plot_size,
           'new_plot_size': new_plot_size,
           'min_plot_size': min_plot_size,
           'estimated_value': {
               'current_property': current_value,
               'after_division': 2 * new_plot_value,
               'value_increase': (2 * new_plot_value) - current_value
           },
           'division_costs': division_costs,
           'roi_percentage': roi * 100,
           'requirements': [
               'Søknad om deling',
               'Godkjent reguleringsplan',
               'Oppmåling av tomt',
               'Infrastruktur til ny tomt'
           ]
       }
       
   def _get_minimum_plot_size(self, municipality_code: str) -> float:
       """Henter minstekrav til tomtestørrelse for en kommune"""
       min_sizes = {
           "0301": 600.0,  # Oslo
           "3024": 800.0,  # Bærum
           "4601": 500.0,  # Bergen
           "5001": 500.0,  # Trondheim
           "3005": 500.0,  # Drammen
           "1103": 450.0,  # Stavanger
           "4204": 500.0,  # Kristiansand
       }
       
       return min_sizes.get(municipality_code, 600.0)  # Standard 600m² hvis ukjent

   async def _generate_development_scenarios(
       self,
       property_data: Dict[str, Any],
       objective: str,
       constraints: Dict[str, Any]
   ) -> List[Dict[str, Any]]:
       """Genererer ulike utviklingsscenarier basert på kundens mål"""
       scenarios = []
       
       # Scenario 1: Maksimal utnyttelse
       scenarios.append(await self._analyze_max_utilization_scenario(
           property_data,
           objective,
           constraints
       ))
       
       # Scenario 2: Kostnadseffektiv utvikling
       scenarios.append(await self._analyze_cost_effective_scenario(
           property_data,
           objective,
           constraints
       ))
       
       # Scenario 3: Optimal leieverdimaksimering
       if objective == 'rental_unit':
           scenarios.append(await self._analyze_rental_optimization_scenario(
               property_data,
               constraints
           ))
       
       # Scenario 4: Energieffektiv oppgradering
       if objective == 'energy_efficiency':
           scenarios.append(await self._analyze_energy_efficiency_scenario(
               property_data,
               constraints
           ))
           
       # Scenario 5: Rask implementering
       scenarios.append(await self._analyze_quick_implementation_scenario(
           property_data,
           objective,
           constraints
       ))
       
       # Scenario 6: Minimalt inngripende (egnet for fredede bygninger)
       if property_data.get('base_info', {}).get('is_protected', False):
           scenarios.append(await self._analyze_minimal_intervention_scenario(
               property_data,
               objective,
               constraints
           ))
       
       # Filtrer bort ugyldige scenarier
       valid_scenarios = [s for s in scenarios if s.get('is_valid', False)]
       
       # Sorter scenarier basert på hvilken som best oppfyller målet
       sorted_scenarios = self._rank_scenarios(valid_scenarios, objective, constraints)
       
       return sorted_scenarios
       
   async def _analyze_max_utilization_scenario(
       self,
       property_data: Dict[str, Any],
       objective: str,
       constraints: Dict[str, Any]
   ) -> Dict[str, Any]:
       """Analyserer scenario for maksimal utnyttelse"""
       zoning = property_data.get('zoning', {})
       base_info = property_data.get('base_info', {})
       
       max_bya_percentage = zoning.get('max_bya_percentage', 0)
       plot_size = base_info.get('plot_size', 0)
       
       if max_bya_percentage <= 0 or plot_size <= 0:
           return {'is_valid': False, 'reason': 'Mangler informasjon om utnyttelsesgrad eller tomtestørrelse'}
       
       max_bya = plot_size * max_bya_percentage / 100
       current_bya = base_info.get('current_bya', 0)
       
       potential_area = max_bya - current_bya
       
       if potential_area <= 0:
           return {'is_valid': False, 'reason': 'Ingen gjenværende utbyggingspotensial'}
       
       # Sjekk mot brukerens budsjettbegrensning
       budget = constraints.get('budget', float('inf'))
       estimated_cost = self._estimate_development_cost(potential_area, 'extension')
       
       if estimated_cost > budget:
           # Juster ned størrelsen for å passe budsjettet
           affordable_area = budget / (estimated_cost / potential_area)
           potential_area = affordable_area
           estimated_cost = budget
       
       # Generer detaljert plan
       extension_plan = self._generate_extension_plan(property_data, potential_area, objective)
       
       return {
           'is_valid': True,
           'type': 'maximum_utilization',
           'title': 'Maksimal utnyttelse av eiendommen',
           'potential_new_area': potential_area,
           'estimated_cost': estimated_cost,
           'potential_value': self._estimate_development_value(
               potential_area,
               property_data.get('property_id', {}).get('municipality_code', '')
           ),
           'potential_rental_income': self._estimate_rental_income(potential_area, property_data) if objective == 'rental_unit' else None,
           'challenges': self._identify_development_challenges(property_data, 'maximum'),
           'requirements': self._get_development_requirements(property_data, 'maximum'),
           'extension_plan': extension_plan,
           'timeline': self._estimate_project_timeline(potential_area, 'extension')
       }

   def _estimate_development_cost(self, area: float, development_type: str) -> float:
       """Estimerer utviklingskostnader basert på areal og type utvikling"""
       # Grunnkostnad per kvadratmeter basert på utviklingstype
       base_costs = {
           'extension': 25000,        # Tilbygg
           'additional_floor': 28000, # Påbygg/ekstra etasje
           'basement_conversion': 15000,  # Kjellerkonvertering
           'attic_conversion': 18000,     # Loftskonvertering
           'outbuilding_conversion': 20000,  # Konvertering av uthus
           'renovation': 12000,       # Renovering
           'subdivision': 3000        # Tomtedeling (kostnad per m²)
       }
       
       base_cost_per_m2 = base_costs.get(development_type, 25000)
       
       # Justeringsfaktorer
       complexity_factor = 1.2  # 20% påslag for kompleksitet
       market_factor = 1.1  # 10% påslag for markedsforhold
       
       return area * base_cost_per_m2 * complexity_factor * market_factor
       
   def _estimate_development_value(self, area: float, municipality_code: str) -> float:
       """Estimerer verdiøkning basert på areal og kommune"""
       # Gjennomsnittlige boligpriser per m² basert på kommune
       avg_prices = {
           "0301": 85000,  # Oslo
           "3024": 70000,  # Bærum
           "4601": 55000,  # Bergen
           "5001": 50000,  # Trondheim
           "3005": 45000,  # Drammen
           "1103": 50000,  # Stavanger
           "4204": 40000,  # Kristiansand
       }
       
       price_per_m2 = avg_prices.get(municipality_code, 50000)  # Standard 50000 kr/m² hvis ukjent
       
       # Justeringsfaktor for nybygget areal vs. eksisterende
       new_build_premium = 1.2  # 20% premie for nytt areal
       
       return area * price_per_m2 * new_build_premium
   
   def _estimate_rental_income(self, area: float, property_data: Dict[str, Any]) -> Dict[str, float]:
       """Estimerer leieinntekter basert på areal og beliggenhet"""
       municipality_code = property_data.get('property_id', {}).get('municipality_code', '')
       
       # Gjennomsnittlige leiepriser per m² per måned basert på kommune
       avg_rental_prices = {
           "0301": 250,  # Oslo
           "3024": 220,  # Bærum
           "4601": 180,  # Bergen
           "5001": 170,  # Trondheim
           "3005": 160,  # Drammen
           "1103": 170,  # Stavanger
           "4204": 150,  # Kristiansand
       }
       
       rental_per_m2 = avg_rental_prices.get(municipality_code, 180)  # Standard 180 kr/m² hvis ukjent
       
       # Månedlig og årlig inntekt
       monthly = area * rental_per_m2
       annual = monthly * 12
       
       # Estimert avkastning
       cost = self._estimate_development_cost(area, 'extension')
       roi_percentage = (annual / cost) * 100 if cost > 0 else 0
       payback_years = cost / annual if annual > 0 else 0
       
       return {
           'monthly': monthly,
           'annual': annual,
           'roi_percentage': roi_percentage,
           'payback_years': payback_years
       }
   
   def _generate_extension_plan(
       self, 
       property_data: Dict[str, Any], 
       area: float, 
       objective: str
   ) -> Dict[str, Any]:
       """Genererer en detaljert plan for tilbygg"""
       # Dette ville være en kompleks funksjon i et virkelig system
       # Her er en forenklet implementasjon
       
       base_info = property_data.get('base_info', {})
       buildings = base_info.get('buildings', [])
       
       if not buildings:
           return {
               'type': 'unknown',
               'description': 'Kan ikke generere plan uten bygningsinformasjon'
           }
           
       main_building = buildings[0]
       
       # Bestem type tilbygg basert på mål
       if objective == 'rental_unit':
           return {
               'type': 'separate_unit',
               'description': f'Separat utleieenhet på {area:.1f} m²',
               'layout': [
                   {'room': 'Stue/kjøkken', 'area': area * 0.5},
                   {'room': 'Soverom', 'area': area * 0.25},
                   {'room': 'Bad', 'area': area * 0.15},
                   {'room': 'Gang', 'area': area * 0.1}
               ],
               'features': [
                   'Separat inngang',
                   'Eget bad og kjøkken',
                   'Brannskille mot hovedbolig'
               ]
           }
       else:
           return {
               'type': 'integrated_extension',
               'description': f'Integrert tilbygg på {area:.1f} m²',
               'layout': [
                   {'room': 'Utvidet stue', 'area': area * 0.7},
                   {'room': 'Ekstra soverom', 'area': area * 0.3}
               ],
               'features': [
                   'Integrert med eksisterende bolig',
                   'Utvidet boligareal',
                   'Forbedret planløsning'
               ]
           }
   
   def _identify_development_challenges(
       self, 
       property_data: Dict[str, Any], 
       scenario_type: str
   ) -> List[Dict[str, Any]]:
       """Identifiserer utfordringer ved utvikling"""
       challenges = []
       
       # Utfordringer for maksimal utnyttelse
       if scenario_type == 'maximum':
           # Sjekk avstand til nabogrense
           if property_data.get('base_info', {}).get('distance_to_neighbor', 0) < 4:
               challenges.append({
                   'type': 'boundary_distance',
                   'description': 'For liten avstand til nabogrense',
                   'severity': 'high',
                   'solution': 'Søk dispensasjon eller juster plassering'
               })
               
           # Sjekk for verneverdige forhold
           if property_data.get('base_info', {}).get('is_protected', False):
               challenges.append({
                   'type': 'conservation',
                   'description': 'Verneverdig bygning med begrensninger',
                   'severity': 'high',
                   'solution': 'Konsulter med kulturminnemyndigheter'
               })
               
           # Sjekk for grunnforhold
           if property_data.get('base_info', {}).get('soil_type', '') == 'clay':
               challenges.append({
                   'type': 'soil_condition',
                   'description': 'Utfordrende grunnforhold (leire)',
                   'severity': 'medium',
                   'solution': 'Geoteknisk undersøkelse anbefales'
               })
       
       # Utfordringer for kostnadseffektiv utvikling
       elif scenario_type == 'cost_effective':
           # Sjekk eksisterende infrastruktur
           if not property_data.get('base_info', {}).get('has_adequate_infrastructure', True):
               challenges.append({
                   'type': 'infrastructure',
                   'description': 'Utilstrekkelig infrastruktur',
                   'severity': 'medium',
                   'solution': 'Oppgradering av vann/avløp nødvendig'
               })
       
       # Generelle utfordringer
       if property_data.get('base_info', {}).get('is_in_flood_zone', False):
           challenges.append({
               'type': 'flood_risk',
               'description': 'Eiendommen ligger i flomsone',
               'severity': 'high',
               'solution': 'Flomsikringstiltak nødvendig'
           })
           
       return challenges
   
   def _get_development_requirements(
       self, 
       property_data: Dict[str, Any], 
       scenario_type: str
   ) -> List[Dict[str, Any]]:
       """Henter krav til utvikling"""
       requirements = []
       
       # Grunnleggende krav til alle utviklingstyper
       requirements.append({
           'type': 'application',
           'description': 'Søknad om tillatelse til tiltak',
           'authority': 'Kommunen',
           'timeline': '2-4 uker behandlingstid'
       })
       
       # Spesifikke krav basert på scenario
       if scenario_type == 'maximum':
           requirements.append({
               'type': 'neighbors',
               'description': 'Nabovarsel til alle berørte',
               'authority': 'Tiltakshaver',
               'timeline': 'Minimum 2 ukers varslingsfrist'
           })
           
           requirements.append({
               'type': 'detailed_plans',
               'description': 'Detaljerte byggeplaner og snitt-tegninger',
               'authority': 'Ansvarlig prosjekterende',
               'timeline': '2-6 uker å utarbeide'
           })
       
       # Krav basert på eiendomstype
       if property_data.get('base_info', {}).get('is_in_regulated_area', False):
           requirements.append({
               'type': 'zoning_compliance',
               'description': 'Tiltaket må være i samsvar med reguleringsplan',
               'authority': 'Kommunen',
               'timeline': 'Sjekkes ved søknad'
           })
           
       # Krav basert på bygningstype
       if property_data.get('base_info', {}).get('is_apartment_building', False):
           requirements.append({
               'type': 'owner_approval',
               'description': 'Godkjenning fra sameie/borettslag',
               'authority': 'Sameiet/borettslaget',
               'timeline': 'Varierer basert på vedtekter'
           })
           
       return requirements
       
   def _estimate_project_timeline(self, area: float, development_type: str) -> Dict[str, Any]:
       """Estimerer tidslinje for prosjektet"""
       # Basistid basert på prosjekttype
       base_times = {
           'extension': 12,        # Tilbygg: 12 uker
           'additional_floor': 16, # Påbygg: 16 uker
           'basement_conversion': 8,  # Kjellerkonvertering: 8 uker
           'attic_conversion': 10,     # Loftskonvertering: 10 uker
           'outbuilding_conversion': 6,  # Konvertering av uthus: 6 uker
           'renovation': 6,       # Renovering: 6 uker
           'subdivision': 20       # Tomtedeling: 20 uker (inkl. søknad)
       }
       
       base_weeks = base_times.get(development_type, 12)
       
       # Juster tid basert på størrelse
       size_factor = 1.0
       if area > 50:
           size_factor = 1.2
       if area > 100:
           size_factor = 1.5
           
       construction_weeks = int(base_weeks * size_factor)
       
       # Legg til tid for søknad og planlegging
       planning_weeks = 4
       application_weeks = 8
       
       return {
           'planning': planning_weeks,
           'application': application_weeks,
           'construction': construction_weeks,
           'total_weeks': planning_weeks + application_weeks + construction_weeks,
           'estimated_start_date': (datetime.now() + timedelta(weeks=planning_weeks + application_weeks)).strftime('%Y-%m-%d'),
           'estimated_completion_date': (datetime.now() + timedelta(weeks=planning_weeks + application_weeks + construction_weeks)).strftime('%Y-%m-%d')
       }
       
   async def _analyze_cost_effective_scenario(
       self,
       property_data: Dict[str, Any],
       objective: str,
       constraints: Dict[str, Any]
   ) -> Dict[str, Any]:
       """Analyserer scenario for kostnadseffektiv utvikling"""
       # Identifiser det mest kostnadseffektive alternativet
       base_info = property_data.get('base_info', {})
       buildings = base_info.get('buildings', [])
       
       if not buildings:
           return {'is_valid': False, 'reason': 'Mangler bygningsinformasjon'}
       
       main_building = buildings[0]
       
       # Identifiser potensialer som kan realiseres med minimal kostnad
       options = []
       
       # Sjekk for uinnredet loft
       if main_building.get('has_attic', False) and main_building.get('attic_finished', False) == False:
           attic_area = main_building.get('attic_area', 0)
           if attic_area >= 15:
               options.append({
                   'type': 'attic_conversion',
                   'description': f'Innredning av uinnredet loft ({attic_area:.1f} m²)',
                   'area': attic_area,
                   'estimated_cost': attic_area * 12000,  # Rimeligere siden det allerede er loft
                   'value_increase': attic_area * 40000,
                   'roi': (attic_area * 40000) / (attic_area * 12000)
               })
       
       # Sjekk for uinnredet kjeller
       if main_building.get('has_basement', False) and main_building.get('basement_finished', False) == False:
           basement_area = main_building.get('basement_area', 0)
           if basement_area >= 15:
               options.append({
                   'type': 'basement_conversion',
                   'description': f'Innredning av uinnredet kjeller ({basement_area:.1f} m²)',
                   'area': basement_area,
                   'estimated_cost': basement_area * 10000,
                   'value_increase': basement_area * 30000,
                   'roi': (basement_area * 30000) / (basement_area * 10000)
               })
       
       # Sjekk for garasje/uthus som kan konverteres
       for i, building in enumerate(buildings[1:], 1):
           if building.get('building_type', '').lower() in ['garage', 'outbuilding', 'annex']:
               building_area = building.get('area_BRA', 0)
               if building_area >= 15:
                   options.append({
                       'type': 'outbuilding_conversion',
                       'description': f'Konvertering av {building.get("building_type", "uthus")} ({building_area:.1f} m²)',
                       'area': building_area,
                       'estimated_cost': building_area * 15000,
                       'value_increase': building_area * 35000,
                       'roi': (building_area * 35000) / (building_area * 15000)
                   })
       
       # Hvis ingen spesifikke alternativer, vurder mindre tilbygg
       if not options and base_info.get('plot_size', 0) > 0:
           zoning = property_data.get('zoning', {})
           max_bya_percentage = zoning.get('max_bya_percentage', 0)
           
           if max_bya_percentage > 0:
               plot_size = base_info.get('plot_size', 0)
               max_bya = plot_size * max_bya_percentage / 100
               current_bya = base_info.get('current_bya', 0)
               
               potential_area = max_bya - current_bya
               
               if potential_area >= 15:
                   # Begrens til 25m² for kostnadseffektivitet
                   extension_area = min(potential_area, 25)
                   options.append({
                       'type': 'small_extension',
                       'description': f'Mindre tilbygg ({extension_area:.1f} m²)',
                       'area': extension_area,
                       'estimated_cost': extension_area * 22000,
                       'value_increase': extension_area * 45000,
                       'roi': (extension_area * 45000) / (extension_area * 22000)
                   })
       
       # Hvis ingen alternativer er funnet
       if not options:
           return {'is_valid': False, 'reason': 'Ingen kostnadseffektive alternativer identifisert'}
       
       # Velg alternativet med best ROI
       best_option = max(options, key=lambda x: x.get('roi', 0))
       
       # Sjekk mot budsjettbegrensning
       budget = constraints.get('budget', float('inf'))
       if best_option['estimated_cost'] > budget:
           return {'is_valid': False, 'reason': 'Best alternativ overstiger budsjettbegrensning'}
           
       # Tilpass til formålet (utleieenhet?)
       if objective == 'rental_unit' and best_option['area'] >= 25:
           rental_potential = self._estimate_rental_income(best_option['area'], property_data)
           best_option['rental_income'] = rental_potential
           best_option['description'] += f' som utleieenhet (est. {rental_potential["monthly"]:.0f} kr/mnd)'
       
       return {
           'is_valid': True,
           'type': 'cost_effective',
           'title': 'Kostnadseffektiv utvikling',
           'best_option': best_option,
           'other_options': [opt for opt in options if opt != best_option],
           'estimated_cost': best_option['estimated_cost'],
           'potential_value': best_option['value_increase'],
           'roi_percentage': (best_option['roi'] * 100),
           'challenges': self._identify_development_challenges(property_data, 'cost_effective'),
           'requirements': self._get_development_requirements(property_data, 'cost_effective'),
           'timeline': self._estimate_project_timeline(best_option['area'], best_option['type'])
       }
       
   async def analyze_rental_optimization_scenario(
    self,
    property_data: Dict[str, Any],
    constraints: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyserer scenario som maksimerer leieinntekter"""
    # Evaluer potensial for utleieenhet(er)
    rental_potential = self._evaluate_rental_unit_potential(property_data)
    
    if not rental_potential.get('feasible', False):
        return {'is_valid': False, 'reason': 'Ingen potensial for utleieenheter identifisert'}
        
    options = rental_potential.get('options', [])
    if not options:
        return {'is_valid': False, 'reason': 'Ingen utleiemuligheter funnet'}
        
    # Sjekk mot budsjettbegrensning
    budget = constraints.get('budget', float('inf'))
    viable_options = [opt for opt in options if opt.get('estimated_cost', float('inf')) <= budget]
    
    if not viable_options:
        return {'is_valid': False, 'reason': 'Ingen utleiemuligheter innenfor budsjett'}
        
    # Finn alternativet med høyest leieinntekt
    best_option = max(viable_options, key=lambda x: x.get('estimated_monthly_rent', 0))
    
    # Detaljert plan for utleieenhet
    rental_plan = self._generate_rental_unit_plan(property_data, best_option)
    
    return {
        'is_valid': True,
        'type': 'rental_optimization',
        'title': 'Optimal utleieløsning',
        'rental_unit': best_option,
        'other_options': [opt for opt in viable_options if opt != best_option],
        'estimated_cost': best_option.get('estimated_cost', 0),
        'monthly_income': best_option.get('estimated_monthly_rent', 0),
        'annual_income': best_option.get('estimated_monthly_rent', 0) * 12,
        'roi_percentage': self._calculate_rental_roi(
            best_option.get('estimated_cost', 0),
            best_option.get('estimated_monthly_rent', 0)
        ),
        'rental_plan': rental_plan,
        'challenges': self._identify_rental_challenges(property_data, best_option),
        'requirements': self._get_rental_requirements(property_data, best_option),
        'timeline': self._estimate_project_timeline(
            best_option.get('area', 0), 
            'rental_conversion'
        )
    }

async def analyze_energy_optimization_scenario(
    self,
    property_data: Dict[str, Any],
    constraints: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyserer scenario som maksimerer energieffektivitet"""
    # Hent nåværende energistatus
    current_energy = property_data.get('energy_rating', {})
    if not current_energy:
        # Hvis energidata mangler, beregn basert på byggeår og eiendomsinfo
        current_energy = await self._estimate_energy_profile(property_data)
    
    # Identifiser potensielle energiforbedringer
    energy_improvements = await self._identify_energy_improvements(property_data, current_energy)
    
    if not energy_improvements:
        return {'is_valid': False, 'reason': 'Ingen energiforbedringsmuligheter funnet'}
    
    # Filtrer basert på budsjett
    budget = constraints.get('budget', float('inf'))
    viable_improvements = [
        imp for imp in energy_improvements 
        if imp.get('estimated_cost', float('inf')) <= budget
    ]
    
    if not viable_improvements:
        return {'is_valid': False, 'reason': 'Ingen energiforbedringer innenfor budsjett'}
    
    # Grupper tiltak i pakker (f.eks. vinduer + isolasjon)
    improvement_packages = self._create_energy_improvement_packages(
        viable_improvements, 
        budget
    )
    
    # Finn den beste pakken basert på ROI og energiforbedring
    best_package = self._select_best_energy_package(improvement_packages)
    
    # Beregn subsidier (som Enova)
    subsidies = self._calculate_energy_subsidies(best_package, property_data)
    net_cost = best_package.get('total_cost', 0) - subsidies.get('total_amount', 0)
    
    return {
        'is_valid': True,
        'type': 'energy_optimization',
        'title': 'Energioptimalisering',
        'current_energy_rating': current_energy.get('rating', 'Ukjent'),
        'potential_energy_rating': best_package.get('new_energy_rating', 'Ukjent'),
        'energy_savings_kwh': best_package.get('annual_energy_savings', 0),
        'energy_savings_percentage': best_package.get('energy_savings_percentage', 0),
        'co2_reduction': best_package.get('annual_co2_reduction', 0),
        'improvements': best_package.get('improvements', []),
        'estimated_cost': best_package.get('total_cost', 0),
        'subsidies': subsidies,
        'net_cost': net_cost,
        'annual_savings': best_package.get('annual_cost_savings', 0),
        'roi_years': self._calculate_energy_roi_years(
            net_cost, 
            best_package.get('annual_cost_savings', 0)
        ),
        'challenges': self._identify_energy_challenges(property_data, best_package),
        'requirements': self._get_energy_requirements(property_data, best_package),
        'timeline': self._estimate_project_timeline(
            property_data.get('area', 0), 
            'energy_renovation'
        )
    }

def _evaluate_rental_unit_potential(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluerer potensial for utleieenheter basert på eiendomsdata"""
    result = {'feasible': False, 'options': []}
    
    # Sjekk om eiendommen er egnet for utleie
    property_type = property_data.get('property_type', '')
    total_area = property_data.get('area', 0)
    rooms = property_data.get('rooms', [])
    floor_count = property_data.get('floor_count', 1)
    
    # Sjekk reguleringsmessige begrensninger
    zoning_allows_rental = self._check_zoning_for_rental(property_data)
    if not zoning_allows_rental:
        return result
    
    # Vurder ulike utleiemuligheter basert på eiendomstype
    if property_type in ['enebolig', 'tomannsbolig', 'rekkehus']:
        # Vurder kjellerleilighet
        basement = next((r for r in rooms if r.get('type') == 'basement'), None)
        if basement and basement.get('area', 0) >= 30:
            result['options'].append(self._create_basement_rental_option(property_data, basement))
        
        # Vurder sokkelleilighet
        if floor_count > 1:
            ground_floor = self._get_ground_floor_data(property_data)
            if ground_floor and ground_floor.get('area', 0) >= 40:
                result['options'].append(self._create_ground_floor_rental_option(property_data, ground_floor))
        
        # Vurder loftsutbygging
        attic = next((r for r in rooms if r.get('type') == 'attic'), None)
        if attic and attic.get('area', 0) >= 25:
            result['options'].append(self._create_attic_rental_option(property_data, attic))
        
        # Vurder garasjekonvertering
        garage = next((r for r in rooms if r.get('type') == 'garage'), None)
        if garage and garage.get('area', 0) >= 20:
            result['options'].append(self._create_garage_conversion_option(property_data, garage))
    
    elif property_type in ['leilighet']:
        # Vurder romdeling for leiligheter
        if total_area >= 80:
            result['options'].append(self._create_room_sharing_option(property_data))
    
    # Vurder Airbnb-potensial for alle eiendomstyper
    if total_area >= 40:
        result['options'].append(self._create_airbnb_option(property_data))
    
    # Hvis minst én mulighet ble funnet, sett feasible til True
    if result['options']:
        result['feasible'] = True
    
    return result

def _create_basement_rental_option(self, property_data: Dict[str, Any], basement: Dict[str, Any]) -> Dict[str, Any]:
    """Oppretter et alternativ for kjellerleilighet"""
    area = basement.get('area', 0)
    need_renovation = basement.get('condition', 'poor') in ['poor', 'average']
    
    # Beregn kostnader basert på tilstand og størrelse
    base_cost_per_sqm = 8000 if need_renovation else 4000
    estimated_cost = area * base_cost_per_sqm
    
    # Tilleggskostnader for spesifikke oppgraderinger
    if basement.get('ceiling_height', 220) < 240:
        estimated_cost += area * 2000  # Kostnad for senking av gulv
    
    if not basement.get('separate_entrance', False):
        estimated_cost += 120000  # Kostnad for ny inngang
    
    if not basement.get('bathroom', False):
        estimated_cost += 150000  # Kostnad for nytt bad
    
    if not basement.get('kitchen', False):
        estimated_cost += 100000  # Kostnad for nytt kjøkken
    
    # Beregn potensiell leieinntekt basert på område og størrelse
    location_factor = self._get_location_rent_factor(property_data.get('location', {}))
    base_rent = area * 150 * location_factor
    
    return {
        'type': 'basement_apartment',
        'name': 'Kjellerleilighet',
        'area': area,
        'estimated_cost': estimated_cost,
        'estimated_monthly_rent': base_rent,
        'description': f'Konvertering av {area}m² kjeller til utleieleilighet',
        'features': self._get_basement_features(basement),
        'limitations': self._get_basement_limitations(basement),
        'upgrades_needed': self._get_basement_upgrades_needed(basement)
    }

def _generate_rental_unit_plan(self, property_data: Dict[str, Any], 
                               rental_option: Dict[str, Any]) -> Dict[str, Any]:
    """Genererer en detaljert plan for utleieenhet"""
    rental_type = rental_option.get('type', '')
    
    # Standardplan for alle utleietyper
    plan = {
        'floor_plan': self._generate_rental_floor_plan(property_data, rental_option),
        'legal_requirements': self._get_rental_legal_requirements(property_data, rental_type),
        'building_code_compliance': self._check_building_code_compliance(property_data, rental_type),
        'fire_safety': self._get_fire_safety_requirements(rental_type),
        'estimated_timeline': self._estimate_rental_conversion_timeline(rental_option),
        'contractor_requirements': self._get_contractor_requirements(rental_type),
        'permit_requirements': self._get_permit_requirements(property_data, rental_type)
    }
    
    # Legg til type-spesifikke detaljer
    if rental_type == 'basement_apartment':
        plan.update(self._get_basement_specific_plan(property_data, rental_option))
    elif rental_type == 'attic_apartment':
        plan.update(self._get_attic_specific_plan(property_data, rental_option))
    elif rental_type == 'ground_floor_apartment':
        plan.update(self._get_ground_floor_specific_plan(property_data, rental_option))
    elif rental_type == 'garage_conversion':
        plan.update(self._get_garage_specific_plan(property_data, rental_option))
    elif rental_type == 'airbnb':
        plan.update(self._get_airbnb_specific_plan(property_data, rental_option))
    
    return plan

def _calculate_rental_roi(self, cost: float, monthly_rent: float) -> float:
    """Beregner ROI for utleieenhet"""
    if cost <= 0:
        return 0
    
    annual_rent = monthly_rent * 12
    # Trekk fra estimert årlige kostnader (vedlikehold, forsikring, etc.)
    maintenance_cost = annual_rent * 0.1
    annual_profit = annual_rent - maintenance_cost
    
    return (annual_profit / cost) * 100  # Returner ROI som prosent

def _identify_rental_challenges(self, property_data: Dict[str, Any], 
                               rental_option: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identifiserer utfordringer med utleieenhet"""
    challenges = []
    rental_type = rental_option.get('type', '')
    
    # Generelle utfordringer for alle utleieenheter
    if property_data.get('year_built', 2020) < 1990:
        challenges.append({
            'type': 'renovation',
            'description': 'Eldre bygning kan kreve omfattende oppgraderinger for å møte moderne standarder',
            'severity': 'medium'
        })
    
    # Type-spesifikke utfordringer
    if rental_type == 'basement_apartment':
        if property_data.get('location', {}).get('flood_risk', 'low') != 'low':
            challenges.append({
                'type': 'flood_risk',
                'description': 'Økt risiko for fuktproblemer i kjeller',
                'severity': 'high'
            })
            
        ceiling_height = rental_option.get('features', {}).get('ceiling_height', 0)
        if ceiling_height < 240:
            challenges.append({
                'type': 'ceiling_height',
                'description': f'Lav takhøyde ({ceiling_height}cm) under byggeforskriftens krav',
                'severity': 'high'
            })
    
    elif rental_type == 'attic_apartment':
        if not property_data.get('roof', {}).get('insulated', False):
            challenges.append({
                'type': 'insulation',
                'description': 'Manglende isolasjon i tak vil kreve omfattende arbeid',
                'severity': 'medium'
            })
    
    return challenges

def _get_rental_requirements(self, property_data: Dict[str, Any], 
                            rental_option: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Henter krav for utleieenhet"""
    requirements = []
    rental_type = rental_option.get('type', '')
    
    # Generelle krav for alle utleieenheter
    requirements.append({
        'type': 'building_permit',
        'description': 'Søknad om bruksendring til kommunen',
        'estimated_cost': 15000,
        'estimated_time': '4-8 uker'
    })
    
    requirements.append({
        'type': 'fire_safety',
        'description': 'Brannsikring med røykvarslere og rømningsvei',
        'estimated_cost': 25000,
        'estimated_time': '1-2 uker'
    })
    
    # Type-spesifikke krav
    if rental_type == 'basement_apartment':
        if not rental_option.get('features', {}).get('separate_entrance', False):
            requirements.append({
                'type': 'separate_entrance',
                'description': 'Etablering av separat inngang for utleieenhet',
                'estimated_cost': 120000,
                'estimated_time': '2-3 uker'
            })
    
    elif rental_type == 'attic_apartment':
        requirements.append({
            'type': 'structural_assessment',
            'description': 'Strukturell vurdering av tak og bjelkelag',
            'estimated_cost': 15000,
            'estimated_time': '1-2 uker'
        })
    
    return requirements

def _estimate_project_timeline(self, area: float, project_type: str) -> Dict[str, Any]:
    """Estimerer tidslinjen for prosjektet"""
    timeline = {
        'phases': [],
        'total_weeks': 0
    }
    
    if project_type == 'rental_conversion':
        # Faser for utleiekonvertering
        planning_weeks = 2
        permit_weeks = 6
        construction_weeks = max(4, int(area / 20))
        inspection_weeks = 2
        
        timeline['phases'] = [
            {'name': 'Planlegging og design', 'duration_weeks': planning_weeks},
            {'name': 'Byggetillatelse', 'duration_weeks': permit_weeks},
            {'name': 'Konstruksjon', 'duration_weeks': construction_weeks},
            {'name': 'Inspeksjon og godkjenning', 'duration_weeks': inspection_weeks}
        ]
        
        timeline['total_weeks'] = planning_weeks + permit_weeks + construction_weeks + inspection_weeks
    
    elif project_type == 'energy_renovation':
        # Faser for energirenovering
        planning_weeks = 1
        construction_weeks = max(2, int(area / 40))
        inspection_weeks = 1
        
        timeline['phases'] = [
            {'name': 'Planlegging', 'duration_weeks': planning_weeks},
            {'name': 'Renovering', 'duration_weeks': construction_weeks},
            {'name': 'Inspeksjon og testing', 'duration_weeks': inspection_weeks}
        ]
        
        timeline['total_weeks'] = planning_weeks + construction_weeks + inspection_weeks
    
    # Legg til estimert start- og sluttdato
    from datetime import datetime, timedelta
    start_date = datetime.now() + timedelta(weeks=2)  # Antar oppstart om 2 uker
    end_date = start_date + timedelta(weeks=timeline['total_weeks'])
    
    timeline['estimated_start_date'] = start_date.strftime('%Y-%m-%d')
    timeline['estimated_end_date'] = end_date.strftime('%Y-%m-%d')
    
    return timeline
