from typing import Dict, Any, List, Optional
import aiohttp
import asyncio
import json
import logging
from datetime import datetime, timedelta
import redis
from bs4 import BeautifulSoup
import re
from dataclasses import dataclass
import hashlib
from aiohttp import ClientTimeout
from concurrent.futures import ThreadPoolExecutor

@dataclass
class CacheConfig:
    """Konfigurasjon for caching"""
    redis_url: str = "redis://localhost"
    default_ttl: int = 3600  # 1 time
    regulations_ttl: int = 86400  # 24 timer
    property_ttl: int = 43200  # 12 timer
    max_parallel_requests: int = 10

class AdvancedMunicipalityService:
    def __init__(self, cache_config: Optional[CacheConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.cache_config = cache_config or CacheConfig()
        self.redis_client = redis.Redis.from_url(
            self.cache_config.redis_url,
            decode_responses=True
        )
        self.session_pool = aiohttp.ClientSession(
            timeout=ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=self.cache_config.max_parallel_requests)
        )
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def get_property_details(
        self,
        address: str,
        municipality_code: str
    ) -> Dict[str, Any]:
        """Henter detaljert informasjon om eiendom"""
        cache_key = f"property:{municipality_code}:{address}"
        
        # Sjekk cache først
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        # Parallell innhenting av data
        tasks = [
            self._get_property_base_info(address, municipality_code),
            self._get_property_regulations(address, municipality_code),
            self._get_property_history(address, municipality_code),
            self._get_zoning_details(address, municipality_code),
            self._get_building_permits(address, municipality_code)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Kombiner resultatene
        property_data = {
            'base_info': results[0] if not isinstance(results[0], Exception) else {},
            'regulations': results[1] if not isinstance(results[1], Exception) else [],
            'history': results[2] if not isinstance(results[2], Exception) else [],
            'zoning': results[3] if not isinstance(results[3], Exception) else {},
            'permits': results[4] if not isinstance(results[4], Exception) else []
        }
        
        # Cache resultatet
        await self._cache_data(
            cache_key,
            property_data,
            self.cache_config.property_ttl
        )
        
        return property_data

    async def analyze_development_potential(
        self,
        property_data: Dict[str, Any],
        municipality_code: str
    ) -> Dict[str, Any]:
        """Analyserer utviklingspotensial basert på kommunale regler"""
        # Parallell analyse av ulike aspekter
        tasks = [
            self._analyze_zoning_potential(property_data, municipality_code),
            self._analyze_building_restrictions(property_data, municipality_code),
            self._analyze_historical_precedents(property_data, municipality_code),
            self._analyze_infrastructure_requirements(property_data, municipality_code)
        ]
        
        results = await asyncio.gather(*tasks)
        
        zoning_potential, restrictions, precedents, infrastructure = results
        
        return {
            'zoning_analysis': zoning_potential,
            'building_restrictions': restrictions,
            'historical_precedents': precedents,
            'infrastructure_requirements': infrastructure,
            'recommended_approaches': self._generate_development_recommendations(
                results
            ),
            'estimated_timeline': self._estimate_development_timeline(
                results
            ),
            'risk_assessment': self._assess_development_risks(
                results
            )
        }

    async def _get_property_base_info(
        self,
        address: str,
        municipality_code: str
    ) -> Dict[str, Any]:
        """Henter grunnleggende eiendomsinformasjon"""
        cache_key = f"base_info:{municipality_code}:{address}"
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        async with self.session_pool.get(
            f"https://ws.geonorge.no/eiendom/v1/sok",
            params={'adresse': address}
        ) as response:
            if response.status == 200:
                data = await response.json()
                result = self._process_property_base_info(data)
                await self._cache_data(
                    cache_key,
                    result,
                    self.cache_config.default_ttl
                )
                return result
            raise Exception(f"Kunne ikke hente eiendomsinfo: {response.status}")

    async def _analyze_zoning_potential(
        self,
        property_data: Dict[str, Any],
        municipality_code: str
    ) -> Dict[str, Any]:
        """Analyserer potensial basert på regulering"""
        current_zoning = property_data.get('zoning', {})
        
        # Beregn utnyttelsesgrad
        max_bya = current_zoning.get('max_bya', 0)
        current_bya = property_data.get('base_info', {}).get('current_bya', 0)
        
        # Beregn høydebegrensninger
        max_height = current_zoning.get('max_height', 0)
        current_height = property_data.get('base_info', {}).get('height', 0)
        
        # Analyser muligheter
        potential_uses = await self._get_potential_uses(
            municipality_code,
            current_zoning.get('zone_type')
        )
        
        return {
            'unutilized_bya': max_bya - current_bya if max_bya > 0 else 0,
            'height_potential': max_height - current_height if max_height > 0 else 0,
            'potential_uses': potential_uses,
            'zoning_restrictions': current_zoning.get('restrictions', []),
            'development_scenarios': await self._generate_development_scenarios(
                property_data,
                municipality_code
            )
        }

    async def _generate_development_scenarios(
        self,
        property_data: Dict[str, Any],
        municipality_code: str
    ) -> List[Dict[str, Any]]:
        """Genererer ulike utviklingsscenarier"""
        scenarios = []
        
        # Scenario 1: Maksimal utnyttelse
        scenarios.append(await self._analyze_max_utilization_scenario(
            property_data,
            municipality_code
        ))
        
        # Scenario 2: Kostnadseffektiv utvikling
        scenarios.append(await self._analyze_cost_effective_scenario(
            property_data,
            municipality_code
        ))
        
        # Scenario 3: Bærekraftig utvikling
        scenarios.append(await self._analyze_sustainable_scenario(
            property_data,
            municipality_code
        ))
        
        return scenarios

    async def _analyze_max_utilization_scenario(
        self,
        property_data: Dict[str, Any],
        municipality_code: str
    ) -> Dict[str, Any]:
        """Analyserer scenario for maksimal utnyttelse"""
        zoning = property_data.get('zoning', {})
        base_info = property_data.get('base_info', {})
        
        max_bya = zoning.get('max_bya', 0)
        plot_size = base_info.get('plot_size', 0)
        
        potential_area = (plot_size * max_bya / 100) - base_info.get('current_bya', 0)
        
        return {
            'type': 'maximum_utilization',
            'potential_new_area': potential_area,
            'estimated_cost': self._estimate_development_cost(potential_area),
            'potential_value': self._estimate_development_value(
                potential_area,
                municipality_code
            ),
            'challenges': self._identify_development_challenges(
                property_data,
                'maximum'
            ),
            'requirements': self._get_development_requirements(
                property_data,
                'maximum'
            )
        }

    def _estimate_development_cost(self, area: float) -> float:
        """Estimerer utviklingskostnader"""
        # Grunnkostnad per kvadratmeter
        base_cost_per_m2 = 25000
        
        # Justeringsfaktorer
        complexity_factor = 1.2  # 20% påslag for kompleksitet
        market_factor = 1.1  # 10% påslag for markedsforhold
        
        return area * base_cost_per_m2 * complexity_factor * market_factor

    async def _get_potential_uses(
        self,
        municipality_code: str,
        zone_type: str
    ) -> List[Dict[str, Any]]:
        """Henter potensielle bruksområder basert på sone"""
        cache_key = f"potential_uses:{municipality_code}:{zone_type}"
        
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        # I en virkelig implementasjon ville dette kalle kommunens API
        # Her returnerer vi predefinerte muligheter basert på sonetype
        uses = []
        
        if zone_type == 'residential':
            uses = [
                {
                    'type': 'bolig',
                    'subtypes': ['enebolig', 'tomannsbolig', 'rekkehus'],
                    'requirements': ['minimum tomtestørrelse', 'parkeringskrav']
                },
                {
                    'type': 'utleie',
                    'subtypes': ['hybel', 'sokkelleilighet'],
                    'requirements': ['egen inngang', 'brannskille']
                }
            ]
        elif zone_type == 'mixed':
            uses = [
                {
                    'type': 'næring',
                    'subtypes': ['kontor', 'butikk', 'verksted'],
                    'requirements': ['universell utforming', 'ventilasjon']
                },
                {
                    'type': 'kombinert',
                    'subtypes': ['bolig/næring', 'bolig/kontor'],
                    'requirements': ['støyskjerming', 'separate innganger']
                }
            ]
        
        await self._cache_data(
            cache_key,
            uses,
            self.cache_config.default_ttl
        )
        return uses

    async def _get_cached_data(self, key: str) -> Optional[Dict[str, Any]]:
        """Henter data fra cache"""
        try:
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            self.logger.error(f"Cache error: {str(e)}")
            return None

    async def _cache_data(
        self,
        key: str,
        data: Dict[str, Any],
        ttl: int
    ) -> None:
        """Lagrer data i cache"""
        try:
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(data)
            )
        except Exception as e:
            self.logger.error(f"Cache error: {str(e)}")

    def _generate_development_recommendations(
        self,
        analysis_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Genererer utviklingsanbefalinger basert på analyseresultater"""
        recommendations = []
        
        zoning_analysis = analysis_results[0]
        restrictions = analysis_results[1]
        precedents = analysis_results[2]
        infrastructure = analysis_results[3]
        
        # Analyser muligheter basert på utnyttelsesgrad
        if zoning_analysis.get('unutilized_bya', 0) > 0:
            recommendations.append({
                'type': 'expansion',
                'description': 'Potensial for utvidelse',
                'details': self._generate_expansion_details(zoning_analysis),
                'priority': 'high' if zoning_analysis['unutilized_bya'] > 50 else 'medium'
            })
            
        # Vurder ombygging basert på historiske presedens
        if len(precedents.get('similar_projects', [])) > 0:
            recommendations.append({
                'type': 'conversion',
                'description': 'Mulighet for ombygging',
                'details': self._generate_conversion_details(precedents),
                'priority': 'medium'
            })
            
        # Infrastrukturbehov
        if infrastructure.get('upgrades_needed', []):
            recommendations.append({
                'type': 'infrastructure',
                'description': 'Nødvendige infrastrukturtiltak',
                'details': infrastructure['upgrades_needed'],
                'priority': 'high'
            })
            
        return self._prioritize_recommendations(recommendations)

    def _prioritize_recommendations(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prioriterer anbefalinger basert på gjennomførbarhet og verdi"""
        # Implementer prioriteringslogikk
        return sorted(
            recommendations,
            key=lambda x: (
                {'high': 0, 'medium': 1, 'low': 2}[x['priority']],
                -len(x['details'])
            )
        )