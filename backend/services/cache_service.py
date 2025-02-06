"""
Cache service for ytelsesoptimalisering
"""
from typing import Any, Optional, Union
import redis
from fastapi import Depends
import json
import logging
from datetime import timedelta
import pickle

logger = logging.getLogger(__name__)

class CacheService:
    """
    Håndterer caching for bedre ytelse og redusert belastning på eksterne tjenester
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = redis.from_url(redis_url)
        self.default_ttl = timedelta(hours=24)
        
    async def get(self, key: str) -> Optional[Any]:
        """Hent verdi fra cache"""
        try:
            value = self.redis.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.error(f"Feil ved henting fra cache: {str(e)}")
            return None
            
    async def set(self, 
                 key: str, 
                 value: Any, 
                 ttl: Optional[timedelta] = None) -> bool:
        """Sett verdi i cache"""
        try:
            ttl = ttl or self.default_ttl
            pickled_value = pickle.dumps(value)
            return self.redis.setex(key, ttl, pickled_value)
        except Exception as e:
            logger.error(f"Feil ved setting i cache: {str(e)}")
            return False
            
    async def delete(self, key: str) -> bool:
        """Slett verdi fra cache"""
        try:
            return bool(self.redis.delete(key))
        except Exception as e:
            logger.error(f"Feil ved sletting fra cache: {str(e)}")
            return False
            
    async def clear_all(self) -> bool:
        """Tøm hele cachen"""
        try:
            return self.redis.flushall()
        except Exception as e:
            logger.error(f"Feil ved tømming av cache: {str(e)}")
            return False
            
class CacheKey:
    """Nøkkelgenerator for cache"""
    
    @staticmethod
    def property_analysis(property_id: int) -> str:
        return f"property:analysis:{property_id}"
        
    @staticmethod
    def municipality_regulations(municipality: str) -> str:
        return f"municipality:regulations:{municipality}"
        
    @staticmethod
    def user_profile(user_id: int) -> str:
        return f"user:profile:{user_id}"
        
    @staticmethod
    def floor_plan_analysis(plan_id: int) -> str:
        return f"floor_plan:analysis:{plan_id}"
        
class CachedResponse:
    """Wrapper for cachede responser"""
    
    def __init__(self, data: Any, timestamp: float):
        self.data = data
        self.timestamp = timestamp
        
class CacheDecorator:
    """Dekoratør for enkel caching av funksjoner"""
    
    def __init__(self, 
                 cache_service: CacheService,
                 ttl: Optional[timedelta] = None):
        self.cache_service = cache_service
        self.ttl = ttl
        
    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            # Generer cache-nøkkel
            key = f"{func.__name__}:{args}:{kwargs}"
            
            # Prøv å hente fra cache
            cached = await self.cache_service.get(key)
            if cached:
                return cached
                
            # Hvis ikke i cache, utfør funksjon
            result = await func(*args, **kwargs)
            
            # Lagre i cache
            await self.cache_service.set(key, result, self.ttl)
            
            return result
        return wrapper
        
class QueryCache:
    """Håndterer caching av databasespørringer"""
    
    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
        
    async def cache_query(self, 
                        query: str, 
                        params: tuple, 
                        result: Any,
                        ttl: Optional[timedelta] = None):
        """Cache en spørring og dens resultat"""
        key = self._generate_query_key(query, params)
        await self.cache_service.set(key, result, ttl)
        
    async def get_cached_query(self, 
                             query: str, 
                             params: tuple) -> Optional[Any]:
        """Hent cachet spørringsresultat"""
        key = self._generate_query_key(query, params)
        return await self.cache_service.get(key)
        
    def _generate_query_key(self, query: str, params: tuple) -> str:
        """Generer unik nøkkel for spørring"""
        return f"query:{hash(query)}:{hash(params)}"
        
class ModelCache:
    """Håndterer caching av 3D-modeller og visualiseringer"""
    
    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
        self.model_ttl = timedelta(days=7)  # Modeller caches i 7 dager
        
    async def cache_model(self, 
                        model_id: str, 
                        model_data: dict,
                        ttl: Optional[timedelta] = None):
        """Cache en 3D-modell"""
        key = f"model:{model_id}"
        await self.cache_service.set(key, model_data, ttl or self.model_ttl)
        
    async def get_cached_model(self, model_id: str) -> Optional[dict]:
        """Hent cachet 3D-modell"""
        key = f"model:{model_id}"
        return await self.cache_service.get(key)
        
    async def update_model_cache(self, 
                              model_id: str, 
                              updates: dict) -> bool:
        """Oppdater en cachet modell"""
        key = f"model:{model_id}"
        current_model = await self.get_cached_model(model_id)
        
        if not current_model:
            return False
            
        # Oppdater modellen
        current_model.update(updates)
        
        # Lagre oppdatert modell
        return await self.cache_model(model_id, current_model)
        
class RateLimiter:
    """Håndterer rate limiting"""
    
    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
        
    async def check_rate_limit(self, 
                             key: str, 
                             limit: int, 
                             window: timedelta) -> bool:
        """Sjekk om en forespørsel er innenfor rate limit"""
        current = await self.get_current_count(key)
        
        if current >= limit:
            return False
            
        await self.increment_count(key, window)
        return True
        
    async def get_current_count(self, key: str) -> int:
        """Hent nåværende antall forespørsler"""
        count = await self.cache_service.get(f"rate:{key}")
        return count or 0
        
    async def increment_count(self, 
                            key: str, 
                            window: timedelta) -> int:
        """Øk telleren for en nøkkel"""
        rate_key = f"rate:{key}"
        
        pipeline = self.redis.pipeline()
        pipeline.incr(rate_key)
        pipeline.expire(rate_key, int(window.total_seconds()))
        
        result = await pipeline.execute()
        return result[0]