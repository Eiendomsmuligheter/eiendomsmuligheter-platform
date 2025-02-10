from redis import asyncio as aioredis
from typing import Any, Optional
import pickle
import json
from datetime import timedelta
import os

class RedisCache:
    """
    Redis-basert caching system for å forbedre ytelsen
    av API-kall og database-spørringer
    """
    
    def __init__(self):
        self.redis = aioredis.from_url(
            os.getenv('REDIS_URL', 'redis://localhost:6379'),
            encoding='utf-8',
            decode_responses=False
        )
        
    async def get(self, key: str) -> Optional[Any]:
        """Hent verdi fra cache"""
        try:
            value = await self.redis.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            print(f"Cache get error: {str(e)}")
            return None
            
    async def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None
    ) -> bool:
        """
        Sett verdi i cache
        expire: Utløpstid i sekunder
        """
        try:
            pickled_value = pickle.dumps(value)
            if expire:
                await self.redis.setex(key, expire, pickled_value)
            else:
                await self.redis.set(key, pickled_value)
            return True
        except Exception as e:
            print(f"Cache set error: {str(e)}")
            return False
            
    async def delete(self, key: str) -> bool:
        """Slett nøkkel fra cache"""
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            print(f"Cache delete error: {str(e)}")
            return False
            
    async def exists(self, key: str) -> bool:
        """Sjekk om nøkkel eksisterer i cache"""
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            print(f"Cache exists error: {str(e)}")
            return False
            
    async def clear(self) -> bool:
        """Tøm hele cachen"""
        try:
            await self.redis.flushdb()
            return True
        except Exception as e:
            print(f"Cache clear error: {str(e)}")
            return False
            
    # Spesialiserte cache-metoder for eiendomsanalyse
    
    async def cache_property_analysis(
        self,
        property_id: str,
        analysis_data: dict,
        expire: int = 3600
    ) -> bool:
        """Cache eiendomsanalyse-resultater"""
        key = f"property_analysis:{property_id}"
        return await self.set(key, analysis_data, expire)
        
    async def get_property_analysis(
        self,
        property_id: str
    ) -> Optional[dict]:
        """Hent cached eiendomsanalyse"""
        key = f"property_analysis:{property_id}"
        return await self.get(key)
        
    async def cache_municipality_rules(
        self,
        municipality_id: str,
        rules_data: dict,
        expire: int = 86400  # 24 timer
    ) -> bool:
        """Cache kommunale regler og forskrifter"""
        key = f"municipality_rules:{municipality_id}"
        return await self.set(key, rules_data, expire)
        
    async def get_municipality_rules(
        self,
        municipality_id: str
    ) -> Optional[dict]:
        """Hent cached kommunale regler"""
        key = f"municipality_rules:{municipality_id}"
        return await self.get(key)
        
    async def cache_regulation_plan(
        self,
        plan_id: str,
        plan_data: dict,
        expire: int = 86400  # 24 timer
    ) -> bool:
        """Cache reguleringsplan"""
        key = f"regulation_plan:{plan_id}"
        return await self.set(key, plan_data, expire)
        
    async def get_regulation_plan(
        self,
        plan_id: str
    ) -> Optional[dict]:
        """Hent cached reguleringsplan"""
        key = f"regulation_plan:{plan_id}"
        return await self.get(key)
        
    # Cache metrics og overvåkning
    
    async def get_cache_stats(self) -> dict:
        """Hent cache-statistikk"""
        try:
            info = await self.redis.info()
            return {
                'hits': info.get('keyspace_hits', 0),
                'misses': info.get('keyspace_misses', 0),
                'memory_used': info.get('used_memory_human', '0B'),
                'total_keys': info.get('db0', {}).get('keys', 0)
            }
        except Exception as e:
            print(f"Cache stats error: {str(e)}")
            return {}