"""
Cache Manager
Manages different cache implementations
"""

from typing import Any, Optional, Dict
from .memory_cache import MemoryCache
from .redis_cache import RedisCache
from .hybrid_cache import HybridCache


class CacheManager:
    """Manages multiple cache implementations"""
    
    def __init__(self):
        self.caches: Dict[str, Any] = {
            "memory": MemoryCache(),
            "redis": RedisCache(),
            "hybrid": HybridCache()
        }
        self.default_cache = "memory"
    
    def get_cache(self, cache_type: Optional[str] = None):
        """Get a specific cache implementation"""
        cache_type = cache_type or self.default_cache
        return self.caches.get(cache_type)
    
    def set_default(self, cache_type: str):
        """Set default cache type"""
        if cache_type in self.caches:
            self.default_cache = cache_type
    
    def get(self, key: str, cache_type: Optional[str] = None) -> Optional[Any]:
        """Get from cache"""
        cache = self.get_cache(cache_type)
        return cache.get(key) if cache else None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        cache_type: Optional[str] = None
    ):
        """Set in cache"""
        cache = self.get_cache(cache_type)
        if cache:
            cache.set(key, value, ttl)
    
    def clear_all(self):
        """Clear all caches"""
        for cache in self.caches.values():
            cache.clear() 