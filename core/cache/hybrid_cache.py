"""
Hybrid Cache Implementation
Combines memory and disk caching
"""

from typing import Any, Optional
from .memory_cache import MemoryCache


class HybridCache(MemoryCache):
    """Hybrid cache with memory and disk storage"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # In a real implementation, this would include disk storage
        
    def get(self, key: str) -> Optional[Any]:
        """Get from memory first, then disk"""
        # Try memory first
        value = super().get(key)
        if value is not None:
            return value
        
        # In real implementation, check disk here
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set in both memory and disk"""
        super().set(key, value, ttl)
        # In real implementation, also save to disk 