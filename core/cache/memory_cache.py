"""
Memory Cache Implementation
In-memory caching with LRU eviction and TTL support
"""

import time
import logging
from typing import Any, Optional, Dict
from collections import OrderedDict
from threading import Lock
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    value: Any
    created_at: float
    ttl: Optional[int]
    access_count: int = 0
    last_accessed: float = 0
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def access(self):
        """Update access metadata"""
        self.access_count += 1
        self.last_accessed = time.time()


class MemoryCache:
    """Thread-safe in-memory cache with LRU eviction"""
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[int] = 3600,
        eviction_policy: str = "lru"
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                self.stats["misses"] += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                self.stats["expired"] += 1
                del self._cache[key]
                return None
            
            # Update access info and move to end (most recent)
            entry.access()
            self._cache.move_to_end(key)
            
            self.stats["hits"] += 1
            return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache"""
        with self._lock:
            # Use provided TTL or default
            entry_ttl = ttl if ttl is not None else self.default_ttl
            
            # Check if we need to evict
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict()
            
            # Create and store entry
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=entry_ttl,
                last_accessed=time.time()
            )
            
            self._cache[key] = entry
            self._cache.move_to_end(key)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")
    
    def _evict(self):
        """Evict entries based on policy"""
        if self.eviction_policy == "lru":
            # Remove least recently used (first item)
            if self._cache:
                evicted_key = next(iter(self._cache))
                del self._cache[evicted_key]
                self.stats["evictions"] += 1
                logger.debug(f"Evicted key: {evicted_key}")
        
        elif self.eviction_policy == "lfu":
            # Remove least frequently used
            if self._cache:
                min_key = min(
                    self._cache.items(),
                    key=lambda x: x[1].access_count
                )[0]
                del self._cache[min_key]
                self.stats["evictions"] += 1
        
        elif self.eviction_policy == "fifo":
            # Remove first in (oldest)
            if self._cache:
                evicted_key = next(iter(self._cache))
                del self._cache[evicted_key]
                self.stats["evictions"] += 1
    
    def cleanup_expired(self):
        """Remove all expired entries"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
                self.stats["expired"] += 1
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (
                self.stats["hits"] / total_requests 
                if total_requests > 0 else 0
            )
            
            return {
                **self.stats,
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                "eviction_policy": self.eviction_policy
            }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Estimate memory usage"""
        import sys
        
        with self._lock:
            total_size = 0
            for key, entry in self._cache.items():
                # Estimate size of key and value
                key_size = sys.getsizeof(key)
                value_size = sys.getsizeof(entry.value)
                metadata_size = sys.getsizeof(entry)
                total_size += key_size + value_size + metadata_size
            
            return {
                "total_bytes": total_size,
                "total_mb": total_size / (1024 * 1024),
                "entries": len(self._cache),
                "avg_entry_size": (
                    total_size / len(self._cache) 
                    if self._cache else 0
                )
            }
    
    def warmup(self, entries: Dict[str, Any]):
        """Pre-populate cache with entries"""
        with self._lock:
            for key, value in entries.items():
                self.set(key, value)
            logger.info(f"Cache warmed up with {len(entries)} entries") 