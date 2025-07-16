"""
Caching Infrastructure for MLBB Coach AI
Provides multiple caching strategies for optimal performance
"""

from .cache_manager import CacheManager
from .redis_cache import RedisCache
from .memory_cache import MemoryCache
from .hybrid_cache import HybridCache
from .decorators import cache_result, cache_async

# Import metrics collector from observability
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..observability.metrics_collector import get_metrics_collector
except ImportError:
    # Fallback for when running from different contexts
    def get_metrics_collector():
        """Stub metrics collector"""
        class StubCollector:
            def increment(self, *args, **kwargs): pass
            def timing(self, *args, **kwargs): pass
            def gauge(self, *args, **kwargs): pass
        return StubCollector()

__all__ = [
    'CacheManager',
    'RedisCache',
    'MemoryCache',
    'HybridCache',
    'cache_result',
    'cache_async',
    'get_metrics_collector'
] 