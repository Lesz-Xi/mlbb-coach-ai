"""
Cache Decorators
Easy-to-use decorators for caching function results
"""

import functools
import hashlib
import json
import logging
from typing import Callable, Optional

from .memory_cache import MemoryCache

logger = logging.getLogger(__name__)

# Global cache instance (can be configured)
_global_cache = MemoryCache(max_size=5000, default_ttl=3600)


def get_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate cache key from function name and arguments"""
    # Create a dictionary of all arguments
    key_data = {
        "func": func_name,
        "args": args,
        "kwargs": kwargs
    }
    
    # Convert to JSON and hash
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    key_hash = hashlib.md5(key_str.encode()).hexdigest()
    
    return f"{func_name}:{key_hash}"


def cache_result(
    ttl: Optional[int] = None,
    cache_instance: Optional[MemoryCache] = None,
    key_prefix: str = "",
    ignore_args: Optional[list] = None
):
    """
    Decorator to cache function results
    
    Args:
        ttl: Time to live in seconds
        cache_instance: Cache instance to use (defaults to global)
        key_prefix: Prefix for cache keys
        ignore_args: List of argument positions to ignore in cache key
    """
    def decorator(func: Callable) -> Callable:
        cache = cache_instance or _global_cache
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Filter out ignored arguments
            if ignore_args:
                cache_args = tuple(
                    arg for i, arg in enumerate(args) 
                    if i not in ignore_args
                )
            else:
                cache_args = args
            
            # Generate cache key
            cache_key = get_cache_key(
                f"{key_prefix}{func.__name__}",
                cache_args,
                kwargs
            )
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value
            
            # Execute function
            logger.debug(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        # Add cache control methods
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_stats = lambda: cache.get_stats()
        
        return wrapper
    
    return decorator


def cache_async(
    ttl: Optional[int] = None,
    cache_instance: Optional[MemoryCache] = None,
    key_prefix: str = "",
    ignore_args: Optional[list] = None
):
    """
    Decorator to cache async function results
    
    Args:
        ttl: Time to live in seconds
        cache_instance: Cache instance to use (defaults to global)
        key_prefix: Prefix for cache keys
        ignore_args: List of argument positions to ignore in cache key
    """
    def decorator(func: Callable) -> Callable:
        cache = cache_instance or _global_cache
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Filter out ignored arguments
            if ignore_args:
                cache_args = tuple(
                    arg for i, arg in enumerate(args) 
                    if i not in ignore_args
                )
            else:
                cache_args = args
            
            # Generate cache key
            cache_key = get_cache_key(
                f"{key_prefix}{func.__name__}",
                cache_args,
                kwargs
            )
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value
            
            # Execute function
            logger.debug(f"Cache miss for {func.__name__}")
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        # Add cache control methods
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_stats = lambda: cache.get_stats()
        
        return wrapper
    
    return decorator


class CachedProperty:
    """
    Cached property descriptor for class properties
    """
    def __init__(self, func: Callable, ttl: Optional[int] = None):
        self.func = func
        self.ttl = ttl
        self.__doc__ = func.__doc__
        self.cache = MemoryCache(max_size=100, default_ttl=ttl)
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        
        # Generate cache key based on object id and property name
        cache_key = f"{id(obj)}:{self.func.__name__}"
        
        # Try to get from cache
        cached_value = self.cache.get(cache_key)
        if cached_value is not None:
            return cached_value
        
        # Compute value
        value = self.func(obj)
        
        # Store in cache
        self.cache.set(cache_key, value, ttl=self.ttl)
        
        return value
    
    def clear_cache(self):
        """Clear the property cache"""
        self.cache.clear()


def cached_property(ttl: Optional[int] = None):
    """
    Decorator to create cached properties
    
    Usage:
        class MyClass:
            @cached_property(ttl=300)
            def expensive_computation(self):
                return sum(range(1000000))
    """
    def decorator(func: Callable) -> CachedProperty:
        return CachedProperty(func, ttl=ttl)
    
    return decorator


# Conditional caching based on result
def cache_on_success(
    ttl: Optional[int] = None,
    success_check: Optional[Callable] = None
):
    """
    Cache only if the result meets success criteria
    
    Args:
        ttl: Time to live in seconds
        success_check: Function to check if result should be cached
    """
    def decorator(func: Callable) -> Callable:
        cache = _global_cache
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = get_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Check if we should cache
            should_cache = True
            if success_check:
                should_cache = success_check(result)
            
            if should_cache:
                cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    
    return decorator


# Batch caching for multiple items
class BatchCache:
    """
    Cache for batch operations
    """
    def __init__(self, cache_instance: Optional[MemoryCache] = None):
        self.cache = cache_instance or _global_cache
    
    def get_many(self, keys: list) -> dict:
        """Get multiple values from cache"""
        results = {}
        for key in keys:
            value = self.cache.get(key)
            if value is not None:
                results[key] = value
        return results
    
    def set_many(self, items: dict, ttl: Optional[int] = None):
        """Set multiple values in cache"""
        for key, value in items.items():
            self.cache.set(key, value, ttl=ttl)
    
    def delete_many(self, keys: list):
        """Delete multiple keys from cache"""
        for key in keys:
            self.cache.delete(key) 