#!/usr/bin/env python3
"""
Quick Redis Connection Test

This script tests if your Redis configuration is working.
Run this before starting the full services.
"""

import os
import sys

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads .env file automatically
except ImportError:
    pass  # Continue without .env loading if dotenv not available


def test_redis_cloud_connection():
    """Test Redis connection using environment variables"""
    
    print("🧪 Testing Redis Connection")
    print("=" * 40)
    
    # Get Redis configuration from environment
    redis_url = os.getenv("REDIS_URL")
    redis_token = os.getenv("REDIS_TOKEN")
    
    if not redis_url:
        print("❌ REDIS_URL environment variable not set")
        print()
        print("💡 To fix this:")
        print("   1. For Upstash: set REDIS_URL and REDIS_TOKEN")
        print("   2. For Redis Cloud: set REDIS_URL with full connection string")
        print("   3. Or create .env file with the variables")
        return False
    
    try:
        if redis_url and redis_token:
            # Upstash Redis format
            from upstash_redis import Redis
            redis_conn = Redis(url=redis_url, token=redis_token)
            print(f"🌐 Testing Upstash Redis: {redis_url}")
        elif redis_url.startswith(('redis://', 'rediss://')):
            # Standard Redis Cloud format
            from redis import Redis
            redis_conn = Redis.from_url(redis_url, decode_responses=False)
            display_url = redis_url.split('@')[1] if '@' in redis_url else redis_url
            print(f"🌐 Testing Redis Cloud: {display_url}")
        else:
            print("❌ Invalid Redis URL format")
            print("Expected: https://... (Upstash) or redis://... (Redis Cloud)")
            return False
        
        # Test connection
        redis_conn.ping()
        print("✅ Redis connection successful!")
        
        # Test basic operations
        redis_conn.set("test_key", "test_value", ex=60)
        redis_conn.get("test_key")  # Test read
        redis_conn.delete("test_key")
        
        print("✅ Redis read/write operations successful!")
        print()
        print("🎉 Your Redis is ready for SkillShift AI!")
        print()
        print("Next steps:")
        print("   1. Run: ./start_all_services.sh")
        print("   2. The services will automatically use your Redis")
        
        return True
        
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print()
        print("💡 Troubleshooting:")
        print("   1. Check your Redis URL and token are correct")
        print("   2. Verify your Redis instance is active")
        print("   3. Check firewall/network connectivity")
        
        return False


def main():
    """Main test function"""
    success = test_redis_cloud_connection()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 