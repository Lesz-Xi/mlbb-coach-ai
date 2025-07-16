#!/usr/bin/env python3
"""
Test Upstash Redis with RQ (Redis Queue) compatibility
"""

import os
from dotenv import load_dotenv
load_dotenv()

def test_upstash_with_rq():
    """Test if our Upstash Redis works with RQ"""
    
    print("🧪 Testing Upstash Redis with RQ")
    print("=" * 40)
    
    # Get configuration
    redis_url = os.getenv("REDIS_URL")
    redis_token = os.getenv("REDIS_TOKEN")
    
    if not (redis_url and redis_token):
        print("❌ REDIS_URL or REDIS_TOKEN not set")
        return False
    
    try:
        # Test 1: Direct Upstash Redis connection
        from upstash_redis import Redis as UpstashRedis
        upstash_conn = UpstashRedis(url=redis_url, token=redis_token)
        upstash_conn.ping()
        print("✅ Direct Upstash Redis connection: OK")
        
        # Test 2: RQ-compatible connection (what our app uses)
        from redis import Redis
        from rq import Queue
        
        # Convert Upstash URL for RQ compatibility
        upstash_url = redis_url.replace("https://", "redis://")
        redis_conn = Redis.from_url(f"{upstash_url}?password={redis_token}", decode_responses=False)
        redis_conn.ping()
        print("✅ RQ-compatible Redis connection: OK")
        
        # Test 3: Create RQ queue
        queue = Queue("test", connection=redis_conn)
        print("✅ RQ Queue creation: OK")
        
        # Test 4: Simple job enqueue/dequeue test
        def dummy_job(message):
            return f"Processed: {message}"
        
        job = queue.enqueue(dummy_job, "Hello Upstash!")
        print(f"✅ Job enqueued: {job.id}")
        
        print()
        print("🎉 Upstash Redis + RQ compatibility confirmed!")
        print("Your async job system should work properly.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_upstash_with_rq()
    exit(0 if success else 1) 