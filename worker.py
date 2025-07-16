#!/usr/bin/env python3
"""
Redis Queue Worker for SkillShift AI
 
This worker process handles heavy analysis jobs in the background,
keeping the main API responsive. It imports the heavy_analysis function
and processes jobs from the 'analysis' queue.

Usage:
    python worker.py

The worker will:
1. Connect to Redis
2. Listen for jobs in the 'analysis' queue  
3. Execute heavy_analysis function for each job
4. Store results in Redis for the API to retrieve
"""

import os
import sys
import time
import socket
from redis import Redis
from rq import Worker

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    """
    Main worker function that connects to Redis and processes jobs.
    """
    print("🔧 Starting SkillShift AI Analysis Worker...")
    
    # Connect to Redis - support multiple formats
    redis_url = os.getenv("REDIS_URL")
    redis_token = os.getenv("REDIS_TOKEN")
    
    try:
        if redis_url and redis_token:
            # Upstash Redis format - convert for RQ compatibility
            upstash_url = redis_url.replace("https://", "redis://")
            redis_conn = Redis.from_url(f"{upstash_url}?password={redis_token}", decode_responses=False)
            print(f"🌐 Worker connecting to Upstash Redis: {redis_url}")
            print("🔄 Using compatibility mode for RQ")
        elif redis_url and redis_url.startswith(('redis://', 'rediss://')):
            # Standard Redis Cloud URL format
            redis_conn = Redis.from_url(redis_url, decode_responses=False)
            display_url = redis_url.split('@')[1] if '@' in redis_url else redis_url
            print(f"🌐 Worker connecting to Redis Cloud: {display_url}")
        else:
            # Fallback to individual parameters for local development
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            redis_db = int(os.getenv("REDIS_DB", "0"))
            redis_password = os.getenv("REDIS_PASSWORD")
            
            redis_conn = Redis(
                host=redis_host, 
                port=redis_port, 
                db=redis_db,
                password=redis_password if redis_password else None,
                decode_responses=False
            )
            print(f"🔧 Worker connecting to local Redis: {redis_host}:{redis_port}")
        
        # Test Redis connection
        redis_conn.ping()
        print("✅ Worker Redis connection established successfully")
        
    except Exception as e:
        print(f"❌ Failed to connect to Redis: {e}")
        print("💡 Tip: Make sure to set REDIS_URL and REDIS_TOKEN for Upstash")
        print("Please ensure Redis is running and accessible")
        sys.exit(1)
    
    # Create worker with connection (modern RQ API) - use unique name
    timestamp = int(time.time())
    hostname = socket.gethostname()
    worker_name = f"skillshift-worker-{hostname}-{timestamp}"
    worker = Worker(
        queues=["analysis"], 
        connection=redis_conn, 
        name=worker_name
    )
    
    print("🚀 Worker ready to process analysis jobs...")
    print("📋 Listening on queue: 'analysis'")
    print("⏹️  Press Ctrl+C to stop")
    
    try:
        # Start processing jobs (burst=False means keep running)
        worker.work(burst=False)
    except KeyboardInterrupt:
        print("\n🛑 Worker stopped by user")
    except Exception as e:
        print(f"\n❌ Worker error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 