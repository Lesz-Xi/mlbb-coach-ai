#!/usr/bin/env python3
"""
Smoke test for Redis Job Queue Integration

This script tests the complete async pipeline:
1. Check Redis connection
2. Test job creation endpoint
3. Test job status endpoint  
4. Verify worker can process jobs
"""

import os
import sys
import time
import requests
import json
from pathlib import Path

# Test configuration
BACKEND_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "Screenshot-Test-Good/screenshot-test-good-1.PNG"
IGN = "Lesz XVII"

def test_redis_connection():
    """Test if Redis is accessible"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis connection: OK")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def test_backend_health():
    """Test if backend is healthy"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/health-isolated", timeout=5)
        if response.status_code == 200:
            print("✅ Backend health: OK")
            return True
        else:
            print(f"❌ Backend health failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend health failed: {e}")
        return False

def test_job_creation():
    """Test creating an analysis job"""
    try:
        # Check if test image exists
        if not Path(TEST_IMAGE_PATH).exists():
            print(f"❌ Test image not found: {TEST_IMAGE_PATH}")
            print("Creating a dummy test file...")
            # Create a simple test file
            Path("test_screenshot.png").touch()
            test_path = "test_screenshot.png"
        else:
            test_path = TEST_IMAGE_PATH
        
        # Create job
        with open(test_path, 'rb') as f:
            files = {'file': f}
            data = {'ign': IGN}
            
            response = requests.post(
                f"{BACKEND_URL}/api/jobs",
                files=files,
                data=data,
                timeout=10
            )
        
        if response.status_code == 202:
            result = response.json()
            job_id = result.get('job_id')
            print(f"✅ Job creation: OK (Job ID: {job_id})")
            return job_id
        else:
            print(f"❌ Job creation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Job creation failed: {e}")
        return None

def test_job_status(job_id):
    """Test getting job status"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/jobs/{job_id}", timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            state = result.get('state', 'unknown')
            print(f"✅ Job status: OK (State: {state})")
            return result
        else:
            print(f"❌ Job status failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Job status failed: {e}")
        return None

def test_worker_processing(job_id, max_wait=60):
    """Test if worker can process the job"""
    print(f"🔄 Waiting for worker to process job {job_id}...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        status = test_job_status(job_id)
        if not status:
            break
            
        state = status.get('state', 'unknown')
        
        if state == 'finished':
            print("✅ Worker processing: OK (Job completed)")
            return True
        elif state == 'failed':
            print(f"❌ Worker processing failed: {status.get('error', 'Unknown error')}")
            return False
        
        print(f"⏳ Job state: {state} (waiting...)")
        time.sleep(3)
    
    print("❌ Worker processing: TIMEOUT")
    return False

def main():
    """Run complete smoke test"""
    print("🧪 Starting Redis Integration Smoke Test")
    print("=" * 50)
    
    # Test 1: Redis connection
    if not test_redis_connection():
        print("💥 Redis not available - stopping test")
        return False
    
    # Test 2: Backend health
    if not test_backend_health():
        print("💥 Backend not available - stopping test")
        return False
    
    # Test 3: Job creation
    job_id = test_job_creation()
    if not job_id:
        print("💥 Job creation failed - stopping test")
        return False
    
    # Test 4: Job status
    if not test_job_status(job_id):
        print("💥 Job status failed - stopping test")
        return False
    
    # Test 5: Worker processing
    if not test_worker_processing(job_id):
        print("💥 Worker processing failed")
        return False
    
    print("\n🎉 All tests passed! Redis integration is working correctly.")
    print("\n📋 Summary:")
    print("   ✅ Redis connection established")
    print("   ✅ Backend API responding")
    print("   ✅ Job creation working")
    print("   ✅ Job status endpoint working")
    print("   ✅ Worker processing jobs")
    print("\n🚀 Async analysis pipeline is fully operational!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 