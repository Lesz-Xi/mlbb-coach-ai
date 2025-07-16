#!/usr/bin/env python3
"""
Minimal test to isolate where numpy.bool_ error occurs
"""

import requests
import json

# Test 1: Health check endpoint (should not use Enhanced Mode logic)
print("🔍 TESTING: Health check endpoint...")
try:
    response = requests.get("http://localhost:8000/health")
    print(f"📥 Health Status: {response.status_code}")
    if response.status_code == 200:
        print("✅ Health endpoint works")
    else:
        print(f"❌ Health endpoint failed: {response.text}")
except Exception as e:
    print(f"❌ Health endpoint error: {e}")

# Test 2: Simple API endpoint (non-Enhanced Mode)
print("\n🔍 TESTING: Simple analyze endpoint...")
try:
    with open("Screenshot-Test/screenshot-test-1.PNG", "rb") as f:
        files = {"file": f}
        data = {"ign": "Lesz XVII"}
        response = requests.post("http://localhost:8000/api/analyze", files=files, data=data)
    
    print(f"📥 Simple API Status: {response.status_code}")
    if response.status_code == 200:
        print("✅ Simple API endpoint works")
    else:
        print(f"❌ Simple API failed: {response.text}")
except Exception as e:
    print(f"❌ Simple API error: {e}")

print("\n🎯 Minimal API test completed!") 