#!/usr/bin/env python3
"""
Test script to debug Enhanced Mode analysis
"""
import requests
import sys
import os

def test_enhanced_analysis():
    """Test the enhanced analysis endpoint with debug output"""
    
    # Check if screenshot exists
    screenshot_path = "Screenshot-Test/screenshot-test-1.PNG"
    if not os.path.exists(screenshot_path):
        print(f"❌ Screenshot not found: {screenshot_path}")
        return
    
    print(f"🔍 Testing Enhanced Mode with: {screenshot_path}")
    print(f"🔍 File size: {os.path.getsize(screenshot_path) / (1024*1024):.1f} MB")
    
    # Prepare the request
    url = "http://127.0.0.1:8000/api/analyze-enhanced/"
    
    with open(screenshot_path, 'rb') as f:
        files = {'file': ('screenshot.PNG', f, 'image/png')}
        data = {'ign': 'Lesz XVII'}
        
        print("🔍 Sending request to Enhanced Mode endpoint...")
        
        try:
            response = requests.post(url, files=files, data=data, timeout=60)
            
            print(f"🔍 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ SUCCESS: Enhanced analysis completed")
                
                # Print key results
                diagnostics = result.get('diagnostics', {})
                print(f"🔍 Analysis Mode: {diagnostics.get('analysis_mode', 'unknown')}")
                print(f"🔍 Confidence Score: {diagnostics.get('confidence_score', 0):.1%}")
                print(f"🔍 Hero Detected: {diagnostics.get('hero_detected', False)}")
                print(f"🔍 Data Completeness: {diagnostics.get('data_completeness', 0):.1%}")
                
                # Check for warnings
                warnings = diagnostics.get('warnings', [])
                if warnings:
                    print("⚠️ Warnings:")
                    for warning in warnings[:5]:  # Show first 5 warnings
                        print(f"  - {warning}")
                
            else:
                print(f"❌ ERROR: {response.status_code}")
                print(f"❌ Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ REQUEST FAILED: {e}")
    
    # Check if debug files were created
    debug_files = [
        "debug_preprocessed_output.png",
        "temp/enhanced_preprocessed_image.png"
    ]
    
    print("\n🔍 Checking for debug files:")
    for debug_file in debug_files:
        if os.path.exists(debug_file):
            size = os.path.getsize(debug_file)
            print(f"✅ {debug_file} - {size} bytes")
        else:
            print(f"❌ {debug_file} - NOT FOUND")

if __name__ == "__main__":
    print("🚀 Starting Enhanced Mode Debug Test")
    print("=" * 50)
    test_enhanced_analysis()
    print("=" * 50)
    print("🔍 Check server console for detailed debug output!") 