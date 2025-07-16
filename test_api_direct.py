#!/usr/bin/env python3
"""
Test the actual Enhanced Mode API endpoint to see what overall_rating it returns
"""
import requests
import json
import os

def test_enhanced_api_direct():
    """Test the Enhanced Mode API directly"""
    
    print("🔍 TESTING Enhanced Mode API Direct")
    print("=" * 60)
    
    # Check if screenshot exists
    screenshot_path = "Screenshot-Test/screenshot-test-1.PNG"
    if not os.path.exists(screenshot_path):
        print(f"❌ Screenshot not found: {screenshot_path}")
        return
    
    print(f"📤 Calling /api/analyze-enhanced/ with {screenshot_path}")
    
    url = "http://127.0.0.1:8000/api/analyze-enhanced/"
    
    try:
        with open(screenshot_path, 'rb') as f:
            files = {'file': ('screenshot.PNG', f, 'image/png')}
            data = {'ign': 'Lesz XVII'}
            
            response = requests.post(url, files=files, data=data, timeout=60)
            
            print(f"📥 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                print("✅ API CALL SUCCESS!")
                print(f"\n📊 OVERALL RATING: '{result.get('overall_rating', 'NOT_FOUND')}'")
                
                # Check key fields
                print(f"\n📊 KEY API RESPONSE FIELDS:")
                print(f"  overall_rating: {result.get('overall_rating')}")
                print(f"  feedback count: {len(result.get('feedback', []))}")
                print(f"  mental_boost: {result.get('mental_boost', 'N/A')}")
                
                # Check diagnostics
                diagnostics = result.get('diagnostics', {})
                print(f"\n📊 DIAGNOSTICS:")
                print(f"  confidence_score: {diagnostics.get('confidence_score', 'N/A')}")
                print(f"  analysis_state: {diagnostics.get('analysis_state', 'N/A')}")
                print(f"  hero_detected: {diagnostics.get('hero_detected', 'N/A')}")
                print(f"  kda_data_complete: {diagnostics.get('kda_data_complete', 'N/A')}")
                print(f"  gold_data_valid: {diagnostics.get('gold_data_valid', 'N/A')}")
                
                # Save full response for inspection
                with open('api_response_debug.json', 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n💾 Full response saved to: api_response_debug.json")
                
                # Check for any signs of error handling
                if 'error' in result:
                    print(f"❌ API Error: {result['error']}")
                
                warnings = result.get('warnings', [])
                if warnings:
                    print(f"\n⚠️ Warnings ({len(warnings)}):")
                    for i, warning in enumerate(warnings[:5]):
                        print(f"  {i+1}. {warning}")
                        
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"❌ Response: {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_enhanced_api_direct() 