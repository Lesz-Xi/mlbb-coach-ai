#!/usr/bin/env python3
"""
Test the fixed API with enhanced parsing
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tempfile
import shutil
from web.app import enhanced_data_collector

def test_api_fixed():
    """Test the API fix for hero detection"""
    print("🔧 TESTING FIXED API")
    print("=" * 50)
    
    screenshot_path = "Screenshot-Test/screenshot-test-1.PNG"
    ign = "Lesz XVII"
    
    # Test the enhanced data collector directly (simulating API call)
    result = enhanced_data_collector.analyze_screenshot_with_session(
        image_path=screenshot_path,
        ign=ign,
        hero_override=None
    )
    
    print("📊 ENHANCED DATA COLLECTOR RESULT:")
    print(f"Data: {result.get('data', {})}")
    print(f"Overall Confidence: {result.get('overall_confidence', 0):.1%}")
    print(f"Hero in data: {result.get('data', {}).get('hero', 'missing')}")
    print(f"Warnings: {result.get('warnings', [])}")
    
    # Test API-style processing (what the frontend sees)
    match_data = result.get("data", {})
    overall_confidence = result.get("overall_confidence", 0.0)
    
    # Test the hero detection logic from the API
    debug_info = result.get("debug_info", {})
    hero_debug = debug_info.get("hero_debug", {})
    
    # Get hero confidence from multiple possible locations
    hero_confidence = 0.0
    if "hero_suggestions" in hero_debug and hero_debug["hero_suggestions"]:
        # If we have suggestions, use the top suggestion confidence
        hero_confidence = hero_debug["hero_suggestions"][0][1]
    else:
        # Fallback: extract from match data or debug info
        hero_confidence = debug_info.get("hero_confidence", 0.0)
    
    # Detect hero based on actual data, not just confidence threshold
    hero_name = match_data.get("hero", "unknown")
    hero_detected = bool(hero_name != "unknown" and hero_name != "" and hero_name is not None)
    
    # If hero is detected, ensure we have reasonable confidence
    if hero_detected and hero_confidence == 0.0:
        hero_confidence = 0.7  # Default reasonable confidence if hero is detected but confidence missing
    
    print(f"\n🎯 API HERO DETECTION LOGIC:")
    print(f"Hero name: {hero_name}")
    print(f"Hero detected: {hero_detected}")
    print(f"Hero confidence: {hero_confidence:.3f}")
    print(f"Overall confidence: {overall_confidence:.3f}")
    
    # Test warning filtering
    from web.app import _filter_warnings_for_high_confidence
    
    filtered_warnings = _filter_warnings_for_high_confidence(
        result.get("warnings", []), 
        overall_confidence, 
        hero_detected
    )
    
    print(f"\n⚠️  WARNING FILTERING:")
    print(f"Original warnings: {len(result.get('warnings', []))}")
    print(f"Filtered warnings: {len(filtered_warnings)}")
    print(f"Filtered list: {filtered_warnings}")
    
    # Final assessment
    if hero_detected and overall_confidence > 0.85:
        print(f"\n✅ SUCCESS: Hero detected with high confidence!")
        print(f"   - Should show: Analysis State: Complete")
        print(f"   - Should show: Hero: {hero_name}")
        print(f"   - Should hide: 'Analysis Quality Issues' banner")
    else:
        print(f"\n⚠️  PARTIAL: Some issues remain")
        print(f"   - Hero detected: {hero_detected}")
        print(f"   - High confidence: {overall_confidence > 0.85}")

if __name__ == "__main__":
    test_api_fixed()