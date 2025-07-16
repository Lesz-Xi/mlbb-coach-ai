#!/usr/bin/env python3
"""
Test Ultimate Parsing System response format
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ultimate_parsing_system import ultimate_parsing_system

def test_ultimate_response():
    print("🧪 Testing Ultimate Parsing System response format...")
    
    # Use a test image (or create a minimal one)
    test_image = "Screenshot-Test/screenshot-test-1.PNG"
    
    try:
        print(f"📸 Analyzing: {test_image}")
        ultimate_result = ultimate_parsing_system.analyze_screenshot_ultimate(
            image_path=test_image,
            ign="Lesz XVII",
            session_id=None,
            hero_override=None
        )
        
        print(f"✅ Ultimate System Response:")
        print(f"   Overall Confidence: {ultimate_result.overall_confidence}")
        print(f"   Analysis Stage: {ultimate_result.analysis_stage}")
        print(f"   Session ID: {ultimate_result.session_id}")
        print(f"   Warnings Count: {len(ultimate_result.warnings)}")
        print(f"   Parsed Data Keys: {list(ultimate_result.parsed_data.keys())}")
        
        # Test the web app conversion
        print(f"\n🔄 Testing Web App Conversion:")
        result = {
            "data": ultimate_result.parsed_data,
            "warnings": ultimate_result.warnings,
            "confidence": ultimate_result.overall_confidence / 100.0,
            "diagnostics": {
                "confidence_score": ultimate_result.overall_confidence / 100.0,
                "analysis_state": ultimate_result.analysis_stage,
                "hero_confidence": ultimate_result.hero_detection.confidence if hasattr(ultimate_result.hero_detection, 'confidence') else 0.0,
                "hero_name": ultimate_result.hero_detection.hero_name if hasattr(ultimate_result.hero_detection, 'hero_name') else "unknown",
            }
        }
        
        print(f"   Converted Confidence: {result['confidence']}")
        print(f"   Diagnostics Confidence: {result['diagnostics']['confidence_score']}")
        print(f"   Analysis State: {result['diagnostics']['analysis_state']}")
        print(f"   Hero: {result['diagnostics']['hero_name']} ({result['diagnostics']['hero_confidence']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Ultimate System Test Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ultimate_response()
    print(f"\n🎯 Test Result: {'✅ SUCCESS' if success else '❌ FAILED'}")