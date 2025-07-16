#!/usr/bin/env python3
"""
Test the enhanced parsing system improvements
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
from core.enhanced_data_collector import enhanced_data_collector
from core.data_collector import get_ocr_reader

def test_enhanced_parsing():
    """Test the enhanced parsing improvements"""
    
    print("🚀 TESTING ENHANCED PARSING SYSTEM")
    print("=" * 60)
    
    # Test with the same screenshot
    screenshot_path = "Screenshot-Test/screenshot-test-1.PNG"
    ign = "Lesz XVII"
    
    print(f"📷 Testing screenshot: {screenshot_path}")
    print(f"👤 Target IGN: {ign}")
    print()
    
    # Test the enhanced parsing
    result = enhanced_data_collector.analyze_screenshot_with_session(
        image_path=screenshot_path,
        ign=ign,
        hero_override=None
    )
    
    print("📊 ENHANCED PARSING RESULTS:")
    print("=" * 40)
    
    if "data" in result:
        data = result["data"]
        print(f"✅ Extracted {len(data)} data fields:")
        for key, value in data.items():
            if not key.startswith('_'):  # Skip debug fields
                print(f"  {key}: {value}")
        
        print(f"\n🎯 Overall Confidence: {result.get('overall_confidence', 0):.1%}")
        print(f"📈 Completeness Score: {result.get('completeness_score', 0):.1%}")
        print(f"🔍 Screenshot Type: {result.get('screenshot_type', 'unknown')}")
        
        if result.get('warnings'):
            print(f"\n⚠️  Warnings ({len(result['warnings'])}):")
            for warning in result['warnings']:
                print(f"  • {warning}")
        
        debug_info = result.get('debug_info', {})
        if debug_info:
            print(f"\n🔧 Debug Info:")
            print(f"  Anchors found: {debug_info.get('_anchors_found', [])}")
            print(f"  Columns found: {debug_info.get('_columns_found', [])}")
            print(f"  Parsing method: {data.get('_parsing_method', 'unknown')}")
        
    else:
        print("❌ No data extracted")
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)
    print("🎉 ENHANCED PARSING TEST COMPLETE")

if __name__ == "__main__":
    test_enhanced_parsing()