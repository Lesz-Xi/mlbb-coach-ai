#!/usr/bin/env python3
"""
Test to trace why Enhanced Mode discards valid data from _parse_player_row
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
from core.data_collector import get_ocr_reader
from core.enhanced_data_collector import EnhancedDataCollector
from core.session_manager import ScreenshotType

def test_enhanced_flow():
    """Test the enhanced flow step by step"""
    
    print("🔍 TESTING ENHANCED MODE FLOW")
    print("=" * 60)
    
    # Get OCR results
    screenshot_path = "Screenshot-Test/screenshot-test-1.PNG"
    image = cv2.imread(screenshot_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    reader = get_ocr_reader()
    ocr_results = reader.readtext(gray, detail=1)
    
    ign = "Lesz XVII"
    collector = EnhancedDataCollector()
    warnings = []
    
    print("1️⃣ Testing _parse_player_row_enhanced...")
    
    # Test _parse_player_row_enhanced directly
    enhanced_data = collector._parse_player_row_enhanced(ign, ocr_results, warnings)
    print(f"📊 _parse_player_row_enhanced result: {len(enhanced_data)} fields")
    if enhanced_data:
        print("✅ Enhanced player row parsing SUCCESS")
        for key, value in enhanced_data.items():
            print(f"  {key}: {value}")
    else:
        print("❌ Enhanced player row parsing FAILED")
    
    print(f"\n2️⃣ Testing _parse_scoreboard_enhanced...")
    
    # Test _parse_scoreboard_enhanced
    scoreboard_data = collector._parse_scoreboard_enhanced(ign, ocr_results, warnings)
    print(f"📊 _parse_scoreboard_enhanced result: {len(scoreboard_data)} fields")
    if scoreboard_data:
        print("✅ Scoreboard parsing SUCCESS")
        for key, value in scoreboard_data.items():
            print(f"  {key}: {value}")
    else:
        print("❌ Scoreboard parsing FAILED")
    
    print(f"\n3️⃣ Testing data completeness validation...")
    
    # Test completeness validation 
    if scoreboard_data:
        completeness = collector._validate_data_completeness(scoreboard_data, warnings)
        print(f"📊 Data completeness score: {completeness:.3f}")
    else:
        print("❌ No data to validate completeness")
    
    print(f"\n4️⃣ Testing confidence calculation...")
    
    # Test confidence calculation
    if scoreboard_data:
        overall_confidence = collector._calculate_overall_confidence(
            hero_confidence=0.0,  # We know hero is unknown
            warning_count=len(warnings),
            data_field_count=len(scoreboard_data),
            completeness_score=completeness if scoreboard_data else 0.0
        )
        print(f"📊 Overall confidence: {overall_confidence:.3f}")
        print(f"📊 Warning count: {len(warnings)}")
        print(f"📊 Data field count: {len(scoreboard_data)}")
    
    print(f"\n📝 Warnings generated:")
    for i, warning in enumerate(warnings):
        print(f"  {i+1}. {warning}")
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSIS:")
    if enhanced_data and not scoreboard_data:
        print("  Issue: _parse_scoreboard_enhanced is losing the data!")
    elif scoreboard_data and overall_confidence < 0.1:
        print("  Issue: Confidence calculation is too harsh!")
    elif not enhanced_data:
        print("  Issue: _parse_player_row_enhanced is failing!")
    else:
        print("  All steps working - need to check integration!")

if __name__ == "__main__":
    test_enhanced_flow() 