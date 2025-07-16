#!/usr/bin/env python3
"""
Test to verify and fix the IGN matching issue in _parse_player_row
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
from core.data_collector import get_ocr_reader, DataCollector

def test_ign_matching():
    """Test the IGN matching issue"""
    
    print("🔍 TESTING IGN MATCHING ISSUE")
    print("=" * 50)
    
    # Get OCR results from our test image
    screenshot_path = "Screenshot-Test/screenshot-test-1.PNG"
    image = cv2.imread(screenshot_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    reader = get_ocr_reader()
    ocr_results = reader.readtext(gray, detail=1)
    
    print(f"🔍 OCR found {len(ocr_results)} text detections")
    
    # Look for our IGN
    ign = "Lesz XVII"
    print(f"\n🔍 Looking for IGN: '{ign}'")
    
    ign_found_texts = []
    for i, (bbox, text, conf) in enumerate(ocr_results):
        if ign.lower() in text.lower():
            print(f"✅ FOUND IGN in text #{i}: '{text}' (conf: {conf:.3f})")
            ign_found_texts.append((i, text, conf))
        elif any(part.lower() in text.lower() for part in ign.split()):
            print(f"🔍 PARTIAL IGN match #{i}: '{text}' (conf: {conf:.3f})")
    
    if not ign_found_texts:
        print("❌ IGN NOT FOUND with current matching logic!")
        print("\n📝 All OCR detections:")
        for i, (bbox, text, conf) in enumerate(ocr_results):
            print(f"  {i:2d}. '{text}' (conf: {conf:.3f})")
    
    # Test original _parse_player_row
    print(f"\n🔍 Testing original _parse_player_row...")
    collector = DataCollector()
    parsed_data = collector._parse_player_row(ign, ocr_results)
    
    print(f"📊 Result: {len(parsed_data)} fields extracted")
    if parsed_data:
        print("✅ _parse_player_row SUCCESS")
        for key, value in parsed_data.items():
            print(f"  {key}: {value}")
    else:
        print("❌ _parse_player_row FAILED - returned empty dict")
    
    # Let's manually check what the issue is
    print(f"\n🔍 Manual debugging...")
    for bbox, text, conf in ocr_results:
        if "lesz" in text.lower() or "xvii" in text.lower():
            print(f"🔍 Found IGN-related text: '{text}' (conf: {conf:.3f})")

if __name__ == "__main__":
    test_ign_matching() 