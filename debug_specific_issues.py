#!/usr/bin/env python3
"""
Debug specific parsing pipeline issues
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
from core.data_collector import get_ocr_reader
from core.enhanced_data_collector import enhanced_data_collector, RobustIGNMatcher, AnchorBasedLayoutParser

def debug_gold_detection():
    """Debug why gold detection is failing"""
    print("🔍 DEBUGGING GOLD DETECTION")
    print("=" * 50)
    
    screenshot_path = "Screenshot-Test/screenshot-test-1.PNG"
    ign = "Lesz XVII"
    
    # Get OCR results
    image = cv2.imread(screenshot_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reader = get_ocr_reader()
    ocr_results = reader.readtext(gray, detail=1)
    
    print(f"OCR found {len(ocr_results)} text detections")
    
    # Test IGN matching
    ign_matcher = RobustIGNMatcher()
    ign_result = ign_matcher.find_ign_in_ocr(ign, ocr_results)
    print(f"IGN found: {ign_result['found']} with confidence {ign_result.get('confidence', 0):.3f}")
    
    if ign_result['found']:
        ign_bbox = ign_result['bbox']
        player_row_y = sum(point[1] for point in ign_bbox) / 4
        print(f"Player row Y position: {player_row_y}")
        
        # Find all numbers near the player row
        row_tolerance = 30
        numbers_near_player = []
        
        for bbox, text, conf in ocr_results:
            text_y = sum(point[1] for point in bbox) / 4
            if abs(text_y - player_row_y) <= row_tolerance:
                # Extract numbers from text
                import re
                numbers = re.findall(r'\b(\d+)\b', text)
                if numbers:
                    for num in numbers:
                        num_val = int(num)
                        if num_val >= 1000:  # Potential gold values
                            numbers_near_player.append((num_val, text, conf))
        
        print(f"Numbers >= 1000 near player row: {numbers_near_player}")
        
        # Test anchor-based parsing
        anchor_parser = AnchorBasedLayoutParser()
        height, width = image.shape[:2]
        anchors = anchor_parser.detect_anchors(ocr_results, (height, width))
        print(f"Anchors detected: {list(anchors.keys())}")
        
        columns = anchor_parser.identify_columns(ocr_results, anchors)
        print(f"Columns identified: {list(columns.keys())}")

def debug_hero_detection():
    """Debug why hero detection is failing"""
    print("\n🔍 DEBUGGING HERO DETECTION")
    print("=" * 50)
    
    screenshot_path = "Screenshot-Test/screenshot-test-1.PNG"
    ign = "Lesz XVII"
    
    from core.advanced_hero_detector import advanced_hero_detector
    
    hero_name, hero_confidence, hero_debug = advanced_hero_detector.detect_hero_comprehensive(
        screenshot_path, ign
    )
    
    print(f"Hero detected: {hero_name}")
    print(f"Confidence: {hero_confidence:.3f}")
    print(f"Strategies tried: {hero_debug.get('strategies_tried', [])}")
    print(f"OCR text found: {hero_debug.get('ocr_text_found', [])[:5]}")
    print(f"Hero suggestions: {hero_debug.get('hero_suggestions', [])[:3]}")

def debug_completeness_calculation():
    """Debug why data completeness is 0%"""
    print("\n🔍 DEBUGGING DATA COMPLETENESS")
    print("=" * 50)
    
    # Sample data that should give > 0% completeness
    test_data = {
        'kills': 0,
        'deaths': 5, 
        'assists': 1,
        'hero': 'roger',
        'match_duration': 10
    }
    
    # Test completeness calculation
    required_fields = ["kills", "deaths", "assists", "hero", "gold"]
    optional_fields = ["hero_damage", "turret_damage", "damage_taken", "teamfight_participation"]
    
    required_score = sum(1 for field in required_fields if test_data.get(field))
    optional_score = sum(1 for field in optional_fields if test_data.get(field))
    
    completeness = (required_score / len(required_fields)) * 0.7 + (optional_score / len(optional_fields)) * 0.3
    
    print(f"Test data: {test_data}")
    print(f"Required fields present: {required_score}/{len(required_fields)}")
    print(f"Optional fields present: {optional_score}/{len(optional_fields)}")
    print(f"Calculated completeness: {completeness:.3f} ({completeness:.1%})")
    
    # Test with missing gold
    test_data_no_gold = test_data.copy()
    test_data_no_gold.pop('hero', None)  # Remove hero too
    
    required_score_ng = sum(1 for field in required_fields if test_data_no_gold.get(field))
    completeness_ng = (required_score_ng / len(required_fields)) * 0.7
    
    print(f"\nWithout hero and gold:")
    print(f"Required fields present: {required_score_ng}/{len(required_fields)}")
    print(f"Calculated completeness: {completeness_ng:.3f} ({completeness_ng:.1%})")

if __name__ == "__main__":
    debug_gold_detection()
    debug_hero_detection()
    debug_completeness_calculation()