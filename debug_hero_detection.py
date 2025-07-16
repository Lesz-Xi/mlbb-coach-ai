#!/usr/bin/env python3
"""
Debug hero detection pipeline specifically
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
from core.enhanced_data_collector import enhanced_data_collector
from core.advanced_hero_detector import advanced_hero_detector
from core.hero_identifier import hero_identifier
from core.data_collector import get_ocr_reader

def debug_hero_pipeline():
    """Debug the complete hero detection pipeline"""
    print("🔍 DEBUGGING HERO DETECTION PIPELINE")
    print("=" * 60)
    
    screenshot_path = "Screenshot-Test/screenshot-test-1.PNG"
    ign = "Lesz XVII"
    
    # Step 1: Test the hero_identifier directly
    print("\n1️⃣ Testing core hero_identifier...")
    
    # Get OCR results
    image = cv2.imread(screenshot_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reader = get_ocr_reader()
    ocr_results = reader.readtext(gray, detail=1)
    
    # Test with known hero names that might be in the image
    test_texts = [
        "roger",
        "Lesz XVII 0 5 1 4671 7586 7 1 13 hindi tayo pwidi",  # Player row
        " ".join([result[1] for result in ocr_results])  # All OCR text
    ]
    
    for i, test_text in enumerate(test_texts):
        print(f"\nTest {i+1}: '{test_text[:50]}...'")
        hero_name, confidence = hero_identifier.identify_hero(test_text)
        print(f"  Result: {hero_name} (confidence: {confidence:.3f})")
        
        if confidence > 0:
            suggestions = hero_identifier.get_hero_suggestions(test_text, top_n=3)
            print(f"  Suggestions: {suggestions}")
    
    # Step 2: Test advanced_hero_detector
    print(f"\n2️⃣ Testing advanced_hero_detector...")
    hero_name, hero_confidence, hero_debug = advanced_hero_detector.detect_hero_comprehensive(
        screenshot_path, ign
    )
    
    print(f"Result: {hero_name} (confidence: {hero_confidence:.3f})")
    print(f"Strategies tried: {hero_debug.get('strategies_tried', [])}")
    print(f"Manual override: {hero_debug.get('manual_override', False)}")
    print(f"Error: {hero_debug.get('error', 'None')}")
    
    # Step 3: Check hero database
    print(f"\n3️⃣ Checking hero database...")
    hero_list = getattr(hero_identifier, 'hero_list', [])
    print(f"Heroes in database: {len(hero_list)}")
    print(f"Sample heroes: {hero_list[:10]}")
    
    # Check if "roger" is in the list
    if hasattr(hero_identifier, 'hero_list'):
        roger_variants = [h for h in hero_identifier.hero_list if 'roger' in h.lower()]
        print(f"Roger variants found: {roger_variants}")
    
    # Step 4: Test fuzzy matching manually
    print(f"\n4️⃣ Testing manual fuzzy matching...")
    all_text = " ".join([result[1] for result in ocr_results])
    
    # Test specific hero names that might match
    test_heroes = ['roger', 'chou', 'franco', 'miya', 'kagura']
    
    for hero in test_heroes:
        # Test if this hero appears in the text
        if hero in all_text.lower():
            print(f"✅ Found '{hero}' in OCR text!")
        else:
            # Test fuzzy matching
            from difflib import SequenceMatcher
            best_similarity = 0
            best_match = ""
            
            for result in ocr_results:
                similarity = SequenceMatcher(None, hero, result[1].lower()).ratio()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = result[1]
            
            print(f"'{hero}' - Best match: '{best_match}' (similarity: {best_similarity:.3f})")

def debug_enhanced_pipeline():
    """Debug the enhanced data collector's hero integration"""
    print(f"\n🔍 DEBUGGING ENHANCED PIPELINE INTEGRATION")
    print("=" * 60)
    
    screenshot_path = "Screenshot-Test/screenshot-test-1.PNG"
    ign = "Lesz XVII"
    
    # Test with manual hero override to verify pipeline
    print("Testing with manual hero override...")
    result_with_override = enhanced_data_collector.analyze_screenshot_with_session(
        image_path=screenshot_path,
        ign=ign,
        hero_override="roger"  # Manual override
    )
    
    print(f"With override - Hero: {result_with_override['data'].get('hero', 'missing')}")
    print(f"With override - Confidence: {result_with_override.get('overall_confidence', 0):.1%}")
    
    # Test without override
    print("\nTesting without manual hero override...")
    result_no_override = enhanced_data_collector.analyze_screenshot_with_session(
        image_path=screenshot_path,
        ign=ign,
        hero_override=None
    )
    
    print(f"No override - Hero: {result_no_override['data'].get('hero', 'missing')}")
    print(f"No override - Confidence: {result_no_override.get('overall_confidence', 0):.1%}")
    
    # Compare the difference
    print(f"\nConfidence difference: {result_with_override.get('overall_confidence', 0) - result_no_override.get('overall_confidence', 0):.1%}")

if __name__ == "__main__":
    debug_hero_pipeline()
    debug_enhanced_pipeline()