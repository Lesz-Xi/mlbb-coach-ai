#!/usr/bin/env python3
"""
Verbose debug test to understand why OCR/parsing is failing
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from core.data_collector import get_ocr_reader
from core.enhanced_data_collector import EnhancedDataCollector

def test_step_by_step():
    """Test each step individually to find the problem"""
    
    screenshot_path = "Screenshot-Test/screenshot-test-1.PNG"
    if not os.path.exists(screenshot_path):
        print(f"❌ Screenshot not found: {screenshot_path}")
        return
    
    print("🔍 STEP-BY-STEP DEBUG TEST")
    print("=" * 60)
    
    # Step 1: Load original image
    print("1️⃣ Loading original image...")
    original = cv2.imread(screenshot_path)
    if original is None:
        print("❌ Failed to load original image")
        return
    print(f"✅ Original image: {original.shape}")
    
    # Step 2: Test quality assessment
    print("\n2️⃣ Testing quality assessment...")
    collector = EnhancedDataCollector()
    quality_score = collector._assess_image_quality(original)
    print(f"🔍 Quality score: {quality_score:.3f}")
    
    processing_type = 'minimal' if quality_score > 0.7 else 'moderate' if quality_score > 0.5 else 'aggressive'
    print(f"🔍 Processing type should be: {processing_type}")
    
    # Step 3: Test preprocessing
    print("\n3️⃣ Testing preprocessing...")
    processed = collector._preprocess_image_enhanced(screenshot_path)
    print(f"✅ Processed image: {processed.shape}")
    print(f"🔍 Processed image dtype: {processed.dtype}")
    print(f"🔍 Processed image range: {processed.min()} - {processed.max()}")
    
    # Check if processed image has content
    if processed.size == 0:
        print("❌ Processed image is EMPTY!")
        return
    elif processed.max() == processed.min():
        print(f"❌ Processed image has uniform values ({processed.max()})")
        return
    else:
        print("✅ Processed image has content")
    
    # Step 4: Test OCR directly
    print("\n4️⃣ Testing OCR directly...")
    reader = get_ocr_reader()
    ocr_results = reader.readtext(processed, detail=1)
    
    print(f"🔍 OCR found {len(ocr_results)} text detections")
    
    if len(ocr_results) == 0:
        print("❌ OCR returned NO TEXT! This is the problem!")
        
        # Try OCR on original image for comparison
        print("\n🔍 Testing OCR on ORIGINAL image...")
        gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        ocr_original = reader.readtext(gray_original, detail=1)
        print(f"🔍 OCR on original: {len(ocr_original)} detections")
        
        if len(ocr_original) > 0:
            print("✅ OCR works on original - preprocessing is destroying text!")
            print("📝 Sample text from original:")
            for i, (bbox, text, conf) in enumerate(ocr_original[:5]):
                print(f"  {i+1}. '{text}' (conf: {conf:.3f})")
        else:
            print("❌ OCR fails on both - might be wrong image type")
            
    else:
        print("✅ OCR found text:")
        for i, (bbox, text, conf) in enumerate(ocr_results[:10]):
            print(f"  {i+1}. '{text}' (conf: {conf:.3f})")
    
    # Step 5: Check if IGN is found
    print(f"\n5️⃣ Looking for IGN 'Lesz XVII' in OCR results...")
    ign = "Lesz XVII"
    ign_found = False
    
    for bbox, text, conf in ocr_results:
        if ign.lower() in text.lower() or any(word in text.lower() for word in ign.lower().split()):
            print(f"✅ Found IGN-related text: '{text}' (conf: {conf:.3f})")
            ign_found = True
    
    if not ign_found:
        print("❌ IGN 'Lesz XVII' NOT FOUND in OCR results")
        print("📝 All detected text:")
        for bbox, text, conf in ocr_results:
            print(f"  - '{text}'")
    
    print("\n" + "=" * 60)
    print("🔍 SUMMARY:")
    print(f"  Quality Score: {quality_score:.3f}")
    print(f"  Processing: {processing_type}")
    print(f"  OCR Detections: {len(ocr_results)}")
    print(f"  IGN Found: {ign_found}")
    
    if len(ocr_results) == 0:
        print("🚨 ROOT CAUSE: OCR is returning empty results!")
        print("🔧 SOLUTION: Need to fix preprocessing or try different OCR settings")

if __name__ == "__main__":
    test_step_by_step() 