#!/usr/bin/env python3
"""
Debug script to isolate numpy.bool_ error in Enhanced Mode
"""

import sys
import traceback
import numpy as np
import cv2

try:
    # Add the project root to the path
    sys.path.append('.')
    
    from core.enhanced_data_collector import enhanced_data_collector
    
    print("🔍 DEBUG: Testing Enhanced Mode pipeline step by step...")
    
    # Test 1: Basic image loading
    print("\n📤 Step 1: Loading image...")
    image_path = "Screenshot-Test/screenshot-test-1.PNG"
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Failed to load image")
        exit(1)
    print("✅ Image loaded successfully")
    
    # Test 2: Image quality assessment
    print("\n📤 Step 2: Testing image quality assessment...")
    try:
        quality_score = enhanced_data_collector._assess_image_quality(image)
        print(f"✅ Quality score: {quality_score}")
        print(f"   Type: {type(quality_score)}")
    except Exception as e:
        print(f"❌ Quality assessment failed: {e}")
        traceback.print_exc()
    
    # Test 3: Image preprocessing
    print("\n📤 Step 3: Testing image preprocessing...")
    try:
        processed_image = enhanced_data_collector._preprocess_image_enhanced(image_path)
        print(f"✅ Preprocessing completed")
        print(f"   Shape: {processed_image.shape}")
        print(f"   Type: {type(processed_image)}")
        print(f"   Dtype: {processed_image.dtype}")
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        traceback.print_exc()
    
    # Test 4: Full analysis (most likely to fail)
    print("\n📤 Step 4: Testing full Enhanced Mode analysis...")
    try:
        result = enhanced_data_collector.analyze_screenshot_with_session(
            image_path=image_path,
            ign="Lesz XVII",
            session_id=None,
            hero_override=None
        )
        print("✅ Full analysis completed")
        print(f"   Result keys: {list(result.keys())}")
        
        # Check for numpy types in the result
        def check_numpy_types(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    check_numpy_types(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_numpy_types(item, f"{path}[{i}]")
            elif isinstance(obj, (np.bool_, np.integer, np.floating)):
                print(f"⚠️  Found numpy type at {path}: {type(obj)} = {obj}")
        
        print("\n📤 Checking result for numpy types...")
        check_numpy_types(result, "result")
        
    except Exception as e:
        print(f"❌ Full analysis failed: {e}")
        traceback.print_exc()
    
    print("\n🎯 Debug test completed!")
    
except Exception as e:
    print(f"❌ Test setup failed: {e}")
    traceback.print_exc() 