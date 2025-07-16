#!/usr/bin/env python3
"""
Test to isolate API response serialization issues
"""

import sys
import json
import numpy as np
import traceback

try:
    # Add the project root to the path
    sys.path.append('.')
    
    from core.enhanced_data_collector import enhanced_data_collector
    from core.advanced_performance_analyzer import advanced_performance_analyzer
    
    print("🔍 TESTING API Response Serialization...")
    
    # Step 1: Get the Enhanced Mode result
    print("\n📤 Step 1: Getting Enhanced Mode analysis result...")
    image_path = "Screenshot-Test/screenshot-test-1.PNG"
    
    result = enhanced_data_collector.analyze_screenshot_with_session(
        image_path=image_path,
        ign="Lesz XVII",
        session_id=None,
        hero_override=None
    )
    
    match_data = result.get("data", {})
    print(f"✅ Enhanced analysis complete")
    
    # Step 2: Test AdvancedPerformanceAnalyzer
    print("\n📤 Step 2: Testing AdvancedPerformanceAnalyzer...")
    performance_report = advanced_performance_analyzer.analyze_comprehensive_performance(match_data)
    print(f"✅ Performance analysis complete")
    print(f"   Overall rating: {performance_report.overall_rating}")
    print(f"   Rating type: {type(performance_report.overall_rating)}")
    print(f"   Rating value: {performance_report.overall_rating.value}")
    print(f"   Value type: {type(performance_report.overall_rating.value)}")
    
    # Step 3: Check for numpy types in performance report
    print("\n📤 Step 3: Checking performance report for numpy types...")
    
    def check_numpy_types_recursive(obj, path=""):
        numpy_types_found = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                numpy_types_found.extend(check_numpy_types_recursive(value, f"{path}.{key}"))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                numpy_types_found.extend(check_numpy_types_recursive(item, f"{path}[{i}]"))
        elif hasattr(obj, '__dict__'):  # Object with attributes
            for attr_name in dir(obj):
                if not attr_name.startswith('_'):  # Skip private attributes
                    try:
                        attr_value = getattr(obj, attr_name)
                        if not callable(attr_value):  # Skip methods
                            numpy_types_found.extend(check_numpy_types_recursive(attr_value, f"{path}.{attr_name}"))
                    except:
                        pass
        elif isinstance(obj, (np.bool_, np.integer, np.floating)):
            numpy_types_found.append((path, type(obj), obj))
        
        return numpy_types_found
    
    numpy_issues = check_numpy_types_recursive(performance_report, "performance_report")
    
    if numpy_issues:
        print("⚠️  Found numpy types in performance report:")
        for path, type_obj, value in numpy_issues:
            print(f"    {path}: {type_obj} = {value}")
    else:
        print("✅ No numpy types found in performance report")
    
    # Step 4: Test JSON serialization of the complete API response structure
    print("\n📤 Step 4: Testing complete API response serialization...")
    
    # Simulate the API response structure from the web app
    api_response = {
        "feedback": [
            {
                "type": "info",
                "message": "Test message",
                "category": "Performance"
            }
        ],
        "mental_boost": "Test boost",
        "overall_rating": performance_report.overall_rating.value.title(),
        "session_info": {
            "session_id": result.get("session_id"),
            "screenshot_type": result.get("screenshot_type"),
            "type_confidence": result.get("type_confidence", 0),
            "session_complete": result.get("session_complete", False),
            "screenshot_count": 1
        },
        "debug_info": result.get("debug_info", {}),
        "warnings": result.get("warnings", []),
        "diagnostics": {
            "hero_detected": True,
            "confidence_score": result.get("overall_confidence", 0.0),
            "analysis_state": "complete"
        }
    }
    
    # Check for numpy types in API response
    print("\n📤 Checking API response for numpy types...")
    numpy_issues = check_numpy_types_recursive(api_response, "api_response")
    
    if numpy_issues:
        print("⚠️  Found numpy types in API response:")
        for path, type_obj, value in numpy_issues:
            print(f"    {path}: {type_obj} = {value}")
    else:
        print("✅ No numpy types found in API response")
    
    # Step 5: Test JSON serialization
    print("\n📤 Step 5: Testing JSON serialization...")
    try:
        json_str = json.dumps(api_response)
        print("✅ JSON serialization successful")
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        traceback.print_exc()
    
    print("\n🎯 API response test completed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    traceback.print_exc() 