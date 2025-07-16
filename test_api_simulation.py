#!/usr/bin/env python3
"""
Test that simulates Enhanced Mode API call without FastAPI
"""

import sys
import traceback
import tempfile
import os
import shutil
import json

try:
    # Add the project root to the path
    sys.path.append('.')
    
    from core.enhanced_data_collector import enhanced_data_collector
    from core.advanced_performance_analyzer import advanced_performance_analyzer
    from coach import generate_feedback
    from core.mental_coach import MentalCoach
    from core.error_handler import error_handler
    
    print("🔍 TESTING Enhanced Mode API simulation...")
    
    # Simulate the exact API call logic
    print("\n📤 Step 1: Simulating file upload handling...")
    
    # Use a temporary file to save the upload (like the API does)
    source_file = "Screenshot-Test/screenshot-test-1.PNG"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        with open(source_file, 'rb') as src:
            shutil.copyfileobj(src, temp_file)
        temp_file_path = temp_file.name
    
    print(f"✅ Temporary file created: {temp_file_path}")
    
    try:
        # Step 2: Validation (like the API does)
        print("\n📤 Step 2: Testing validation...")
        validation_result = error_handler.validate_request(
            ign="Lesz XVII",
            file_path=temp_file_path
        )
        
        if not validation_result.is_valid:
            print(f"❌ Validation failed: {validation_result.error_message}")
            exit(1)
        
        print("✅ Validation passed")
        
        # Step 3: Enhanced analysis (like the API does)
        print("\n📤 Step 3: Testing enhanced analysis...")
        result = enhanced_data_collector.analyze_screenshot_with_session(
            image_path=temp_file_path,
            ign="Lesz XVII",
            session_id=None,
            hero_override=None
        )
        
        match_data = result.get("data", {})
        print("✅ Enhanced analysis completed")
        
        # Step 4: Generate feedback (like the API does)
        print("\n📤 Step 4: Testing feedback generation...")
        statistical_feedback = generate_feedback(match_data, include_severity=True)
        
        # Load player history for mental coach
        mental_coach = MentalCoach(player_history=[], player_goal="general_improvement")
        mental_boost = mental_coach.get_mental_boost(match_data)
        
        # Use AdvancedPerformanceAnalyzer for context-aware rating
        performance_report = advanced_performance_analyzer.analyze_comprehensive_performance(match_data)
        overall_rating = performance_report.overall_rating.value.title()
        
        print("✅ Feedback generation completed")
        
        # Step 5: Build the exact response structure (like the API does)
        print("\n📤 Step 5: Building API response structure...")
        
        # Extract hero detection info from debug data
        hero_debug = result.get("hero_debug", {})
        hero_confidence = hero_debug.get("hero_confidence", result.get("debug_info", {}).get("hero_confidence", 0.0))
        
        # Only mark hero as detected if confidence is above threshold (ensure Python bool)
        hero_detected = bool(match_data.get("hero", "unknown") != "unknown" and float(hero_confidence) >= 0.3)
        
        # Calculate granular confidence scores for each attribute
        kda_fields = ["kills", "deaths", "assists"]
        kda_present = [k in match_data for k in kda_fields]
        kda_confidence = float(sum(kda_present) / len(kda_fields)) if kda_fields else 0.0
        
        gold_value = match_data.get("gold", 0)
        gold_confidence = float(1.0 if gold_value > 1000 else 0.5 if gold_value > 0 else 0.0)
        
        # Consistency check: if overall confidence is 0, don't claim anything is detected
        overall_confidence = result.get("overall_confidence", 0.0)
        
        diagnostics = {
            # Hero Detection - ensure Python bool
            "hero_detected": bool(hero_detected if float(overall_confidence) > 0 else False),
            "hero_name": match_data.get("hero", "unknown") if hero_detected else "unknown",
            "hero_confidence": float(hero_confidence),
            
            # Match Info - ensure Python bool
            "match_duration_detected": bool(match_data.get("match_duration")) and bool(float(overall_confidence) > 0),
            "match_result_detected": bool(match_data.get("match_result")) and bool(float(overall_confidence) > 0),
            "player_rank_detected": bool(match_data.get("player_rank")) and bool(float(overall_confidence) > 0),
            
            # Core Stats with Confidence - ensure Python bool
            "gold_data_valid": bool(float(gold_confidence) > 0.5 and float(overall_confidence) > 0),
            "gold_confidence": float(gold_confidence),
            "kda_data_complete": bool(float(kda_confidence) >= 1.0 and float(overall_confidence) > 0),
            "kda_confidence": float(kda_confidence),
            
            # Additional Stats - ensure Python bool
            "damage_data_available": bool(match_data.get("hero_damage")) and bool(float(overall_confidence) > 0),
            "teamfight_data_available": bool(match_data.get("teamfight_participation")) and bool(float(overall_confidence) > 0),
            
            # Overall Metrics
            "ign_found": True,  # Enhanced mode has better IGN validation
            "confidence_score": float(overall_confidence),
            "warnings": result.get("warnings", []),
            "data_completeness": float(result.get("completeness_score", 0.0)),
            
            # Analysis Info
            "analysis_mode": "enhanced",
            "screenshot_type": result.get("screenshot_type", "unknown"),
            "type_confidence": float(result.get("type_confidence", 0.0)),
            "session_complete": bool(result.get("session_complete", False)),
            "hero_suggestions": result.get("hero_suggestions", [])[:3],  # Top 3 suggestions
            
            # Analysis State
            "analysis_state": "partial" if 10.0 < (overall_confidence * 100 if overall_confidence <= 1.0 else overall_confidence) < 60.0 else "complete" if (overall_confidence * 100 if overall_confidence <= 1.0 else overall_confidence) >= 60.0 else "failed"
        }
        
        print("✅ Response structure built")
        
        # Step 6: Test JSON serialization (like FastAPI does)
        print("\n📤 Step 6: Testing JSON serialization...")
        
        response = {
            "feedback": [{"type": "info", "message": "Test"}],
            "mental_boost": mental_boost,
            "overall_rating": overall_rating,
            "session_info": {
                "session_id": result.get("session_id"),
                "screenshot_type": result.get("screenshot_type"),
                "type_confidence": result.get("type_confidence", 0),
                "session_complete": result.get("session_complete", False),
                "screenshot_count": 1
            },
            "debug_info": result.get("debug_info", {}),
            "warnings": result.get("warnings", []),
            "diagnostics": diagnostics
        }
        
        # Try to serialize to JSON
        json_str = json.dumps(response)
        print("✅ JSON serialization successful")
        print(f"   Response size: {len(json_str)} characters")
        
        print("\n🎯 API simulation completed successfully!")
        
    finally:
        # Cleanup temporary file (like the API does)
        try:
            os.unlink(temp_file_path)
            print(f"✅ Temporary file cleaned up")
        except Exception:
            pass

except Exception as e:
    print(f"❌ API simulation failed: {e}")
    traceback.print_exc() 