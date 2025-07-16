#!/usr/bin/env python3
"""
Incremental test to isolate Enhanced Mode numpy.bool_ error
"""

import sys
import traceback

try:
    # Add the project root to the path
    sys.path.append('.')
    
    print("🔍 TESTING Enhanced Mode step by step...")
    
    # Step 1: Test imports
    print("\n📤 Step 1: Testing imports...")
    try:
        from core.enhanced_data_collector import enhanced_data_collector
        print("✅ enhanced_data_collector imported")
        
        from core.advanced_performance_analyzer import advanced_performance_analyzer
        print("✅ advanced_performance_analyzer imported")
        
        from coach import generate_feedback
        print("✅ generate_feedback imported")
        
        from core.mental_coach import MentalCoach
        print("✅ MentalCoach imported")
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        exit(1)
    
    # Step 2: Test Enhanced Mode analysis
    print("\n📤 Step 2: Testing Enhanced Mode analysis...")
    try:
        image_path = "Screenshot-Test/screenshot-test-1.PNG"
        result = enhanced_data_collector.analyze_screenshot_with_session(
            image_path=image_path,
            ign="Lesz XVII",
            session_id=None,
            hero_override=None
        )
        match_data = result.get("data", {})
        print("✅ Enhanced analysis completed")
    except Exception as e:
        print(f"❌ Enhanced analysis failed: {e}")
        traceback.print_exc()
        exit(1)
    
    # Step 3: Test generate_feedback
    print("\n📤 Step 3: Testing generate_feedback...")
    try:
        statistical_feedback = generate_feedback(match_data, include_severity=True)
        print("✅ generate_feedback completed")
        print(f"   Feedback type: {type(statistical_feedback)}")
        print(f"   Feedback length: {len(statistical_feedback) if hasattr(statistical_feedback, '__len__') else 'N/A'}")
    except Exception as e:
        print(f"❌ generate_feedback failed: {e}")
        traceback.print_exc()
        exit(1)
    
    # Step 4: Test MentalCoach
    print("\n📤 Step 4: Testing MentalCoach...")
    try:
        mental_coach = MentalCoach(player_history=[], player_goal="general_improvement")
        mental_boost = mental_coach.get_mental_boost(match_data)
        print("✅ MentalCoach completed")
        print(f"   Mental boost type: {type(mental_boost)}")
    except Exception as e:
        print(f"❌ MentalCoach failed: {e}")
        traceback.print_exc()
        exit(1)
    
    # Step 5: Test AdvancedPerformanceAnalyzer
    print("\n📤 Step 5: Testing AdvancedPerformanceAnalyzer...")
    try:
        performance_report = advanced_performance_analyzer.analyze_comprehensive_performance(match_data)
        overall_rating = performance_report.overall_rating.value.title()
        print("✅ AdvancedPerformanceAnalyzer completed")
        print(f"   Overall rating: {overall_rating}")
    except Exception as e:
        print(f"❌ AdvancedPerformanceAnalyzer failed: {e}")
        traceback.print_exc()
        exit(1)
    
    # Step 6: Test feedback processing
    print("\n📤 Step 6: Testing feedback processing...")
    try:
        feedback_items = []
        for item in statistical_feedback:
            if isinstance(item, tuple) and len(item) == 2:
                severity, message = item
                feedback_items.append({
                    "type": severity,
                    "message": message,
                    "category": "Performance"
                })
            elif isinstance(item, dict):
                feedback_items.append({
                    "type": item.get("severity", "info"),
                    "message": item.get("feedback", ""),
                    "category": item.get("category", "General")
                })
            else:
                feedback_items.append({
                    "type": "info",
                    "message": str(item),
                    "category": "General"
                })
        print("✅ Feedback processing completed")
        print(f"   Feedback items: {len(feedback_items)}")
    except Exception as e:
        print(f"❌ Feedback processing failed: {e}")
        traceback.print_exc()
        exit(1)
    
    print("\n🎯 All Enhanced Mode steps completed successfully!")
    
except Exception as e:
    print(f"❌ Test setup failed: {e}")
    traceback.print_exc() 