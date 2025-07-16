#!/usr/bin/env python3
"""
Debug test to see what data the AdvancedPerformanceAnalyzer receives and returns
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from core.advanced_performance_analyzer import advanced_performance_analyzer

def test_analyzer_with_our_data():
    """Test the analyzer with the data we know is being extracted"""
    
    print("🔍 TESTING AdvancedPerformanceAnalyzer")
    print("=" * 60)
    
    # This is the data we know is being extracted successfully
    test_match_data = {
        "hero": "unknown",
        "gold": 7586,
        "kills": 0,
        "deaths": 5,
        "assists": 1,
        "match_duration": 10,
        "match_result": "defeat",
        "player_rank": "Unknown",
        "screenshot_confidence": 0.65,
        "hero_damage": 0,
        "turret_damage": 0,
        "damage_taken": 0,
        "teamfight_participation": 0,
        "positioning_rating": "average",
        "ult_usage": "average",
        "gold_per_min": round(7586 / 10)  # 759
    }
    
    print("📊 INPUT DATA:")
    for key, value in test_match_data.items():
        print(f"  {key}: {value}")
    
    print(f"\n🔍 Calling advanced_performance_analyzer.analyze_comprehensive_performance...")
    
    try:
        # Call the analyzer
        performance_report = advanced_performance_analyzer.analyze_comprehensive_performance(test_match_data)
        
        print("✅ ANALYZER SUCCESS!")
        print(f"\n📊 PERFORMANCE REPORT:")
        print(f"  Hero: {performance_report.hero}")
        print(f"  Role: {performance_report.role}")
        print(f"  Overall Rating: {performance_report.overall_rating.value}")
        print(f"  Overall Score: {performance_report.overall_score:.3f}")
        
        print(f"\n📊 CORE METRICS:")
        print(f"  Combat Efficiency: {performance_report.combat_efficiency.value:.3f} (benchmark: {performance_report.combat_efficiency.benchmark:.3f}) -> {performance_report.combat_efficiency.category.value}")
        print(f"  Objective Participation: {performance_report.objective_participation.value:.3f} (benchmark: {performance_report.objective_participation.benchmark:.3f}) -> {performance_report.objective_participation.category.value}")
        print(f"  Economic Efficiency: {performance_report.economic_efficiency.value:.3f} (benchmark: {performance_report.economic_efficiency.benchmark:.3f}) -> {performance_report.economic_efficiency.category.value}")
        print(f"  Survival Rating: {performance_report.survival_rating.value:.3f} (benchmark: {performance_report.survival_rating.benchmark:.3f}) -> {performance_report.survival_rating.category.value}")
        
        print(f"\n📊 ANALYSIS:")
        print(f"  Strengths: {performance_report.strengths}")
        print(f"  Weaknesses: {performance_report.weaknesses}")
        print(f"  Improvement Priorities: {performance_report.improvement_priorities}")
        
        # Test different scenarios
        print(f"\n🔍 TESTING EDGE CASES:")
        
        # Test with minimal data
        minimal_data = {
            "hero": "unknown",
            "kills": 0,
            "deaths": 1,
            "assists": 0,
            "gold": 0
        }
        
        print(f"\n📊 Testing with minimal data...")
        minimal_report = advanced_performance_analyzer.analyze_comprehensive_performance(minimal_data)
        print(f"  Overall Rating: {minimal_report.overall_rating.value}")
        print(f"  Overall Score: {minimal_report.overall_score:.3f}")
        
        # Test with empty data
        empty_data = {}
        print(f"\n📊 Testing with empty data...")
        empty_report = advanced_performance_analyzer.analyze_comprehensive_performance(empty_data)
        print(f"  Overall Rating: {empty_report.overall_rating.value}")
        print(f"  Overall Score: {empty_report.overall_score:.3f}")
        
    except Exception as e:
        print(f"❌ ANALYZER FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_analyzer_with_our_data() 