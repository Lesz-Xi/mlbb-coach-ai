#!/usr/bin/env python3
"""
Test script to validate that our fixes are working:
1. Row-specific hero detection
2. MVP performance boost
3. Result structure fixes
"""
import sys
sys.path.append('.')

from core.enhanced_data_collector import EnhancedDataCollector
from core.mvp_performance_booster import mvp_performance_booster
from core.advanced_performance_analyzer import AdvancedPerformanceAnalyzer, PerformanceCategory

def test_fixes():
    """Test all critical fixes."""
    print("🔧 TESTING CRITICAL FIXES")
    print("=" * 50)
    
    # Test 1: Row-specific hero detection
    print("🎯 TEST 1: Row-Specific Hero Detection")
    print("-" * 40)
    
    try:
        collector = EnhancedDataCollector()
        result = collector.analyze_screenshot_with_session(
            image_path="Screenshot-Test/screenshot-test-1.PNG",
            ign="Lesz XVII",
            session_id=None,
            hero_override=None
        )
        
        # Extract data correctly
        match_data = result.get('data', {})
        hero_detected = match_data.get('hero', 'unknown')
        
        print(f"✅ Hero detection test completed")
        print(f"   Hero detected: {hero_detected}")
        print(f"   Data fields: {list(match_data.keys())}")
        
        # Check if we have Lesz XVII's actual data
        kda = f"{match_data.get('kills', 0)}/{match_data.get('deaths', 0)}/{match_data.get('assists', 0)}"
        print(f"   KDA extracted: {kda}")
        print(f"   Gold extracted: {match_data.get('gold', 0)}")
        
    except Exception as e:
        print(f"❌ Row-specific hero detection test failed: {str(e)}")
        match_data = {}
    
    # Test 2: MVP Performance Boost
    print("\n🏆 TEST 2: MVP Performance Boost")
    print("-" * 40)
    
    # Create test data simulating Mathilda MVP scenario
    test_mvp_data = {
        'hero': 'mathilda',
        'kills': 2,
        'deaths': 2,
        'assists': 17,
        'teamfight_participation': 90,  # High TFP - MVP indicator
        'match_result': 'victory',
        'gold': 7344,
        'hero_damage': 20959,
        'damage_taken': 44655
    }
    
    try:
        analyzer = AdvancedPerformanceAnalyzer()
        
        # Test original rating (would be Poor)
        original_report = analyzer.analyze_comprehensive_performance(test_mvp_data)
        
        print(f"✅ MVP performance boost test completed")
        print(f"   Original rating: {original_report.overall_rating.value}")
        print(f"   TFP: {test_mvp_data['teamfight_participation']}%")
        print(f"   Match result: {test_mvp_data['match_result']}")
        print(f"   Hero: {test_mvp_data['hero']} (support)")
        
        # Test MVP boost directly
        boost_result = mvp_performance_booster.analyze_and_boost_performance(
            test_mvp_data, 
            PerformanceCategory.POOR,  # Simulating original poor rating
            0.5,  # Low original score
            ocr_results=None
        )
        
        boosted_rating, boosted_score, boost_analysis = boost_result
        
        print(f"\n   MVP Boost Analysis:")
        print(f"   - MVP detected: {boost_analysis['mvp_detected']}")
        print(f"   - High TFP: {boost_analysis['high_teamfight_participation']}")
        print(f"   - Victory bonus: {boost_analysis['victory_bonus']}")
        print(f"   - Boost applied: {boost_analysis['boost_applied']}")
        print(f"   - Final rating: {boosted_rating.value}")
        
        if boost_analysis['boost_reasons']:
            print(f"   - Boost reasons: {', '.join(boost_analysis['boost_reasons'])}")
        
    except Exception as e:
        print(f"❌ MVP performance boost test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Result structure validation
    print("\n📊 TEST 3: Result Structure Validation")
    print("-" * 40)
    
    try:
        if 'result' in locals():
            print(f"✅ Result structure validation")
            print(f"   Result type: {type(result)}")
            print(f"   Keys available: {list(result.keys())}")
            
            if 'data' in result:
                data = result['data']
                print(f"   Data type: {type(data)}")
                print(f"   Data fields: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                
                # Check if we have real extracted data
                if isinstance(data, dict) and data.get('gold', 0) > 0:
                    print(f"   ✅ Real data extracted: Gold={data.get('gold')}")
                else:
                    print(f"   ❌ No real data extracted")
            
            if 'session_id' in result:
                print(f"   Session ID: {result['session_id']}")
        else:
            print("❌ No result object to validate")
            
    except Exception as e:
        print(f"❌ Result structure validation failed: {str(e)}")
    
    # Test 4: Support hero criteria
    print("\n🛡️ TEST 4: Support Hero Criteria")
    print("-" * 40)
    
    support_scenarios = [
        {
            'name': 'Mathilda MVP',
            'data': {
                'hero': 'mathilda',
                'teamfight_participation': 90,
                'assists': 17,
                'deaths': 2,
                'match_result': 'victory'
            }
        },
        {
            'name': 'Estes High Support',
            'data': {
                'hero': 'estes',
                'teamfight_participation': 85,
                'assists': 15,
                'deaths': 3,
                'match_result': 'victory'
            }
        }
    ]
    
    for scenario in support_scenarios:
        try:
            boost_result = mvp_performance_booster.analyze_and_boost_performance(
                scenario['data'],
                PerformanceCategory.POOR,
                0.5
            )
            
            boosted_rating, boosted_score, boost_analysis = boost_result
            
            print(f"   {scenario['name']}:")
            print(f"     Rating: Poor -> {boosted_rating.value}")
            print(f"     TFP boost: {boost_analysis.get('tfp_boost_factor', 1.0)}x")
            print(f"     Support detected: {boost_analysis.get('detected_role') == 'support'}")
            
        except Exception as e:
            print(f"   ❌ {scenario['name']} failed: {str(e)}")
    
    print("\n✅ ALL TESTS COMPLETED")
    print("=" * 50)

if __name__ == "__main__":
    test_fixes() 