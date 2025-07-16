#!/usr/bin/env python3
"""
Debug script to analyze the specific user screenshots and identify
why Mathilda is detected as Miya and rated as Poor.
"""
import sys
sys.path.append('.')

from core.enhanced_data_collector import EnhancedDataCollector
from core.advanced_performance_analyzer import AdvancedPerformanceAnalyzer
import json

def debug_user_screenshots():
    """Debug the specific user issue."""
    print("🔍 DEBUGGING USER SCREENSHOT ISSUES")
    print("=" * 50)
    
    # Initialize systems
    collector = EnhancedDataCollector()
    
    # Correct paths (screenshots are in same directory)
    screenshot1 = "Screenshot-Test/screenshot-test-1.PNG"  # Scoreboard
    screenshot2 = "Screenshot-Test/screenshot-test-2.PNG"  # Stats
    
    print("📊 ANALYZING SCREENSHOT 1 (Scoreboard):")
    print("-" * 40)
    
    try:
        result1 = collector.analyze_screenshot_with_session(
            image_path=screenshot1,
            ign="Lesz XVII",
            session_id=None,
            hero_override=None
        )
        
        print("✅ Screenshot 1 analysis completed")
        
        # Extract result data properly
        if hasattr(result1, 'match_data'):
            match_data1 = result1.match_data
        elif isinstance(result1, dict):
            match_data1 = result1.get('match_data', result1)
        else:
            match_data1 = result1
            
        print(f"Hero detected: {match_data1.get('hero', 'unknown')}")
        print(f"KDA: {match_data1.get('kills', 0)}/{match_data1.get('deaths', 0)}/{match_data1.get('assists', 0)}")
        print(f"Gold: {match_data1.get('gold', 0)}")
        print(f"Match result: {match_data1.get('match_result', 'unknown')}")
        
        # Get session info
        session_id = None
        if hasattr(result1, 'session_info'):
            session_id = result1.session_info.get('session_id')
        elif isinstance(result1, dict) and 'session_info' in result1:
            session_id = result1['session_info'].get('session_id')
            
        print(f"Session ID: {session_id}")
        
    except Exception as e:
        print(f"❌ Error analyzing screenshot 1: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n📊 ANALYZING SCREENSHOT 2 (Stats Page):")
    print("-" * 40)
    
    try:
        result2 = collector.analyze_screenshot_with_session(
            image_path=screenshot2,
            ign="Lesz XVII",
            session_id=session_id,
            hero_override=None
        )
        
        print("✅ Screenshot 2 analysis completed")
        
        # Extract result data properly
        if hasattr(result2, 'match_data'):
            match_data2 = result2.match_data
        elif isinstance(result2, dict):
            match_data2 = result2.get('match_data', result2)
        else:
            match_data2 = result2
            
        print(f"Hero detected: {match_data2.get('hero', 'unknown')}")
        print(f"Hero damage: {match_data2.get('hero_damage', 0)}")
        print(f"Damage taken: {match_data2.get('damage_taken', 0)}")
        print(f"Teamfight participation: {match_data2.get('teamfight_participation', 0)}%")
        
        # Check session reuse
        session_id2 = None
        if hasattr(result2, 'session_info'):
            session_id2 = result2.session_info.get('session_id')
        elif isinstance(result2, dict) and 'session_info' in result2:
            session_id2 = result2['session_info'].get('session_id')
            
        print(f"Session reused: {session_id == session_id2}")
        
    except Exception as e:
        print(f"❌ Error analyzing screenshot 2: {str(e)}")
        result2 = None
    
    print("\n🔍 PERFORMANCE ANALYSIS:")
    print("-" * 40)
    
    # Use the final result
    final_match_data = match_data2 if result2 else match_data1
    
    try:
        # Analyze with AdvancedPerformanceAnalyzer
        analyzer = AdvancedPerformanceAnalyzer()
        performance_report = analyzer.analyze_comprehensive_performance(final_match_data)
        
        print(f"Overall Rating: {performance_report.overall_rating.value}")
        print(f"Overall Score: {performance_report.overall_score:.2f}")
        
        # Print key metrics
        print("\nKey Metrics:")
        print(f"- Combat Efficiency: {performance_report.combat_efficiency:.2f}")
        print(f"- Objective Participation: {performance_report.objective_participation:.2f}")
        print(f"- Economic Efficiency: {performance_report.economic_efficiency:.2f}")
        print(f"- Survival Rating: {performance_report.survival_rating:.2f}")
        
        # Print strengths and weaknesses
        if performance_report.strengths:
            print(f"\nStrengths: {', '.join(performance_report.strengths)}")
        if performance_report.weaknesses:
            print(f"Weaknesses: {', '.join(performance_report.weaknesses)}")
        
    except Exception as e:
        print(f"❌ Error in performance analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n🚨 ISSUE ANALYSIS:")
    print("-" * 40)
    
    # Analyze the specific issues
    detected_hero = final_match_data.get('hero', 'unknown')
    kda_kills = final_match_data.get('kills', 0)
    kda_deaths = final_match_data.get('deaths', 1)
    kda_assists = final_match_data.get('assists', 0)
    tfp = final_match_data.get('teamfight_participation', 0)
    match_result = final_match_data.get('match_result', 'unknown')
    
    print(f"Detected Hero: {detected_hero}")
    print(f"Expected Hero: Mathilda (support)")
    print(f"KDA: {kda_kills}/{kda_deaths}/{kda_assists}")
    print(f"KDA Ratio: {(kda_kills + kda_assists) / max(kda_deaths, 1):.2f}")
    print(f"Teamfight Participation: {tfp}%")
    print(f"Match Result: {match_result}")
    
    # Issue 1: Hero Detection
    if detected_hero == 'miya':
        print("\n❌ ISSUE 1: Hero Misidentification")
        print("   - System detected Miya (marksman)")
        print("   - Player actually used Mathilda (support)")
        print("   - This affects performance criteria and expectations")
    
    # Issue 2: MVP Performance Rating
    if tfp >= 80 and match_result == 'victory':
        print("\n❌ ISSUE 2: MVP Performance Rated as Poor")
        print("   - High teamfight participation (90%) indicates MVP play")
        print("   - Victory match should boost rating")
        print("   - Support with 90% TFP should be rated highly")
    
    # Issue 3: Support vs Marksman Criteria
    if detected_hero == 'miya':
        print("\n❌ ISSUE 3: Wrong Performance Criteria Applied")
        print("   - Miya (marksman) judged on damage/kills")
        print("   - Mathilda (support) should be judged on assists/teamfight")
        print("   - Performance analysis using wrong hero expectations")
    
    print("\n💡 SOLUTIONS NEEDED:")
    print("-" * 40)
    print("1. Fix hero detection to recognize Mathilda correctly")
    print("2. Implement MVP/high TFP performance boost")
    print("3. Apply support-specific performance criteria")
    print("4. Ensure multi-screenshot session merges data properly")
    print("5. Validate player row identification for 'Lesz XVII'")

if __name__ == "__main__":
    debug_user_screenshots() 