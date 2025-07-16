#!/usr/bin/env python3
"""
Test script to analyze the user's screenshots and identify root causes
of the Mathilda -> Miya misclassification and Poor rating issues.
"""
import sys
import os
sys.path.append('.')

from core.enhanced_data_collector import EnhancedDataCollector
from core.ultimate_parsing_system import ultimate_parsing_system
from core.performance_analyzer import PerformanceAnalyzer

def analyze_screenshots():
    """Analyze both screenshots to identify issues."""
    print("🔍 COMPREHENSIVE SCREENSHOT ANALYSIS")
    print("=" * 60)
    
    # Initialize systems
    enhanced_collector = EnhancedDataCollector()
    performance_analyzer = PerformanceAnalyzer()
    
    # Screenshot paths (assuming they're in Screenshot-Test directory)
    screenshot1 = "../Screenshot-Test/screenshot-test-1.PNG"  # Scoreboard
    screenshot2 = "../Screenshot-Test/screenshot-test-2.PNG"  # Stats
    
    print("📊 ANALYZING SCREENSHOT 1 (Scoreboard):")
    print("-" * 40)
    
    try:
        result1 = enhanced_collector.analyze_screenshot_with_session(
            image_path=screenshot1,
            ign="Lesz XVII",
            session_id=None,
            hero_override=None
        )
        
        print(f"✅ Analysis successful")
        print(f"Hero detected: {result1.match_data.get('hero', 'unknown')}")
        print(f"Hero confidence: {result1.debug_info.get('hero_confidence', 0):.1%}")
        print(f"KDA: {result1.match_data.get('kills', 0)}/{result1.match_data.get('deaths', 0)}/{result1.match_data.get('assists', 0)}")
        print(f"Gold: {result1.match_data.get('gold', 0)}")
        print(f"Match result: {result1.match_data.get('match_result', 'unknown')}")
        print(f"Overall confidence: {result1.confidence:.1%}")
        print(f"Session ID: {result1.session_info.get('session_id', 'None')}")
        
        # Check for MVP mentions
        all_text = str(result1.debug_info)
        mvp_detected = "mvp" in all_text.lower()
        print(f"MVP detected in text: {mvp_detected}")
        
    except Exception as e:
        print(f"❌ Error analyzing screenshot 1: {str(e)}")
        return
    
    print("\n📊 ANALYZING SCREENSHOT 2 (Stats Page):")
    print("-" * 40)
    
    try:
        result2 = enhanced_collector.analyze_screenshot_with_session(
            image_path=screenshot2,
            ign="Lesz XVII",
            session_id=result1.session_info.get('session_id'),
            hero_override=None
        )
        
        print(f"✅ Analysis successful")
        print(f"Hero detected: {result2.match_data.get('hero', 'unknown')}")
        print(f"Hero damage: {result2.match_data.get('hero_damage', 0)}")
        print(f"Damage taken: {result2.match_data.get('damage_taken', 0)}")
        print(f"Teamfight participation: {result2.match_data.get('teamfight_participation', 0)}%")
        print(f"Overall confidence: {result2.confidence:.1%}")
        print(f"Session reused: {result1.session_info.get('session_id') == result2.session_info.get('session_id')}")
        
    except Exception as e:
        print(f"❌ Error analyzing screenshot 2: {str(e)}")
        result2 = None
    
    print("\n🔍 PERFORMANCE ANALYSIS:")
    print("-" * 40)
    
    # Use the final result for performance analysis
    final_result = result2 if result2 else result1
    
    try:
        # Generate performance feedback
        feedback = performance_analyzer.analyze_performance(final_result.match_data)
        print(f"Performance rating: {feedback.get('overall_performance', 'Unknown')}")
        print(f"Rating explanation: {feedback.get('performance_explanation', 'None')}")
        
        # Check if MVP status was considered
        match_data = final_result.match_data
        kda = (match_data.get('kills', 0) + match_data.get('assists', 0)) / max(match_data.get('deaths', 1), 1)
        tfp = match_data.get('teamfight_participation', 0)
        
        print(f"KDA ratio: {kda:.2f}")
        print(f"Teamfight participation: {tfp}%")
        
        # MVP analysis
        if tfp >= 80:
            print("⭐ HIGH TEAMFIGHT PARTICIPATION - Should boost rating!")
        if match_data.get('match_result') == 'victory':
            print("🏆 VICTORY - Should boost rating!")
            
    except Exception as e:
        print(f"❌ Error in performance analysis: {str(e)}")
    
    print("\n🚨 ROOT CAUSE ANALYSIS:")
    print("-" * 40)
    
    # Hero detection analysis
    detected_hero = final_result.match_data.get('hero', 'unknown')
    if detected_hero == 'miya' and 'mathilda' not in detected_hero.lower():
        print("❌ ISSUE 1: Hero misidentification")
        print("   - Detected: Miya")
        print("   - Expected: Mathilda")
        print("   - Impact: Wrong hero affects performance expectations")
    
    # Row mapping analysis
    ign_found = result1.debug_info.get('ign_found', False)
    if not ign_found:
        print("❌ ISSUE 2: IGN not found")
        print("   - Could not locate 'Lesz XVII' in screenshot")
        print("   - May be analyzing wrong player row")
    
    # MVP analysis
    tfp = final_result.match_data.get('teamfight_participation', 0)
    match_result = final_result.match_data.get('match_result', 'unknown')
    
    if tfp >= 80 and match_result == 'victory':
        print("❌ ISSUE 3: MVP performance rated as 'Poor'")
        print(f"   - High teamfight participation: {tfp}%")
        print(f"   - Victory match")
        print("   - Should receive higher rating")
    
    # Session management analysis
    if result2:
        session_worked = result1.session_info.get('session_id') == result2.session_info.get('session_id')
        if not session_worked:
            print("❌ ISSUE 4: Multi-screenshot session failed")
            print("   - Screenshots analyzed separately")
            print("   - Missing cross-screenshot data enhancement")
    
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 40)
    print("1. Improve hero detection for support heroes")
    print("2. Enhance IGN matching for player row identification")
    print("3. Add MVP detection and rating boost logic")
    print("4. Fix multi-screenshot session management")
    print("5. Implement support-specific performance criteria")

if __name__ == "__main__":
    analyze_screenshots() 