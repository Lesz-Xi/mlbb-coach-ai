#!/usr/bin/env python3
"""
Test script to verify the enhanced performance rating system.
Tests various scenarios including Bronze defeats to ensure proper rating logic.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.advanced_performance_analyzer import AdvancedPerformanceAnalyzer, PerformanceCategory

def test_bronze_defeat_scenario():
    """Test that Bronze defeats with poor performance get appropriate ratings."""
    analyzer = AdvancedPerformanceAnalyzer()
    
    # Simulate Bronze defeat with poor KDA
    bronze_defeat_data = {
        "hero": "miya",
        "kills": 0,
        "deaths": 5,
        "assists": 1,
        "gold_per_min": 300,
        "hero_damage": 15000,
        "damage_taken": 25000,
        "teamfight_participation": 60,
        "turret_damage": 2000,
        "gold": 4500,
        "match_duration": 15,
        "match_result": "defeat",
        "player_rank": "Bronze II"
    }
    
    report = analyzer.analyze_comprehensive_performance(bronze_defeat_data)
    
    print("=== Bronze Defeat Test ===")
    print(f"Overall Rating: {report.overall_rating.value}")
    print(f"Overall Score: {report.overall_score:.2f}")
    print(f"KDA: {(bronze_defeat_data['kills'] + bronze_defeat_data['assists']) / bronze_defeat_data['deaths']:.2f}")
    print(f"Match Result: {bronze_defeat_data['match_result']}")
    print(f"Player Rank: {bronze_defeat_data['player_rank']}")
    
    # Should NOT be Excellent
    assert report.overall_rating != PerformanceCategory.EXCELLENT, "Bronze defeat should not be rated Excellent!"
    
    # Should be Average or lower due to poor KDA and defeat
    assert report.overall_rating in [
        PerformanceCategory.POOR, 
        PerformanceCategory.NEEDS_WORK, 
        PerformanceCategory.AVERAGE
    ], f"Expected lower rating for Bronze defeat, got {report.overall_rating.value}"
    
    print("✅ Bronze defeat test passed!")
    return True

def test_victory_scenario():
    """Test that good performance in victory gets appropriate rating."""
    analyzer = AdvancedPerformanceAnalyzer()
    
    # Simulate good performance in victory
    victory_data = {
        "hero": "chou",
        "kills": 8,
        "deaths": 2,
        "assists": 12,
        "gold_per_min": 480,
        "hero_damage": 45000,
        "damage_taken": 18000,
        "teamfight_participation": 85,
        "turret_damage": 8000,
        "gold": 14400,
        "match_duration": 18,
        "match_result": "victory",
        "player_rank": "Epic III"
    }
    
    report = analyzer.analyze_comprehensive_performance(victory_data)
    
    print("\n=== Victory Test ===")
    print(f"Overall Rating: {report.overall_rating.value}")
    print(f"Overall Score: {report.overall_score:.2f}")
    print(f"KDA: {(victory_data['kills'] + victory_data['assists']) / victory_data['deaths']:.2f}")
    print(f"Match Result: {victory_data['match_result']}")
    print(f"Player Rank: {victory_data['player_rank']}")
    
    # Should be Good or Excellent for strong performance in victory
    assert report.overall_rating in [
        PerformanceCategory.GOOD, 
        PerformanceCategory.EXCELLENT
    ], f"Expected higher rating for strong victory, got {report.overall_rating.value}"
    
    print("✅ Victory test passed!")
    return True

def test_poor_kda_defeat():
    """Test that very poor KDA in defeat is rated appropriately regardless of other metrics."""
    analyzer = AdvancedPerformanceAnalyzer()
    
    # Simulate poor KDA in defeat (should cap rating)
    poor_kda_data = {
        "hero": "layla",
        "kills": 1,
        "deaths": 8,
        "assists": 2,
        "gold_per_min": 420,  # Decent farming
        "hero_damage": 35000,  # Decent damage
        "damage_taken": 22000,
        "teamfight_participation": 75,  # Good participation
        "turret_damage": 5000,
        "gold": 12600,
        "match_duration": 18,
        "match_result": "defeat",
        "player_rank": "Epic I"
    }
    
    report = analyzer.analyze_comprehensive_performance(poor_kda_data)
    
    print("\n=== Poor KDA Defeat Test ===")
    print(f"Overall Rating: {report.overall_rating.value}")
    print(f"Overall Score: {report.overall_score:.2f}")
    print(f"KDA: {(poor_kda_data['kills'] + poor_kda_data['assists']) / poor_kda_data['deaths']:.2f}")
    print(f"Match Result: {poor_kda_data['match_result']}")
    
    # Should be capped at Average or lower due to poor KDA (0.375)
    assert report.overall_rating in [
        PerformanceCategory.POOR, 
        PerformanceCategory.NEEDS_WORK, 
        PerformanceCategory.AVERAGE
    ], f"Expected capped rating for poor KDA defeat, got {report.overall_rating.value}"
    
    print("✅ Poor KDA defeat test passed!")
    return True

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    analyzer = AdvancedPerformanceAnalyzer()
    
    # Test with missing context (should still work)
    minimal_data = {
        "hero": "gusion",
        "kills": 5,
        "deaths": 3,
        "assists": 7,
        "gold_per_min": 400,
        "hero_damage": 28000,
        "damage_taken": 20000,
        "teamfight_participation": 70,
        "turret_damage": 3000,
        "gold": 10000,
        "match_duration": 15
        # No match_result or player_rank
    }
    
    report = analyzer.analyze_comprehensive_performance(minimal_data)
    
    print("\n=== Edge Case Test (Missing Context) ===")
    print(f"Overall Rating: {report.overall_rating.value}")
    print(f"Overall Score: {report.overall_score:.2f}")
    
    # Should still provide a rating
    assert report.overall_rating is not None, "Should provide rating even with missing context"
    
    print("✅ Edge case test passed!")
    return True

def main():
    """Run all rating system tests."""
    print("Testing Enhanced Performance Rating System")
    print("=" * 50)
    
    tests = [
        test_bronze_defeat_scenario,
        test_victory_scenario,
        test_poor_kda_defeat,
        test_edge_cases
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Rating system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the rating logic.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 