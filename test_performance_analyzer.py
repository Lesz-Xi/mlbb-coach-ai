#!/usr/bin/env python3
"""Test script for the Performance Analyzer."""

from core.performance_analyzer import PerformanceAnalyzer
from core.meta_analyzer import MetaAnalyzer


def main():
    print("📊 SkillShift Performance Analyzer Demo")
    print("=" * 45)
    
    # Initialize systems
    meta_analyzer = MetaAnalyzer()
    performance_analyzer = PerformanceAnalyzer(meta_analyzer)
    
    # Sample player statistics
    player_stats = {
        "Franco": {"win_rate": 45.0, "games_played": 25},
        "Angela": {"win_rate": 62.0, "games_played": 15},
        "Tigreal": {"win_rate": 48.0, "games_played": 30},
        "Layla": {"win_rate": 40.0, "games_played": 20},
        "Melissa": {"win_rate": 58.0, "games_played": 8},
    }
    
    print("\n📈 Player Statistics:")
    for hero, stats in player_stats.items():
        print(f"  {hero}: {stats['win_rate']:.1f}% WR ({stats['games_played']} games)")
    
    # Test 1: Individual performance analysis
    print("\n🎯 Individual Performance Analysis:")
    comparisons = performance_analyzer.analyze_player_performance(player_stats)
    
    for hero, comp in comparisons.items():
        print(f"\n{hero}:")
        print(f"  Player WR: {comp.player_winrate:.1f}%")
        print(f"  Meta WR: {comp.meta_winrate:.1f}%")
        print(f"  Gap: {comp.performance_gap:+.1f}%")
        print(f"  Percentile: {comp.percentile_rank}th")
        if comp.improvement_areas:
            print(f"  Improvement areas: {', '.join(comp.improvement_areas)}")
    
    # Test 2: Performance summary
    print("\n📋 Performance Summary:")
    summary = performance_analyzer.get_performance_summary(comparisons)
    
    overall = summary.get("overall_performance", {})
    print(f"  Overall gap: {overall.get('average_gap', 0):.1f}%")
    print(f"  Category: {overall.get('performance_category', 'unknown')}")
    print(f"  Heroes analyzed: {overall.get('heroes_analyzed', 0)}")
    
    if summary.get("strong_heroes"):
        print(f"\n  Strong heroes:")
        for hero, gap in summary["strong_heroes"]:
            print(f"    • {hero}: +{gap:.1f}%")
    
    if summary.get("weak_heroes"):
        print(f"\n  Weak heroes:")
        for hero, gap in summary["weak_heroes"]:
            print(f"    • {hero}: {gap:.1f}%")
    
    # Test 3: Improvement priorities
    print("\n🎯 Improvement Priorities:")
    priorities = performance_analyzer.identify_improvement_priorities(comparisons)
    
    for i, priority in enumerate(priorities, 1):
        print(f"\n{i}. {priority['hero']} (Priority: {priority['priority_score']:.2f})")
        print(f"   Gap: {priority['performance_gap']:.1f}%")
        print(f"   Meta Tier: {priority['meta_tier']}")
        print(f"   Reasoning: {priority['reasoning']}")
    
    # Test 4: Rank bracket comparison
    print("\n🏆 Rank Bracket Comparison (Epic Rank):")
    rank_analysis = performance_analyzer.compare_vs_rank_bracket(player_stats, "Epic")
    
    print(f"  Rank: {rank_analysis['rank_bracket']}")
    print(f"  Modifier: {rank_analysis['modifier_applied']:+d}% (higher expectations)")
    
    rank_summary = rank_analysis["summary"]["overall_performance"]
    print(f"  Adjusted gap: {rank_summary.get('average_gap', 0):.1f}%")
    print(f"  Adjusted category: {rank_summary.get('performance_category', 'unknown')}")
    
    # Test 5: Coaching insights
    print("\n💡 Coaching Insights:")
    insights = performance_analyzer.generate_coaching_insights(comparisons)
    
    for insight in insights:
        print(f"  {insight}")
    
    # Test 6: Historical trend analysis (simulated)
    print("\n📈 Historical Trend Analysis:")
    historical_data = [
        {
            "date": "2024-01-01",
            "player_stats": {
                "Franco": {"win_rate": 42.0, "games_played": 20},
                "Angela": {"win_rate": 58.0, "games_played": 12},
            }
        },
        {
            "date": "2024-01-15",
            "player_stats": {
                "Franco": {"win_rate": 45.0, "games_played": 25},
                "Angela": {"win_rate": 62.0, "games_played": 15},
            }
        }
    ]
    
    trends = performance_analyzer.track_performance_trends(historical_data)
    
    for hero, trend in trends.items():
        print(f"\n  {hero}:")
        print(f"    Trend: {trend['trend_direction']}")
        print(f"    Improvement rate: {trend['improvement_rate']:+.1f}% per period")
        print(f"    Consistency: {trend['consistency']:.2f}")


if __name__ == "__main__":
    main()