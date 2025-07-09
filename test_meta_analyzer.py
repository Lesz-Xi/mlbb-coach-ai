#!/usr/bin/env python3
"""Test script to demonstrate the meta analyzer functionality."""

from core.meta_analyzer import MetaAnalyzer
from core.schemas import RecommendationRequest


def main():
    # Initialize meta analyzer
    print("🎮 SkillShift Meta Analyzer Demo")
    print("=" * 40)
    
    analyzer = MetaAnalyzer()
    
    # Test 1: Show tier list
    print("\n📊 Current Tier List:")
    tier_list = analyzer.generate_tier_list()
    for tier, heroes in tier_list.items():
        if heroes:
            print(f"\n{tier} Tier:")
            for hero in heroes[:3]:  # Show top 3 in each tier
                print(f"  • {hero.hero} - {hero.win_rate:.1f}% WR, {hero.pick_rate:.1f}% PR")
    
    # Test 2: Counter-pick recommendations
    print("\n🎯 Counter-Pick Recommendations:")
    enemy_draft = ["Angela", "Chou", "Melissa"]
    print(f"Enemy team: {', '.join(enemy_draft)}")
    
    counter_recs = analyzer.get_counter_recommendations(enemy_draft)
    for i, rec in enumerate(counter_recs[:5], 1):
        print(f"{i}. {rec.hero} ({rec.confidence:.2f} confidence)")
        print(f"   {rec.reasoning}")
    
    # Test 3: Meta recommendations
    print("\n🔥 Top Meta Picks:")
    meta_recs = analyzer.get_meta_recommendations()
    for i, rec in enumerate(meta_recs[:5], 1):
        print(f"{i}. {rec.hero} - Tier {rec.meta_data.tier}")
        print(f"   Win Rate: {rec.meta_data.win_rate:.1f}%, Pick Rate: {rec.meta_data.pick_rate:.1f}%")
    
    # Test 4: Performance comparison
    print("\n📈 Performance Comparison:")
    comparison = analyzer.compare_performance("Franco", 45.0)
    if comparison:
        print(f"Your Franco: {comparison.player_winrate:.1f}% WR")
        print(f"Meta average: {comparison.meta_winrate:.1f}% WR")
        print(f"Performance gap: {comparison.performance_gap:.1f}%")
        print(f"Percentile rank: {comparison.percentile_rank}th percentile")
        if comparison.improvement_areas:
            print("Improvement areas:")
            for area in comparison.improvement_areas:
                print(f"  • {area}")
    
    # Test 5: Ban priority list
    print("\n🚫 Ban Priority List:")
    ban_priorities = analyzer.get_ban_priority_list()
    for i, hero in enumerate(ban_priorities[:5], 1):
        print(f"{i}. {hero.hero} - {hero.win_rate:.1f}% WR, {hero.ban_rate:.1f}% BR")
    
    # Test 6: Individual hero analysis
    print("\n🦸 Individual Hero Analysis:")
    hero_name = "Lolita"
    hero_data = analyzer.get_hero_meta(hero_name)
    if hero_data:
        print(f"{hero_name}:")
        print(f"  Ranking: #{hero_data.ranking}")
        print(f"  Win Rate: {hero_data.win_rate:.1f}%")
        print(f"  Pick Rate: {hero_data.pick_rate:.1f}%")
        print(f"  Ban Rate: {hero_data.ban_rate:.1f}%")
        print(f"  Tier: {hero_data.tier}")
        print(f"  Meta Score: {hero_data.meta_score:.1f}")
        print(f"  Countered by: {', '.join(hero_data.counter_heroes[:5])}")


if __name__ == "__main__":
    main()