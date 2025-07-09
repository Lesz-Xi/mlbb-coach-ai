#!/usr/bin/env python3
"""Test script for the Patch Monitor."""

import json
from datetime import datetime
from pathlib import Path

from core.patch_monitor import PatchMonitor
from core.meta_analyzer import MetaAnalyzer
from core.schemas import HeroMetaData, MetaLeaderboard


def create_sample_patch_data():
    """Create sample patch data for testing."""
    # Create patches directory
    patches_dir = Path("data/patches")
    patches_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample data for patch 1.7.8 (older patch)
    patch_178_data = {
        "patch_version": "1.7.8",
        "timestamp": "2024-01-01T00:00:00",
        "heroes": [
            {
                "ranking": 1, "hero": "Angela", "pick_rate": 0.8, "win_rate": 55.0,
                "ban_rate": 65.0, "counter_heroes": ["Chou", "Franco", "Jawhead"],
                "tier": "S", "meta_score": 35.0
            },
            {
                "ranking": 2, "hero": "Franco", "pick_rate": 0.2, "win_rate": 52.0,
                "ban_rate": 0.5, "counter_heroes": ["Karrie", "Claude", "Wanwan"],
                "tier": "A", "meta_score": 32.0
            },
            {
                "ranking": 3, "hero": "Melissa", "pick_rate": 0.5, "win_rate": 53.0,
                "ban_rate": 8.0, "counter_heroes": ["Chou", "Gusion", "Ling"],
                "tier": "A", "meta_score": 33.0
            },
            {
                "ranking": 4, "hero": "Tigreal", "pick_rate": 0.3, "win_rate": 51.0,
                "ban_rate": 5.0, "counter_heroes": ["Karrie", "Claude", "Wanwan"],
                "tier": "B", "meta_score": 31.0
            },
            {
                "ranking": 5, "hero": "Layla", "pick_rate": 0.4, "win_rate": 50.0,
                "ban_rate": 2.0, "counter_heroes": ["Chou", "Gusion", "Ling"],
                "tier": "B", "meta_score": 30.0
            }
        ]
    }
    
    # Sample data for patch 1.7.9 (newer patch) with some changes
    patch_179_data = {
        "patch_version": "1.7.9",
        "timestamp": "2024-01-15T00:00:00",
        "heroes": [
            {
                "ranking": 1, "hero": "Angela", "pick_rate": 1.0, "win_rate": 58.0,
                "ban_rate": 75.0, "counter_heroes": ["Chou", "Franco", "Jawhead"],
                "tier": "S", "meta_score": 40.0
            },
            {
                "ranking": 2, "hero": "Melissa", "pick_rate": 0.4, "win_rate": 55.0,
                "ban_rate": 12.0, "counter_heroes": ["Chou", "Gusion", "Ling"],
                "tier": "S", "meta_score": 35.0
            },
            {
                "ranking": 3, "hero": "Franco", "pick_rate": 0.1, "win_rate": 54.0,
                "ban_rate": 0.2, "counter_heroes": ["Karrie", "Claude", "Wanwan"],
                "tier": "B", "meta_score": 33.0
            },
            {
                "ranking": 4, "hero": "Tigreal", "pick_rate": 0.4, "win_rate": 53.0,
                "ban_rate": 8.0, "counter_heroes": ["Karrie", "Claude", "Wanwan"],
                "tier": "A", "meta_score": 34.0
            },
            {
                "ranking": 5, "hero": "Layla", "pick_rate": 0.3, "win_rate": 48.0,
                "ban_rate": 1.0, "counter_heroes": ["Chou", "Gusion", "Ling"],
                "tier": "C", "meta_score": 28.0
            }
        ]
    }
    
    # Save sample patches
    with open(patches_dir / "patch_1_7_8.json", 'w') as f:
        json.dump(patch_178_data, f, indent=2)
    
    with open(patches_dir / "patch_1_7_9.json", 'w') as f:
        json.dump(patch_179_data, f, indent=2)
    
    print("Created sample patch data")


def main():
    print("🔄 SkillShift Patch Monitor Demo")
    print("=" * 40)
    
    # Create sample data
    create_sample_patch_data()
    
    # Initialize patch monitor
    patch_monitor = PatchMonitor()
    
    # Test 1: Save current meta as a patch snapshot
    print("\n💾 Saving Current Meta as Patch Snapshot:")
    meta_analyzer = MetaAnalyzer()
    current_meta = meta_analyzer.meta_data
    
    if current_meta:
        patch_monitor.save_patch_snapshot("1.8.0", current_meta)
        print("  ✅ Current meta saved as patch 1.8.0")
    
    # Test 2: Get patch history
    print("\n📋 Available Patch History:")
    patch_history = patch_monitor.get_patch_history()
    for patch in patch_history:
        print(f"  📦 Patch {patch}")
    
    # Test 3: Compare patches
    print("\n🔍 Patch Comparison (1.7.8 → 1.7.9):")
    comparison = patch_monitor.compare_patches("1.7.8", "1.7.9")
    
    if comparison:
        print(f"  Patch: {comparison.patch_version}")
        print(f"  Release Date: {comparison.release_date}")
        print(f"  Heroes Changed: {comparison.total_heroes_changed}")
        print(f"  Meta Stability: {comparison.meta_stability:.2f}")
        
        print(f"\n  🎯 Major Changes:")
        for change in comparison.major_changes[:5]:
            print(f"    • {change.hero}: {change.old_tier} → {change.new_tier}")
            print(f"      WR: {change.old_win_rate:.1f}% → {change.new_win_rate:.1f}%")
            print(f"      Impact: {change.impact_score:.2f} ({change.change_type})")
            print()
        
        print(f"  📊 Tier Shifts:")
        if comparison.tier_shifts["promoted"]:
            print(f"    Promoted: {', '.join(comparison.tier_shifts['promoted'])}")
        if comparison.tier_shifts["demoted"]:
            print(f"    Demoted: {', '.join(comparison.tier_shifts['demoted'])}")
        
        print(f"  🎭 Affected Roles:")
        for role, count in comparison.affected_roles.items():
            print(f"    {role}: {count} heroes")
    
    # Test 4: Track hero evolution
    print("\n📈 Hero Evolution Tracking (Angela):")
    if len(patch_history) >= 2:
        evolution = patch_monitor.track_hero_evolution("Angela", patch_history)
        
        for metric, data in evolution.items():
            if data:
                print(f"  {metric.replace('_', ' ').title()}:")
                for point in data:
                    print(f"    {point['patch']}: {point['value']:.1f}")
    
    # Test 5: Meta trends analysis
    print("\n🔄 Meta Trends Analysis:")
    if len(patch_history) >= 2:
        trends = patch_monitor.identify_meta_trends(patch_history)
        
        if trends["rising_stars"]:
            print(f"  🌟 Rising Stars:")
            for star in trends["rising_stars"]:
                print(f"    • {star['hero']}: +{star['rank_change']} rank, +{star['wr_change']:.1f}% WR")
        
        if trends["falling_powers"]:
            print(f"  📉 Falling Powers:")
            for fallen in trends["falling_powers"]:
                print(f"    • {fallen['hero']}: {fallen['rank_change']} rank, {fallen['wr_change']:.1f}% WR")
        
        if trends["stable_picks"]:
            print(f"  ⚖️ Stable Picks:")
            for stable in trends["stable_picks"][:3]:
                print(f"    • {stable['hero']}: {stable['consistency']:.2f} consistency")
    
    # Test 6: Patch notes analysis
    print("\n📝 Patch Notes Analysis:")
    if comparison:
        analysis = patch_monitor.generate_patch_notes_analysis(comparison)
        
        if analysis["headline_changes"]:
            print(f"  Headlines:")
            for headline in analysis["headline_changes"]:
                print(f"    {headline}")
        
        if analysis["role_impact"]:
            print(f"  Role Impact:")
            for role, impact in analysis["role_impact"].items():
                print(f"    {role}: {impact}")
        
        if analysis["ban_priority_shifts"]:
            print(f"  Ban Priority Shifts:")
            for shift in analysis["ban_priority_shifts"]:
                print(f"    {shift}")
        
        if analysis["recommendation_updates"]:
            print(f"  Recommendation Updates:")
            for update in analysis["recommendation_updates"]:
                print(f"    ⚠️ {update}")


if __name__ == "__main__":
    main()