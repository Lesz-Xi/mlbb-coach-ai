#!/usr/bin/env python3
"""
Team Behavior Analysis Demo for MLBB Coach AI
==============================================

This script demonstrates how to analyze behavioral data from a full 5-player squad
in a single match, identifying patterns of team synergy, coordination breakdowns,
role overlap, timing mismatches, and team comp dynamics.

Usage:
    python team_behavior_analysis_demo.py
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the team behavior analysis system
from core.services.team_behavior_service import create_team_behavior_service
from core.analytics.team_behavior_analyzer import create_team_behavior_analyzer
from core.utils.synergy_matrix_builder import create_synergy_matrix_builder


def create_sample_team_data() -> List[Dict[str, Any]]:
    """Create sample match data for a 5-player team."""
    return [
        {
            "player_id": "TankMaster",
            "hero": "franco",
            "kills": 2,
            "deaths": 6,
            "assists": 18,
            "gold_per_min": 2800,
            "hero_damage": 42500,
            "turret_damage": 8500,
            "damage_taken": 58000,
            "teamfight_participation": 95,
            "positioning_rating": "good",
            "ult_usage": "high",
            "match_duration": 18,
            "hooks_landed": 12,
            "team_engages": 8,
            "vision_score": 45
        },
        {
            "player_id": "FighterTop",
            "hero": "chou",
            "kills": 8,
            "deaths": 4,
            "assists": 6,
            "gold_per_min": 3200,
            "hero_damage": 67500,
            "turret_damage": 15000,
            "damage_taken": 35000,
            "teamfight_participation": 78,
            "positioning_rating": "good",
            "ult_usage": "high",
            "match_duration": 18
        },
        {
            "player_id": "JungleKing",
            "hero": "hayabusa",
            "kills": 15,
            "deaths": 3,
            "assists": 8,
            "gold_per_min": 4200,
            "hero_damage": 89500,
            "turret_damage": 12000,
            "damage_taken": 28000,
            "teamfight_participation": 85,
            "positioning_rating": "good",
            "ult_usage": "high",
            "match_duration": 18
        },
        {
            "player_id": "MidLaner",
            "hero": "kagura",
            "kills": 9,
            "deaths": 2,
            "assists": 12,
            "gold_per_min": 3800,
            "hero_damage": 78000,
            "turret_damage": 8000,
            "damage_taken": 25000,
            "teamfight_participation": 88,
            "positioning_rating": "good",
            "ult_usage": "high",
            "match_duration": 18
        },
        {
            "player_id": "ADCPro",
            "hero": "miya",
            "kills": 12,
            "deaths": 1,
            "assists": 9,
            "gold_per_min": 4500,
            "hero_damage": 95000,
            "turret_damage": 18000,
            "damage_taken": 18000,
            "teamfight_participation": 82,
            "positioning_rating": "good",
            "ult_usage": "average",
            "match_duration": 18
        }
    ]


async def demonstrate_team_behavior_analysis():
    """Demonstrate comprehensive team behavior analysis."""
    print("🎮 MLBB Coach AI - Team Behavior Analysis Demo")
    print("=" * 60)
    
    # Step 1: Create sample team data
    print("\n📊 Step 1: Creating Sample Team Data")
    team_data = create_sample_team_data()
    print(f"✅ Created data for {len(team_data)} players")
    
    # Display team composition
    print("\n👥 Team Composition:")
    for i, player in enumerate(team_data, 1):
        print(f"  {i}. {player['player_id']} - {player['hero']} "
              f"(KDA: {player['kills']}/{player['deaths']}/{player['assists']})")
    
    # Step 2: Initialize team behavior service
    print("\n🔧 Step 2: Initializing Team Behavior Service")
    try:
        team_service = await create_team_behavior_service()
        print("✅ Team behavior service initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize service: {e}")
        return
    
    # Step 3: Perform team behavior analysis
    print("\n🔍 Step 3: Analyzing Team Behavior Patterns")
    try:
        analysis_result = await team_service.analyze_team_behavior(
            match_data=team_data,
            match_id="demo_match_001",
            team_id="team_alpha",
            include_synergy_matrix=True
        )
        print("✅ Team behavior analysis completed")
        
        # Display analysis duration
        duration = analysis_result.get("team_behavior", {}).get("analysis_duration", 0)
        print(f"⏱️ Analysis completed in {duration:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Team behavior analysis failed: {e}")
        return
    
    # Step 4: Display comprehensive results
    print("\n📈 Step 4: Team Analysis Results")
    print("-" * 40)
    
    # Team Overview
    team_overview = analysis_result["formatted_output"]["team_overview"]
    print(f"\n🌟 **Team Overview**:")
    print(f"  • Synergy Level: {team_overview['synergy_level']}")
    print(f"  • Coordination Pattern: {team_overview['coordination_pattern']}")
    print(f"  • Role Overlap Severity: {team_overview['role_overlap_severity']}")
    print(f"  • Overall Effectiveness: {team_overview['overall_effectiveness']}")
    
    # Performance Metrics
    metrics = analysis_result["formatted_output"]["performance_metrics"]
    print(f"\n📊 **Performance Metrics**:")
    print(f"  • Teamfight Spacing: {metrics['teamfight_spacing']}")
    print(f"  • Objective Control: {metrics['objective_control']}")
    print(f"  • Rotation Sync: {metrics['rotation_sync']}")
    print(f"  • Role Synergy: {metrics['role_synergy']}")
    
    # Key Insights
    insights = analysis_result["formatted_output"]["key_insights"]
    print(f"\n💡 **Key Insights**:")
    for i, insight in enumerate(insights[:3], 1):  # Show top 3
        print(f"  {i}. [{insight['category']}] {insight['insight']}")
        print(f"     Severity: {insight['severity']}")
        print(f"     Recommendations: {', '.join(insight['recommendations'])}")
        print()
    
    # Player Compatibility
    compatibility = analysis_result["formatted_output"]["player_compatibility"]
    print(f"🤝 **Player Compatibility** (Top 3 Pairs):")
    for i, pair in enumerate(compatibility, 1):
        print(f"  {i}. {pair['players']} - {pair['compatibility']}")
        print(f"     Top Synergy: {pair['top_synergy']}")
        if pair['top_conflict'] != "None":
            print(f"     Top Conflict: {pair['top_conflict']}")
        print()
    
    # Step 5: Synergy Matrix Analysis
    if "synergy_matrix" in analysis_result:
        print("\n🔗 Step 5: Synergy Matrix Analysis")
        print("-" * 40)
        
        synergy_data = analysis_result["formatted_output"]["synergy_analysis"]
        print(f"Overall Team Synergy: {synergy_data['overall_synergy']}")
        
        print(f"\n💪 **Strongest Pairs**:")
        for pair in synergy_data["strongest_pairs"]:
            print(f"  • {pair['players']}: {pair['synergy_score']}")
        
        print(f"\n⚠️ **Weakest Pairs**:")
        for pair in synergy_data["weakest_pairs"]:
            print(f"  • {pair['players']}: {pair['synergy_score']}")
        
        print(f"\n📊 **Synergy Distribution**:")
        distribution = synergy_data["synergy_distribution"]
        for level, count in distribution.items():
            print(f"  • {level.title()}: {count} pairs")
    
    # Step 6: Collective Feedback
    print("\n📝 Step 6: Collective Feedback")
    print("-" * 40)
    
    feedback = analysis_result["formatted_output"]["collective_feedback"]
    print(feedback)
    
    # Step 7: Additional Analysis Options
    print("\n🔬 Step 7: Additional Analysis Options")
    print("-" * 40)
    
    # Get specific insight types
    insight_types = ["synergy", "coordination", "summary"]
    for insight_type in insight_types:
        try:
            specific_insights = await team_service.get_team_insights(
                match_id="demo_match_001",
                team_id="team_alpha",
                insight_type=insight_type
            )
            print(f"\n🎯 **{insight_type.title()} Insights**:")
            
            if insight_type == "synergy":
                synergy_overview = specific_insights.get("synergy_overview", {})
                if synergy_overview:
                    print(f"  • Overall Synergy: {synergy_overview.get('overall_synergy', 'N/A')}")
            
            elif insight_type == "coordination":
                coord_pattern = specific_insights.get("coordination_pattern", "N/A")
                print(f"  • Coordination Pattern: {coord_pattern}")
            
            elif insight_type == "summary":
                top_insight = specific_insights.get("top_insight")
                if top_insight:
                    print(f"  • Top Insight: [{top_insight['category']}] {top_insight['insight']}")
            
        except Exception as e:
            print(f"  ❌ Failed to get {insight_type} insights: {e}")
    
    # Step 8: Service Health Check
    print("\n🏥 Step 8: Service Health Check")
    print("-" * 40)
    
    try:
        health_status = await team_service.get_service_health()
        print(f"Service Status: {health_status['status']}")
        print(f"Service Version: {health_status['version']}")
        print(f"Cache Status: {health_status.get('cache_status', {}).get('status', 'N/A')}")
        
        metrics = health_status.get('metrics', {})
        if metrics:
            print(f"Total Analyses: {metrics.get('total_team_analyses', 0)}")
            print(f"Success Rate: {metrics.get('successful_analyses', 0)}/{metrics.get('total_team_analyses', 0)}")
            print(f"Avg Processing Time: {metrics.get('avg_processing_time', 0):.2f}s")
    
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Step 9: Save Results
    print("\n💾 Step 9: Saving Results")
    print("-" * 40)
    
    try:
        # Save to JSON file
        output_file = "team_analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(analysis_result, f, indent=2, default=str)
        print(f"✅ Results saved to {output_file}")
        
        # Save formatted summary
        summary_file = "team_analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("MLBB Coach AI - Team Behavior Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Match ID: {analysis_result['match_id']}\n")
            f.write(f"Team ID: {analysis_result['team_id']}\n")
            f.write(f"Analysis Date: {analysis_result['analysis_timestamp']}\n\n")
            f.write("TEAM OVERVIEW\n")
            f.write("-" * 20 + "\n")
            for key, value in team_overview.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n" + feedback)
        
        print(f"✅ Summary saved to {summary_file}")
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
    
    print("\n🎉 Team Behavior Analysis Demo Completed!")
    print("=" * 60)


async def demonstrate_individual_components():
    """Demonstrate individual system components."""
    print("\n🔧 Individual Component Demonstration")
    print("=" * 50)
    
    # Demonstrate team analyzer alone
    print("\n1. Team Behavior Analyzer")
    try:
        analyzer = await create_team_behavior_analyzer()
        team_data = create_sample_team_data()
        
        analysis = await analyzer.analyze_team_behavior(
            match_data=team_data,
            match_id="component_test",
            team_id="test_team"
        )
        
        print(f"✅ Analyzer completed: {analysis.synergy_level.value}")
        print(f"   Coordination: {analysis.coordination_pattern.value}")
        print(f"   Team Effectiveness: {analysis.team_effectiveness_rating:.1%}")
        
    except Exception as e:
        print(f"❌ Analyzer test failed: {e}")
    
    # Demonstrate synergy matrix builder
    print("\n2. Synergy Matrix Builder")
    try:
        builder = create_synergy_matrix_builder()
        
        # Create mock behavioral profiles
        from core.behavioral_modeling import BehavioralFingerprint, PlayStyle, RiskProfile, GameTempo
        from datetime import datetime
        
        mock_profiles = [
            BehavioralFingerprint(
                player_id="Player1",
                play_style=PlayStyle.AGGRESSIVE_ROAMER,
                risk_profile=RiskProfile.CALCULATED_RISK,
                game_tempo=GameTempo.EARLY_AGGRESSIVE,
                map_awareness_score=0.8,
                synergy_with_team=0.7,
                adaptability_score=0.6,
                mechanical_skill_score=0.9,
                decision_making_score=0.7,
                preferred_lane="jungle",
                preferred_role="assassin",
                preferred_heroes=["hayabusa"],
                behavior_tags=["aggressive", "roaming"],
                identified_flaws=["overextending"],
                strength_areas=["mechanics"],
                behavioral_metrics=None,
                confidence_score=0.85,
                analysis_date=datetime.now(),
                matches_analyzed=10
            ),
            BehavioralFingerprint(
                player_id="Player2",
                play_style=PlayStyle.SUPPORT_ORIENTED,
                risk_profile=RiskProfile.CONSERVATIVE,
                game_tempo=GameTempo.ADAPTIVE,
                map_awareness_score=0.9,
                synergy_with_team=0.95,
                adaptability_score=0.8,
                mechanical_skill_score=0.7,
                decision_making_score=0.85,
                preferred_lane="bottom",
                preferred_role="support",
                preferred_heroes=["estes"],
                behavior_tags=["supportive", "team-focused"],
                identified_flaws=["low damage"],
                strength_areas=["team coordination"],
                behavioral_metrics=None,
                confidence_score=0.88,
                analysis_date=datetime.now(),
                matches_analyzed=12
            )
        ]
        
        synergy_matrix = builder.build_synergy_matrix(
            behavioral_profiles=mock_profiles,
            team_id="test_synergy"
        )
        
        print(f"✅ Synergy matrix built: {synergy_matrix.overall_team_synergy:.1%}")
        print(f"   Pairs analyzed: {len(synergy_matrix.player_synergies)}")
        
    except Exception as e:
        print(f"❌ Synergy matrix test failed: {e}")


def print_usage_guide():
    """Print usage guide for the team behavior analysis system."""
    print("\n📖 Usage Guide - Team Behavior Analysis System")
    print("=" * 60)
    print("""
🎯 PURPOSE:
   Analyze behavioral data from a full 5-player squad to identify:
   • Team synergy patterns
   • Coordination breakdowns
   • Role overlap and timing mismatches
   • Team composition dynamics
   • Player compatibility matrix

📊 INPUT REQUIREMENTS:
   • Match data for exactly 5 players
   • Each player must have: kills, deaths, assists, hero
   • Optional: behavioral profiles for enhanced analysis

🔍 ANALYSIS OUTPUTS:
   • Team Overview: synergy level, coordination pattern, role overlap
   • Performance Metrics: teamfight spacing, objective control, rotation sync
   • Key Insights: categorized findings with actionable recommendations
   • Player Compatibility: pairwise compatibility scores and factors
   • Collective Feedback: comprehensive team coaching summary

🚀 USAGE EXAMPLES:
   
   # Basic team analysis
   service = await create_team_behavior_service()
   result = await service.analyze_team_behavior(
       match_data=team_data,
       match_id="match_001",
       team_id="team_alpha"
   )
   
   # Get specific insights
   insights = await service.get_team_insights(
       match_id="match_001",
       team_id="team_alpha",
       insight_type="synergy"
   )
   
   # Build compatibility matrix
   matrix = await service.build_compatibility_matrix(
       behavioral_profiles=profiles,
       team_id="team_alpha"
   )

🎮 INTEGRATION:
   • Seamlessly integrates with existing MLBB Coach AI system
   • Uses existing behavioral modeling and caching infrastructure
   • Provides service-oriented architecture compatibility
   • Supports event-driven monitoring and metrics collection

⚡ PERFORMANCE:
   • Sub-second analysis times for typical team data
   • Intelligent caching for repeated analyses
   • Concurrent analysis support
   • Comprehensive error handling and diagnostics
""")


async def main():
    """Main demonstration function."""
    try:
        # Print usage guide
        print_usage_guide()
        
        # Run comprehensive demonstration
        await demonstrate_team_behavior_analysis()
        
        # Demonstrate individual components
        await demonstrate_individual_components()
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.exception("Demo failed")


if __name__ == "__main__":
    print("🎮 MLBB Coach AI - Team Behavior Analysis System")
    print("Starting comprehensive demonstration...")
    print("Press Ctrl+C to interrupt at any time\n")
    
    asyncio.run(main()) 