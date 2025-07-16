"""
Behavioral Modeling System - Complete Usage Example
==================================================

This example demonstrates how to use the behavioral modeling system to analyze
player behavior patterns from match history and video replay data.

Key Features Demonstrated:
- Single player behavioral analysis
- Batch analysis of multiple players
- Video behavior analysis integration
- Insight generation and recommendations
- Player comparison and benchmarking
- Service health monitoring

Usage:
    python examples/behavioral_modeling_example.py
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

# Import behavioral modeling components
from core.services.behavioral_modeling_service import (
    create_behavioral_modeling_service
)
from core.cache.hybrid_cache import HybridCache
from core.events.event_bus import EventBus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BehavioralModelingDemo:
    """Comprehensive demonstration of behavioral modeling capabilities."""
    
    def __init__(self):
        self.service = None
        self.sample_data = self._generate_sample_data()
    
    async def initialize(self):
        """Initialize the behavioral modeling service."""
        logger.info("Initializing Behavioral Modeling Demo...")
        
        # Create service with caching and event monitoring
        self.service = await create_behavioral_modeling_service(
            cache_manager=HybridCache(),
            event_bus=EventBus(),
            max_concurrent_analyses=3
        )
        
        logger.info("Service initialized successfully!")
    
    async def run_complete_demo(self):
        """Run complete demonstration of behavioral modeling features."""
        logger.info("🧠 Starting Behavioral Modeling System Demo")
        
        # 1. Single Player Analysis
        await self._demo_single_player_analysis()
        
        # 2. Batch Player Analysis
        await self._demo_batch_analysis()
        
        # 3. Video Behavior Analysis
        await self._demo_video_analysis()
        
        # 4. Insight Generation
        await self._demo_insight_generation()
        
        # 5. Player Comparison
        await self._demo_player_comparison()
        
        # 6. Service Health Check
        await self._demo_service_health()
        
        logger.info("🎉 Demo completed successfully!")
    
    async def _demo_single_player_analysis(self):
        """Demonstrate single player behavioral analysis."""
        logger.info("\n" + "="*50)
        logger.info("📊 Single Player Behavioral Analysis")
        logger.info("="*50)
        
        player_id = "aggressive_player_001"
        match_history = self.sample_data["aggressive_player"]
        
        try:
            # Analyze player behavior
            fingerprint = await self.service.analyze_player_behavior(
                player_id=player_id,
                match_history=match_history
            )
            
            # Display results
            logger.info(f"🎯 Analysis Results for {player_id}:")
            logger.info(f"   Play Style: {fingerprint.play_style.value}")
            logger.info(f"   Risk Profile: {fingerprint.risk_profile.value}")
            logger.info(f"   Game Tempo: {fingerprint.game_tempo.value}")
            logger.info(f"   Map Awareness: {fingerprint.map_awareness_score:.2f}")
            logger.info(f"   Team Synergy: {fingerprint.synergy_with_team:.2f}")
            logger.info(f"   Confidence Score: {fingerprint.confidence_score:.2f}")
            logger.info(f"   Preferred Heroes: {fingerprint.preferred_heroes}")
            logger.info(f"   Behavioral Tags: {fingerprint.behavior_tags}")
            logger.info(f"   Identified Flaws: {fingerprint.identified_flaws}")
            logger.info(f"   Strength Areas: {fingerprint.strength_areas}")
            
                                      # Test caching (second call should be faster)
             start_time = datetime.now()
             await self.service.analyze_player_behavior(
                 player_id=player_id,
                 match_history=match_history
             )
             cache_time = (datetime.now() - start_time).total_seconds()
             logger.info(f"🚀 Cache Performance: {cache_time:.3f}s (cached)")
            
        except Exception as e:
            logger.error(f"❌ Single player analysis failed: {str(e)}")
    
    async def _demo_batch_analysis(self):
        """Demonstrate batch analysis of multiple players."""
        logger.info("\n" + "="*50)
        logger.info("📈 Batch Player Analysis")
        logger.info("="*50)
        
        # Prepare batch requests
        batch_requests = [
            {
                "player_id": "aggressive_player_001",
                "match_history": self.sample_data["aggressive_player"]
            },
            {
                "player_id": "passive_player_002",
                "match_history": self.sample_data["passive_player"]
            },
            {
                "player_id": "strategic_player_003",
                "match_history": self.sample_data["strategic_player"]
            }
        ]
        
        try:
            # Batch analyze players
            start_time = datetime.now()
            results = await self.service.batch_analyze_players(batch_requests)
            batch_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"🚀 Batch Analysis Performance: {batch_time:.2f}s for {len(batch_requests)} players")
            
            # Display results
            successful_analyses = 0
            for player_id, fingerprint, error in results:
                if fingerprint:
                    logger.info(f"✅ {player_id}: {fingerprint.play_style.value} | {fingerprint.risk_profile.value}")
                    successful_analyses += 1
                else:
                    logger.error(f"❌ {player_id}: {error}")
            
            logger.info(f"📊 Success Rate: {successful_analyses}/{len(batch_requests)} ({successful_analyses/len(batch_requests)*100:.1f}%)")
            
        except Exception as e:
            logger.error(f"❌ Batch analysis failed: {str(e)}")
    
    async def _demo_video_analysis(self):
        """Demonstrate video behavior analysis integration."""
        logger.info("\n" + "="*50)
        logger.info("🎬 Video Behavior Analysis")
        logger.info("="*50)
        
        # Note: This would work with real video files
        # For demo purposes, we'll simulate the scenario
        player_id = "video_player_001"
        match_history = self.sample_data["aggressive_player"]
        
        # Simulate video paths (in real usage, these would be actual video files)
        video_paths = [
            "replays/match_001.mp4",
            "replays/match_002.mp4"
        ]
        
        logger.info(f"📹 Analyzing player behavior with video data...")
        logger.info(f"   Player: {player_id}")
        logger.info(f"   Match History: {len(match_history)} matches")
        logger.info(f"   Video Files: {len(video_paths)} replays")
        
        try:
            # This would analyze video data if files existed
            # For demo, we'll just show the capability
            logger.info("🎯 Video Analysis Capabilities:")
            logger.info("   - Positioning pattern recognition")
            logger.info("   - Movement behavior analysis")
            logger.info("   - Decision-making assessment")
            logger.info("   - Team coordination evaluation")
            logger.info("   - Real-time tactical analysis")
            
            # Show how it would integrate with match data
            fingerprint = await self.service.analyze_player_behavior(
                player_id=player_id,
                match_history=match_history
                # video_paths=video_paths  # Would uncomment for real videos
            )
            
            logger.info(f"✅ Enhanced analysis completed with confidence: {fingerprint.confidence_score:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Video analysis failed: {str(e)}")
    
    async def _demo_insight_generation(self):
        """Demonstrate insight generation capabilities."""
        logger.info("\n" + "="*50)
        logger.info("💡 Behavioral Insights Generation")
        logger.info("="*50)
        
        player_id = "aggressive_player_001"
        
        try:
            # Generate different types of insights
            insights_types = ["summary", "recommendations", "comparison"]
            
            for insight_type in insights_types:
                logger.info(f"\n🔍 {insight_type.title()} Insights:")
                
                insights = await self.service.get_behavior_insights(
                    player_id=player_id,
                    insight_type=insight_type
                )
                
                if insight_type == "summary":
                    logger.info(f"   Play Style: {insights['play_style']}")
                    logger.info(f"   Risk Profile: {insights['risk_profile']}")
                    logger.info(f"   Key Strengths: {insights['key_strengths']}")
                    logger.info(f"   Improvement Areas: {insights['improvement_areas']}")
                
                elif insight_type == "recommendations":
                    logger.info(f"   Priority Focus: {insights['priority_focus']}")
                    for rec in insights['recommendations']:
                        logger.info(f"   - {rec['category']}: {rec['suggestion']}")
                
                elif insight_type == "comparison":
                    logger.info(f"   Meta Alignment: {insights['meta_comparison']}")
                    logger.info(f"   Recommendations: {insights['recommendations']}")
            
        except Exception as e:
            logger.error(f"❌ Insight generation failed: {str(e)}")
    
    async def _demo_player_comparison(self):
        """Demonstrate player comparison capabilities."""
        logger.info("\n" + "="*50)
        logger.info("⚖️ Player Comparison Analysis")
        logger.info("="*50)
        
        # Compare multiple players
        player_ids = ["aggressive_player_001", "passive_player_002", "strategic_player_003"]
        
        try:
            comparison = await self.service.compare_players(
                player_ids=player_ids,
                comparison_metrics=["map_awareness_score", "synergy_with_team", "mechanical_skill_score"]
            )
            
            logger.info(f"👥 Comparing {len(player_ids)} players:")
            
            # Display comparison results
            for metric, values in comparison['comparison_metrics'].items():
                logger.info(f"\n📊 {metric.replace('_', ' ').title()}:")
                for player_id, score in values.items():
                    logger.info(f"   {player_id}: {score:.2f}")
            
            # Show summary insights
            summary = comparison['summary']
            logger.info(f"\n🎯 Summary Insights:")
            logger.info(f"   Most Aggressive: {summary['most_aggressive']}")
            logger.info(f"   Most Strategic: {summary['most_strategic']}")
            logger.info(f"   Play Styles: {summary['play_styles']}")
            logger.info(f"   Risk Profiles: {summary['risk_profiles']}")
            
        except Exception as e:
            logger.error(f"❌ Player comparison failed: {str(e)}")
    
    async def _demo_service_health(self):
        """Demonstrate service health monitoring."""
        logger.info("\n" + "="*50)
        logger.info("🏥 Service Health Monitoring")
        logger.info("="*50)
        
        try:
            health = await self.service.get_service_health()
            
            logger.info(f"🔧 Service Information:")
            logger.info(f"   Service: {health['service_name']} v{health['version']}")
            logger.info(f"   Status: {health['status']}")
            logger.info(f"   Analyzer Initialized: {health['analyzer_initialized']}")
            
            logger.info(f"\n📊 Performance Metrics:")
            metrics = health['metrics']
            logger.info(f"   Total Analyses: {metrics['total_analyses']}")
            logger.info(f"   Success Rate: {metrics['successful_analyses']}/{metrics['total_analyses']}")
            logger.info(f"   Avg Processing Time: {metrics['avg_processing_time']:.2f}s")
            logger.info(f"   Cache Hit Rate: {metrics['cache_hit_rate']:.2f}")
            
            logger.info(f"\n💾 Cache Status:")
            cache_status = health['cache_status']
            logger.info(f"   Status: {cache_status['status']}")
            logger.info(f"   Hit Rate: {cache_status['hit_rate']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Service health check failed: {str(e)}")
    
    def _generate_sample_data(self) -> Dict[str, Any]:
        """Generate sample match data for demonstration."""
        return {
            "aggressive_player": [
                {
                    "hero": "hayabusa",
                    "kills": 15,
                    "deaths": 8,
                    "assists": 3,
                    "gold_per_min": 4200,
                    "hero_damage": 89500,
                    "turret_damage": 12000,
                    "damage_taken": 35000,
                    "teamfight_participation": 85,
                    "positioning_rating": "low",
                    "ult_usage": "high",
                    "match_duration": 18
                },
                {
                    "hero": "lancelot",
                    "kills": 12,
                    "deaths": 6,
                    "assists": 5,
                    "gold_per_min": 4500,
                    "hero_damage": 76000,
                    "turret_damage": 8000,
                    "damage_taken": 28000,
                    "teamfight_participation": 75,
                    "positioning_rating": "low",
                    "ult_usage": "high",
                    "match_duration": 16
                },
                {
                    "hero": "hayabusa",
                    "kills": 18,
                    "deaths": 10,
                    "assists": 2,
                    "gold_per_min": 4800,
                    "hero_damage": 95000,
                    "turret_damage": 15000,
                    "damage_taken": 42000,
                    "teamfight_participation": 80,
                    "positioning_rating": "low",
                    "ult_usage": "high",
                    "match_duration": 20
                },
                {
                    "hero": "lancelot",
                    "kills": 14,
                    "deaths": 7,
                    "assists": 4,
                    "gold_per_min": 4300,
                    "hero_damage": 82000,
                    "turret_damage": 10000,
                    "damage_taken": 32000,
                    "teamfight_participation": 70,
                    "positioning_rating": "average",
                    "ult_usage": "high",
                    "match_duration": 17
                },
                {
                    "hero": "hayabusa",
                    "kills": 20,
                    "deaths": 12,
                    "assists": 3,
                    "gold_per_min": 5000,
                    "hero_damage": 102000,
                    "turret_damage": 18000,
                    "damage_taken": 48000,
                    "teamfight_participation": 90,
                    "positioning_rating": "low",
                    "ult_usage": "high",
                    "match_duration": 22
                }
            ],
            "passive_player": [
                {
                    "hero": "miya",
                    "kills": 8,
                    "deaths": 2,
                    "assists": 12,
                    "gold_per_min": 5200,
                    "hero_damage": 98000,
                    "turret_damage": 25000,
                    "damage_taken": 18000,
                    "teamfight_participation": 95,
                    "positioning_rating": "good",
                    "ult_usage": "average",
                    "match_duration": 25
                },
                {
                    "hero": "miya",
                    "kills": 6,
                    "deaths": 1,
                    "assists": 15,
                    "gold_per_min": 5500,
                    "hero_damage": 105000,
                    "turret_damage": 32000,
                    "damage_taken": 15000,
                    "teamfight_participation": 90,
                    "positioning_rating": "good",
                    "ult_usage": "average",
                    "match_duration": 28
                },
                {
                    "hero": "miya",
                    "kills": 10,
                    "deaths": 3,
                    "assists": 18,
                    "gold_per_min": 5800,
                    "hero_damage": 112000,
                    "turret_damage": 35000,
                    "damage_taken": 22000,
                    "teamfight_participation": 88,
                    "positioning_rating": "good",
                    "ult_usage": "high",
                    "match_duration": 32
                },
                {
                    "hero": "miya",
                    "kills": 7,
                    "deaths": 2,
                    "assists": 14,
                    "gold_per_min": 5300,
                    "hero_damage": 98000,
                    "turret_damage": 28000,
                    "damage_taken": 16000,
                    "teamfight_participation": 85,
                    "positioning_rating": "good",
                    "ult_usage": "average",
                    "match_duration": 26
                },
                {
                    "hero": "miya",
                    "kills": 9,
                    "deaths": 1,
                    "assists": 16,
                    "gold_per_min": 5600,
                    "hero_damage": 108000,
                    "turret_damage": 30000,
                    "damage_taken": 14000,
                    "teamfight_participation": 92,
                    "positioning_rating": "good",
                    "ult_usage": "high",
                    "match_duration": 29
                }
            ],
            "strategic_player": [
                {
                    "hero": "tigreal",
                    "kills": 3,
                    "deaths": 4,
                    "assists": 18,
                    "gold_per_min": 3800,
                    "hero_damage": 45000,
                    "turret_damage": 8000,
                    "damage_taken": 65000,
                    "teamfight_participation": 98,
                    "positioning_rating": "good",
                    "ult_usage": "high",
                    "match_duration": 24
                },
                {
                    "hero": "franco",
                    "kills": 2,
                    "deaths": 5,
                    "assists": 20,
                    "gold_per_min": 3500,
                    "hero_damage": 38000,
                    "turret_damage": 6000,
                    "damage_taken": 72000,
                    "teamfight_participation": 95,
                    "positioning_rating": "good",
                    "ult_usage": "high",
                    "match_duration": 26
                },
                {
                    "hero": "tigreal",
                    "kills": 4,
                    "deaths": 3,
                    "assists": 22,
                    "gold_per_min": 4000,
                    "hero_damage": 52000,
                    "turret_damage": 10000,
                    "damage_taken": 68000,
                    "teamfight_participation": 100,
                    "positioning_rating": "good",
                    "ult_usage": "high",
                    "match_duration": 28
                },
                {
                    "hero": "franco",
                    "kills": 1,
                    "deaths": 6,
                    "assists": 19,
                    "gold_per_min": 3300,
                    "hero_damage": 35000,
                    "turret_damage": 5000,
                    "damage_taken": 78000,
                    "teamfight_participation": 92,
                    "positioning_rating": "average",
                    "ult_usage": "high",
                    "match_duration": 30
                },
                {
                    "hero": "tigreal",
                    "kills": 5,
                    "deaths": 2,
                    "assists": 24,
                    "gold_per_min": 4200,
                    "hero_damage": 48000,
                    "turret_damage": 12000,
                    "damage_taken": 62000,
                    "teamfight_participation": 96,
                    "positioning_rating": "good",
                    "ult_usage": "high",
                    "match_duration": 25
                }
            ]
        }


async def main():
    """Run the complete behavioral modeling demonstration."""
    demo = BehavioralModelingDemo()
    
    try:
        await demo.initialize()
        await demo.run_complete_demo()
    except Exception as e:
        logger.error(f"❌ Demo failed: {str(e)}")
    finally:
        logger.info("🏁 Demo completed.")


if __name__ == "__main__":
    asyncio.run(main()) 