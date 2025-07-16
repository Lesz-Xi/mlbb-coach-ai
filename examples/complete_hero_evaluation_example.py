#!/usr/bin/env python3
"""
Complete Hero Evaluation Module Example

This example demonstrates the full hero evaluation system in action:
- Role-specific evaluators
- Hero-specific overrides
- Caching and async processing
- Event-driven architecture
- Performance optimization

Run this example to see the complete evaluation pipeline.
"""

import asyncio
import logging
from typing import List
from datetime import datetime

# Import our evaluation system
from core.services.hero_evaluation_orchestrator import (
    HeroEvaluationOrchestrator,
    HeroEvaluationRequest,
    HeroEvaluationResponse,
    get_orchestrator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_single_hero_evaluation():
    """Example: Single hero evaluation with comprehensive feedback."""
    print("\n" + "="*60)
    print("🎯 SINGLE HERO EVALUATION EXAMPLE")
    print("="*60)
    
    # Initialize orchestrator
    orchestrator = get_orchestrator()
    
    # Example match data for Tigreal (Tank)
    tigreal_match_data = {
        'kills': 2,
        'deaths': 5,
        'assists': 15,
        'hero_damage': 28000,
        'damage_taken': 45000,
        'gold_per_min': 420,
        'teamfight_participation': 85,
        'crowd_control_score': 75,
        'vision_score': 40,
        'match_duration': 18,
        'damage_percentage': 18,
        'turret_damage': 3200,
        'lord_participation': 90,
        # Tigreal-specific metrics
        'ultimate_usage_frequency': 6,
        'avg_targets_per_ultimate': 2.8,
        'skill_combo_success_rate': 0.75,
        'ally_damage_prevented': 22000,
        'positioning_rating': 'aggressive'
    }
    
    # Create evaluation request
    request = HeroEvaluationRequest(
        hero="Tigreal",
        match_data=tigreal_match_data,
        player_ign="ExamplePlayer",
        evaluation_mode="comprehensive",
        cache_enabled=True,
        emit_events=True
    )
    
    # Execute evaluation
    print("🔍 Evaluating Tigreal performance...")
    result = await orchestrator.evaluate_hero(request)
    
    # Display results
    print_evaluation_results(result)
    
    return result


async def example_batch_evaluation():
    """Example: Batch evaluation of multiple heroes."""
    print("\n" + "="*60)
    print("📊 BATCH EVALUATION EXAMPLE")
    print("="*60)
    
    orchestrator = get_orchestrator()
    
    # Create multiple evaluation requests
    requests = []
    
    # Lancelot (Assassin)
    lancelot_data = {
        'kills': 12,
        'deaths': 3,
        'assists': 6,
        'hero_damage': 78000,
        'damage_taken': 18000,
        'gold_per_min': 850,
        'teamfight_participation': 65,
        'match_duration': 16,
        'damage_percentage': 35
    }
    requests.append(HeroEvaluationRequest(
        hero="Lancelot",
        match_data=lancelot_data,
        player_ign="ExamplePlayer",
        evaluation_mode="quick"
    ))
    
    # Miya (Marksman)
    miya_data = {
        'kills': 8,
        'deaths': 2,
        'assists': 10,
        'hero_damage': 95000,
        'damage_taken': 12000,
        'gold_per_min': 920,
        'teamfight_participation': 75,
        'match_duration': 16,
        'damage_percentage': 42
    }
    requests.append(HeroEvaluationRequest(
        hero="Miya",
        match_data=miya_data,
        player_ign="ExamplePlayer",
        evaluation_mode="quick"
    ))
    
    # Kagura (Mage)
    kagura_data = {
        'kills': 7,
        'deaths': 4,
        'assists': 12,
        'hero_damage': 68000,
        'damage_taken': 25000,
        'gold_per_min': 720,
        'teamfight_participation': 70,
        'match_duration': 16,
        'damage_percentage': 30
    }
    requests.append(HeroEvaluationRequest(
        hero="Kagura",
        match_data=kagura_data,
        player_ign="ExamplePlayer",
        evaluation_mode="quick"
    ))
    
    # Execute batch evaluation
    print(f"🔍 Evaluating {len(requests)} heroes in batch...")
    results = await orchestrator.evaluate_heroes_batch(requests)
    
    # Display batch results summary
    print_batch_results_summary(results)
    
    return results


async def example_role_specific_evaluation():
    """Example: Demonstrating different role-specific evaluations."""
    print("\n" + "="*60)
    print("🎭 ROLE-SPECIFIC EVALUATION EXAMPLE")
    print("="*60)
    
    orchestrator = get_orchestrator()
    
    # Same base stats, different heroes/roles
    base_stats = {
        'kills': 5,
        'deaths': 3,
        'assists': 8,
        'hero_damage': 45000,
        'damage_taken': 25000,
        'gold_per_min': 650,
        'teamfight_participation': 70,
        'match_duration': 15,
        'damage_percentage': 25
    }
    
    heroes_to_test = ["Tigreal", "Lancelot", "Miya", "Kagura", "Chou", "Estes"]
    
    print("🧪 Testing same stats across different roles:")
    
    for hero in heroes_to_test:
        request = HeroEvaluationRequest(
            hero=hero,
            match_data=base_stats.copy(),
            player_ign="RoleTestPlayer",
            evaluation_mode="quick",
            emit_events=False
        )
        
        result = await orchestrator.evaluate_hero(request)
        
        print(f"  {hero:<12} ({result.role:<8}): {result.overall_score:.2f} - {result.performance_rating}")
        
        # Show top feedback for each role
        if result.feedback:
            top_feedback = result.feedback[0]
            print(f"    └─ {top_feedback[0]}: {top_feedback[1][:60]}...")


async def example_performance_optimization():
    """Example: Performance optimization features."""
    print("\n" + "="*60)
    print("⚡ PERFORMANCE OPTIMIZATION EXAMPLE")
    print("="*60)
    
    # Configure orchestrator with performance settings
    config = {
        'batch_size': 5,
        'max_concurrent': 3,
        'timeout_seconds': 10
    }
    
    orchestrator = HeroEvaluationOrchestrator(config)
    
    # Warmup cache
    common_heroes = ["Tigreal", "Franco", "Lancelot", "Hayabusa", "Miya"]
    print("🔥 Warming up cache...")
    await orchestrator.warmup_cache(common_heroes)
    
    # Test cache hit
    sample_data = {
        'kills': 5, 'deaths': 3, 'assists': 8,
        'hero_damage': 50000, 'damage_taken': 30000,
        'gold_per_min': 600, 'match_duration': 15
    }
    
    # First evaluation (cache miss)
    print("\n📊 First evaluation (cache miss):")
    start_time = datetime.now()
    
    request = HeroEvaluationRequest(
        hero="Tigreal",
        match_data=sample_data,
        player_ign="warmup_user",
        evaluation_mode="quick"
    )
    
    result1 = await orchestrator.evaluate_hero(request)
    first_time = (datetime.now() - start_time).total_seconds() * 1000
    
    print(f"  Time: {first_time:.2f}ms, Cache Hit: {result1.cache_hit}")
    
    # Second evaluation (cache hit)
    print("\n📊 Second evaluation (cache hit):")
    start_time = datetime.now()
    
    result2 = await orchestrator.evaluate_hero(request)
    second_time = (datetime.now() - start_time).total_seconds() * 1000
    
    print(f"  Time: {second_time:.2f}ms, Cache Hit: {result2.cache_hit}")
    print(f"  Speedup: {first_time/second_time:.1f}x faster")
    
    # Get orchestrator statistics
    stats = await orchestrator.get_evaluation_stats()
    print("\n📈 Orchestrator Statistics:")
    print(f"  Hero mappings: {stats['hero_mappings_count']}")
    print(f"  Available roles: {', '.join(stats['available_roles'])}")
    print(f"  Batch size: {stats['config']['batch_size']}")
    
    await orchestrator.cleanup()


def print_evaluation_results(result: HeroEvaluationResponse):
    """Pretty print evaluation results."""
    print(f"\n📋 EVALUATION RESULTS FOR {result.hero.upper()}")
    print("=" * 50)
    
    # Basic info
    print(f"Hero: {result.hero}")
    print(f"Role: {result.role.title()}")
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"Performance Rating: {result.performance_rating}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Processing Time: {result.processing_time_ms}ms")
    print(f"Cache Hit: {result.cache_hit}")
    
    # Role-specific metrics
    print(f"\n📊 Role-Specific Metrics:")
    for metric, value in result.role_specific_metrics.items():
        bar_length = int(value * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"  {metric.replace('_', ' ').title():<20}: {bar} {value:.2f}")
    
    # Feedback
    print(f"\n💬 Feedback ({len(result.feedback)} items):")
    for severity, message in result.feedback[:5]:  # Show top 5
        icon = {"success": "✅", "warning": "⚠️", "critical": "🚨", "info": "💡"}.get(severity, "📝")
        print(f"  {icon} {message}")
    
    if len(result.feedback) > 5:
        print(f"  ... and {len(result.feedback) - 5} more")
    
    # Suggestions
    print(f"\n💡 Suggestions ({len(result.suggestions)} items):")
    for i, suggestion in enumerate(result.suggestions[:3], 1):  # Show top 3
        print(f"  {i}. {suggestion}")
    
    if len(result.suggestions) > 3:
        print(f"  ... and {len(result.suggestions) - 3} more")
    
    # Hero insights
    if result.hero_specific_insights:
        print(f"\n🔍 Hero-Specific Insights:")
        
        # Match context
        match_context = result.hero_specific_insights.get('match_context', {})
        if match_context:
            duration_info = match_context.get('duration_assessment', {})
            if duration_info:
                print(f"  Match Type: {duration_info.get('description', 'N/A')}")
            
            team_impact = match_context.get('team_performance_impact', {})
            if team_impact:
                print(f"  Team Impact: {team_impact.get('description', 'N/A')}")
        
        # Optimization focus
        opt_focus = result.hero_specific_insights.get('optimization_focus', [])
        if opt_focus:
            focus = opt_focus[0]
            print(f"  Priority Focus: {focus.get('metric', 'N/A').replace('_', ' ').title()}")
            print(f"  Current: {focus.get('current_value', 0):.2f} → Target: {focus.get('target_value', 0):.2f}")


def print_batch_results_summary(results: List[HeroEvaluationResponse]):
    """Print summary of batch evaluation results."""
    print(f"\n📊 BATCH EVALUATION SUMMARY")
    print(f"{'='*50}")
    
    total_time = sum(r.processing_time_ms for r in results)
    cache_hits = sum(1 for r in results if r.cache_hit)
    
    print(f"Total Heroes Evaluated: {len(results)}")
    print(f"Total Processing Time: {total_time}ms")
    print(f"Average Time per Hero: {total_time/len(results):.1f}ms")
    print(f"Cache Hits: {cache_hits}/{len(results)} ({cache_hits/len(results)*100:.1f}%)")
    
    print(f"\n📈 Results by Performance:")
    performance_counts = {}
    for result in results:
        performance_counts[result.performance_rating] = performance_counts.get(result.performance_rating, 0) + 1
    
    for rating, count in sorted(performance_counts.items()):
        print(f"  {rating}: {count} heroes")
    
    print(f"\n🏆 Top Performers:")
    sorted_results = sorted(results, key=lambda x: x.overall_score, reverse=True)
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"  {i}. {result.hero} ({result.role}): {result.overall_score:.3f}")


async def main():
    """Run all examples."""
    print("🚀 MLBB Coach AI - Complete Hero Evaluation Module Examples")
    print("="*70)
    
    try:
        # Run examples
        await example_single_hero_evaluation()
        await example_batch_evaluation()
        await example_role_specific_evaluation()
        await example_performance_optimization()
        
        print("\n" + "="*70)
        print("✅ All examples completed successfully!")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"\n❌ Example failed: {e}")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main()) 