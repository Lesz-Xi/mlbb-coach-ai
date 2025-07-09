from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path

from .meta_analyzer import MetaAnalyzer
from .schemas import PerformanceComparison, HeroMetaData


class PerformanceAnalyzer:
    """Advanced performance analysis system for comparing player stats against meta benchmarks."""
    
    def __init__(self, meta_analyzer: Optional[MetaAnalyzer] = None):
        """Initialize with meta analyzer."""
        self.meta_analyzer = meta_analyzer or MetaAnalyzer()
    
    def analyze_player_performance(self, player_stats: Dict[str, Dict[str, float]]) -> Dict[str, PerformanceComparison]:
        """
        Analyze player performance across multiple heroes.
        
        Args:
            player_stats: Dict of hero_name -> stats dict
                         Example: {"Franco": {"win_rate": 45.0, "games_played": 20}}
        
        Returns:
            Dict of hero_name -> PerformanceComparison
        """
        comparisons = {}
        
        for hero_name, stats in player_stats.items():
            player_winrate = stats.get("win_rate", 0.0)
            games_played = stats.get("games_played", 0)
            
            comparison = self.meta_analyzer.compare_performance(hero_name, player_winrate)
            
            if comparison:
                # Enhance with games played consideration
                if games_played < 10:
                    comparison.improvement_areas.append("Play more games for better statistical reliability")
                
                # Add percentile calculation based on games played
                comparison.percentile_rank = self._calculate_adjusted_percentile(
                    comparison.performance_gap, games_played
                )
                
                comparisons[hero_name] = comparison
        
        return comparisons
    
    def get_performance_summary(self, comparisons: Dict[str, PerformanceComparison]) -> Dict[str, any]:
        """Generate performance summary across all heroes."""
        if not comparisons:
            return {}
        
        # Calculate overall stats
        total_gap = sum(comp.performance_gap for comp in comparisons.values())
        avg_gap = total_gap / len(comparisons)
        
        # Categorize heroes
        strong_heroes = []
        weak_heroes = []
        balanced_heroes = []
        
        for hero, comp in comparisons.items():
            if comp.performance_gap > 5:
                strong_heroes.append((hero, comp.performance_gap))
            elif comp.performance_gap < -5:
                weak_heroes.append((hero, comp.performance_gap))
            else:
                balanced_heroes.append((hero, comp.performance_gap))
        
        # Sort by performance gap
        strong_heroes.sort(key=lambda x: x[1], reverse=True)
        weak_heroes.sort(key=lambda x: x[1])
        
        return {
            "overall_performance": {
                "average_gap": avg_gap,
                "heroes_analyzed": len(comparisons),
                "performance_category": self._get_performance_category(avg_gap)
            },
            "strong_heroes": strong_heroes[:5],  # Top 5
            "weak_heroes": weak_heroes[:5],      # Bottom 5
            "balanced_heroes": balanced_heroes,
            "recommendations": self._generate_performance_recommendations(comparisons)
        }
    
    def compare_vs_rank_bracket(self, player_stats: Dict[str, Dict[str, float]], 
                               rank_bracket: str) -> Dict[str, any]:
        """Compare player performance against specific rank bracket."""
        # This would require rank-specific data
        # For now, return general comparison with rank context
        comparisons = self.analyze_player_performance(player_stats)
        
        rank_modifiers = {
            "warrior": -5,    # Lower expectations
            "elite": -3,
            "master": 0,      # Baseline
            "grandmaster": 2,
            "epic": 5,
            "legend": 8,
            "mythic": 12     # Higher expectations
        }
        
        modifier = rank_modifiers.get(rank_bracket.lower(), 0)
        
        # Adjust performance gaps based on rank
        for comp in comparisons.values():
            comp.performance_gap -= modifier
            comp.percentile_rank = self._calculate_adjusted_percentile(comp.performance_gap, 50)
        
        return {
            "rank_bracket": rank_bracket,
            "modifier_applied": modifier,
            "comparisons": comparisons,
            "summary": self.get_performance_summary(comparisons)
        }
    
    def identify_improvement_priorities(self, comparisons: Dict[str, PerformanceComparison]) -> List[Dict[str, any]]:
        """Identify which heroes need the most improvement."""
        priorities = []
        
        for hero, comp in comparisons.items():
            if comp.performance_gap < -3:  # Below average
                hero_meta = self.meta_analyzer.get_hero_meta(hero)
                if hero_meta:
                    priority_score = self._calculate_priority_score(comp, hero_meta)
                    
                    priorities.append({
                        "hero": hero,
                        "performance_gap": comp.performance_gap,
                        "priority_score": priority_score,
                        "meta_tier": hero_meta.tier,
                        "meta_winrate": hero_meta.win_rate,
                        "improvement_areas": comp.improvement_areas,
                        "reasoning": self._generate_improvement_reasoning(comp, hero_meta)
                    })
        
        # Sort by priority score
        priorities.sort(key=lambda x: x["priority_score"], reverse=True)
        return priorities
    
    def track_performance_trends(self, historical_data: List[Dict[str, any]]) -> Dict[str, any]:
        """Track performance trends over time."""
        if len(historical_data) < 2:
            return {"error": "Need at least 2 data points for trend analysis"}
        
        trends = {}
        
        # Extract hero performance over time
        for hero in historical_data[0].get("player_stats", {}).keys():
            hero_trends = []
            
            for data_point in historical_data:
                stats = data_point.get("player_stats", {}).get(hero, {})
                if stats:
                    hero_trends.append({
                        "date": data_point.get("date"),
                        "win_rate": stats.get("win_rate", 0),
                        "games_played": stats.get("games_played", 0)
                    })
            
            if len(hero_trends) >= 2:
                trends[hero] = {
                    "trend_direction": self._calculate_trend_direction(hero_trends),
                    "improvement_rate": self._calculate_improvement_rate(hero_trends),
                    "consistency": self._calculate_consistency(hero_trends)
                }
        
        return trends
    
    def generate_coaching_insights(self, comparisons: Dict[str, PerformanceComparison]) -> List[str]:
        """Generate coaching insights based on performance analysis."""
        insights = []
        
        # Overall performance insights
        summary = self.get_performance_summary(comparisons)
        avg_gap = summary.get("overall_performance", {}).get("average_gap", 0)
        
        if avg_gap > 5:
            insights.append("🎯 Strong overall performance! You're consistently above meta averages.")
        elif avg_gap < -5:
            insights.append("📈 Focus on fundamentals. Your performance is below meta averages across heroes.")
        else:
            insights.append("⚖️ Balanced performance. Some heroes strong, others need work.")
        
        # Hero-specific insights
        for hero, comp in comparisons.items():
            hero_meta = self.meta_analyzer.get_hero_meta(hero)
            if hero_meta:
                if comp.performance_gap > 10:
                    insights.append(f"🔥 {hero}: Exceptional performance! Consider this a pocket pick.")
                elif comp.performance_gap < -10:
                    insights.append(f"⚠️ {hero}: Significant underperformance. Focus on core mechanics.")
                elif hero_meta.tier in ["S", "A"] and comp.performance_gap < -5:
                    insights.append(f"🎯 {hero}: Strong meta pick but you're underperforming. High impact potential.")
        
        return insights
    
    def _calculate_adjusted_percentile(self, performance_gap: float, games_played: int) -> int:
        """Calculate percentile with games played consideration."""
        base_percentile = 50
        
        if performance_gap > 10:
            base_percentile = 95
        elif performance_gap > 5:
            base_percentile = 80
        elif performance_gap > 0:
            base_percentile = 65
        elif performance_gap > -5:
            base_percentile = 40
        elif performance_gap > -10:
            base_percentile = 25
        else:
            base_percentile = 10
        
        # Adjust for sample size
        if games_played < 10:
            base_percentile = max(base_percentile - 10, 10)
        elif games_played > 50:
            base_percentile = min(base_percentile + 5, 95)
        
        return base_percentile
    
    def _get_performance_category(self, avg_gap: float) -> str:
        """Get performance category based on average gap."""
        if avg_gap > 8:
            return "exceptional"
        elif avg_gap > 3:
            return "above_average"
        elif avg_gap > -3:
            return "average"
        elif avg_gap > -8:
            return "below_average"
        else:
            return "needs_improvement"
    
    def _calculate_priority_score(self, comp: PerformanceComparison, hero_meta: HeroMetaData) -> float:
        """Calculate improvement priority score."""
        # Base score on performance gap
        score = abs(comp.performance_gap) * 0.4
        
        # Meta tier bonus (higher tier = higher priority)
        tier_bonus = {"S": 1.0, "A": 0.8, "B": 0.6, "C": 0.4, "D": 0.2}
        score += tier_bonus.get(hero_meta.tier, 0.2) * 0.3
        
        # Pick rate bonus (popular heroes get higher priority)
        if hero_meta.pick_rate > 2:
            score += 0.3
        elif hero_meta.pick_rate > 1:
            score += 0.2
        
        return score
    
    def _generate_improvement_reasoning(self, comp: PerformanceComparison, hero_meta: HeroMetaData) -> str:
        """Generate reasoning for improvement priority."""
        reasoning = f"Performance gap: {comp.performance_gap:.1f}%. "
        reasoning += f"Meta tier: {hero_meta.tier}. "
        
        if hero_meta.tier in ["S", "A"]:
            reasoning += "High meta impact - mastering this hero will improve your rank significantly."
        elif hero_meta.pick_rate > 2:
            reasoning += "Popular pick - opponents likely familiar with counters."
        
        return reasoning
    
    def _calculate_trend_direction(self, trends: List[Dict[str, any]]) -> str:
        """Calculate trend direction."""
        if len(trends) < 2:
            return "insufficient_data"
        
        first_wr = trends[0]["win_rate"]
        last_wr = trends[-1]["win_rate"]
        
        if last_wr > first_wr + 3:
            return "improving"
        elif last_wr < first_wr - 3:
            return "declining"
        else:
            return "stable"
    
    def _calculate_improvement_rate(self, trends: List[Dict[str, any]]) -> float:
        """Calculate improvement rate per time period."""
        if len(trends) < 2:
            return 0.0
        
        first_wr = trends[0]["win_rate"]
        last_wr = trends[-1]["win_rate"]
        periods = len(trends) - 1
        
        return (last_wr - first_wr) / periods
    
    def _calculate_consistency(self, trends: List[Dict[str, any]]) -> float:
        """Calculate performance consistency."""
        if len(trends) < 3:
            return 0.5
        
        win_rates = [t["win_rate"] for t in trends]
        avg_wr = sum(win_rates) / len(win_rates)
        variance = sum((wr - avg_wr) ** 2 for wr in win_rates) / len(win_rates)
        
        # Normalize consistency score (lower variance = higher consistency)
        return max(0, 1 - (variance / 100))
    
    def _generate_performance_recommendations(self, comparisons: Dict[str, PerformanceComparison]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        # Find patterns in weak performance
        weak_heroes = [hero for hero, comp in comparisons.items() if comp.performance_gap < -5]
        
        if len(weak_heroes) > 3:
            recommendations.append("Focus on fundamentals that apply across all heroes")
            recommendations.append("Consider coaching or guides for macro gameplay")
        
        # Find meta heroes with poor performance
        meta_heroes_weak = []
        for hero, comp in comparisons.items():
            hero_meta = self.meta_analyzer.get_hero_meta(hero)
            if hero_meta and hero_meta.tier in ["S", "A"] and comp.performance_gap < -5:
                meta_heroes_weak.append(hero)
        
        if meta_heroes_weak:
            recommendations.append(f"Priority heroes to improve: {', '.join(meta_heroes_weak)}")
        
        return recommendations