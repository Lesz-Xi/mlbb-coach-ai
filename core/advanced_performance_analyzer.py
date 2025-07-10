"""
Advanced Performance Analysis System
Analyzes player performance beyond KDA metrics including damage efficiency, 
objective participation, positioning, and role-specific metrics.
"""

import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from .hero_database import hero_database

logger = logging.getLogger(__name__)


class PerformanceCategory(Enum):
    """Performance evaluation categories."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    NEEDS_WORK = "needs_work"
    POOR = "poor"


@dataclass
class PerformanceMetric:
    """Individual performance metric with benchmarks."""
    name: str
    value: float
    benchmark: float
    weight: float = 1.0
    category: PerformanceCategory = PerformanceCategory.AVERAGE
    description: str = ""
    improvement_tip: str = ""
    
    def __post_init__(self):
        """Calculate category based on value vs benchmark."""
        ratio = self.value / self.benchmark if self.benchmark > 0 else 0
        
        if ratio >= 1.3:
            self.category = PerformanceCategory.EXCELLENT
        elif ratio >= 1.1:
            self.category = PerformanceCategory.GOOD
        elif ratio >= 0.9:
            self.category = PerformanceCategory.AVERAGE
        elif ratio >= 0.7:
            self.category = PerformanceCategory.NEEDS_WORK
        else:
            self.category = PerformanceCategory.POOR


@dataclass
class AdvancedPerformanceReport:
    """Comprehensive performance report."""
    hero: str
    role: str
    overall_rating: PerformanceCategory
    overall_score: float
    
    # Core metrics
    combat_efficiency: PerformanceMetric
    objective_participation: PerformanceMetric
    economic_efficiency: PerformanceMetric
    survival_rating: PerformanceMetric
    
    # Role-specific metrics
    role_specific_metrics: Dict[str, PerformanceMetric] = field(default_factory=dict)
    
    # Analysis
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_priorities: List[str] = field(default_factory=list)
    advanced_insights: List[str] = field(default_factory=list)


class AdvancedPerformanceAnalyzer:
    """Advanced performance analyzer with comprehensive metrics."""
    
    def __init__(self):
        """Initialize with role-specific benchmarks."""
        self.role_benchmarks = self._load_role_benchmarks()
        self.performance_thresholds = self._load_performance_thresholds()
    
    def analyze_comprehensive_performance(self, match_data: Dict[str, Any]) -> AdvancedPerformanceReport:
        """Perform comprehensive performance analysis."""
        hero_name = match_data.get("hero", "unknown")
        hero_info = hero_database.get_hero_info(hero_name)
        role = hero_info.role if hero_info else "unknown"
        
        # Calculate core metrics
        combat_efficiency = self._calculate_combat_efficiency(match_data, role)
        objective_participation = self._calculate_objective_participation(match_data, role)
        economic_efficiency = self._calculate_economic_efficiency(match_data, role)
        survival_rating = self._calculate_survival_rating(match_data, role)
        
        # Calculate role-specific metrics
        role_specific_metrics = self._calculate_role_specific_metrics(match_data, role)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            combat_efficiency, objective_participation, economic_efficiency, 
            survival_rating, role_specific_metrics
        )
        
        # Determine overall rating
        overall_rating = self._determine_overall_rating(overall_score, match_data)
        
        # ENHANCED: Apply MVP performance boost to fix Poor rating issue
        from .mvp_performance_booster import mvp_performance_booster
        
        boosted_rating, boosted_score, boost_analysis = mvp_performance_booster.analyze_and_boost_performance(
            match_data, overall_rating, overall_score, ocr_results=None
        )
        
        # Use boosted rating if it's higher than original
        if boosted_rating != overall_rating:
            logger.info(f"Performance boosted: {overall_rating.value} -> {boosted_rating.value}")
            overall_rating = boosted_rating
            overall_score = boosted_score
        
        # Generate insights
        strengths, weaknesses = self._analyze_strengths_weaknesses(
            combat_efficiency, objective_participation, economic_efficiency,
            survival_rating, role_specific_metrics
        )
        
        # Add MVP-specific insights
        if boost_analysis.get("boost_applied", False):
            boost_reasons = boost_analysis.get("boost_reasons", [])
            strengths.extend(boost_reasons)
        
        improvement_priorities = self._generate_improvement_priorities(
            weaknesses, role, match_data
        )
        
        advanced_insights = self._generate_advanced_insights(
            match_data, role, overall_score
        )
        
        return AdvancedPerformanceReport(
            hero=hero_name,
            role=role,
            overall_rating=overall_rating,
            overall_score=overall_score,
            combat_efficiency=combat_efficiency,
            objective_participation=objective_participation,
            economic_efficiency=economic_efficiency,
            survival_rating=survival_rating,
            role_specific_metrics=role_specific_metrics,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_priorities=improvement_priorities,
            advanced_insights=advanced_insights
        )
    
    def _calculate_combat_efficiency(self, match_data: Dict[str, Any], role: str) -> PerformanceMetric:
        """Calculate combat efficiency score."""
        kills = match_data.get("kills", 0)
        deaths = max(match_data.get("deaths", 1), 1)
        assists = match_data.get("assists", 0)
        hero_damage = match_data.get("hero_damage", 0)
        damage_taken = match_data.get("damage_taken", 1)
        
        # KDA component
        kda = (kills + assists) / deaths
        
        # Damage efficiency component
        damage_efficiency = hero_damage / damage_taken if damage_taken > 0 else 0
        
        # Role-specific weights
        if role == "marksman":
            combat_score = (kda * 0.4) + (damage_efficiency * 0.6)
            benchmark = 2.8
        elif role == "assassin":
            combat_score = (kda * 0.6) + (damage_efficiency * 0.4)
            benchmark = 3.2
        elif role == "mage":
            combat_score = (kda * 0.4) + (damage_efficiency * 0.6)
            benchmark = 2.5
        elif role == "fighter":
            combat_score = (kda * 0.5) + (damage_efficiency * 0.5)
            benchmark = 2.3
        elif role == "tank":
            # For tanks, focus more on assists and survival
            combat_score = (assists * 0.3) + ((assists + kills) / deaths * 0.7)
            benchmark = 2.0
        elif role == "support":
            combat_score = (assists * 0.4) + ((assists + kills) / deaths * 0.6)
            benchmark = 2.2
        else:
            combat_score = kda
            benchmark = 2.0
        
        return PerformanceMetric(
            name="Combat Efficiency",
            value=combat_score,
            benchmark=benchmark,
            weight=0.3,
            description="Overall combat effectiveness combining KDA and damage efficiency",
            improvement_tip="Focus on trading efficiently and positioning for optimal damage output"
        )
    
    def _calculate_objective_participation(self, match_data: Dict[str, Any], role: str) -> PerformanceMetric:
        """Calculate objective participation score."""
        teamfight_participation = match_data.get("teamfight_participation", 0)
        turret_damage = match_data.get("turret_damage", 0)
        gold = match_data.get("gold", 0)
        
        # Base participation from teamfight percentage
        base_participation = teamfight_participation / 100 if teamfight_participation > 0 else 0
        
        # Objective damage contribution (normalized)
        objective_contribution = min(turret_damage / 5000, 1.0) if turret_damage > 0 else 0
        
        # Gold contribution (indicates map presence)
        gold_contribution = min(gold / 12000, 1.0) if gold > 0 else 0
        
        # Role-specific calculation
        if role in ["marksman", "mage"]:
            participation_score = (base_participation * 0.4) + (objective_contribution * 0.4) + (gold_contribution * 0.2)
            benchmark = 0.75
        elif role in ["tank", "support"]:
            participation_score = (base_participation * 0.6) + (objective_contribution * 0.2) + (gold_contribution * 0.2)
            benchmark = 0.8
        else:
            participation_score = (base_participation * 0.5) + (objective_contribution * 0.3) + (gold_contribution * 0.2)
            benchmark = 0.7
        
        return PerformanceMetric(
            name="Objective Participation",
            value=participation_score,
            benchmark=benchmark,
            weight=0.25,
            description="Participation in teamfights and objective control",
            improvement_tip="Join more teamfights and focus on objective control"
        )
    
    def _calculate_economic_efficiency(self, match_data: Dict[str, Any], role: str) -> PerformanceMetric:
        """Calculate economic efficiency (GPM relative to role)."""
        gold_per_min = match_data.get("gold_per_min", 0)
        match_duration = match_data.get("match_duration", 1)
        
        # Role-specific GPM benchmarks
        role_benchmarks = {
            "marksman": 450,
            "mage": 420,
            "assassin": 400,
            "fighter": 380,
            "tank": 320,
            "support": 300
        }
        
        benchmark = role_benchmarks.get(role, 380)
        efficiency_score = gold_per_min / benchmark if benchmark > 0 else 0
        
        return PerformanceMetric(
            name="Economic Efficiency",
            value=efficiency_score,
            benchmark=1.0,
            weight=0.2,
            description="Gold per minute relative to role expectations",
            improvement_tip="Focus on farming efficiency and avoid unnecessary deaths"
        )
    
    def _calculate_survival_rating(self, match_data: Dict[str, Any], role: str) -> PerformanceMetric:
        """Calculate survival rating based on deaths and damage taken."""
        deaths = match_data.get("deaths", 1)
        damage_taken = match_data.get("damage_taken", 1)
        match_duration = match_data.get("match_duration", 1)
        
        # Deaths per minute
        deaths_per_min = deaths / match_duration if match_duration > 0 else deaths
        
        # Damage taken efficiency (less is better for carries, more acceptable for tanks)
        if role in ["marksman", "mage", "assassin"]:
            # Carries should take less damage
            damage_efficiency = max(0, 1 - (damage_taken / 15000))
            death_penalty = max(0, 1 - (deaths_per_min / 0.3))
            benchmark = 0.8
        elif role in ["tank", "support"]:
            # Tanks/supports can take more damage
            damage_efficiency = max(0, 1 - (damage_taken / 25000))
            death_penalty = max(0, 1 - (deaths_per_min / 0.4))
            benchmark = 0.7
        else:
            damage_efficiency = max(0, 1 - (damage_taken / 20000))
            death_penalty = max(0, 1 - (deaths_per_min / 0.35))
            benchmark = 0.75
        
        survival_score = (damage_efficiency * 0.4) + (death_penalty * 0.6)
        
        return PerformanceMetric(
            name="Survival Rating",
            value=survival_score,
            benchmark=benchmark,
            weight=0.15,
            description="Ability to stay alive and avoid unnecessary damage",
            improvement_tip="Improve positioning and map awareness to reduce deaths"
        )
    
    def _calculate_role_specific_metrics(self, match_data: Dict[str, Any], role: str) -> Dict[str, PerformanceMetric]:
        """Calculate role-specific performance metrics."""
        metrics = {}
        
        if role == "tank":
            # Tank-specific metrics
            if "hooks_landed" in match_data:
                hook_accuracy = match_data.get("hooks_landed", 0) / max(match_data.get("hooks_attempted", 1), 1)
                metrics["initiation_success"] = PerformanceMetric(
                    name="Initiation Success",
                    value=hook_accuracy,
                    benchmark=0.35,
                    weight=0.1,
                    description="Accuracy of skill shots and initiations",
                    improvement_tip="Practice skill shot timing and positioning"
                )
            
            # Engagement score
            assists = match_data.get("assists", 0)
            teamfight_participation = match_data.get("teamfight_participation", 0)
            engage_score = (assists * 0.1) + (teamfight_participation / 100 * 0.9)
            metrics["engagement_score"] = PerformanceMetric(
                name="Engagement Score",
                value=engage_score,
                benchmark=0.8,
                weight=0.1,
                description="Effectiveness in team engagements",
                improvement_tip="Focus on timing engagements and protecting carries"
            )
        
        elif role == "marksman":
            # Marksman-specific metrics
            hero_damage = match_data.get("hero_damage", 0)
            gold = match_data.get("gold", 1)
            damage_per_gold = hero_damage / gold if gold > 0 else 0
            
            metrics["damage_per_gold"] = PerformanceMetric(
                name="Damage per Gold",
                value=damage_per_gold,
                benchmark=2.5,
                weight=0.1,
                description="Damage output efficiency per gold earned",
                improvement_tip="Focus on item optimization and positioning in fights"
            )
            
            # Late game effectiveness
            match_duration = match_data.get("match_duration", 1)
            if match_duration > 15:
                late_game_factor = min(match_duration / 20, 1.5)
                damage_scaling = hero_damage * late_game_factor / 30000
                metrics["late_game_effectiveness"] = PerformanceMetric(
                    name="Late Game Effectiveness",
                    value=damage_scaling,
                    benchmark=1.0,
                    weight=0.1,
                    description="Performance scaling in late game",
                    improvement_tip="Focus on farming and positioning for late game teamfights"
                )
        
        elif role == "assassin":
            # Assassin-specific metrics
            kills = match_data.get("kills", 0)
            deaths = max(match_data.get("deaths", 1), 1)
            elimination_ratio = kills / deaths
            
            metrics["elimination_ratio"] = PerformanceMetric(
                name="Elimination Ratio",
                value=elimination_ratio,
                benchmark=2.0,
                weight=0.1,
                description="Kill to death ratio for target elimination",
                improvement_tip="Focus on target selection and escape timing"
            )
        
        elif role == "support":
            # Support-specific metrics
            assists = match_data.get("assists", 0)
            teamfight_participation = match_data.get("teamfight_participation", 0)
            support_score = (assists * 0.2) + (teamfight_participation / 100 * 0.8)
            
            metrics["support_effectiveness"] = PerformanceMetric(
                name="Support Effectiveness",
                value=support_score,
                benchmark=1.0,
                weight=0.1,
                description="Effectiveness in supporting team",
                improvement_tip="Focus on positioning to maximize team support"
            )
        
        return metrics
    
    def _calculate_overall_score(self, combat_efficiency: PerformanceMetric, 
                               objective_participation: PerformanceMetric,
                               economic_efficiency: PerformanceMetric,
                               survival_rating: PerformanceMetric,
                               role_specific_metrics: Dict[str, PerformanceMetric]) -> float:
        """Calculate weighted overall performance score."""
        score = 0
        total_weight = 0
        
        # Core metrics
        for metric in [combat_efficiency, objective_participation, economic_efficiency, survival_rating]:
            metric_score = metric.value / metric.benchmark if metric.benchmark > 0 else 0
            score += metric_score * metric.weight
            total_weight += metric.weight
        
        # Role-specific metrics
        for metric in role_specific_metrics.values():
            metric_score = metric.value / metric.benchmark if metric.benchmark > 0 else 0
            score += metric_score * metric.weight
            total_weight += metric.weight
        
        return score / total_weight if total_weight > 0 else 0
    
    def _determine_overall_rating(self, score: float, match_data: Dict[str, Any] = None) -> PerformanceCategory:
        """
        Determine overall performance rating with context awareness.
        
        Considers match outcome, player rank, and prevents inflated ratings
        for poor performance regardless of metric ratios.
        """
        # Base thresholds (more conservative than before)
        base_thresholds = {
            PerformanceCategory.EXCELLENT: 1.4,  # Raised from 1.25
            PerformanceCategory.GOOD: 1.15,      # Raised from 1.05  
            PerformanceCategory.AVERAGE: 0.9,    # Raised from 0.85
            PerformanceCategory.NEEDS_WORK: 0.7, # Raised from 0.65
        }
        
        # Context-aware adjustments
        if match_data:
            # Check match outcome - defeats heavily penalize rating
            match_result = match_data.get("match_result", "").lower()
            is_defeat = match_result in ["defeat", "loss", "lose"]
            
            # Player rank context (if available)
            player_rank = match_data.get("player_rank", "").lower()
            is_bronze_tier = any(tier in player_rank for tier in ["bronze", "warrior", "elite"])
            
            # KDA context for objective assessment
            kills = match_data.get("kills", 0)
            deaths = match_data.get("deaths", 1)
            assists = match_data.get("assists", 0)
            kda = (kills + assists) / max(deaths, 1)
            
            # Apply defeat penalty
            if is_defeat:
                # Heavy penalty for defeats - max rating becomes "Good"
                score *= 0.7  # 30% score reduction
                # Further cap the maximum possible rating
                max_rating_on_defeat = PerformanceCategory.GOOD
            else:
                max_rating_on_defeat = PerformanceCategory.EXCELLENT
            
            # Apply rank-appropriate expectations
            if is_bronze_tier:
                # Lower tier players need higher scores for same rating
                score *= 0.85  # More stringent requirements
                
                # Bronze defeats with poor KDA should never be "Excellent" or "Good"
                if is_defeat and kda < 1.5:
                    max_rating_on_defeat = PerformanceCategory.AVERAGE
            
            # Objective performance floor - prevent rating inflation
            # If KDA is very poor, cap the rating regardless of other metrics
            if kda < 0.8:  # Very poor KDA
                max_rating_on_defeat = PerformanceCategory.NEEDS_WORK
            elif kda < 1.2 and is_defeat:  # Poor KDA in defeat
                max_rating_on_defeat = PerformanceCategory.AVERAGE
        else:
            max_rating_on_defeat = PerformanceCategory.EXCELLENT
        
        # Determine rating from adjusted score
        if score >= base_thresholds[PerformanceCategory.EXCELLENT]:
            rating = PerformanceCategory.EXCELLENT
        elif score >= base_thresholds[PerformanceCategory.GOOD]:
            rating = PerformanceCategory.GOOD
        elif score >= base_thresholds[PerformanceCategory.AVERAGE]:
            rating = PerformanceCategory.AVERAGE
        elif score >= base_thresholds[PerformanceCategory.NEEDS_WORK]:
            rating = PerformanceCategory.NEEDS_WORK
        else:
            rating = PerformanceCategory.POOR
        
        # Apply context ceiling
        rating_order = [
            PerformanceCategory.POOR,
            PerformanceCategory.NEEDS_WORK, 
            PerformanceCategory.AVERAGE,
            PerformanceCategory.GOOD,
            PerformanceCategory.EXCELLENT
        ]
        
        max_index = rating_order.index(max_rating_on_defeat)
        current_index = rating_order.index(rating)
        
        if current_index > max_index:
            rating = max_rating_on_defeat
            
        return rating
    
    def _analyze_strengths_weaknesses(self, *metrics) -> Tuple[List[str], List[str]]:
        """Analyze strengths and weaknesses from metrics."""
        strengths = []
        weaknesses = []
        
        all_metrics = []
        
        # Add core metrics
        for metric in metrics[:4]:  # First 4 are core metrics
            all_metrics.append(metric)
        
        # Add role-specific metrics
        if len(metrics) > 4 and isinstance(metrics[4], dict):
            for metric in metrics[4].values():
                all_metrics.append(metric)
        
        for metric in all_metrics:
            if metric.category == PerformanceCategory.EXCELLENT:
                strengths.append(f"Outstanding {metric.name.lower()}")
            elif metric.category == PerformanceCategory.GOOD:
                strengths.append(f"Strong {metric.name.lower()}")
            elif metric.category == PerformanceCategory.NEEDS_WORK:
                weaknesses.append(f"Weak {metric.name.lower()}")
            elif metric.category == PerformanceCategory.POOR:
                weaknesses.append(f"Poor {metric.name.lower()}")
        
        return strengths, weaknesses
    
    def _generate_improvement_priorities(self, weaknesses: List[str], role: str, 
                                       match_data: Dict[str, Any]) -> List[str]:
        """Generate improvement priorities based on weaknesses."""
        priorities = []
        
        # Priority based on role and weaknesses
        if "poor combat efficiency" in [w.lower() for w in weaknesses]:
            if role in ["marksman", "assassin", "mage"]:
                priorities.append("Focus on positioning and damage output in teamfights")
            elif role == "tank":
                priorities.append("Improve engagement timing and target selection")
        
        if "poor survival rating" in [w.lower() for w in weaknesses]:
            priorities.append("Work on map awareness and positioning to avoid deaths")
        
        if "poor economic efficiency" in [w.lower() for w in weaknesses]:
            priorities.append("Improve farming patterns and avoid unnecessary recalls")
        
        if "poor objective participation" in [w.lower() for w in weaknesses]:
            priorities.append("Join more teamfights and focus on objective control")
        
        # Add general improvement if multiple weaknesses
        if len(weaknesses) > 2:
            priorities.append("Consider reviewing fundamentals and game mechanics")
        
        return priorities[:3]  # Top 3 priorities
    
    def _generate_advanced_insights(self, match_data: Dict[str, Any], role: str, 
                                  overall_score: float) -> List[str]:
        """Generate advanced performance insights."""
        insights = []
        
        # Match duration insights
        duration = match_data.get("match_duration", 0)
        if duration > 20:
            insights.append("Long match duration suggests good scaling or defensive play")
        elif duration < 10:
            insights.append("Short match indicates either strong early game or poor resistance")
        
        # Gold efficiency insights
        gold = match_data.get("gold", 0)
        gold_per_min = match_data.get("gold_per_min", 0)
        if gold_per_min > 450:
            insights.append("Excellent farming efficiency - maintain this economic advantage")
        elif gold_per_min < 300:
            insights.append("Low gold generation - focus on improving farming patterns")
        
        # Damage insights
        hero_damage = match_data.get("hero_damage", 0)
        damage_taken = match_data.get("damage_taken", 0)
        if hero_damage > 0 and damage_taken > 0:
            damage_ratio = hero_damage / damage_taken
            if damage_ratio > 2.0:
                insights.append("Excellent damage trading - high output with low risk")
            elif damage_ratio < 0.8:
                insights.append("Poor damage trading - taking more damage than dealing")
        
        # Role-specific insights
        if role == "tank":
            assists = match_data.get("assists", 0)
            if assists > 15:
                insights.append("High assist count indicates good team support and engagement")
        elif role == "marksman":
            if hero_damage > 25000:
                insights.append("High damage output - good positioning and target selection")
        
        return insights
    
    def _load_role_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Load role-specific performance benchmarks."""
        return {
            "tank": {
                "combat_efficiency": 2.0,
                "objective_participation": 0.8,
                "economic_efficiency": 320,
                "survival_rating": 0.7
            },
            "marksman": {
                "combat_efficiency": 2.8,
                "objective_participation": 0.75,
                "economic_efficiency": 450,
                "survival_rating": 0.8
            },
            "mage": {
                "combat_efficiency": 2.5,
                "objective_participation": 0.75,
                "economic_efficiency": 420,
                "survival_rating": 0.8
            },
            "assassin": {
                "combat_efficiency": 3.2,
                "objective_participation": 0.7,
                "economic_efficiency": 400,
                "survival_rating": 0.75
            },
            "fighter": {
                "combat_efficiency": 2.3,
                "objective_participation": 0.7,
                "economic_efficiency": 380,
                "survival_rating": 0.75
            },
            "support": {
                "combat_efficiency": 2.2,
                "objective_participation": 0.8,
                "economic_efficiency": 300,
                "survival_rating": 0.7
            }
        }
    
    def _load_performance_thresholds(self) -> Dict[str, float]:
        """Load performance category thresholds."""
        return {
            "excellent": 1.25,
            "good": 1.05,
            "average": 0.85,
            "needs_work": 0.65,
            "poor": 0.0
        }
    
    def generate_detailed_feedback(self, report: AdvancedPerformanceReport) -> List[str]:
        """Generate detailed feedback based on performance report."""
        feedback = []
        
        # Overall performance
        feedback.append(f"Overall Performance: {report.overall_rating.value.title()}")
        
        # Strengths
        if report.strengths:
            feedback.append(f"Strengths: {', '.join(report.strengths)}")
        
        # Areas for improvement
        if report.improvement_priorities:
            feedback.append("Priority improvements:")
            for priority in report.improvement_priorities:
                feedback.append(f"  • {priority}")
        
        # Specific metric feedback
        metrics = [
            report.combat_efficiency,
            report.objective_participation,
            report.economic_efficiency,
            report.survival_rating
        ]
        
        for metric in metrics:
            if metric.category in [PerformanceCategory.POOR, PerformanceCategory.NEEDS_WORK]:
                feedback.append(f"{metric.name}: {metric.improvement_tip}")
        
        # Advanced insights
        if report.advanced_insights:
            feedback.append("Advanced insights:")
            for insight in report.advanced_insights:
                feedback.append(f"  • {insight}")
        
        return feedback


# Global instance
advanced_performance_analyzer = AdvancedPerformanceAnalyzer()