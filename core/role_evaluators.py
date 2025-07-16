"""
Role-specific evaluator classes for MLBB Coach AI.
Each role has unique evaluation logic and priorities.

Architecture:
- Inherits from BaseEvaluator for common functionality
- Implements role-specific evaluation methods
- Supports caching and async processing
- Event-driven feedback generation
"""

import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from abc import abstractmethod

from .base_evaluator import BaseEvaluator
from .cache.decorators import cache_result
from .events.event_types import EvaluationEvent


logger = logging.getLogger(__name__)


@dataclass
class RoleEvaluationResult:
    """Structured result from role evaluation."""
    role: str
    hero: str
    overall_score: float
    feedback: List[Tuple[str, str]]
    role_specific_metrics: Dict[str, float]
    suggestions: List[str]
    confidence: float


class RoleEvaluator(BaseEvaluator):
    """Base class for role-specific evaluators."""
    
    def __init__(self, role: str):
        super().__init__()
        self.role = role
        self.role_thresholds = self.roles.get(role, {})
    
    async def evaluate_async(self, data: Dict[str, Any],
                             minutes: int = None) -> RoleEvaluationResult:
        """Async evaluation for better performance."""
        # Emit evaluation start event
        await self._emit_event(EvaluationEvent(
            type="evaluation_start",
            role=self.role,
            hero=data.get('hero', 'Unknown'),
            timestamp=self._get_timestamp()
        ))
        
        # Perform evaluation
        result = await self._evaluate_role_specific(data, minutes)
        
        # Emit evaluation complete event
        await self._emit_event(EvaluationEvent(
            type="evaluation_complete",
            role=self.role,
            hero=data.get('hero', 'Unknown'),
            score=result.overall_score,
            timestamp=self._get_timestamp()
        ))
        
        return result
    
    @abstractmethod
    async def _evaluate_role_specific(self, data: Dict[str, Any],
                                      minutes: int) -> RoleEvaluationResult:
        """Role-specific evaluation logic."""
        pass
    
    def _calculate_role_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall role performance score."""
        formula_cfg = self.config.get('evaluation_formulas', {})
        weights = formula_cfg.get('role_weights', {}).get(self.role, {})
        
        if not weights:
            return 0.5  # Default neutral score
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                total_score += metrics[metric] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    async def _emit_event(self, event: Any):
        """Emit evaluation event."""
        # In production, this would emit to event bus
        logger.debug(f"Event emitted: {event}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


class TankEvaluator(RoleEvaluator):
    """
    Tank evaluator - The Shield Wall
    
    Tanks are the backbone of team composition. They initiate fights,
    absorb damage, and create space for allies. Think of them as the
    castle gates that must hold the line while enabling victory.
    """
    
    def __init__(self):
        super().__init__("tank")
    
    @cache_result(ttl=60)
    async def _evaluate_role_specific(self, data: Dict[str, Any],
                                      minutes: int = None) -> RoleEvaluationResult:
        """Tank-specific evaluation focusing on initiation and protection."""
        if minutes is None:
            minutes = data.get('match_duration', 15)
        
        # Get tank thresholds
        tank_thresholds = self.role_thresholds
        
        # Calculate core tank metrics
        metrics = await self._calculate_tank_metrics(
            data, minutes, tank_thresholds)
        
        # Generate tank-specific feedback
        feedback = await self._generate_tank_feedback(
            data, metrics, tank_thresholds)
        
        # Calculate overall score
        overall_score = self._calculate_role_score(metrics)
        
        # Generate improvement suggestions
        suggestions = await self._generate_tank_suggestions(
            metrics, tank_thresholds)
        
        return RoleEvaluationResult(
            role="tank",
            hero=data.get('hero', 'Unknown'),
            overall_score=overall_score,
            feedback=feedback,
            role_specific_metrics=metrics,
            suggestions=suggestions,
            confidence=self._calculate_confidence(data, metrics)
        )
    
    async def _calculate_tank_metrics(self, data: Dict[str, Any],
                                      minutes: int,
                                      thresholds: Dict) -> Dict[str, float]:
        """Calculate tank-specific performance metrics."""
        metrics = {}
        
        # Survivability (30% weight)
        damage_taken = data.get('damage_taken', 0)
        combat_cfg = thresholds.get('combat', {})
        expected_damage = combat_cfg.get('damage_taken_base', 4500) * minutes
        if expected_damage > 0:
            survivability = min(1.0, damage_taken / expected_damage)
        else:
            survivability = 0.5
        metrics['survivability'] = survivability
        
        # Initiation effectiveness (25% weight)
        teamfight_participation = data.get('teamfight_participation', 0) / 100
        engage_success = data.get('engage_success_rate', 60) / 100
        initiation = (teamfight_participation * 0.6 + engage_success * 0.4)
        metrics['initiation'] = initiation
        
        # Protection score (25% weight)
        assists = data.get('assists', 0)
        deaths = max(1, data.get('deaths', 1))
        protection = min(1.0, assists / (deaths * 2))
        metrics['protection'] = protection
        
        # Utility effectiveness (20% weight)
        cc_score = data.get('crowd_control_score', 50) / 100
        vision_score = data.get('vision_score', 30) / 100
        utility = (cc_score * 0.7 + vision_score * 0.3)
        metrics['utility'] = utility
        
        return metrics
    
    async def _generate_tank_feedback(self, data: Dict[str, Any],
                                      metrics: Dict[str, float],
                                      thresholds: Dict) -> List[Tuple[str, str]]:
        """Generate tank-specific feedback."""
        feedback = []
        
        # Survivability feedback
        if metrics['survivability'] < 0.6:
            feedback.append(("warning",
                           "Low damage absorption. Position more aggressively to "
                           "draw fire away from your carries."))
        elif metrics['survivability'] > 0.9:
            feedback.append(("success",
                           "Excellent damage absorption! Your frontline presence "
                           "is creating space for your team."))
        
        # Initiation feedback
        if metrics['initiation'] < 0.5:
            feedback.append(("warning",
                           "Improve fight initiation. Look for opportunities to "
                           "engage when your team is ready to follow up."))
        elif metrics['initiation'] > 0.8:
            feedback.append(("success",
                           "Outstanding initiation! Your engages are creating "
                           "winning opportunities for your team."))
        
        # Protection feedback
        deaths = data.get('deaths', 0)
        if deaths > 8:
            feedback.append(("critical",
                           f"{deaths} deaths is excessive. Even tanks need to "
                           f"choose their moments - don't feed the enemy."))
        elif deaths <= 3:
            feedback.append(("success",
                           f"Great survival with {deaths} deaths. You're making "
                           f"impactful plays without feeding."))
        
        # KDA context for tanks
        kills = data.get('kills', 0)
        assists = data.get('assists', 0)
        if assists > kills * 2:
            feedback.append(("success",
                           f"Excellent support with {assists} assists. You're "
                           f"enabling your team's success."))
        
        return feedback
    
    async def _generate_tank_suggestions(self, metrics: Dict[str, float],
                                         thresholds: Dict) -> List[str]:
        """Generate tank improvement suggestions."""
        suggestions = []
        
        if metrics['survivability'] < 0.6:
            suggestions.append("Build more defensive items early game")
            suggestions.append("Practice positioning at the edge of enemy range")
        
        if metrics['initiation'] < 0.5:
            suggestions.append("Communicate with team before engaging")
            suggestions.append("Look for isolated enemies to hook/stun")
        
        if metrics['protection'] < 0.4:
            suggestions.append("Stay close to your carries in teamfights")
            suggestions.append("Use your skills to peel for allies")
        
        if metrics['utility'] < 0.5:
            suggestions.append("Place more wards in key locations")
            suggestions.append("Time your crowd control with team combos")
        
        return suggestions
    
    def _calculate_confidence(self, data: Dict[str, Any],
                              metrics: Dict[str, float]) -> float:
        """Calculate confidence in evaluation."""
        # High confidence if we have good data coverage
        required_fields = ['damage_taken', 'teamfight_participation',
                          'assists', 'deaths']
        present_fields = sum(1 for field in required_fields
                            if data.get(field) is not None)
        
        base_confidence = present_fields / len(required_fields)
        
        # Adjust for metric consistency
        metric_variance = sum(abs(m - 0.5) for m in metrics.values())
        metric_variance = metric_variance / len(metrics)
        confidence_adjustment = min(0.2, metric_variance)
        
        return min(1.0, base_confidence + confidence_adjustment)


class AssassinEvaluator(RoleEvaluator):
    """
    Assassin evaluator - The Shadow Striker
    
    Assassins are surgical instruments that eliminate key targets.
    They need to snowball early and maintain kill pressure while
    staying alive through superior positioning and timing.
    """
    
    def __init__(self):
        super().__init__("assassin")
    
    @cache_result(ttl=60)
    async def _evaluate_role_specific(self, data: Dict[str, Any],
                                      minutes: int = None) -> RoleEvaluationResult:
        """Assassin-specific evaluation focusing on elimination."""
        if minutes is None:
            minutes = data.get('match_duration', 15)
        
        # Get assassin thresholds
        assassin_thresholds = self.role_thresholds
        
        # Calculate core assassin metrics
        metrics = await self._calculate_assassin_metrics(
            data, minutes, assassin_thresholds)
        
        # Generate assassin-specific feedback
        feedback = await self._generate_assassin_feedback(
            data, metrics, assassin_thresholds)
        
        # Calculate overall score
        overall_score = self._calculate_role_score(metrics)
        
        # Generate improvement suggestions
        suggestions = await self._generate_assassin_suggestions(
            metrics, assassin_thresholds)
        
        return RoleEvaluationResult(
            role="assassin",
            hero=data.get('hero', 'Unknown'),
            overall_score=overall_score,
            feedback=feedback,
            role_specific_metrics=metrics,
            suggestions=suggestions,
            confidence=self._calculate_confidence(data, metrics)
        )
    
    async def _calculate_assassin_metrics(self, data: Dict[str, Any],
                                          minutes: int,
                                          thresholds: Dict) -> Dict[str, float]:
        """Calculate assassin-specific performance metrics."""
        metrics = {}
        
        # Elimination effectiveness (35% weight)
        kills = data.get('kills', 0)
        assists = data.get('assists', 0)
        total_eliminations = kills + assists
        expected_eliminations = max(6, minutes * 0.5)
        elimination = min(1.0, total_eliminations / expected_eliminations)
        metrics['elimination'] = elimination
        
        # Mobility usage (25% weight)
        deaths = max(1, data.get('deaths', 1))
        damage_dealt = data.get('hero_damage', 0)
        mobility_proxy = min(1.0, (damage_dealt / 1000) / (deaths * 2))
        metrics['mobility'] = mobility_proxy
        
        # Snowball potential (25% weight)
        gold_per_min = data.get('gold_per_min', 0)
        expected_gpm = thresholds.get('economy', {}).get('gpm_base', 700)
        snowball = min(1.0, gold_per_min / expected_gpm)
        metrics['snowball'] = snowball
        
        # Positioning (15% weight)
        positioning = max(0.1, 1.0 - (deaths / 10))
        metrics['positioning'] = positioning
        
        return metrics
    
    async def _generate_assassin_feedback(self, data: Dict[str, Any],
                                          metrics: Dict[str, float],
                                          thresholds: Dict) -> List[Tuple[str, str]]:
        """Generate assassin-specific feedback."""
        feedback = []
        
        # Elimination feedback
        kills = data.get('kills', 0)
        assists = data.get('assists', 0)
        if kills < 3:
            feedback.append(("warning",
                           f"Only {kills} kills. Assassins need to secure "
                           f"eliminations to stay relevant. Look for isolated "
                           f"targets."))
        elif kills > 8:
            feedback.append(("success",
                           f"Excellent {kills} kills! You're successfully "
                           f"eliminating key targets and creating advantages."))
        
        # Mobility feedback
        deaths = data.get('deaths', 0)
        if deaths > 5:
            feedback.append(("critical",
                           f"{deaths} deaths is too high for an assassin. "
                           f"Use your mobility to engage and escape safely."))
        elif deaths <= 2:
            feedback.append(("success",
                           f"Great survival with {deaths} deaths. Your "
                           f"positioning and timing are on point."))
        
        # Snowball feedback
        if metrics['snowball'] < 0.7:
            feedback.append(("warning",
                           "Low gold income. Farm efficiently between ganks "
                           "and prioritize objectives for more gold."))
        
        # Damage ratio feedback
        damage_dealt = data.get('hero_damage', 0)
        damage_taken = data.get('damage_taken', 1)
        if damage_taken > 0:
            damage_ratio = damage_dealt / damage_taken
            if damage_ratio < 1.5:
                feedback.append(("warning",
                               f"Damage ratio {damage_ratio:.1f} is low. "
                               f"Use hit-and-run tactics to maximize damage "
                               f"while minimizing exposure."))
            elif damage_ratio > 3.0:
                feedback.append(("success",
                               f"Excellent damage ratio {damage_ratio:.1f}! "
                               f"Your positioning and timing are creating "
                               f"maximum impact."))
        
        return feedback
    
    async def _generate_assassin_suggestions(self, metrics: Dict[str, float],
                                             thresholds: Dict) -> List[str]:
        """Generate assassin improvement suggestions."""
        suggestions = []
        
        if metrics['elimination'] < 0.6:
            suggestions.append("Focus on eliminating squishy targets "
                              "(mages, marksmen)")
            suggestions.append("Time your engages when enemies are isolated")
        
        if metrics['mobility'] < 0.5:
            suggestions.append("Practice using mobility skills for both "
                              "engage and escape")
            suggestions.append("Learn to cancel skill animations for "
                              "faster combos")
        
        if metrics['snowball'] < 0.6:
            suggestions.append("Farm jungle monsters efficiently between ganks")
            suggestions.append("Participate in early objectives for "
                              "gold advantage")
        
        if metrics['positioning'] < 0.5:
            suggestions.append("Wait for tank initiation before engaging")
            suggestions.append("Use brush and terrain to approach unseen")
        
        return suggestions
    
    def _calculate_confidence(self, data: Dict[str, Any],
                              metrics: Dict[str, float]) -> float:
        """Calculate confidence in evaluation."""
        required_fields = ['kills', 'deaths', 'assists', 'hero_damage',
                          'gold_per_min']
        present_fields = sum(1 for field in required_fields
                            if data.get(field) is not None)
        
        base_confidence = present_fields / len(required_fields)
        
        # Higher confidence for assassins with clear performance patterns
        kills = data.get('kills', 0)
        deaths = data.get('deaths', 0)
        if kills > 5 or deaths > 5:  # Clear performance indicators
            base_confidence += 0.1
        
        return min(1.0, base_confidence)


class MarksmanEvaluator(RoleEvaluator):
    """
    Marksman evaluator - The Precision Artillery
    
    Marksmen are the primary damage dealers who need protection
    to deliver consistent damage. They scale with items and need
    to position safely while maximizing damage output.
    """
    
    def __init__(self):
        super().__init__("marksman")
    
    @cache_result(ttl=60)
    async def _evaluate_role_specific(self, data: Dict[str, Any],
                                      minutes: int = None) -> RoleEvaluationResult:
        """Marksman-specific evaluation focusing on damage output and positioning."""
        if minutes is None:
            minutes = data.get('match_duration', 15)
        
        # Get marksman thresholds
        mm_thresholds = self.role_thresholds
        
        # Calculate core marksman metrics
        metrics = await self._calculate_marksman_metrics(
            data, minutes, mm_thresholds)
        
        # Generate marksman-specific feedback
        feedback = await self._generate_marksman_feedback(
            data, metrics, mm_thresholds)
        
        # Calculate overall score
        overall_score = self._calculate_role_score(metrics)
        
        # Generate improvement suggestions
        suggestions = await self._generate_marksman_suggestions(
            metrics, mm_thresholds)
        
        return RoleEvaluationResult(
            role="marksman",
            hero=data.get('hero', 'Unknown'),
            overall_score=overall_score,
            feedback=feedback,
            role_specific_metrics=metrics,
            suggestions=suggestions,
            confidence=self._calculate_confidence(data, metrics)
        )
    
    async def _calculate_marksman_metrics(self, data: Dict[str, Any],
                                          minutes: int,
                                          thresholds: Dict) -> Dict[str, float]:
        """Calculate marksman-specific performance metrics."""
        metrics = {}
        
        # Damage output (35% weight)
        damage_dealt = data.get('hero_damage', 0)
        expected_damage = thresholds.get('combat', {}).get('damage_base', 5500)
        expected_damage *= minutes
        if expected_damage > 0:
            damage_output = min(1.0, damage_dealt / expected_damage)
        else:
            damage_output = 0.5
        metrics['damage_output'] = damage_output
        
        # Positioning (25% weight)
        deaths = data.get('deaths', 0)
        positioning = max(0.1, 1.0 - (deaths / 8))
        metrics['positioning'] = positioning
        
        # Economy (25% weight)
        gold_per_min = data.get('gold_per_min', 0)
        expected_gpm = thresholds.get('economy', {}).get('gpm_base', 750)
        if expected_gpm > 0:
            economy = min(1.0, gold_per_min / expected_gpm)
        else:
            economy = 0.5
        metrics['economy'] = economy
        
        # Scaling (15% weight)
        damage_per_min = damage_dealt / max(1, minutes)
        scaling = min(1.0, damage_per_min / 4000)
        metrics['scaling'] = scaling
        
        return metrics
    
    async def _generate_marksman_feedback(self, data: Dict[str, Any],
                                          metrics: Dict[str, float],
                                          thresholds: Dict) -> List[Tuple[str, str]]:
        """Generate marksman-specific feedback."""
        feedback = []
        
        # Damage output feedback
        team_damage_pct = data.get('damage_percentage', 0)
        
        if team_damage_pct > 35:
            feedback.append(("success",
                           f"Excellent damage output! {team_damage_pct}% of "
                           f"team damage shows you're carrying effectively."))
        elif team_damage_pct < 25:
            feedback.append(("warning",
                           f"Low damage contribution ({team_damage_pct}%). "
                           f"Focus on consistent damage output in teamfights."))
        
        # Positioning feedback
        deaths = data.get('deaths', 0)
        if deaths > 4:
            feedback.append(("critical",
                           f"{deaths} deaths is too high for a marksman. "
                           f"Stay behind your frontline and use your range "
                           f"advantage."))
        elif deaths <= 1:
            feedback.append(("success",
                           f"Outstanding positioning with {deaths} deaths. "
                           f"You're maximizing your damage while staying safe."))
        
        # Economy feedback
        gold_per_min = data.get('gold_per_min', 0)
        if gold_per_min < 600:
            feedback.append(("warning",
                           f"Low GPM ({gold_per_min}). Farm side lanes and "
                           f"jungle between teamfights to accelerate your "
                           f"items."))
        elif gold_per_min > 800:
            feedback.append(("success",
                           f"Excellent economy with {gold_per_min} GPM. "
                           f"Your farming efficiency is creating a significant "
                           f"item advantage."))
        
        # KDA feedback for marksmen
        kills = data.get('kills', 0)
        assists = data.get('assists', 0)
        if kills + assists < 5:
            feedback.append(("warning",
                           f"Low kill participation ({kills + assists}). "
                           f"Join more teamfights to contribute to eliminations."))
        
        return feedback
    
    async def _generate_marksman_suggestions(self, metrics: Dict[str, float],
                                             thresholds: Dict) -> List[str]:
        """Generate marksman improvement suggestions."""
        suggestions = []
        
        if metrics['damage_output'] < 0.7:
            suggestions.append("Focus on hitting the nearest enemy consistently")
            suggestions.append("Build damage items before defensive items")
        
        if metrics['positioning'] < 0.6:
            suggestions.append("Stay at maximum attack range from enemies")
            suggestions.append("Use your tank as a shield in teamfights")
        
        if metrics['economy'] < 0.6:
            suggestions.append("Clear jungle camps when safe")
            suggestions.append("Farm side lanes when not teamfighting")
        
        if metrics['scaling'] < 0.5:
            suggestions.append("Prioritize core damage items early")
            suggestions.append("Join teamfights once you have key items")
        
        return suggestions
    
    def _calculate_confidence(self, data: Dict[str, Any],
                              metrics: Dict[str, float]) -> float:
        """Calculate confidence in evaluation."""
        required_fields = ['hero_damage', 'deaths', 'gold_per_min',
                          'damage_percentage']
        present_fields = sum(1 for field in required_fields
                            if data.get(field) is not None)
        
        base_confidence = present_fields / len(required_fields)
        
        # Higher confidence for clear damage patterns
        damage_pct = data.get('damage_percentage', 0)
        if damage_pct > 30 or damage_pct < 15:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)


class MageEvaluator(RoleEvaluator):
    """Mage evaluator - The Elemental Master"""
    
    def __init__(self):
        super().__init__("mage")
    
    @cache_result(ttl=60)
    async def _evaluate_role_specific(self, data: Dict[str, Any],
                                      minutes: int = None) -> RoleEvaluationResult:
        """Mage-specific evaluation focusing on burst damage and area control."""
        if minutes is None:
            minutes = data.get('match_duration', 15)
        
        # Calculate mage metrics (simplified for brevity)
        metrics = {
            'burst_damage': min(1.0, data.get('hero_damage', 0) / (5000 * minutes)),
            'area_control': 0.7,  # Placeholder
            'combo_execution': 0.8,  # Placeholder
            'resource_management': 0.6  # Placeholder
        }
        
        feedback = [
            ("info", "Mage evaluation - focus on burst damage and area control"),
            ("success", f"Damage dealt: {data.get('hero_damage', 0):,}")
        ]
        
        suggestions = ["Practice skill shot accuracy", "Time your burst combos"]
        
        return RoleEvaluationResult(
            role="mage",
            hero=data.get('hero', 'Unknown'),
            overall_score=self._calculate_role_score(metrics),
            feedback=feedback,
            role_specific_metrics=metrics,
            suggestions=suggestions,
            confidence=0.8
        )
    
    def _calculate_confidence(self, data: Dict[str, Any],
                              metrics: Dict[str, float]) -> float:
        return 0.8  # Placeholder


class FighterEvaluator(RoleEvaluator):
    """Fighter evaluator - The Versatile Warrior"""
    
    def __init__(self):
        super().__init__("fighter")
    
    @cache_result(ttl=60)
    async def _evaluate_role_specific(self, data: Dict[str, Any],
                                      minutes: int = None) -> RoleEvaluationResult:
        """Fighter-specific evaluation focusing on versatility and sustain."""
        if minutes is None:
            minutes = data.get('match_duration', 15)
        
        # Calculate fighter metrics (simplified for brevity)
        metrics = {
            'damage_trade': 0.7,  # Placeholder
            'versatility': 0.8,   # Placeholder
            'objective_control': 0.6,  # Placeholder
            'sustain': 0.7  # Placeholder
        }
        
        feedback = [
            ("info", "Fighter evaluation - balance damage and durability"),
            ("success", f"Versatile performance across multiple areas")
        ]
        
        suggestions = ["Balance offensive and defensive items",
                      "Control objectives"]
        
        return RoleEvaluationResult(
            role="fighter",
            hero=data.get('hero', 'Unknown'),
            overall_score=self._calculate_role_score(metrics),
            feedback=feedback,
            role_specific_metrics=metrics,
            suggestions=suggestions,
            confidence=0.8
        )
    
    def _calculate_confidence(self, data: Dict[str, Any],
                              metrics: Dict[str, float]) -> float:
        return 0.8  # Placeholder


class SupportEvaluator(RoleEvaluator):
    """Support evaluator - The Guardian Angel"""
    
    def __init__(self):
        super().__init__("support")
    
    @cache_result(ttl=60)
    async def _evaluate_role_specific(self, data: Dict[str, Any],
                                      minutes: int = None) -> RoleEvaluationResult:
        """Support-specific evaluation focusing on team enablement."""
        if minutes is None:
            minutes = data.get('match_duration', 15)
        
        # Calculate support metrics (simplified for brevity)
        metrics = {
            'enablement': 0.8,    # Placeholder
            'utility': 0.7,       # Placeholder
            'vision': 0.6,        # Placeholder
            'protection': 0.8     # Placeholder
        }
        
        feedback = [
            ("info", "Support evaluation - focus on team enablement"),
            ("success", f"High assist participation: {data.get('assists', 0)}")
        ]
        
        suggestions = ["Improve ward placement", "Time utility skills better"]
        
        return RoleEvaluationResult(
            role="support",
            hero=data.get('hero', 'Unknown'),
            overall_score=self._calculate_role_score(metrics),
            feedback=feedback,
            role_specific_metrics=metrics,
            suggestions=suggestions,
            confidence=0.8
        )
    
    def _calculate_confidence(self, data: Dict[str, Any],
                              metrics: Dict[str, float]) -> float:
        return 0.8  # Placeholder


class RoleEvaluatorFactory:
    """Factory for creating role-specific evaluators."""
    
    _evaluators = {
        'tank': TankEvaluator,
        'assassin': AssassinEvaluator,
        'marksman': MarksmanEvaluator,
        'mage': MageEvaluator,
        'fighter': FighterEvaluator,
        'support': SupportEvaluator
    }
    
    @classmethod
    def get_evaluator(cls, role: str) -> RoleEvaluator:
        """Get evaluator for specific role."""
        evaluator_class = cls._evaluators.get(role.lower())
        if evaluator_class:
            return evaluator_class()
        else:
            # Default to base evaluator
            return RoleEvaluator(role)
    
    @classmethod
    def get_available_roles(cls) -> List[str]:
        """Get list of available roles."""
        return list(cls._evaluators.keys()) 