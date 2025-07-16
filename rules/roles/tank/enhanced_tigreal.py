"""
Enhanced Tigreal evaluator using the new role-specific evaluation system.
Demonstrates hero-specific overrides and advanced evaluation logic.
"""

from typing import Dict, List, Tuple, Any
from core.role_evaluators import TankEvaluator, RoleEvaluationResult
from core.cache.decorators import cache_result


class EnhancedTigrealEvaluator(TankEvaluator):
    """
    Enhanced Tigreal evaluator with hero-specific mechanics.
    
    Tigreal is a pure initiator tank - he excels at starting fights
    and can afford higher death counts if those deaths create
    winning opportunities for his team.
    
    Special considerations:
    - Lower KDA tolerance due to initiation role
    - Higher damage absorption expectations
    - Ultimate usage effectiveness tracking
    - Multi-target crowd control evaluation
    """
    
    def __init__(self):
        super().__init__()
        self.hero_name = "Tigreal"
        # Override hero-specific thresholds
        self.hero_thresholds = self.heroes.get('tigreal', {})
    
    @cache_result(ttl=60)
    async def _evaluate_role_specific(self, data: Dict[str, Any],
                                      minutes: int = None) -> RoleEvaluationResult:
        """Enhanced Tigreal evaluation with hero mechanics."""
        if minutes is None:
            minutes = data.get('match_duration', 15)
        
        # Get base tank evaluation
        base_result = await super()._evaluate_role_specific(data, minutes)
        
        # Add Tigreal-specific evaluations
        tigreal_feedback = await self._evaluate_tigreal_specifics(
            data, minutes, base_result.role_specific_metrics)
        
        # Combine feedback
        combined_feedback = base_result.feedback + tigreal_feedback
        
        # Add Tigreal-specific suggestions
        tigreal_suggestions = await self._generate_tigreal_suggestions(
            data, base_result.role_specific_metrics)
        
        combined_suggestions = base_result.suggestions + tigreal_suggestions
        
        # Calculate enhanced score with hero-specific weights
        enhanced_score = await self._calculate_tigreal_score(
            data, base_result.role_specific_metrics)
        
        return RoleEvaluationResult(
            role="tank",
            hero="Tigreal",
            overall_score=enhanced_score,
            feedback=combined_feedback,
            role_specific_metrics=base_result.role_specific_metrics,
            suggestions=combined_suggestions,
            confidence=self._calculate_tigreal_confidence(data)
        )
    
    async def _evaluate_tigreal_specifics(self, data: Dict[str, Any],
                                          minutes: int,
                                          metrics: Dict[str, float]) -> List[Tuple[str, str]]:
        """Evaluate Tigreal-specific mechanics and abilities."""
        feedback = []
        
        # Ultimate usage evaluation
        ult_usage = data.get('ultimate_usage_frequency', 0)
        if ult_usage < 3 and minutes > 10:
            feedback.append(("warning",
                           f"Low ultimate usage ({ult_usage}). Tigreal's ultimate "
                           f"is a game-changer - use it to initiate team fights."))
        elif ult_usage > 5:
            feedback.append(("success",
                           f"Great ultimate usage ({ult_usage})! You're maximizing "
                           f"your crowd control potential."))
        
        # Multi-target crowd control
        avg_targets_per_ult = data.get('avg_targets_per_ultimate', 0)
        if avg_targets_per_ult < 2:
            feedback.append(("warning",
                           f"Low ultimate targets ({avg_targets_per_ult:.1f}). "
                           f"Position to catch multiple enemies in Sacred Hammer."))
        elif avg_targets_per_ult > 3:
            feedback.append(("success",
                           f"Excellent ultimate positioning! Catching "
                           f"{avg_targets_per_ult:.1f} enemies on average."))
        
        # Initiation timing
        deaths = data.get('deaths', 0)
        assists = data.get('assists', 0)
        if deaths > 6 and assists < deaths * 1.5:
            feedback.append(("critical",
                           f"{deaths} deaths with only {assists} assists. "
                           f"Ensure your team follows up before engaging."))
        elif deaths <= 4 and assists > 8:
            feedback.append(("success",
                           f"Perfect initiation balance! {deaths} deaths but "
                           f"{assists} assists shows excellent timing."))
        
        # Damage absorption vs team protection
        damage_taken = data.get('damage_taken', 0)
        ally_damage_prevented = data.get('ally_damage_prevented', 0)
        if ally_damage_prevented > damage_taken * 0.5:
            feedback.append(("success",
                           f"Excellent protection! You're absorbing damage "
                           f"effectively and keeping allies safe."))
        
        # Positioning aggression
        positioning_rating = data.get('positioning_rating', 'medium')
        if positioning_rating == 'defensive':
            feedback.append(("warning",
                           "Tigreal should be more aggressive in positioning. "
                           "Look for flanks and engage opportunities."))
        elif positioning_rating == 'aggressive':
            feedback.append(("success",
                           "Great aggressive positioning! You're creating "
                           "pressure and openings for your team."))
        
        return feedback
    
    async def _generate_tigreal_suggestions(self, data: Dict[str, Any],
                                            metrics: Dict[str, float]) -> List[str]:
        """Generate Tigreal-specific improvement suggestions."""
        suggestions = []
        
        # Ultimate usage suggestions
        ult_usage = data.get('ultimate_usage_frequency', 0)
        if ult_usage < 3:
            suggestions.append("Use Sacred Hammer more frequently in team fights")
            suggestions.append("Practice Sacred Hammer + Attack Wave combos")
        
        # Positioning suggestions
        deaths = data.get('deaths', 0)
        if deaths > 7:
            suggestions.append("Communicate with team before engaging")
            suggestions.append("Use bushes to set up surprise initiations")
        
        # Item build suggestions based on performance
        damage_taken = data.get('damage_taken', 0)
        if damage_taken < 15000:  # Low for a tank
            suggestions.append("Build more defensive items early")
            suggestions.append("Consider Oracle for magic resistance")
        
        # Skill combo suggestions
        skill_combo_success = data.get('skill_combo_success_rate', 0)
        if skill_combo_success < 0.6:
            suggestions.append("Practice Attack Wave -> Sacred Hammer combo")
            suggestions.append("Use Attack Wave to position before ultimate")
        
        return suggestions
    
    async def _calculate_tigreal_score(self, data: Dict[str, Any],
                                       base_metrics: Dict[str, float]) -> float:
        """Calculate Tigreal-specific performance score."""
        # Start with base tank score
        base_score = self._calculate_role_score(base_metrics)
        
        # Tigreal-specific adjustments
        tigreal_multiplier = 1.0
        
        # Bonus for good ultimate usage
        ult_usage = data.get('ultimate_usage_frequency', 0)
        if ult_usage > 4:
            tigreal_multiplier += 0.1
        
        # Bonus for multi-target ultimates
        avg_targets = data.get('avg_targets_per_ultimate', 0)
        if avg_targets > 2.5:
            tigreal_multiplier += 0.15
        
        # Penalty for poor death-to-assist ratio
        deaths = data.get('deaths', 0)
        assists = data.get('assists', 0)
        if deaths > 0:
            death_assist_ratio = assists / deaths
            if death_assist_ratio < 1.0:
                tigreal_multiplier -= 0.1
            elif death_assist_ratio > 2.0:
                tigreal_multiplier += 0.1
        
        # Bonus for damage absorption
        damage_taken = data.get('damage_taken', 0)
        if damage_taken > 20000:  # High absorption
            tigreal_multiplier += 0.1
        
        return min(1.0, base_score * tigreal_multiplier)
    
    def _calculate_tigreal_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence in Tigreal evaluation."""
        # Start with base tank confidence
        base_confidence = super()._calculate_confidence(data, {})
        
        # Tigreal-specific data availability
        tigreal_fields = ['ultimate_usage_frequency', 'avg_targets_per_ultimate',
                         'skill_combo_success_rate', 'ally_damage_prevented']
        
        present_tigreal_fields = sum(1 for field in tigreal_fields
                                   if data.get(field) is not None)
        
        tigreal_confidence = present_tigreal_fields / len(tigreal_fields)
        
        # Weighted combination
        return (base_confidence * 0.7 + tigreal_confidence * 0.3)
    
    def get_hero_tips(self) -> List[str]:
        """Get general Tigreal tips."""
        return [
            "Use Attack Wave to position before Sacred Hammer",
            "Sacred Hammer can go through walls - use for surprise initiations",
            "Build defensive items early to maximize damage absorption",
            "Communicate with team before engaging - timing is everything",
            "Use bushes and fog of war for better positioning",
            "Sacred Hammer + Attack Wave combo is devastating",
            "Don't be afraid to sacrifice yourself for team advantages",
            "Ward enemy jungle to set up better initiations"
        ]
    
    def get_counter_strategies(self) -> Dict[str, List[str]]:
        """Get strategies against common Tigreal counters."""
        return {
            "mobility_heroes": [
                "Wait for them to use escape skills before engaging",
                "Use Attack Wave to close gaps quickly",
                "Coordinate with team for chain CC"
            ],
            "long_range_heroes": [
                "Use bushes to avoid poke damage",
                "Build magic resist early",
                "Engage from unexpected angles"
            ],
            "crowd_control_heroes": [
                "Build Tough Boots for CC reduction",
                "Wait for enemy CC to be used on others",
                "Use Sacred Hammer's unstoppable effect"
            ]
        }
    
    def get_synergy_suggestions(self) -> List[str]:
        """Get hero synergy suggestions for Tigreal."""
        return [
            "Works well with AOE damage dealers (Odette, Vale, Aurora)",
            "Great with follow-up CC heroes (Franco, Kaja)",
            "Excellent with burst mages who can capitalize on initiations",
            "Strong with marksmen who need front-line protection",
            "Pairs well with healers who can keep him alive longer"
        ] 