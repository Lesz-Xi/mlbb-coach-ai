"""
Synergy Matrix Builder for MLBB Coach AI
========================================

This utility module provides tools for building player compatibility matrices,
analyzing team synergy patterns, and generating synergy-based recommendations.

Key Features:
- Player compatibility matrix generation
- Synergy pattern analysis
- Team composition optimization
- Role-based synergy calculations
- Playstyle compatibility assessment
- Temporal synergy analysis
"""

import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import existing system components
from ..behavioral_modeling import BehavioralFingerprint

logger = logging.getLogger(__name__)


class SynergyStrength(Enum):
    """Synergy strength levels."""
    EXCELLENT = "excellent"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    INCOMPATIBLE = "incompatible"


@dataclass
class SynergyFactor:
    """Represents a synergy factor between players."""
    factor_type: str
    description: str
    weight: float
    score: float
    reasoning: str


@dataclass
class PlayerSynergy:
    """Represents synergy between two players."""
    player_a: str
    player_b: str
    synergy_score: float
    synergy_strength: SynergyStrength
    synergy_factors: List[SynergyFactor] = field(default_factory=list)
    anti_synergy_factors: List[SynergyFactor] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "player_a": self.player_a,
            "player_b": self.player_b,
            "synergy_score": self.synergy_score,
            "synergy_strength": self.synergy_strength.value,
            "synergy_factors": [
                {
                    "factor_type": factor.factor_type,
                    "description": factor.description,
                    "weight": factor.weight,
                    "score": factor.score,
                    "reasoning": factor.reasoning
                }
                for factor in self.synergy_factors
            ],
            "anti_synergy_factors": [
                {
                    "factor_type": factor.factor_type,
                    "description": factor.description,
                    "weight": factor.weight,
                    "score": factor.score,
                    "reasoning": factor.reasoning
                }
                for factor in self.anti_synergy_factors
            ],
            "recommendations": self.recommendations
        }


@dataclass
class TeamSynergyMatrix:
    """Complete team synergy matrix."""
    team_id: str
    player_synergies: List[PlayerSynergy]
    overall_team_synergy: float
    strongest_pairs: List[Tuple[str, str, float]]
    weakest_pairs: List[Tuple[str, str, float]]
    synergy_distribution: Dict[str, int]
    team_recommendations: List[str]
    
         def to_dict(self) -> Dict[str, Any]:
         """Convert to dictionary format."""
                  return {
             "team_id": self.team_id,
             "player_synergies": [
                 synergy.to_dict() for synergy in self.player_synergies
             ],
             "overall_team_synergy": self.overall_team_synergy,
             "strongest_pairs": [
                 {"player_a": pair[0], "player_b": pair[1], "score": pair[2]}
                 for pair in self.strongest_pairs
             ],
             "weakest_pairs": [
                 {"player_a": pair[0], "player_b": pair[1], "score": pair[2]}
                 for pair in self.weakest_pairs
             ],
             "synergy_distribution": self.synergy_distribution,
             "team_recommendations": self.team_recommendations
         }


class SynergyMatrixBuilder:
    """
    Builds player compatibility matrices and analyzes team synergy patterns.
    
    This builder provides comprehensive synergy analysis including:
    - Playstyle compatibility assessment
    - Role-based synergy calculations
    - Temporal alignment analysis
    - Risk profile balance evaluation
    - Performance consistency analysis
    """
    
    def __init__(self):
        """Initialize the synergy matrix builder."""
        self.playstyle_synergy_matrix = self._build_playstyle_synergy_matrix()
        self.role_synergy_matrix = self._build_role_synergy_matrix()
        self.tempo_synergy_matrix = self._build_tempo_synergy_matrix()
        self.risk_balance_matrix = self._build_risk_balance_matrix()
        
        # Weights for different synergy factors
        self.synergy_weights = {
            "playstyle": 0.25,
            "role": 0.30,
            "tempo": 0.20,
            "risk_balance": 0.15,
            "performance_consistency": 0.10
        }
        
        logger.info("Synergy Matrix Builder initialized")
    
    def _build_playstyle_synergy_matrix(self) -> Dict[Tuple[str, str], float]:
        """Build playstyle synergy matrix."""
        return {
            ("aggressive-roamer", "support-oriented"): 0.9,
            ("aggressive-roamer", "team-fighter"): 0.8,
            ("aggressive-roamer", "passive-farmer"): 0.2,
            ("objective-focused", "team-fighter"): 0.95,
            ("objective-focused", "support-oriented"): 0.85,
            ("objective-focused", "split-pusher"): 0.4,
            ("carry-focused", "support-oriented"): 0.9,
            ("carry-focused", "team-fighter"): 0.7,
            ("carry-focused", "aggressive-roamer"): 0.5,
            ("assassin-style", "team-fighter"): 0.8,
            ("assassin-style", "support-oriented"): 0.7,
            ("assassin-style", "passive-farmer"): 0.3,
            ("split-pusher", "team-fighter"): 0.3,
            ("split-pusher", "objective-focused"): 0.6,
            ("split-pusher", "support-oriented"): 0.4,
            ("passive-farmer", "support-oriented"): 0.6,
            ("passive-farmer", "team-fighter"): 0.5,
            ("passive-farmer", "objective-focused"): 0.7,
            ("team-fighter", "support-oriented"): 0.85,
            ("team-fighter", "carry-focused"): 0.7,
        }
    
    def _build_role_synergy_matrix(self) -> Dict[Tuple[str, str], float]:
        """Build role synergy matrix."""
        return {
            ("tank", "marksman"): 0.95,
            ("tank", "mage"): 0.85,
            ("tank", "assassin"): 0.8,
            ("tank", "support"): 0.7,
            ("tank", "fighter"): 0.6,
            ("tank", "tank"): 0.2,  # Role overlap
            ("support", "marksman"): 0.9,
            ("support", "mage"): 0.85,
            ("support", "carry"): 0.8,
            ("support", "assassin"): 0.7,
            ("support", "support"): 0.3,  # Role overlap
            ("fighter", "assassin"): 0.75,
            ("fighter", "mage"): 0.6,
            ("fighter", "marksman"): 0.5,
            ("fighter", "fighter"): 0.4,  # Some overlap acceptable
            ("assassin", "mage"): 0.6,
            ("assassin", "marksman"): 0.5,
            ("assassin", "assassin"): 0.3,  # Role overlap
            ("mage", "marksman"): 0.7,
            ("mage", "mage"): 0.4,  # Some overlap acceptable
            ("marksman", "marksman"): 0.2,  # Role overlap
        }
    
    def _build_tempo_synergy_matrix(self) -> Dict[Tuple[str, str], float]:
        """Build tempo synergy matrix."""
        return {
            ("early-aggressive", "early-aggressive"): 0.9,
            ("early-aggressive", "mid-game-focused"): 0.7,
            ("early-aggressive", "late-game-scaling"): 0.4,
            ("early-aggressive", "adaptive"): 0.8,
            ("mid-game-focused", "mid-game-focused"): 0.85,
            ("mid-game-focused", "late-game-scaling"): 0.6,
            ("mid-game-focused", "adaptive"): 0.8,
            ("late-game-scaling", "late-game-scaling"): 0.8,
            ("late-game-scaling", "adaptive"): 0.7,
            ("adaptive", "adaptive"): 0.9,
        }
    
    def _build_risk_balance_matrix(self) -> Dict[Tuple[str, str], float]:
        """Build risk balance matrix."""
        return {
            ("high-risk-high-reward", "conservative"): 0.8,
            ("high-risk-high-reward", "calculated-risk"): 0.7,
            ("high-risk-high-reward", "opportunistic"): 0.6,
            ("high-risk-high-reward", "high-risk-high-reward"): 0.3,
            ("calculated-risk", "calculated-risk"): 0.85,
            ("calculated-risk", "conservative"): 0.9,
            ("calculated-risk", "opportunistic"): 0.8,
            ("conservative", "conservative"): 0.6,
            ("conservative", "opportunistic"): 0.7,
            ("opportunistic", "opportunistic"): 0.75,
        }
    
    def build_synergy_matrix(
        self, 
        behavioral_profiles: List[BehavioralFingerprint],
        team_id: str = "team_1"
    ) -> TeamSynergyMatrix:
        """
        Build complete team synergy matrix.
        
        Args:
            behavioral_profiles: List of behavioral profiles for team members
            team_id: Team identifier
            
        Returns:
            TeamSynergyMatrix with comprehensive synergy analysis
        """
        player_synergies = []
        
        # Calculate pairwise synergies
        for i in range(len(behavioral_profiles)):
            for j in range(i + 1, len(behavioral_profiles)):
                synergy = self._calculate_player_synergy(
                    behavioral_profiles[i], 
                    behavioral_profiles[j]
                )
                player_synergies.append(synergy)
        
        # Calculate overall team synergy
        overall_synergy = self._calculate_overall_team_synergy(player_synergies)
        
        # Find strongest and weakest pairs
        strongest_pairs = self._find_strongest_pairs(player_synergies)
        weakest_pairs = self._find_weakest_pairs(player_synergies)
        
        # Calculate synergy distribution
        synergy_distribution = self._calculate_synergy_distribution(player_synergies)
        
        # Generate team recommendations
        team_recommendations = self._generate_team_recommendations(
            player_synergies, overall_synergy, behavioral_profiles
        )
        
        return TeamSynergyMatrix(
            team_id=team_id,
            player_synergies=player_synergies,
            overall_team_synergy=overall_synergy,
            strongest_pairs=strongest_pairs,
            weakest_pairs=weakest_pairs,
            synergy_distribution=synergy_distribution,
            team_recommendations=team_recommendations
        )
    
    def _calculate_player_synergy(
        self, 
        profile_a: BehavioralFingerprint, 
        profile_b: BehavioralFingerprint
    ) -> PlayerSynergy:
        """Calculate synergy between two players."""
        synergy_factors = []
        anti_synergy_factors = []
        
        # Playstyle synergy
        playstyle_score = self._calculate_playstyle_synergy(profile_a, profile_b)
        synergy_factors.append(SynergyFactor(
            factor_type="playstyle",
            description="Playstyle compatibility",
            weight=self.synergy_weights["playstyle"],
            score=playstyle_score,
            reasoning=self._get_playstyle_reasoning(profile_a, profile_b, playstyle_score)
        ))
        
        # Role synergy
        role_score = self._calculate_role_synergy(profile_a, profile_b)
        synergy_factors.append(SynergyFactor(
            factor_type="role",
            description="Role compatibility",
            weight=self.synergy_weights["role"],
            score=role_score,
            reasoning=self._get_role_reasoning(profile_a, profile_b, role_score)
        ))
        
        # Tempo synergy
        tempo_score = self._calculate_tempo_synergy(profile_a, profile_b)
        synergy_factors.append(SynergyFactor(
            factor_type="tempo",
            description="Game tempo alignment",
            weight=self.synergy_weights["tempo"],
            score=tempo_score,
            reasoning=self._get_tempo_reasoning(profile_a, profile_b, tempo_score)
        ))
        
        # Risk balance
        risk_score = self._calculate_risk_balance(profile_a, profile_b)
        synergy_factors.append(SynergyFactor(
            factor_type="risk_balance",
            description="Risk profile balance",
            weight=self.synergy_weights["risk_balance"],
            score=risk_score,
            reasoning=self._get_risk_reasoning(profile_a, profile_b, risk_score)
        ))
        
        # Performance consistency
        consistency_score = self._calculate_performance_consistency(profile_a, profile_b)
        synergy_factors.append(SynergyFactor(
            factor_type="performance_consistency",
            description="Performance consistency",
            weight=self.synergy_weights["performance_consistency"],
            score=consistency_score,
            reasoning=self._get_consistency_reasoning(profile_a, profile_b, consistency_score)
        ))
        
        # Calculate weighted synergy score
        weighted_score = sum(
            factor.score * factor.weight for factor in synergy_factors
        )
        
        # Identify anti-synergy factors
        anti_synergy_factors = self._identify_anti_synergy_factors(profile_a, profile_b)
        
        # Generate recommendations
        recommendations = self._generate_player_recommendations(
            profile_a, profile_b, synergy_factors, anti_synergy_factors
        )
        
        # Classify synergy strength
        synergy_strength = self._classify_synergy_strength(weighted_score)
        
        return PlayerSynergy(
            player_a=profile_a.player_id,
            player_b=profile_b.player_id,
            synergy_score=weighted_score,
            synergy_strength=synergy_strength,
            synergy_factors=synergy_factors,
            anti_synergy_factors=anti_synergy_factors,
            recommendations=recommendations
        )
    
    def _calculate_playstyle_synergy(
        self, profile_a: BehavioralFingerprint, profile_b: BehavioralFingerprint
    ) -> float:
        """Calculate playstyle synergy score."""
        style_a = profile_a.play_style.value
        style_b = profile_b.play_style.value
        
        # Check direct match
        key = (style_a, style_b)
        reverse_key = (style_b, style_a)
        
        direct_score = self.playstyle_synergy_matrix.get(key, 
                        self.playstyle_synergy_matrix.get(reverse_key, 0.5))
        
        # Adjust based on behavioral scores
        synergy_adjustment = (
            profile_a.synergy_with_team + profile_b.synergy_with_team
        ) / 2
        
        return min(direct_score * (0.8 + synergy_adjustment * 0.2), 1.0)
    
    def _calculate_role_synergy(
        self, profile_a: BehavioralFingerprint, profile_b: BehavioralFingerprint
    ) -> float:
        """Calculate role synergy score."""
        role_a = profile_a.preferred_role
        role_b = profile_b.preferred_role
        
        # Check direct match
        key = (role_a, role_b)
        reverse_key = (role_b, role_a)
        
        return self.role_synergy_matrix.get(key, 
                self.role_synergy_matrix.get(reverse_key, 0.5))
    
    def _calculate_tempo_synergy(
        self, profile_a: BehavioralFingerprint, profile_b: BehavioralFingerprint
    ) -> float:
        """Calculate tempo synergy score."""
        tempo_a = profile_a.game_tempo.value
        tempo_b = profile_b.game_tempo.value
        
        # Check direct match
        key = (tempo_a, tempo_b)
        reverse_key = (tempo_b, tempo_a)
        
        return self.tempo_synergy_matrix.get(key, 
                self.tempo_synergy_matrix.get(reverse_key, 0.5))
    
    def _calculate_risk_balance(
        self, profile_a: BehavioralFingerprint, profile_b: BehavioralFingerprint
    ) -> float:
        """Calculate risk balance score."""
        risk_a = profile_a.risk_profile.value
        risk_b = profile_b.risk_profile.value
        
        # Check direct match
        key = (risk_a, risk_b)
        reverse_key = (risk_b, risk_a)
        
        return self.risk_balance_matrix.get(key, 
                self.risk_balance_matrix.get(reverse_key, 0.5))
    
    def _calculate_performance_consistency(
        self, profile_a: BehavioralFingerprint, profile_b: BehavioralFingerprint
    ) -> float:
        """Calculate performance consistency score."""
        # Use confidence scores as proxy for consistency
        avg_confidence = (profile_a.confidence_score + profile_b.confidence_score) / 2
        
        # Use mechanical skill similarity as consistency factor
        skill_diff = abs(profile_a.mechanical_skill_score - profile_b.mechanical_skill_score)
        consistency_factor = 1 - skill_diff
        
        return (avg_confidence + consistency_factor) / 2
    
    def _identify_anti_synergy_factors(
        self, profile_a: BehavioralFingerprint, profile_b: BehavioralFingerprint
    ) -> List[SynergyFactor]:
        """Identify anti-synergy factors."""
        anti_factors = []
        
        # Role conflicts
        if profile_a.preferred_role == profile_b.preferred_role:
            if profile_a.preferred_role in ["tank", "marksman", "support"]:
                anti_factors.append(SynergyFactor(
                    factor_type="role_conflict",
                    description="Role overlap conflict",
                    weight=0.3,
                    score=0.2,
                    reasoning=f"Both players prefer {profile_a.preferred_role} role"
                ))
        
        # Playstyle conflicts
        if (profile_a.play_style.value == "split-pusher" and 
            profile_b.play_style.value == "team-fighter"):
            anti_factors.append(SynergyFactor(
                factor_type="playstyle_conflict",
                description="Contradictory playstyles",
                weight=0.2,
                score=0.3,
                reasoning="Split-pusher vs team-fighter conflict"
            ))
        
        # Risk profile imbalance
        if (profile_a.risk_profile.value == "high-risk-high-reward" and
            profile_b.risk_profile.value == "high-risk-high-reward"):
            anti_factors.append(SynergyFactor(
                factor_type="risk_imbalance",
                description="Excessive risk-taking",
                weight=0.15,
                score=0.3,
                reasoning="Both players are high-risk takers"
            ))
        
        return anti_factors
    
    def _generate_player_recommendations(
        self, profile_a: BehavioralFingerprint, profile_b: BehavioralFingerprint,
        synergy_factors: List[SynergyFactor], anti_synergy_factors: List[SynergyFactor]
    ) -> List[str]:
        """Generate recommendations for player pair."""
        recommendations = []
        
        # Strong synergy recommendations
        if any(factor.score > 0.8 for factor in synergy_factors):
            recommendations.append("Leverage strong synergy with coordinated plays")
        
        # Address anti-synergy factors
        for anti_factor in anti_synergy_factors:
            if anti_factor.factor_type == "role_conflict":
                recommendations.append("Consider role flexibility or position swaps")
            elif anti_factor.factor_type == "playstyle_conflict":
                recommendations.append("Improve communication to align strategies")
            elif anti_factor.factor_type == "risk_imbalance":
                recommendations.append("Balance risk-taking with calculated plays")
        
        # Tempo-specific recommendations
        tempo_factor = next((f for f in synergy_factors if f.factor_type == "tempo"), None)
        if tempo_factor and tempo_factor.score < 0.5:
            recommendations.append("Synchronize game tempo and objective timing")
        
        # Performance consistency recommendations
        consistency_factor = next((f for f in synergy_factors if f.factor_type == "performance_consistency"), None)
        if consistency_factor and consistency_factor.score < 0.6:
            recommendations.append("Work on consistent performance patterns")
        
        return recommendations
    
    def _calculate_overall_team_synergy(self, player_synergies: List[PlayerSynergy]) -> float:
        """Calculate overall team synergy score."""
        if not player_synergies:
            return 0.0
        
        synergy_scores = [synergy.synergy_score for synergy in player_synergies]
        return sum(synergy_scores) / len(synergy_scores)
    
    def _find_strongest_pairs(self, player_synergies: List[PlayerSynergy]) -> List[Tuple[str, str, float]]:
        """Find strongest synergy pairs."""
        sorted_synergies = sorted(player_synergies, key=lambda x: x.synergy_score, reverse=True)
        return [(s.player_a, s.player_b, s.synergy_score) for s in sorted_synergies[:3]]
    
    def _find_weakest_pairs(self, player_synergies: List[PlayerSynergy]) -> List[Tuple[str, str, float]]:
        """Find weakest synergy pairs."""
        sorted_synergies = sorted(player_synergies, key=lambda x: x.synergy_score)
        return [(s.player_a, s.player_b, s.synergy_score) for s in sorted_synergies[:3]]
    
    def _calculate_synergy_distribution(self, player_synergies: List[PlayerSynergy]) -> Dict[str, int]:
        """Calculate synergy strength distribution."""
        distribution = {
            "excellent": 0,
            "strong": 0,
            "moderate": 0,
            "weak": 0,
            "incompatible": 0
        }
        
        for synergy in player_synergies:
            distribution[synergy.synergy_strength.value] += 1
        
        return distribution
    
    def _generate_team_recommendations(
        self, player_synergies: List[PlayerSynergy], overall_synergy: float,
        behavioral_profiles: List[BehavioralFingerprint]
    ) -> List[str]:
        """Generate team-level recommendations."""
        recommendations = []
        
        # Overall synergy recommendations
        if overall_synergy >= 0.8:
            recommendations.append("Excellent team synergy - maintain current dynamics")
        elif overall_synergy >= 0.6:
            recommendations.append("Good synergy base - focus on weak pairs")
        elif overall_synergy >= 0.4:
            recommendations.append("Moderate synergy - significant improvement needed")
        else:
            recommendations.append("Poor synergy - consider role redistributions")
        
        # Role distribution recommendations
        roles = [p.preferred_role for p in behavioral_profiles]
        role_counts = {role: roles.count(role) for role in set(roles)}
        
        for role, count in role_counts.items():
            if count > 1 and role in ["tank", "marksman", "support"]:
                recommendations.append(f"Address {role} role overlap")
        
        # Synergy distribution recommendations
        weak_pairs = [s for s in player_synergies if s.synergy_strength in [SynergyStrength.WEAK, SynergyStrength.INCOMPATIBLE]]
        if len(weak_pairs) > 2:
            recommendations.append("Multiple weak synergies - focus on communication")
        
        return recommendations
    
    def _classify_synergy_strength(self, score: float) -> SynergyStrength:
        """Classify synergy strength based on score."""
        if score >= 0.8:
            return SynergyStrength.EXCELLENT
        elif score >= 0.65:
            return SynergyStrength.STRONG
        elif score >= 0.45:
            return SynergyStrength.MODERATE
        elif score >= 0.25:
            return SynergyStrength.WEAK
        else:
            return SynergyStrength.INCOMPATIBLE
    
    # Reasoning methods
    def _get_playstyle_reasoning(self, profile_a: BehavioralFingerprint, profile_b: BehavioralFingerprint, score: float) -> str:
        """Get playstyle synergy reasoning."""
        style_a = profile_a.play_style.value
        style_b = profile_b.play_style.value
        
        if score >= 0.8:
            return f"{style_a} and {style_b} complement each other excellently"
        elif score >= 0.6:
            return f"{style_a} and {style_b} work well together"
        elif score >= 0.4:
            return f"{style_a} and {style_b} have moderate compatibility"
        else:
            return f"{style_a} and {style_b} have conflicting approaches"
    
    def _get_role_reasoning(self, profile_a: BehavioralFingerprint, profile_b: BehavioralFingerprint, score: float) -> str:
        """Get role synergy reasoning."""
        role_a = profile_a.preferred_role
        role_b = profile_b.preferred_role
        
        if score >= 0.8:
            return f"{role_a} and {role_b} roles have excellent synergy"
        elif score >= 0.6:
            return f"{role_a} and {role_b} roles complement each other"
        elif score >= 0.4:
            return f"{role_a} and {role_b} roles have moderate compatibility"
        else:
            return f"{role_a} and {role_b} roles conflict or overlap"
    
    def _get_tempo_reasoning(self, profile_a: BehavioralFingerprint, profile_b: BehavioralFingerprint, score: float) -> str:
        """Get tempo synergy reasoning."""
        tempo_a = profile_a.game_tempo.value
        tempo_b = profile_b.game_tempo.value
        
        if score >= 0.8:
            return f"{tempo_a} and {tempo_b} tempo preferences align perfectly"
        elif score >= 0.6:
            return f"{tempo_a} and {tempo_b} tempo preferences work well together"
        elif score >= 0.4:
            return f"{tempo_a} and {tempo_b} tempo preferences have some alignment"
        else:
            return f"{tempo_a} and {tempo_b} tempo preferences conflict"
    
    def _get_risk_reasoning(self, profile_a: BehavioralFingerprint, profile_b: BehavioralFingerprint, score: float) -> str:
        """Get risk balance reasoning."""
        risk_a = profile_a.risk_profile.value
        risk_b = profile_b.risk_profile.value
        
        if score >= 0.8:
            return f"{risk_a} and {risk_b} risk profiles create excellent balance"
        elif score >= 0.6:
            return f"{risk_a} and {risk_b} risk profiles complement each other"
        elif score >= 0.4:
            return f"{risk_a} and {risk_b} risk profiles have moderate balance"
        else:
            return f"{risk_a} and {risk_b} risk profiles create imbalance"
    
    def _get_consistency_reasoning(self, profile_a: BehavioralFingerprint, profile_b: BehavioralFingerprint, score: float) -> str:
        """Get consistency reasoning."""
        if score >= 0.8:
            return "Both players show excellent performance consistency"
        elif score >= 0.6:
            return "Players show good performance consistency"
        elif score >= 0.4:
            return "Players show moderate performance consistency"
        else:
            return "Players show inconsistent performance patterns"


def create_synergy_matrix_builder() -> SynergyMatrixBuilder:
    """Create and initialize a synergy matrix builder."""
    return SynergyMatrixBuilder() 