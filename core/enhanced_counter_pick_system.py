"""
Enhanced Counter-Pick System with comprehensive team composition analysis.
Provides intelligent counter-pick suggestions based on enemy team composition,
meta analysis, and synergy calculations.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import json

from .hero_database import hero_database, HeroData
from .counter_pick_assistant import CounterPickAssistant, DraftState, CounterPickSuggestion

logger = logging.getLogger(__name__)


class PickPriority(Enum):
    """Counter-pick priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TeamComposition:
    """Team composition analysis."""
    heroes: List[str]
    roles: Dict[str, List[str]] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    team_synergy: float = 0.0
    early_game_power: float = 0.0
    late_game_power: float = 0.0
    teamfight_strength: float = 0.0
    split_push_potential: float = 0.0
    
    def __post_init__(self):
        """Analyze team composition after initialization."""
        self._analyze_roles()
        self._analyze_team_strengths()
        self._calculate_power_spikes()
    
    def _analyze_roles(self):
        """Analyze role distribution in team."""
        for hero_name in self.heroes:
            hero_info = hero_database.get_hero_info(hero_name)
            if hero_info:
                role = hero_info.role
                if role not in self.roles:
                    self.roles[role] = []
                self.roles[role].append(hero_name)
    
    def _analyze_team_strengths(self):
        """Analyze team strengths and weaknesses."""
        # Check role balance
        essential_roles = {"tank", "marksman", "mage"}
        missing_roles = essential_roles - set(self.roles.keys())
        
        if not missing_roles:
            self.strengths.append("Balanced team composition")
        else:
            self.weaknesses.append(f"Missing roles: {', '.join(missing_roles)}")
        
        # Check for multiple carries
        carry_count = len(self.roles.get("marksman", [])) + len(self.roles.get("mage", []))
        if carry_count > 2:
            self.strengths.append("High damage potential")
            self.weaknesses.append("May lack frontline")
        
        # Check for crowd control
        cc_heroes = []
        for hero_name in self.heroes:
            hero_info = hero_database.get_hero_info(hero_name)
            if hero_info and any(keyword in hero_info.detection_keywords for keyword in ["stun", "hook", "cc"]):
                cc_heroes.append(hero_name)
        
        if len(cc_heroes) >= 2:
            self.strengths.append("Strong crowd control")
        else:
            self.weaknesses.append("Limited crowd control")
    
    def _calculate_power_spikes(self):
        """Calculate team power spikes."""
        early_game_heroes = []
        late_game_heroes = []
        
        for hero_name in self.heroes:
            hero_info = hero_database.get_hero_info(hero_name)
            if hero_info:
                # Simple heuristic based on role
                if hero_info.role in ["assassin", "fighter"]:
                    early_game_heroes.append(hero_name)
                elif hero_info.role in ["marksman", "mage"]:
                    late_game_heroes.append(hero_name)
        
        self.early_game_power = len(early_game_heroes) / len(self.heroes)
        self.late_game_power = len(late_game_heroes) / len(self.heroes)
        self.teamfight_strength = 0.7 if "tank" in self.roles and len(self.roles.get("mage", [])) > 0 else 0.4


@dataclass
class CounterPickAnalysis:
    """Comprehensive counter-pick analysis."""
    enemy_composition: TeamComposition
    recommended_picks: List[CounterPickSuggestion]
    draft_strategy: str
    timing_recommendations: List[str]
    ban_suggestions: List[str]
    meta_insights: List[str]


class EnhancedCounterPickSystem:
    """Enhanced counter-pick system with comprehensive analysis."""
    
    def __init__(self):
        """Initialize counter-pick system."""
        self.counter_pick_assistant = CounterPickAssistant()
        self.synergy_matrix = self._load_synergy_matrix()
        self.counter_matrix = self._load_counter_matrix()
    
    def analyze_enemy_team_and_suggest_counters(
        self, 
        enemy_heroes: List[str], 
        allied_heroes: List[str] = None,
        match_result: Optional[str] = None
    ) -> CounterPickAnalysis:
        """Analyze enemy team and provide comprehensive counter-pick recommendations."""
        allied_heroes = allied_heroes or []
        
        # Analyze enemy team composition
        enemy_composition = TeamComposition(enemy_heroes)
        
        # Generate counter-pick suggestions
        counter_suggestions = self._generate_counter_suggestions(
            enemy_heroes, allied_heroes, enemy_composition
        )
        
        # Determine draft strategy
        draft_strategy = self._determine_draft_strategy(enemy_composition, allied_heroes)
        
        # Generate timing recommendations
        timing_recommendations = self._generate_timing_recommendations(enemy_composition)
        
        # Generate ban suggestions
        ban_suggestions = self._generate_ban_suggestions(enemy_heroes, allied_heroes)
        
        # Generate meta insights
        meta_insights = self._generate_meta_insights(enemy_composition, match_result)
        
        return CounterPickAnalysis(
            enemy_composition=enemy_composition,
            recommended_picks=counter_suggestions,
            draft_strategy=draft_strategy,
            timing_recommendations=timing_recommendations,
            ban_suggestions=ban_suggestions,
            meta_insights=meta_insights
        )
    
    def _generate_counter_suggestions(
        self, 
        enemy_heroes: List[str], 
        allied_heroes: List[str],
        enemy_composition: TeamComposition
    ) -> List[CounterPickSuggestion]:
        """Generate intelligent counter-pick suggestions."""
        suggestions = []
        
        # Get basic counters for each enemy hero
        counter_candidates = set()
        
        for enemy_hero in enemy_heroes:
            hero_info = hero_database.get_hero_info(enemy_hero)
            if hero_info:
                counter_candidates.update(hero_info.countered_by)
        
        # Filter out already picked heroes
        unavailable_heroes = set(enemy_heroes + allied_heroes)
        available_counters = counter_candidates - unavailable_heroes
        
        # Analyze each counter candidate
        for counter_hero in available_counters:
            counter_info = hero_database.get_hero_info(counter_hero)
            if not counter_info:
                continue
            
            # Calculate effectiveness against enemy team
            effectiveness = self._calculate_counter_effectiveness(
                counter_hero, enemy_heroes, enemy_composition
            )
            
            # Calculate synergy with allied heroes
            synergy = self._calculate_team_synergy(counter_hero, allied_heroes)
            
            # Determine priority
            priority = self._determine_counter_priority(effectiveness, synergy, counter_info)
            
            # Generate reasoning
            reasoning = self._generate_counter_reasoning(
                counter_hero, enemy_heroes, effectiveness, synergy
            )
            
            # Find what this hero counters
            counters = [hero for hero in enemy_heroes if hero in counter_info.counters]
            
            # Find what counters this hero
            vulnerable_to = [hero for hero in enemy_heroes if counter_hero in hero_database.get_hero_info(hero).counters]
            
            suggestion = CounterPickSuggestion(
                hero=counter_hero,
                priority=priority.value,
                reasoning=reasoning,
                counters=counters,
                vulnerable_to=vulnerable_to,
                synergy_score=synergy,
                meta_strength=counter_info.meta_tier,
                confidence=effectiveness
            )
            
            suggestions.append(suggestion)
        
        # Sort by priority and effectiveness
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        suggestions.sort(
            key=lambda x: (priority_order.get(x.priority, 0), x.confidence, x.synergy_score),
            reverse=True
        )
        
        return suggestions[:8]  # Top 8 suggestions
    
    def _calculate_counter_effectiveness(
        self, 
        counter_hero: str, 
        enemy_heroes: List[str],
        enemy_composition: TeamComposition
    ) -> float:
        """Calculate how effective a counter hero is against the enemy team."""
        effectiveness = 0.0
        
        counter_info = hero_database.get_hero_info(counter_hero)
        if not counter_info:
            return 0.0
        
        # Base effectiveness from direct counters
        direct_counters = sum(1 for enemy in enemy_heroes if enemy in counter_info.counters)
        effectiveness += direct_counters * 0.3
        
        # Role-based effectiveness
        counter_role = counter_info.role
        role_effectiveness = self._calculate_role_effectiveness(counter_role, enemy_composition)
        effectiveness += role_effectiveness * 0.4
        
        # Meta strength bonus
        meta_bonus = {"S": 0.2, "A": 0.15, "B": 0.1, "C": 0.05, "D": 0.0}
        effectiveness += meta_bonus.get(counter_info.meta_tier, 0.0)
        
        # Power spike compatibility
        if enemy_composition.early_game_power > 0.6 and counter_role in ["assassin", "fighter"]:
            effectiveness += 0.1
        elif enemy_composition.late_game_power > 0.6 and counter_role in ["marksman", "mage"]:
            effectiveness += 0.1
        
        return min(effectiveness, 1.0)
    
    def _calculate_role_effectiveness(self, counter_role: str, enemy_composition: TeamComposition) -> float:
        """Calculate role-based effectiveness against enemy composition."""
        effectiveness = 0.0
        
        enemy_roles = enemy_composition.roles
        
        # Tank effectiveness
        if counter_role == "tank":
            if "assassin" in enemy_roles or "fighter" in enemy_roles:
                effectiveness += 0.3
            if len(enemy_roles.get("marksman", [])) > 0:
                effectiveness += 0.2
        
        # Assassin effectiveness
        elif counter_role == "assassin":
            if "mage" in enemy_roles:
                effectiveness += 0.4
            if "marksman" in enemy_roles:
                effectiveness += 0.3
            if "support" in enemy_roles:
                effectiveness += 0.2
        
        # Marksman effectiveness
        elif counter_role == "marksman":
            if "tank" in enemy_roles:
                effectiveness += 0.2
            if len(enemy_roles.get("fighter", [])) > 0:
                effectiveness += 0.3
        
        # Mage effectiveness
        elif counter_role == "mage":
            if "fighter" in enemy_roles:
                effectiveness += 0.3
            if "tank" in enemy_roles:
                effectiveness += 0.2
        
        # Fighter effectiveness
        elif counter_role == "fighter":
            if "marksman" in enemy_roles:
                effectiveness += 0.3
            if "mage" in enemy_roles:
                effectiveness += 0.2
        
        return min(effectiveness, 1.0)
    
    def _calculate_team_synergy(self, hero: str, allied_heroes: List[str]) -> float:
        """Calculate synergy between hero and allied heroes."""
        if not allied_heroes:
            return 0.5
        
        synergy_score = 0.0
        hero_info = hero_database.get_hero_info(hero)
        
        if not hero_info:
            return 0.5
        
        for ally in allied_heroes:
            # Check if heroes have explicit synergy
            if ally in hero_info.synergies:
                synergy_score += 0.3
            
            # Check role synergies
            ally_info = hero_database.get_hero_info(ally)
            if ally_info:
                role_synergy = self._calculate_role_synergy(hero_info.role, ally_info.role)
                synergy_score += role_synergy * 0.2
        
        return min(synergy_score / len(allied_heroes), 1.0)
    
    def _calculate_role_synergy(self, role1: str, role2: str) -> float:
        """Calculate synergy between two roles."""
        synergy_map = {
            ("tank", "marksman"): 0.8,
            ("tank", "mage"): 0.7,
            ("support", "marksman"): 0.9,
            ("support", "mage"): 0.6,
            ("assassin", "tank"): 0.5,
            ("fighter", "marksman"): 0.4,
            ("fighter", "mage"): 0.3
        }
        
        return synergy_map.get((role1, role2), 0.3)
    
    def _determine_counter_priority(
        self, 
        effectiveness: float, 
        synergy: float, 
        hero_info: HeroData
    ) -> PickPriority:
        """Determine counter-pick priority."""
        # Calculate combined score
        score = effectiveness * 0.6 + synergy * 0.3
        
        # Meta tier bonus
        meta_bonus = {"S": 0.1, "A": 0.08, "B": 0.05, "C": 0.02, "D": 0.0}
        score += meta_bonus.get(hero_info.meta_tier, 0.0)
        
        if score >= 0.8:
            return PickPriority.CRITICAL
        elif score >= 0.6:
            return PickPriority.HIGH
        elif score >= 0.4:
            return PickPriority.MEDIUM
        else:
            return PickPriority.LOW
    
    def _generate_counter_reasoning(
        self, 
        counter_hero: str, 
        enemy_heroes: List[str], 
        effectiveness: float, 
        synergy: float
    ) -> str:
        """Generate reasoning for counter-pick suggestion."""
        counter_info = hero_database.get_hero_info(counter_hero)
        if not counter_info:
            return f"{counter_hero} provides general team balance"
        
        reasoning = f"{counter_hero} ({counter_info.role}) "
        
        # Direct counters
        direct_counters = [hero for hero in enemy_heroes if hero in counter_info.counters]
        if direct_counters:
            reasoning += f"directly counters {', '.join(direct_counters)}. "
        
        # Role advantage
        if effectiveness > 0.6:
            reasoning += f"Has strong role advantage against enemy composition. "
        
        # Meta strength
        if counter_info.meta_tier in ["S", "A"]:
            reasoning += f"Currently {counter_info.meta_tier}-tier meta pick. "
        
        # Synergy
        if synergy > 0.6:
            reasoning += "Good synergy with your team composition."
        
        return reasoning.strip()
    
    def _determine_draft_strategy(
        self, 
        enemy_composition: TeamComposition, 
        allied_heroes: List[str]
    ) -> str:
        """Determine overall draft strategy."""
        # Analyze enemy strengths and weaknesses
        if enemy_composition.early_game_power > 0.6:
            return "defensive_early_game"
        elif enemy_composition.late_game_power > 0.6:
            return "aggressive_early_game"
        elif enemy_composition.teamfight_strength > 0.7:
            return "split_push_focus"
        elif "Strong crowd control" in enemy_composition.strengths:
            return "mobility_focus"
        else:
            return "balanced_approach"
    
    def _generate_timing_recommendations(self, enemy_composition: TeamComposition) -> List[str]:
        """Generate timing-based recommendations."""
        recommendations = []
        
        if enemy_composition.early_game_power > 0.6:
            recommendations.append("Focus on farming safely in early game")
            recommendations.append("Avoid extended trades until item power spikes")
        
        if enemy_composition.late_game_power > 0.6:
            recommendations.append("Apply pressure in early-mid game")
            recommendations.append("Force teamfights before enemy reaches full build")
        
        if enemy_composition.teamfight_strength > 0.7:
            recommendations.append("Avoid grouped teamfights")
            recommendations.append("Focus on picks and split pushing")
        
        return recommendations
    
    def _generate_ban_suggestions(self, enemy_heroes: List[str], allied_heroes: List[str]) -> List[str]:
        """Generate ban suggestions based on team compositions."""
        ban_suggestions = []
        
        # Get top meta heroes that would threaten our composition
        meta_threats = hero_database.get_ban_worthy_heroes(10)
        
        # Filter out already picked heroes
        unavailable = set(enemy_heroes + allied_heroes)
        available_bans = [hero for hero in meta_threats if hero not in unavailable]
        
        # Add specific threats based on our composition
        for allied_hero in allied_heroes:
            hero_info = hero_database.get_hero_info(allied_hero)
            if hero_info:
                for threat in hero_info.countered_by:
                    if threat not in unavailable and threat not in ban_suggestions:
                        ban_suggestions.append(threat)
        
        # Combine and prioritize
        combined_bans = available_bans[:3] + ban_suggestions[:2]
        return list(dict.fromkeys(combined_bans))[:5]  # Remove duplicates, keep order
    
    def _generate_meta_insights(
        self, 
        enemy_composition: TeamComposition, 
        match_result: str = None
    ) -> List[str]:
        """Generate meta-based insights."""
        insights = []
        
        # Composition type insights
        if enemy_composition.early_game_power > 0.6:
            insights.append("Enemy has strong early game - expect aggressive laning")
        
        if enemy_composition.late_game_power > 0.6:
            insights.append("Enemy scales well into late game - end quickly if possible")
        
        if enemy_composition.teamfight_strength > 0.7:
            insights.append("Enemy has strong teamfight potential - avoid grouped fights")
        
        # Role distribution insights
        if "tank" not in enemy_composition.roles:
            insights.append("Enemy lacks frontline - aggressive positioning may work")
        
        if len(enemy_composition.roles.get("marksman", [])) > 1:
            insights.append("Multiple enemy carries - focus on dive/assassination")
        
        # Match result insights
        if match_result == "defeat":
            insights.append("Consider these counters for next match against similar composition")
            if enemy_composition.early_game_power > 0.6:
                insights.append("Enemy early game advantage likely contributed to loss")
        
        return insights
    
    def _load_synergy_matrix(self) -> Dict[str, Dict[str, float]]:
        """Load hero synergy matrix."""
        # Simplified synergy matrix - in production, this would be more comprehensive
        return {
            "Franco": {"Odette": 0.9, "Aurora": 0.8, "Eudora": 0.8},
            "Tigreal": {"Odette": 0.9, "Aurora": 0.8, "Cyclops": 0.7},
            "Angela": {"Fanny": 0.9, "Hayabusa": 0.8, "Gusion": 0.8},
            "Estes": {"Fanny": 0.8, "Hayabusa": 0.7, "Chou": 0.7}
        }
    
    def _load_counter_matrix(self) -> Dict[str, Dict[str, float]]:
        """Load hero counter matrix."""
        # Simplified counter matrix - in production, this would be more comprehensive
        return {
            "Franco": {"Fanny": 0.8, "Hayabusa": 0.7, "Gusion": 0.7},
            "Diggie": {"Franco": 0.9, "Tigreal": 0.8, "Atlas": 0.8},
            "Khufra": {"Fanny": 0.9, "Gusion": 0.8, "Harith": 0.8},
            "Chou": {"Fanny": 0.8, "Hayabusa": 0.7, "Gusion": 0.7}
        }
    
    def export_counter_analysis(self, analysis: CounterPickAnalysis) -> Dict[str, Any]:
        """Export counter-pick analysis to dictionary format."""
        return {
            "enemy_composition": {
                "heroes": analysis.enemy_composition.heroes,
                "roles": analysis.enemy_composition.roles,
                "strengths": analysis.enemy_composition.strengths,
                "weaknesses": analysis.enemy_composition.weaknesses,
                "power_spikes": {
                    "early_game": analysis.enemy_composition.early_game_power,
                    "late_game": analysis.enemy_composition.late_game_power,
                    "teamfight": analysis.enemy_composition.teamfight_strength
                }
            },
            "recommended_picks": [
                {
                    "hero": pick.hero,
                    "priority": pick.priority,
                    "reasoning": pick.reasoning,
                    "counters": pick.counters,
                    "vulnerable_to": pick.vulnerable_to,
                    "synergy_score": pick.synergy_score,
                    "meta_strength": pick.meta_strength,
                    "confidence": pick.confidence
                }
                for pick in analysis.recommended_picks
            ],
            "draft_strategy": analysis.draft_strategy,
            "timing_recommendations": analysis.timing_recommendations,
            "ban_suggestions": analysis.ban_suggestions,
            "meta_insights": analysis.meta_insights
        }


# Global enhanced counter-pick system instance
enhanced_counter_pick_system = EnhancedCounterPickSystem()