from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .meta_analyzer import MetaAnalyzer
from .schemas import HeroRecommendation, HeroMetaData


@dataclass
class MatchResult:
    """Represents a completed match result."""
    player_hero: str
    ally_team: List[str]
    enemy_team: List[str]
    match_outcome: str  # "victory", "defeat"
    match_duration: int  # in minutes
    player_performance: Dict[str, float]  # KDA, damage, etc.


@dataclass
class ThreatAssessment:
    """Assessment of enemy hero threat level."""
    hero: str
    threat_level: float  # 0-1 scale
    threat_reasons: List[str]
    suggested_counters: List[str]
    meta_data: HeroMetaData


@dataclass
class PostMatchInsight:
    """Retrospective insight for future improvement."""
    insight_type: str  # "counter_pick", "team_comp", "meta_awareness"
    priority: str  # "high", "medium", "low"
    title: str
    description: str
    actionable_advice: str
    relevant_heroes: List[str]


class PostMatchAnalyzer:
    """Post-match analysis system focused on retrospective learning."""
    
    def __init__(self, meta_analyzer: Optional[MetaAnalyzer] = None):
        """Initialize with meta analyzer."""
        self.meta_analyzer = meta_analyzer or MetaAnalyzer()
        self.role_mapping = self._init_role_mapping()
    
    def analyze_match(self, match_result: MatchResult) -> Dict[str, any]:
        """Perform comprehensive post-match analysis."""
        analysis = {
            "match_summary": self._generate_match_summary(match_result),
            "threat_assessment": self._assess_enemy_threats(match_result),
            "counter_suggestions": self._suggest_retrospective_counters(match_result),
            "team_composition_analysis": self._analyze_team_composition(match_result),
            "learning_points": self._generate_learning_points(match_result),
            "meta_awareness": self._analyze_meta_awareness(match_result)
        }
        
        return analysis
    
    def _suggest_retrospective_counters(self, match_result: MatchResult) -> List[HeroRecommendation]:
        """Suggest what heroes could have been picked to counter enemy threats."""
        # Get counter recommendations for the enemy team
        enemy_counters = self.meta_analyzer.get_counter_recommendations(match_result.enemy_team)
        
        # Filter out heroes that were already picked by allies
        unavailable_heroes = set(match_result.ally_team + match_result.enemy_team)
        available_counters = [rec for rec in enemy_counters if rec.hero not in unavailable_heroes]
        
        # Enhance recommendations with retrospective context
        enhanced_recommendations = []
        for rec in available_counters:
            # Calculate how much this could have helped
            threat_coverage = self._calculate_threat_coverage(rec.hero, match_result.enemy_team)
            
            # Adjust reasoning for post-match context
            retrospective_reasoning = f"Could have countered {len(rec.meta_data.counter_heroes)} enemy heroes. "
            retrospective_reasoning += f"This {rec.meta_data.tier}-tier hero would have provided {threat_coverage:.1f}% threat coverage. "
            
            if match_result.match_outcome == "defeat":
                retrospective_reasoning += "Consider this pick for similar enemy compositions in future matches."
            else:
                retrospective_reasoning += "Alternative strong pick for this enemy composition."
            
            enhanced_rec = HeroRecommendation(
                hero=rec.hero,
                confidence=rec.confidence,
                reasoning=retrospective_reasoning,
                meta_data=rec.meta_data,
                counter_effectiveness=rec.counter_effectiveness
            )
            enhanced_recommendations.append(enhanced_rec)
        
        return enhanced_recommendations[:8]  # Top 8 suggestions
    
    def _assess_enemy_threats(self, match_result: MatchResult) -> List[ThreatAssessment]:
        """Assess threat level of enemy heroes based on meta and performance."""
        threats = []
        
        for enemy_hero in match_result.enemy_team:
            hero_data = self.meta_analyzer.get_hero_meta(enemy_hero)
            if not hero_data:
                continue
            
            # Calculate threat level
            threat_level = self._calculate_threat_level(hero_data, match_result)
            
            # Generate threat reasons
            threat_reasons = []
            if hero_data.tier in ["S", "A"]:
                threat_reasons.append(f"High meta tier ({hero_data.tier})")
            if hero_data.win_rate > 55:
                threat_reasons.append(f"High win rate ({hero_data.win_rate:.1f}%)")
            if hero_data.ban_rate > 20:
                threat_reasons.append(f"Frequently banned ({hero_data.ban_rate:.1f}%)")
            
            # Check if this hero counters our team
            countered_allies = []
            for ally in match_result.ally_team:
                ally_data = self.meta_analyzer.get_hero_meta(ally)
                if ally_data and enemy_hero.lower() in [c.lower() for c in ally_data.counter_heroes]:
                    countered_allies.append(ally)
            
            if countered_allies:
                threat_reasons.append(f"Counters {len(countered_allies)} ally heroes")
            
            # Get suggested counters
            suggested_counters = []
            if hero_data.counter_heroes:
                # Filter to heroes not picked in match
                used_heroes = set(match_result.ally_team + match_result.enemy_team)
                available_counters = [c for c in hero_data.counter_heroes if c not in used_heroes]
                suggested_counters = available_counters[:3]  # Top 3
            
            threat = ThreatAssessment(
                hero=enemy_hero,
                threat_level=threat_level,
                threat_reasons=threat_reasons,
                suggested_counters=suggested_counters,
                meta_data=hero_data
            )
            threats.append(threat)
        
        # Sort by threat level
        threats.sort(key=lambda x: x.threat_level, reverse=True)
        return threats
    
    def _generate_learning_points(self, match_result: MatchResult) -> List[PostMatchInsight]:
        """Generate actionable learning points for future matches."""
        insights = []
        
        # Counter-pick learning
        enemy_threats = self._assess_enemy_threats(match_result)
        high_threats = [t for t in enemy_threats if t.threat_level > 0.7]
        
        if high_threats:
            counter_insight = PostMatchInsight(
                insight_type="counter_pick",
                priority="high",
                title="Counter-Pick Opportunities Missed",
                description=f"Enemy team had {len(high_threats)} high-threat heroes that could have been countered.",
                actionable_advice=f"In future matches against {', '.join([t.hero for t in high_threats])}, consider picking {', '.join(high_threats[0].suggested_counters[:2])}.",
                relevant_heroes=[t.hero for t in high_threats] + high_threats[0].suggested_counters[:2]
            )
            insights.append(counter_insight)
        
        # Team composition learning
        team_comp_analysis = self._analyze_team_composition(match_result)
        if team_comp_analysis.get("weaknesses"):
            comp_insight = PostMatchInsight(
                insight_type="team_comp",
                priority="medium",
                title="Team Composition Analysis",
                description="Your team composition had structural weaknesses.",
                actionable_advice=f"Address these weaknesses: {', '.join(team_comp_analysis['weaknesses'])}",
                relevant_heroes=match_result.ally_team
            )
            insights.append(comp_insight)
        
        # Meta awareness learning
        meta_picks = self.meta_analyzer.get_meta_recommendations()
        enemy_meta_heroes = []
        for enemy in match_result.enemy_team:
            enemy_data = self.meta_analyzer.get_hero_meta(enemy)
            if enemy_data and enemy_data.tier in ["S", "A"]:
                enemy_meta_heroes.append(enemy)
        
        if len(enemy_meta_heroes) > len([h for h in match_result.ally_team if self.meta_analyzer.get_hero_meta(h) and self.meta_analyzer.get_hero_meta(h).tier in ["S", "A"]]):
            meta_insight = PostMatchInsight(
                insight_type="meta_awareness",
                priority="medium",
                title="Meta Hero Disadvantage",
                description=f"Enemy team had more meta heroes ({len(enemy_meta_heroes)}) than your team.",
                actionable_advice=f"Consider learning strong meta heroes: {', '.join([rec.hero for rec in meta_picks[:3]])}",
                relevant_heroes=enemy_meta_heroes + [rec.hero for rec in meta_picks[:3]]
            )
            insights.append(meta_insight)
        
        return insights
    
    def _analyze_team_composition(self, match_result: MatchResult) -> Dict[str, any]:
        """Analyze team composition strengths and weaknesses."""
        ally_roles = {}
        enemy_roles = {}
        
        # Analyze ally team
        for hero in match_result.ally_team:
            role = self.role_mapping.get(hero, "unknown")
            ally_roles[role] = ally_roles.get(role, 0) + 1
        
        # Analyze enemy team
        for hero in match_result.enemy_team:
            role = self.role_mapping.get(hero, "unknown")
            enemy_roles[role] = enemy_roles.get(role, 0) + 1
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        essential_roles = ["tank", "marksman", "mage"]
        for role in essential_roles:
            if ally_roles.get(role, 0) == 0:
                weaknesses.append(f"Missing {role}")
            elif ally_roles.get(role, 0) > 1:
                strengths.append(f"Multiple {role}s")
        
        # Compare with enemy
        if enemy_roles.get("tank", 0) > ally_roles.get("tank", 0):
            weaknesses.append("Tank disadvantage")
        
        return {
            "ally_roles": ally_roles,
            "enemy_roles": enemy_roles,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "balance_score": self._calculate_balance_score(ally_roles)
        }
    
    def _analyze_meta_awareness(self, match_result: MatchResult) -> Dict[str, any]:
        """Analyze meta awareness and adaptation."""
        ally_meta_score = 0
        enemy_meta_score = 0
        
        for hero in match_result.ally_team:
            hero_data = self.meta_analyzer.get_hero_meta(hero)
            if hero_data:
                ally_meta_score += hero_data.meta_score
        
        for hero in match_result.enemy_team:
            hero_data = self.meta_analyzer.get_hero_meta(hero)
            if hero_data:
                enemy_meta_score += hero_data.meta_score
        
        meta_advantage = ally_meta_score - enemy_meta_score
        
        return {
            "ally_meta_score": ally_meta_score,
            "enemy_meta_score": enemy_meta_score,
            "meta_advantage": meta_advantage,
            "meta_awareness_rating": "high" if meta_advantage > 20 else "medium" if meta_advantage > -20 else "low"
        }
    
    def _generate_match_summary(self, match_result: MatchResult) -> Dict[str, any]:
        """Generate match summary with key metrics."""
        return {
            "outcome": match_result.match_outcome,
            "duration": match_result.match_duration,
            "player_hero": match_result.player_hero,
            "ally_team": match_result.ally_team,
            "enemy_team": match_result.enemy_team,
            "enemy_threat_level": sum(self._calculate_threat_level(
                self.meta_analyzer.get_hero_meta(hero), match_result
            ) for hero in match_result.enemy_team if self.meta_analyzer.get_hero_meta(hero)) / len(match_result.enemy_team)
        }
    
    def _calculate_threat_level(self, hero_data: HeroMetaData, match_result: MatchResult) -> float:
        """Calculate threat level of an enemy hero."""
        threat = 0
        
        # Base threat from meta strength
        tier_threats = {"S": 0.8, "A": 0.6, "B": 0.4, "C": 0.2, "D": 0.1}
        threat += tier_threats.get(hero_data.tier, 0.1)
        
        # Threat from win rate
        if hero_data.win_rate > 55:
            threat += 0.2
        elif hero_data.win_rate > 50:
            threat += 0.1
        
        # Threat from countering our team
        for ally in match_result.ally_team:
            ally_data = self.meta_analyzer.get_hero_meta(ally)
            if ally_data and hero_data.hero.lower() in [c.lower() for c in ally_data.counter_heroes]:
                threat += 0.15
        
        return min(threat, 1.0)
    
    def _calculate_threat_coverage(self, counter_hero: str, enemy_team: List[str]) -> float:
        """Calculate how much threat this counter hero would have covered."""
        coverage = 0
        
        for enemy in enemy_team:
            enemy_data = self.meta_analyzer.get_hero_meta(enemy)
            if enemy_data and counter_hero.lower() in [c.lower() for c in enemy_data.counter_heroes]:
                coverage += 1
        
        return (coverage / len(enemy_team)) * 100
    
    def _calculate_balance_score(self, roles: Dict[str, int]) -> float:
        """Calculate team composition balance score."""
        essential_roles = ["tank", "marksman", "mage"]
        score = sum(1 for role in essential_roles if roles.get(role, 0) > 0) / len(essential_roles)
        return score
    
    def _init_role_mapping(self) -> Dict[str, str]:
        """Initialize hero to role mapping."""
        return {
            "Franco": "tank", "Tigreal": "tank", "Lolita": "tank", "Akai": "tank",
            "Atlas": "tank", "Belerick": "tank", "Gatotkaca": "tank", "Grock": "tank",
            "Hilda": "tank", "Hylos": "tank", "Johnson": "tank", "Khufra": "tank",
            "Minotaur": "tank", "Angela": "support", "Estes": "support", "Mathilda": "support",
            "Carmilla": "support", "Diggie": "support", "Faramis": "support", "Rafaela": "support",
            "Layla": "marksman", "Beatrix": "marksman", "Melissa": "marksman", "Yi Sun-shin": "marksman",
            "Popol and Kupa": "marksman", "Cyclops": "mage", "Bane": "fighter", "Ruby": "fighter",
            "Sun": "fighter"
        }