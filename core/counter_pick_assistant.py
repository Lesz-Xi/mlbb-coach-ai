from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from .meta_analyzer import MetaAnalyzer
from .schemas import HeroRecommendation, HeroMetaData


@dataclass
class DraftState:
    """Represents the current state of hero draft."""
    enemy_picks: List[str]
    ally_picks: List[str]
    enemy_bans: List[str]
    ally_bans: List[str]
    current_phase: str  # "ban1", "pick1", "ban2", "pick2", etc.


@dataclass
class CounterPickSuggestion:
    """Enhanced counter-pick suggestion with detailed analysis."""
    hero: str
    priority: str  # "high", "medium", "low"
    reasoning: str
    counters: List[str]  # Which enemy heroes this counters
    vulnerable_to: List[str]  # Enemy heroes that counter this pick
    synergy_score: float
    meta_strength: str  # "S", "A", "B", "C", "D"
    confidence: float


class CounterPickAssistant:
    """Advanced counter-pick assistant for draft phase."""
    
    def __init__(self, meta_analyzer: Optional[MetaAnalyzer] = None):
        """Initialize with meta analyzer."""
        self.meta_analyzer = meta_analyzer or MetaAnalyzer()
        self.role_mapping = self._init_role_mapping()
    
    def _init_role_mapping(self) -> Dict[str, str]:
        """Initialize hero to role mapping."""
        # This would ideally be loaded from a configuration file
        return {
            "Franco": "tank",
            "Tigreal": "tank",
            "Lolita": "tank",
            "Akai": "tank",
            "Atlas": "tank",
            "Belerick": "tank",
            "Gatotkaca": "tank",
            "Grock": "tank",
            "Hilda": "tank",
            "Hylos": "tank",
            "Johnson": "tank",
            "Khufra": "tank",
            "Minotaur": "tank",
            "Angela": "support",
            "Estes": "support",
            "Mathilda": "support",
            "Carmilla": "support",
            "Diggie": "support",
            "Faramis": "support",
            "Rafaela": "support",
            "Layla": "marksman",
            "Beatrix": "marksman",
            "Melissa": "marksman",
            "Cyclops": "mage",
            "Bane": "fighter",
            "Ruby": "fighter",
            "Sun": "fighter",
            "Popol and Kupa": "marksman",
            "Yi Sun-shin": "marksman",
            # Add more heroes as needed
        }
    
    def analyze_draft_state(self, draft_state: DraftState) -> Dict[str, any]:
        """Analyze current draft state and provide insights."""
        analysis = {
            "team_composition": self._analyze_team_composition(draft_state.ally_picks),
            "enemy_threats": self._analyze_enemy_threats(draft_state.enemy_picks),
            "missing_roles": self._identify_missing_roles(draft_state.ally_picks),
            "counter_opportunities": self._find_counter_opportunities(draft_state),
            "ban_recommendations": self._suggest_strategic_bans(draft_state)
        }
        return analysis
    
    def get_counter_pick_suggestions(self, draft_state: DraftState, 
                                   role_filter: Optional[str] = None) -> List[CounterPickSuggestion]:
        """Get prioritized counter-pick suggestions."""
        suggestions = []
        
        # Get available heroes (not banned or picked)
        unavailable_heroes = set(draft_state.enemy_picks + draft_state.ally_picks + 
                               draft_state.enemy_bans + draft_state.ally_bans)
        
        # Get counter recommendations from meta analyzer
        counter_recs = self.meta_analyzer.get_counter_recommendations(draft_state.enemy_picks)
        
        for rec in counter_recs:
            if rec.hero in unavailable_heroes:
                continue
                
            # Filter by role if specified
            if role_filter and self.role_mapping.get(rec.hero) != role_filter:
                continue
            
            # Calculate enhanced metrics
            counters = self._get_countered_enemies(rec.hero, draft_state.enemy_picks)
            vulnerable_to = self._get_vulnerabilities(rec.hero, draft_state.enemy_picks)
            synergy_score = self._calculate_synergy_score(rec.hero, draft_state.ally_picks)
            
            # Determine priority
            priority = self._calculate_priority(rec, counters, vulnerable_to, synergy_score)
            
            suggestion = CounterPickSuggestion(
                hero=rec.hero,
                priority=priority,
                reasoning=rec.reasoning,
                counters=counters,
                vulnerable_to=vulnerable_to,
                synergy_score=synergy_score,
                meta_strength=rec.meta_data.tier,
                confidence=rec.confidence
            )
            
            suggestions.append(suggestion)
        
        # Sort by priority and confidence
        priority_order = {"high": 3, "medium": 2, "low": 1}
        suggestions.sort(key=lambda x: (priority_order.get(x.priority, 0), x.confidence), reverse=True)
        
        return suggestions[:10]
    
    def suggest_ban_picks(self, draft_state: DraftState) -> List[HeroMetaData]:
        """Suggest heroes to ban based on current draft state."""
        ban_priorities = self.meta_analyzer.get_ban_priority_list()
        
        # Filter out already banned heroes
        banned_heroes = set(draft_state.enemy_bans + draft_state.ally_bans)
        available_bans = [hero for hero in ban_priorities if hero.hero not in banned_heroes]
        
        # Consider enemy team composition for targeted bans
        enhanced_bans = []
        for hero in available_bans:
            # Check if this hero would counter our current picks
            threat_level = self._assess_threat_level(hero, draft_state.ally_picks)
            if threat_level > 0.5:  # High threat
                enhanced_bans.append(hero)
        
        return enhanced_bans[:5]
    
    def _analyze_team_composition(self, ally_picks: List[str]) -> Dict[str, any]:
        """Analyze ally team composition."""
        roles = {}
        for hero in ally_picks:
            role = self.role_mapping.get(hero, "unknown")
            roles[role] = roles.get(role, 0) + 1
        
        return {
            "roles": roles,
            "balance_score": self._calculate_balance_score(roles),
            "strengths": self._identify_comp_strengths(ally_picks),
            "weaknesses": self._identify_comp_weaknesses(ally_picks)
        }
    
    def _analyze_enemy_threats(self, enemy_picks: List[str]) -> List[Dict[str, any]]:
        """Analyze enemy threats."""
        threats = []
        for hero in enemy_picks:
            hero_data = self.meta_analyzer.get_hero_meta(hero)
            if hero_data:
                threats.append({
                    "hero": hero,
                    "threat_level": self._calculate_threat_level(hero_data),
                    "role": self.role_mapping.get(hero, "unknown"),
                    "counters": hero_data.counter_heroes
                })
        return threats
    
    def _identify_missing_roles(self, ally_picks: List[str]) -> List[str]:
        """Identify missing roles in team composition."""
        essential_roles = ["tank", "marksman", "mage"]
        current_roles = [self.role_mapping.get(hero, "unknown") for hero in ally_picks]
        return [role for role in essential_roles if role not in current_roles]
    
    def _find_counter_opportunities(self, draft_state: DraftState) -> List[Dict[str, any]]:
        """Find specific counter opportunities."""
        opportunities = []
        for enemy_hero in draft_state.enemy_picks:
            enemy_data = self.meta_analyzer.get_hero_meta(enemy_hero)
            if enemy_data:
                # Find available counters
                available_counters = []
                for counter in enemy_data.counter_heroes:
                    if counter not in (draft_state.ally_picks + draft_state.enemy_picks + 
                                     draft_state.ally_bans + draft_state.enemy_bans):
                        counter_data = self.meta_analyzer.get_hero_meta(counter)
                        if counter_data:
                            available_counters.append({
                                "hero": counter,
                                "tier": counter_data.tier,
                                "win_rate": counter_data.win_rate
                            })
                
                if available_counters:
                    opportunities.append({
                        "target": enemy_hero,
                        "counters": available_counters
                    })
        
        return opportunities
    
    def _suggest_strategic_bans(self, draft_state: DraftState) -> List[str]:
        """Suggest strategic bans."""
        # Get top meta heroes that would be problematic
        ban_priorities = self.meta_analyzer.get_ban_priority_list()
        
        # Filter by availability
        available_bans = [hero.hero for hero in ban_priorities 
                         if hero.hero not in (draft_state.enemy_bans + draft_state.ally_bans)]
        
        return available_bans[:5]
    
    def _get_countered_enemies(self, hero: str, enemy_picks: List[str]) -> List[str]:
        """Get list of enemy heroes that this hero counters."""
        counters = []
        for enemy in enemy_picks:
            enemy_data = self.meta_analyzer.get_hero_meta(enemy)
            if enemy_data and hero.lower() in [c.lower() for c in enemy_data.counter_heroes]:
                counters.append(enemy)
        return counters
    
    def _get_vulnerabilities(self, hero: str, enemy_picks: List[str]) -> List[str]:
        """Get list of enemy heroes that counter this hero."""
        hero_data = self.meta_analyzer.get_hero_meta(hero)
        if not hero_data:
            return []
        
        vulnerabilities = []
        for counter in hero_data.counter_heroes:
            if counter.lower() in [e.lower() for e in enemy_picks]:
                vulnerabilities.append(counter)
        return vulnerabilities
    
    def _calculate_synergy_score(self, hero: str, ally_picks: List[str]) -> float:
        """Calculate synergy score with ally picks."""
        # Simplified synergy calculation
        # This would be enhanced with actual synergy data
        return 0.5  # Placeholder
    
    def _calculate_priority(self, rec: HeroRecommendation, counters: List[str], 
                          vulnerable_to: List[str], synergy_score: float) -> str:
        """Calculate pick priority."""
        score = 0
        
        # Counter effectiveness
        score += len(counters) * 0.3
        
        # Meta strength
        tier_scores = {"S": 1.0, "A": 0.8, "B": 0.6, "C": 0.4, "D": 0.2}
        score += tier_scores.get(rec.meta_data.tier, 0.2) * 0.3
        
        # Vulnerability penalty
        score -= len(vulnerable_to) * 0.2
        
        # Synergy bonus
        score += synergy_score * 0.2
        
        if score >= 0.8:
            return "high"
        elif score >= 0.5:
            return "medium"
        else:
            return "low"
    
    def _assess_threat_level(self, hero: HeroMetaData, ally_picks: List[str]) -> float:
        """Assess threat level of a hero against our team."""
        threat = 0
        
        # Check how many of our picks this hero counters
        for ally in ally_picks:
            ally_data = self.meta_analyzer.get_hero_meta(ally)
            if ally_data and hero.hero.lower() in [c.lower() for c in ally_data.counter_heroes]:
                threat += 0.3
        
        # Add meta strength factor
        tier_threats = {"S": 0.4, "A": 0.3, "B": 0.2, "C": 0.1, "D": 0.05}
        threat += tier_threats.get(hero.tier, 0.05)
        
        return min(threat, 1.0)
    
    def _calculate_balance_score(self, roles: Dict[str, int]) -> float:
        """Calculate team composition balance score."""
        # Simplified balance calculation
        essential_roles = ["tank", "marksman", "mage"]
        score = sum(1 for role in essential_roles if roles.get(role, 0) > 0) / len(essential_roles)
        return score
    
    def _identify_comp_strengths(self, ally_picks: List[str]) -> List[str]:
        """Identify team composition strengths."""
        # Placeholder - would analyze actual team synergies
        return ["Balanced composition"]
    
    def _identify_comp_weaknesses(self, ally_picks: List[str]) -> List[str]:
        """Identify team composition weaknesses."""
        # Placeholder - would analyze actual team weaknesses
        return ["Needs more crowd control"]
    
    def _calculate_threat_level(self, hero_data: HeroMetaData) -> float:
        """Calculate threat level of an enemy hero."""
        # Simple threat calculation based on meta strength
        return hero_data.meta_score / 100