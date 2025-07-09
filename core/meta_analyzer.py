import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

from .schemas import (
    HeroMetaData, 
    MetaLeaderboard, 
    HeroRecommendation, 
    RecommendationRequest,
    PerformanceComparison
)


class MetaAnalyzer:
    """Core system for meta analysis, hero recommendations, and performance comparisons."""
    
    def __init__(self, data_path: str = "data/mlbb-winrate-leaderboard-complete.json"):
        """Initialize with winrate leaderboard data."""
        self.data_path = Path(data_path)
        self.meta_data: Optional[MetaLeaderboard] = None
        self.hero_lookup: Dict[str, HeroMetaData] = {}
        self.load_meta_data()
    
    def load_meta_data(self) -> None:
        """Load and validate meta data from JSON file."""
        try:
            with open(self.data_path, 'r') as f:
                raw_data = json.load(f)
            
            # Convert raw data to structured format
            heroes = [HeroMetaData(**hero_data) for hero_data in raw_data]
            
            self.meta_data = MetaLeaderboard(
                data=heroes,
                last_updated=datetime.now()
            )
            
            # Create lookup dictionary for fast hero access
            self.hero_lookup = {hero.hero.lower(): hero for hero in heroes}
            
            print(f"Loaded {len(heroes)} heroes from meta data")
            
        except Exception as e:
            print(f"Error loading meta data: {e}")
            raise
    
    def get_hero_meta(self, hero_name: str) -> Optional[HeroMetaData]:
        """Get meta data for a specific hero."""
        return self.hero_lookup.get(hero_name.lower())
    
    def generate_tier_list(self) -> Dict[str, List[HeroMetaData]]:
        """Generate dynamic tier list based on current meta."""
        if not self.meta_data:
            return {}
        
        tier_list = {"S": [], "A": [], "B": [], "C": [], "D": []}
        
        for hero in self.meta_data.data:
            tier_list[hero.tier].append(hero)
        
        # Sort each tier by meta score
        for tier in tier_list:
            tier_list[tier].sort(key=lambda x: x.meta_score, reverse=True)
        
        return tier_list
    
    def get_counter_recommendations(self, enemy_heroes: List[str]) -> List[HeroRecommendation]:
        """Get hero recommendations based on enemy draft."""
        if not self.meta_data:
            return []
        
        recommendations = []
        counter_scores = {}
        
        # Calculate counter effectiveness for each hero
        for hero in self.meta_data.data:
            counter_effectiveness = 0
            counter_count = 0
            
            # Check how many enemy heroes this hero counters
            for enemy in enemy_heroes:
                enemy_data = self.get_hero_meta(enemy)
                if enemy_data and hero.hero.lower() in [c.lower() for c in enemy_data.counter_heroes]:
                    counter_effectiveness += 1
                    counter_count += 1
            
            if counter_count > 0:
                # Normalize counter effectiveness
                counter_effectiveness = counter_effectiveness / len(enemy_heroes)
                
                # Calculate overall recommendation score
                meta_score = hero.meta_score / 100  # Normalize to 0-1
                recommendation_score = (counter_effectiveness * 0.7) + (meta_score * 0.3)
                
                reasoning = f"Counters {counter_count} enemy heroes. "
                reasoning += f"Current meta tier: {hero.tier}. "
                reasoning += f"Win rate: {hero.win_rate:.1f}%"
                
                recommendations.append(HeroRecommendation(
                    hero=hero.hero,
                    confidence=min(recommendation_score, 1.0),
                    reasoning=reasoning,
                    meta_data=hero,
                    counter_effectiveness=counter_effectiveness
                ))
        
        # Sort by confidence and return top recommendations
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations[:10]
    
    def get_meta_recommendations(self, role_preference: Optional[str] = None) -> List[HeroRecommendation]:
        """Get general meta recommendations based on current tier list."""
        if not self.meta_data:
            return []
        
        recommendations = []
        
        # Get top tier heroes
        top_heroes = self.meta_data.top_tier_heroes
        
        for hero in top_heroes:
            confidence = min(hero.meta_score / 100, 1.0)
            
            reasoning = f"Tier {hero.tier} hero with {hero.win_rate:.1f}% win rate. "
            reasoning += f"Pick rate: {hero.pick_rate:.1f}%. "
            
            if hero.ban_rate > 20:
                reasoning += "High ban rate indicates strong meta presence."
            
            recommendations.append(HeroRecommendation(
                hero=hero.hero,
                confidence=confidence,
                reasoning=reasoning,
                meta_data=hero
            ))
        
        return recommendations[:15]
    
    def compare_performance(self, hero_name: str, player_winrate: float) -> Optional[PerformanceComparison]:
        """Compare player performance vs meta average."""
        hero_data = self.get_hero_meta(hero_name)
        if not hero_data:
            return None
        
        performance_gap = player_winrate - hero_data.win_rate
        
        # Calculate percentile rank (simplified)
        if performance_gap > 10:
            percentile_rank = 90
        elif performance_gap > 5:
            percentile_rank = 75
        elif performance_gap > 0:
            percentile_rank = 60
        elif performance_gap > -5:
            percentile_rank = 40
        elif performance_gap > -10:
            percentile_rank = 25
        else:
            percentile_rank = 10
        
        # Generate improvement areas
        improvement_areas = []
        if performance_gap < -5:
            improvement_areas.extend([
                "Focus on optimal item builds",
                "Improve positioning in team fights",
                "Practice combo execution"
            ])
        elif performance_gap < 0:
            improvement_areas.extend([
                "Refine macro decision making",
                "Optimize farming efficiency"
            ])
        
        return PerformanceComparison(
            hero=hero_name,
            player_winrate=player_winrate,
            meta_winrate=hero_data.win_rate,
            performance_gap=performance_gap,
            percentile_rank=percentile_rank,
            improvement_areas=improvement_areas
        )
    
    def get_ban_priority_list(self) -> List[HeroMetaData]:
        """Get heroes that should be prioritized for banning."""
        if not self.meta_data:
            return []
        
        # Heroes with high win rate AND high pick rate are ban priorities
        ban_priorities = []
        
        for hero in self.meta_data.data:
            if hero.win_rate >= 52 and hero.pick_rate >= 1.0:
                ban_priorities.append(hero)
        
        # Sort by meta score (combination of win rate, pick rate, ban rate)
        ban_priorities.sort(key=lambda x: x.meta_score, reverse=True)
        
        return ban_priorities[:10]
    
    def analyze_meta_shifts(self, previous_data_path: Optional[str] = None) -> Dict[str, List[HeroMetaData]]:
        """Analyze meta shifts compared to previous patch data."""
        if not previous_data_path:
            return {"rising": [], "falling": [], "stable": []}
        
        # This would compare current data with previous patch
        # For now, return empty structure
        return {
            "rising": [],    # Heroes gaining popularity/winrate
            "falling": [],   # Heroes losing popularity/winrate  
            "stable": []     # Heroes with minimal changes
        }
    
    def get_role_recommendations(self, role: str) -> List[HeroRecommendation]:
        """Get recommendations for a specific role (tank, marksman, etc.)."""
        # This would require role data in the JSON
        # For now, return general meta recommendations
        return self.get_meta_recommendations()
    
    def get_synergy_recommendations(self, ally_heroes: List[str]) -> List[HeroRecommendation]:
        """Get hero recommendations based on ally synergies."""
        # This would require synergy data
        # For now, return general meta recommendations
        return self.get_meta_recommendations()