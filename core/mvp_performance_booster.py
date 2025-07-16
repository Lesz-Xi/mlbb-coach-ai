"""
MVP Performance Booster

Detects MVP status and high teamfight participation to boost performance ratings.
Addresses the issue where MVP players are incorrectly rated as "Poor".
"""
import logging
from typing import Dict, Any, List, Tuple
import re

from .advanced_performance_analyzer import PerformanceCategory

logger = logging.getLogger(__name__)


class MVPPerformanceBooster:
    """
    Analyzes match data to detect MVP status and apply appropriate performance boosts.
    
    Fixes the issue where players with high teamfight participation and MVP status
    are incorrectly rated as "Poor" due to standard metrics not accounting for
    support/role-specific excellence.
    """
    
    def __init__(self):
        self.mvp_indicators = [
            'mvp', 'most valuable player', 'best player', 'star player',
            'top performer', 'match mvp', 'game mvp'
        ]
        
        self.excellence_thresholds = {
            'teamfight_participation': 80,  # 80%+ is excellent
            'high_participation': 70,       # 70%+ is high
            'good_participation': 50,       # 50%+ is good
            'assists_ratio': 0.7,          # Assists/Deaths ratio
            'support_damage_ratio': 0.3     # Support damage expectations
        }
        
        self.role_specific_criteria = {
            'support': {
                'primary_metrics': ['teamfight_participation', 'assists', 'damage_taken'],
                'secondary_metrics': ['healing_done', 'damage_dealt'],
                'performance_weight': {
                    'teamfight_participation': 0.4,
                    'assists': 0.3,
                    'survival': 0.2,
                    'damage': 0.1
                }
            },
            'tank': {
                'primary_metrics': ['teamfight_participation', 'damage_taken', 'assists'],
                'secondary_metrics': ['turret_damage', 'healing_done'],
                'performance_weight': {
                    'teamfight_participation': 0.35,
                    'damage_taken': 0.3,
                    'assists': 0.25,
                    'damage': 0.1
                }
            },
            'marksman': {
                'primary_metrics': ['hero_damage', 'kills', 'gold'],
                'secondary_metrics': ['teamfight_participation', 'turret_damage'],
                'performance_weight': {
                    'damage': 0.4,
                    'kills': 0.3,
                    'gold': 0.2,
                    'teamfight_participation': 0.1
                }
            }
        }
    
    def analyze_and_boost_performance(
        self, 
        match_data: Dict[str, Any], 
        original_rating: PerformanceCategory,
        original_score: float,
        ocr_results: List = None
    ) -> Tuple[PerformanceCategory, float, Dict[str, Any]]:
        """
        Analyze match data for MVP indicators and boost performance if warranted.
        
        Args:
            match_data: Match performance data
            original_rating: Original performance rating
            original_score: Original performance score
            ocr_results: OCR results for MVP text detection
            
        Returns:
            Tuple of (boosted_rating, boosted_score, boost_analysis)
        """
        boost_analysis = {
            "mvp_detected": False,
            "high_teamfight_participation": False,
            "victory_bonus": False,
            "role_specific_excellence": False,
            "boost_applied": False,
            "boost_reasons": [],
            "original_rating": original_rating.value if hasattr(original_rating, 'value') else str(original_rating),
            "original_score": original_score
        }
        
        # Detect MVP status
        mvp_detected = self._detect_mvp_status(match_data, ocr_results)
        boost_analysis["mvp_detected"] = mvp_detected
        
        # Analyze teamfight participation
        tfp_analysis = self._analyze_teamfight_participation(match_data)
        boost_analysis.update(tfp_analysis)
        
        # Check victory bonus
        victory_bonus = self._check_victory_bonus(match_data)
        boost_analysis["victory_bonus"] = victory_bonus
        
        # Role-specific excellence analysis
        role_excellence = self._analyze_role_specific_excellence(match_data)
        boost_analysis.update(role_excellence)
        
        # Calculate performance boost
        boost_score = self._calculate_performance_boost(
            original_score, boost_analysis, match_data
        )
        
        # Determine new rating
        new_rating = self._determine_boosted_rating(
            original_rating, boost_score, boost_analysis
        )
        
        # Record boost reasoning
        if boost_score > original_score or new_rating != original_rating:
            boost_analysis["boost_applied"] = True
            self._record_boost_reasons(boost_analysis, match_data)
        
        logger.info(f"MVP Analysis: {original_rating} -> {new_rating} (score: {original_score:.2f} -> {boost_score:.2f})")
        
        return new_rating, boost_score, boost_analysis
    
    def _detect_mvp_status(self, match_data: Dict[str, Any], ocr_results: List = None) -> bool:
        """Detect if player achieved MVP status."""
        # Check OCR results for MVP text
        if ocr_results:
            for bbox, text, conf in ocr_results:
                text_lower = text.lower()
                if any(indicator in text_lower for indicator in self.mvp_indicators):
                    logger.info(f"MVP detected in OCR text: '{text}'")
                    return True
        
        # Check match data for MVP indicators
        for key, value in match_data.items():
            if isinstance(value, str):
                value_lower = value.lower()
                if any(indicator in value_lower for indicator in self.mvp_indicators):
                    logger.info(f"MVP detected in match data: {key}='{value}'")
                    return True
        
        # Infer MVP from exceptional performance
        tfp = match_data.get('teamfight_participation', 0)
        match_result = match_data.get('match_result', '').lower()
        
        if tfp >= 85 and 'victory' in match_result:
            logger.info(f"MVP inferred from high TFP ({tfp}%) + victory")
            return True
        
        return False
    
    def _analyze_teamfight_participation(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze teamfight participation for performance boost."""
        tfp = match_data.get('teamfight_participation', 0)
        
        analysis = {
            "teamfight_participation": tfp,
            "high_teamfight_participation": tfp >= self.excellence_thresholds['teamfight_participation'],
            "tfp_category": "poor",
            "tfp_boost_factor": 1.0
        }
        
        if tfp >= self.excellence_thresholds['teamfight_participation']:
            analysis["tfp_category"] = "excellent"
            analysis["tfp_boost_factor"] = 1.4
        elif tfp >= self.excellence_thresholds['high_participation']:
            analysis["tfp_category"] = "high"
            analysis["tfp_boost_factor"] = 1.2
        elif tfp >= self.excellence_thresholds['good_participation']:
            analysis["tfp_category"] = "good"
            analysis["tfp_boost_factor"] = 1.1
        
        return analysis
    
    def _check_victory_bonus(self, match_data: Dict[str, Any]) -> bool:
        """Check if player won the match for victory bonus."""
        match_result = match_data.get('match_result', '').lower()
        return 'victory' in match_result or 'win' in match_result
    
    def _analyze_role_specific_excellence(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze role-specific performance excellence."""
        hero = match_data.get('hero', 'unknown').lower()
        role = self._get_hero_role(hero)
        
        analysis = {
            "detected_role": role,
            "role_specific_excellence": False,
            "role_performance_score": 0.0,
            "excels_in_primary_metrics": False
        }
        
        if role in self.role_specific_criteria:
            criteria = self.role_specific_criteria[role]
            role_score = self._calculate_role_performance(match_data, criteria)
            
            analysis["role_performance_score"] = role_score
            analysis["role_specific_excellence"] = role_score >= 1.2  # 20% above average
            analysis["excels_in_primary_metrics"] = role_score >= 1.1
        
        return analysis
    
    def _calculate_performance_boost(
        self, 
        original_score: float, 
        boost_analysis: Dict[str, Any], 
        match_data: Dict[str, Any]
    ) -> float:
        """Calculate the boosted performance score."""
        boost_score = original_score
        
        # MVP boost
        if boost_analysis["mvp_detected"]:
            boost_score *= 1.5  # 50% boost for MVP
        
        # Teamfight participation boost
        tfp_boost = boost_analysis.get("tfp_boost_factor", 1.0)
        if tfp_boost > 1.0:
            boost_score *= tfp_boost
        
        # Victory bonus
        if boost_analysis["victory_bonus"]:
            boost_score *= 1.2  # 20% boost for victory
        
        # Role-specific excellence boost
        if boost_analysis["role_specific_excellence"]:
            boost_score *= 1.3  # 30% boost for role excellence
        
        # Support hero special consideration
        hero = match_data.get('hero', '').lower()
        if self._is_support_hero(hero):
            tfp = match_data.get('teamfight_participation', 0)
            assists = match_data.get('assists', 0)
            deaths = max(match_data.get('deaths', 1), 1)
            
            # Support heroes with high TFP and good KDA should be rated highly
            if tfp >= 80 and assists >= deaths:
                boost_score *= 1.4  # Special support boost
        
        return min(boost_score, 2.0)  # Cap at 2.0 to prevent extreme scores
    
    def _determine_boosted_rating(
        self, 
        original_rating: PerformanceCategory, 
        boost_score: float,
        boost_analysis: Dict[str, Any]
    ) -> PerformanceCategory:
        """Determine the new performance rating after boosts."""
        # Force minimum rating for MVP with high teamfight participation
        if (boost_analysis["mvp_detected"] and 
            boost_analysis["high_teamfight_participation"] and
            boost_analysis["victory_bonus"]):
            return PerformanceCategory.EXCELLENT
        
        # Force minimum rating for MVP
        if boost_analysis["mvp_detected"]:
            if original_rating == PerformanceCategory.POOR:
                return PerformanceCategory.GOOD
            elif original_rating == PerformanceCategory.NEEDS_WORK:
                return PerformanceCategory.GOOD
        
        # High teamfight participation should never be "Poor"
        if boost_analysis["high_teamfight_participation"]:
            if original_rating == PerformanceCategory.POOR:
                return PerformanceCategory.AVERAGE
        
        # Use boosted score for rating
        if boost_score >= 1.4:
            return PerformanceCategory.EXCELLENT
        elif boost_score >= 1.15:
            return PerformanceCategory.GOOD
        elif boost_score >= 0.9:
            return PerformanceCategory.AVERAGE
        elif boost_score >= 0.7:
            return PerformanceCategory.NEEDS_WORK
        else:
            return PerformanceCategory.POOR
    
    def _record_boost_reasons(self, boost_analysis: Dict[str, Any], match_data: Dict[str, Any]):
        """Record reasons for performance boost."""
        reasons = []
        
        if boost_analysis["mvp_detected"]:
            reasons.append("MVP status detected")
        
        if boost_analysis["high_teamfight_participation"]:
            tfp = boost_analysis["teamfight_participation"]
            reasons.append(f"Excellent teamfight participation ({tfp}%)")
        
        if boost_analysis["victory_bonus"]:
            reasons.append("Victory match bonus")
        
        if boost_analysis["role_specific_excellence"]:
            role = boost_analysis["detected_role"]
            reasons.append(f"Excellent {role} performance")
        
        hero = match_data.get('hero', '').lower()
        if self._is_support_hero(hero):
            reasons.append("Support hero performance criteria applied")
        
        boost_analysis["boost_reasons"] = reasons
    
    def _calculate_role_performance(self, match_data: Dict[str, Any], criteria: Dict[str, Any]) -> float:
        """Calculate role-specific performance score."""
        weights = criteria["performance_weight"]
        score = 0.0
        
        # Teamfight participation
        if "teamfight_participation" in weights:
            tfp = match_data.get('teamfight_participation', 0) / 100.0
            score += tfp * weights["teamfight_participation"]
        
        # Assists relative to deaths
        if "assists" in weights:
            assists = match_data.get('assists', 0)
            deaths = max(match_data.get('deaths', 1), 1)
            assist_ratio = min(assists / deaths, 2.0)  # Cap at 2.0
            score += assist_ratio * weights["assists"]
        
        # Damage metrics
        if "damage" in weights:
            damage = match_data.get('hero_damage', 0)
            damage_score = min(damage / 50000, 1.5)  # Normalize and cap
            score += damage_score * weights["damage"]
        
        # Survival metrics
        if "survival" in weights:
            deaths = match_data.get('deaths', 1)
            survival_score = max(0, 1.0 - (deaths / 10))  # Better with fewer deaths
            score += survival_score * weights["survival"]
        
        return score
    
    def _get_hero_role(self, hero: str) -> str:
        """Get hero role for role-specific analysis."""
        support_heroes = ['mathilda', 'estes', 'angela', 'diggie', 'rafaela']
        tank_heroes = ['tigreal', 'franco', 'johnson', 'grock', 'khufra']
        marksman_heroes = ['miya', 'layla', 'bruno', 'clint', 'moskov']
        
        if hero in support_heroes:
            return 'support'
        elif hero in tank_heroes:
            return 'tank'
        elif hero in marksman_heroes:
            return 'marksman'
        else:
            return 'unknown'
    
    def _is_support_hero(self, hero: str) -> bool:
        """Check if hero is a support."""
        support_heroes = ['mathilda', 'estes', 'angela', 'diggie', 'rafaela']
        return hero.lower() in support_heroes


# Create global instance
mvp_performance_booster = MVPPerformanceBooster() 