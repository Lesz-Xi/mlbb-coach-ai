"""
Behavioral Modeling System for MLBB Coach AI
============================================

This module provides comprehensive behavioral analysis capabilities that extract 
player behavior patterns from screenshot match results and replay footage.

Key Features:
- Play style classification (aggressive-roamer, passive-farmer, etc.)
- Strategic tendency identification (objective-focused, kill-focused, etc.)
- Decision-making trait analysis (risk-taking, positioning, etc.)
- Behavioral fingerprint generation
- Integration with existing SOA architecture

Architecture:
- Extends existing screenshot analysis and video processing
- Uses hybrid caching strategy (memory + Redis)
- Integrates with event system for real-time monitoring
- Follows async/await patterns for performance
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

# Import existing system components
from .enhanced_data_collector import EnhancedDataCollector
from .video_reader import VideoReader
from .cache.hybrid_cache import HybridCache
from .events.event_bus import EventBus
from .diagnostic_logger import diagnostic_logger
from .schemas import BaseMatch, AnyMatch

logger = logging.getLogger(__name__)


class PlayStyle(Enum):
    """Play style classifications."""
    AGGRESSIVE_ROAMER = "aggressive-roamer"
    PASSIVE_FARMER = "passive-farmer"
    OBJECTIVE_FOCUSED = "objective-focused"
    TEAM_FIGHTER = "team-fighter"
    SPLIT_PUSHER = "split-pusher"
    SUPPORT_ORIENTED = "support-oriented"
    CARRY_FOCUSED = "carry-focused"
    ASSASSIN_STYLE = "assassin-style"


class RiskProfile(Enum):
    """Risk-taking behavior classifications."""
    HIGH_RISK_HIGH_REWARD = "high-risk-high-reward"
    CALCULATED_RISK = "calculated-risk"
    CONSERVATIVE = "conservative"
    OPPORTUNISTIC = "opportunistic"


class GameTempo(Enum):
    """Game tempo preferences."""
    EARLY_AGGRESSIVE = "early-aggressive"
    MID_GAME_FOCUSED = "mid-game-focused"
    LATE_GAME_SCALING = "late-game-scaling"
    ADAPTIVE = "adaptive"


@dataclass
class BehavioralMetrics:
    """Core behavioral metrics extracted from match data."""
    # Combat behavior
    kill_participation_rate: float = 0.0
    death_avoidance_score: float = 0.0
    assist_contribution: float = 0.0
    
    # Economic behavior
    farming_efficiency: float = 0.0
    gold_per_minute_consistency: float = 0.0
    resource_prioritization: float = 0.0
    
    # Strategic behavior
    objective_participation: float = 0.0
    map_control_score: float = 0.0
    team_coordination: float = 0.0
    
    # Positioning behavior
    positioning_safety: float = 0.0
    engagement_timing: float = 0.0
    escape_success_rate: float = 0.0
    
    # Tempo behavior
    early_game_impact: float = 0.0
    mid_game_impact: float = 0.0
    late_game_impact: float = 0.0


@dataclass
class BehavioralFingerprint:
    """Complete behavioral fingerprint for a player."""
    player_id: str
    
    # Core classifications
    play_style: PlayStyle
    risk_profile: RiskProfile
    game_tempo: GameTempo
    
    # Behavioral scores (0.0 to 1.0)
    map_awareness_score: float
    synergy_with_team: float
    adaptability_score: float
    mechanical_skill_score: float
    decision_making_score: float
    
    # Preferred patterns
    preferred_lane: str
    preferred_role: str
    preferred_heroes: List[str]
    
    # Identified patterns
    behavior_tags: List[str]
    identified_flaws: List[str]
    strength_areas: List[str]
    
    # Metrics
    behavioral_metrics: BehavioralMetrics
    
    # Confidence and metadata
    confidence_score: float
    analysis_date: datetime
    matches_analyzed: int
    
    # Temporal patterns
    performance_by_time: Dict[str, float] = field(default_factory=dict)
    hero_performance_patterns: Dict[str, Dict[str, float]] = field(default_factory=dict)


class BehavioralAnalyzer:
    """Main behavioral analysis engine."""
    
    def __init__(self, cache_manager: Optional[HybridCache] = None):
        """Initialize the behavioral analyzer."""
        self.cache_manager = cache_manager or HybridCache()
        self.event_bus = EventBus()
        self.data_collector = EnhancedDataCollector()
        self.video_reader = VideoReader()
        
        # Analysis parameters
        self.min_matches_for_analysis = 5
        self.behavioral_weights = self._load_behavioral_weights()
        self.classification_thresholds = self._load_classification_thresholds()
        
        # Performance tracking
        self.analysis_history = []
        
    def _load_behavioral_weights(self) -> Dict[str, float]:
        """Load behavioral analysis weights."""
        return {
            "kda_weight": 0.25,
            "economic_weight": 0.20,
            "objective_weight": 0.20,
            "positioning_weight": 0.15,
            "tempo_weight": 0.10,
            "team_coordination_weight": 0.10
        }
    
    def _load_classification_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load classification thresholds for behavioral patterns."""
        return {
            "play_style": {
                "aggressive_threshold": 0.7,
                "passive_threshold": 0.3,
                "objective_threshold": 0.6,
                "team_fight_threshold": 0.65
            },
            "risk_profile": {
                "high_risk_threshold": 0.75,
                "conservative_threshold": 0.35,
                "calculated_threshold": 0.6
            },
            "tempo": {
                "early_game_threshold": 0.6,
                "late_game_threshold": 0.65,
                "adaptive_threshold": 0.5
            }
        }
    
    async def analyze_player_behavior(
        self,
        player_id: str,
        match_history: List[Dict[str, Any]],
        video_paths: Optional[List[str]] = None
    ) -> BehavioralFingerprint:
        """
        Analyze player behavior from match history and optional video data.
        
        Args:
            player_id: Unique player identifier
            match_history: List of match data dictionaries
            video_paths: Optional list of video file paths for deep analysis
            
        Returns:
            BehavioralFingerprint with comprehensive behavioral analysis
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = f"behavioral_analysis:{player_id}"
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result:
            logger.info(f"Retrieved cached behavioral analysis for {player_id}")
            return BehavioralFingerprint(**cached_result)
        
        # Validate input data
        if len(match_history) < self.min_matches_for_analysis:
            raise ValueError(f"Insufficient match history. Need at least {self.min_matches_for_analysis} matches.")
        
        # Extract behavioral metrics from match history
        behavioral_metrics = await self._extract_behavioral_metrics(match_history)
        
        # Analyze video data if provided
        video_insights = {}
        if video_paths:
            video_insights = await self._analyze_video_behavior(video_paths, player_id)
        
        # Generate behavioral classifications
        play_style = await self._classify_play_style(behavioral_metrics, video_insights)
        risk_profile = await self._classify_risk_profile(behavioral_metrics, video_insights)
        game_tempo = await self._classify_game_tempo(behavioral_metrics, video_insights)
        
        # Calculate behavioral scores
        scores = await self._calculate_behavioral_scores(behavioral_metrics, video_insights)
        
        # Identify patterns and insights
        behavior_tags = await self._identify_behavior_tags(behavioral_metrics, video_insights)
        identified_flaws = await self._identify_behavioral_flaws(behavioral_metrics, video_insights)
        strength_areas = await self._identify_strength_areas(behavioral_metrics, video_insights)
        
        # Generate behavioral fingerprint
        fingerprint = BehavioralFingerprint(
            player_id=player_id,
            play_style=play_style,
            risk_profile=risk_profile,
            game_tempo=game_tempo,
            map_awareness_score=scores.get("map_awareness", 0.5),
            synergy_with_team=scores.get("team_synergy", 0.5),
            adaptability_score=scores.get("adaptability", 0.5),
            mechanical_skill_score=scores.get("mechanical_skill", 0.5),
            decision_making_score=scores.get("decision_making", 0.5),
            preferred_lane=await self._determine_preferred_lane(match_history),
            preferred_role=await self._determine_preferred_role(match_history),
            preferred_heroes=await self._determine_preferred_heroes(match_history),
            behavior_tags=behavior_tags,
            identified_flaws=identified_flaws,
            strength_areas=strength_areas,
            behavioral_metrics=behavioral_metrics,
            confidence_score=await self._calculate_confidence_score(behavioral_metrics, video_insights),
            analysis_date=datetime.now(),
            matches_analyzed=len(match_history),
            performance_by_time=await self._analyze_temporal_patterns(match_history),
            hero_performance_patterns=await self._analyze_hero_patterns(match_history)
        )
        
        # Cache the result
        await self.cache_manager.set(
            cache_key,
            fingerprint.__dict__,
            ttl=3600  # Cache for 1 hour
        )
        
        # Emit event for monitoring
        await self.event_bus.emit("behavioral_analysis_completed", {
            "player_id": player_id,
            "processing_time": time.time() - start_time,
            "confidence_score": fingerprint.confidence_score,
            "matches_analyzed": len(match_history)
        })
        
        # Log performance metrics
        processing_time = time.time() - start_time
        logger.info(f"Behavioral analysis completed for {player_id} in {processing_time:.2f}s")
        
        return fingerprint
    
    async def _extract_behavioral_metrics(self, match_history: List[Dict[str, Any]]) -> BehavioralMetrics:
        """Extract behavioral metrics from match history."""
        if not match_history:
            return BehavioralMetrics()
        
        # Calculate aggregate metrics
        total_matches = len(match_history)
        
        # Combat behavior metrics
        kills = [m.get("kills", 0) for m in match_history]
        deaths = [max(m.get("deaths", 1), 1) for m in match_history]  # Avoid division by zero
        assists = [m.get("assists", 0) for m in match_history]
        
        kill_participation_rate = sum(kills) / (sum(kills) + sum(assists) + 1)
        death_avoidance_score = 1.0 - (sum(deaths) / (sum(kills) + sum(assists) + sum(deaths)))
        assist_contribution = sum(assists) / (sum(kills) + sum(assists) + 1)
        
        # Economic behavior metrics
        gold_values = [m.get("gold_per_min", 0) for m in match_history]
        farming_efficiency = np.mean(gold_values) / 6000 if gold_values else 0.0  # Normalize to 6k GPM
        gold_per_minute_consistency = 1.0 - (np.std(gold_values) / (np.mean(gold_values) + 1))
        
        # Strategic behavior metrics
        teamfight_participation = [m.get("teamfight_participation", 0) for m in match_history]
        objective_participation = np.mean(teamfight_participation) / 100.0 if teamfight_participation else 0.0
        
        # Positioning behavior metrics
        positioning_ratings = [m.get("positioning_rating", "average") for m in match_history]
        positioning_safety = self._calculate_positioning_score(positioning_ratings)
        
        # Tempo behavior metrics
        match_durations = [m.get("match_duration", 0) for m in match_history]
        early_game_impact = self._calculate_early_game_impact(match_history)
        mid_game_impact = self._calculate_mid_game_impact(match_history)
        late_game_impact = self._calculate_late_game_impact(match_history)
        
        return BehavioralMetrics(
            kill_participation_rate=kill_participation_rate,
            death_avoidance_score=max(0.0, death_avoidance_score),
            assist_contribution=assist_contribution,
            farming_efficiency=min(1.0, farming_efficiency),
            gold_per_minute_consistency=max(0.0, gold_per_minute_consistency),
            objective_participation=objective_participation,
            positioning_safety=positioning_safety,
            early_game_impact=early_game_impact,
            mid_game_impact=mid_game_impact,
            late_game_impact=late_game_impact,
            map_control_score=0.5,  # Placeholder - would need more detailed data
            team_coordination=0.5,  # Placeholder - would need more detailed data
            resource_prioritization=0.5,  # Placeholder - would need more detailed data
            engagement_timing=0.5,  # Placeholder - would need more detailed data
            escape_success_rate=0.5  # Placeholder - would need more detailed data
        )
    
    def _calculate_positioning_score(self, positioning_ratings: List[str]) -> float:
        """Calculate positioning safety score from ratings."""
        if not positioning_ratings:
            return 0.5
        
        score_map = {"low": 0.2, "average": 0.5, "good": 0.8}
        scores = [score_map.get(rating, 0.5) for rating in positioning_ratings]
        return np.mean(scores)
    
    def _calculate_early_game_impact(self, match_history: List[Dict[str, Any]]) -> float:
        """Calculate early game impact score."""
        # Simplified calculation based on KDA ratio in shorter matches
        short_matches = [m for m in match_history if m.get("match_duration", 0) < 15]
        if not short_matches:
            return 0.5
        
        total_impact = 0.0
        for match in short_matches:
            kills = match.get("kills", 0)
            deaths = max(match.get("deaths", 1), 1)
            assists = match.get("assists", 0)
            kda = (kills + assists) / deaths
            total_impact += min(kda / 3.0, 1.0)  # Normalize to 0-1 range
        
        return total_impact / len(short_matches)
    
    def _calculate_mid_game_impact(self, match_history: List[Dict[str, Any]]) -> float:
        """Calculate mid game impact score."""
        # Simplified calculation based on teamfight participation
        teamfight_scores = [m.get("teamfight_participation", 0) for m in match_history]
        return np.mean(teamfight_scores) / 100.0 if teamfight_scores else 0.5
    
    def _calculate_late_game_impact(self, match_history: List[Dict[str, Any]]) -> float:
        """Calculate late game impact score."""
        # Simplified calculation based on damage output in longer matches
        long_matches = [m for m in match_history if m.get("match_duration", 0) > 20]
        if not long_matches:
            return 0.5
        
        total_impact = 0.0
        for match in long_matches:
            hero_damage = match.get("hero_damage", 0)
            # Normalize damage based on match duration
            duration = match.get("match_duration", 1)
            dps = hero_damage / (duration * 60)  # Damage per second
            total_impact += min(dps / 1000, 1.0)  # Normalize to 0-1 range
        
        return total_impact / len(long_matches)
    
    async def _analyze_video_behavior(self, video_paths: List[str], player_id: str) -> Dict[str, Any]:
        """Analyze behavioral patterns from video replay data."""
        video_insights = {
            "positioning_patterns": {},
            "movement_behavior": {},
            "decision_making": {},
            "team_coordination": {}
        }
        
        for video_path in video_paths:
            try:
                # Validate video file
                is_valid, error_msg = self.video_reader.validate_video_file(video_path)
                if not is_valid:
                    logger.warning(f"Invalid video file {video_path}: {error_msg}")
                    continue
                
                # Extract frames for analysis
                frames = self.video_reader.extract_frames(video_path, sample_rate=0.5)  # 0.5 FPS
                
                # Analyze positioning patterns
                positioning_data = await self._analyze_positioning_from_frames(frames)
                video_insights["positioning_patterns"][video_path] = positioning_data
                
                # Analyze movement behavior
                movement_data = await self._analyze_movement_from_frames(frames)
                video_insights["movement_behavior"][video_path] = movement_data
                
                # Clean up temporary frames
                self.video_reader._cleanup_temp_frames()
                
            except Exception as e:
                logger.error(f"Error analyzing video {video_path}: {str(e)}")
                continue
        
        return video_insights
    
    async def _analyze_positioning_from_frames(self, frames: List[str]) -> Dict[str, Any]:
        """Analyze positioning patterns from video frames."""
        # This would involve computer vision analysis of player position
        # For now, return placeholder data
        return {
            "aggressive_positioning": 0.5,
            "safe_positioning": 0.5,
            "flanking_behavior": 0.3,
            "team_positioning": 0.6
        }
    
    async def _analyze_movement_from_frames(self, frames: List[str]) -> Dict[str, Any]:
        """Analyze movement behavior from video frames."""
        # This would involve tracking player movement patterns
        # For now, return placeholder data
        return {
            "roaming_frequency": 0.4,
            "farming_focus": 0.6,
            "objective_priority": 0.5,
            "escape_behavior": 0.7
        }
    
    async def _classify_play_style(self, metrics: BehavioralMetrics, video_insights: Dict[str, Any]) -> PlayStyle:
        """Classify player's play style based on behavioral metrics."""
        thresholds = self.classification_thresholds["play_style"]
        
        # Calculate play style indicators
        aggression_score = (metrics.kill_participation_rate + (1 - metrics.death_avoidance_score)) / 2
        objective_score = metrics.objective_participation
        team_fight_score = metrics.mid_game_impact
        
        # Classify based on thresholds
        if aggression_score > thresholds["aggressive_threshold"]:
            return PlayStyle.AGGRESSIVE_ROAMER
        elif objective_score > thresholds["objective_threshold"]:
            return PlayStyle.OBJECTIVE_FOCUSED
        elif team_fight_score > thresholds["team_fight_threshold"]:
            return PlayStyle.TEAM_FIGHTER
        elif metrics.farming_efficiency > 0.7:
            return PlayStyle.PASSIVE_FARMER
        else:
            return PlayStyle.CARRY_FOCUSED
    
    async def _classify_risk_profile(self, metrics: BehavioralMetrics, video_insights: Dict[str, Any]) -> RiskProfile:
        """Classify player's risk-taking behavior."""
        thresholds = self.classification_thresholds["risk_profile"]
        
        # Calculate risk indicators
        risk_score = (1 - metrics.death_avoidance_score) + (1 - metrics.positioning_safety)
        risk_score = risk_score / 2  # Normalize to 0-1
        
        if risk_score > thresholds["high_risk_threshold"]:
            return RiskProfile.HIGH_RISK_HIGH_REWARD
        elif risk_score < thresholds["conservative_threshold"]:
            return RiskProfile.CONSERVATIVE
        elif risk_score > thresholds["calculated_threshold"]:
            return RiskProfile.CALCULATED_RISK
        else:
            return RiskProfile.OPPORTUNISTIC
    
    async def _classify_game_tempo(self, metrics: BehavioralMetrics, video_insights: Dict[str, Any]) -> GameTempo:
        """Classify player's game tempo preferences."""
        thresholds = self.classification_thresholds["tempo"]
        
        # Calculate tempo indicators
        early_strength = metrics.early_game_impact
        late_strength = metrics.late_game_impact
        
        if early_strength > thresholds["early_game_threshold"]:
            return GameTempo.EARLY_AGGRESSIVE
        elif late_strength > thresholds["late_game_threshold"]:
            return GameTempo.LATE_GAME_SCALING
        elif abs(early_strength - late_strength) < thresholds["adaptive_threshold"]:
            return GameTempo.ADAPTIVE
        else:
            return GameTempo.MID_GAME_FOCUSED
    
    async def _calculate_behavioral_scores(self, metrics: BehavioralMetrics, video_insights: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive behavioral scores."""
        return {
            "map_awareness": min(metrics.positioning_safety + 0.2, 1.0),
            "team_synergy": metrics.objective_participation,
            "adaptability": (metrics.early_game_impact + metrics.mid_game_impact + metrics.late_game_impact) / 3,
            "mechanical_skill": (metrics.kill_participation_rate + metrics.farming_efficiency) / 2,
            "decision_making": (metrics.positioning_safety + metrics.death_avoidance_score) / 2
        }
    
    async def _identify_behavior_tags(self, metrics: BehavioralMetrics, video_insights: Dict[str, Any]) -> List[str]:
        """Identify behavioral tags based on metrics."""
        tags = []
        
        if metrics.kill_participation_rate > 0.7:
            tags.append("high kill focus")
        if metrics.objective_participation < 0.4:
            tags.append("ignores objectives")
        if metrics.early_game_impact > 0.6:
            tags.append("early roams")
        if metrics.farming_efficiency > 0.8:
            tags.append("efficient farmer")
        if metrics.positioning_safety < 0.4:
            tags.append("aggressive positioning")
        if metrics.death_avoidance_score < 0.3:
            tags.append("high death rate")
        
        return tags
    
    async def _identify_behavioral_flaws(self, metrics: BehavioralMetrics, video_insights: Dict[str, Any]) -> List[str]:
        """Identify behavioral flaws that need improvement."""
        flaws = []
        
        if metrics.positioning_safety < 0.4:
            flaws.append("overextending")
        if metrics.death_avoidance_score < 0.3:
            flaws.append("poor survival instincts")
        if metrics.objective_participation < 0.3:
            flaws.append("low objective participation")
        if metrics.farming_efficiency < 0.4:
            flaws.append("inefficient farming")
        if metrics.gold_per_minute_consistency < 0.5:
            flaws.append("inconsistent economy")
        
        return flaws
    
    async def _identify_strength_areas(self, metrics: BehavioralMetrics, video_insights: Dict[str, Any]) -> List[str]:
        """Identify behavioral strengths."""
        strengths = []
        
        if metrics.kill_participation_rate > 0.7:
            strengths.append("strong team fighting")
        if metrics.positioning_safety > 0.7:
            strengths.append("excellent positioning")
        if metrics.farming_efficiency > 0.7:
            strengths.append("efficient resource management")
        if metrics.objective_participation > 0.7:
            strengths.append("strong objective focus")
        if metrics.death_avoidance_score > 0.7:
            strengths.append("excellent survival skills")
        
        return strengths
    
    async def _determine_preferred_lane(self, match_history: List[Dict[str, Any]]) -> str:
        """Determine player's preferred lane based on hero choices."""
        # This would analyze hero picks and infer lane preference
        # For now, return a placeholder
        return "gold"
    
    async def _determine_preferred_role(self, match_history: List[Dict[str, Any]]) -> str:
        """Determine player's preferred role based on hero choices."""
        # This would analyze hero picks and infer role preference
        # For now, return a placeholder
        return "marksman"
    
    async def _determine_preferred_heroes(self, match_history: List[Dict[str, Any]]) -> List[str]:
        """Determine player's preferred heroes based on frequency."""
        hero_counts = {}
        for match in match_history:
            hero = match.get("hero", "unknown")
            hero_counts[hero] = hero_counts.get(hero, 0) + 1
        
        # Return top 3 most played heroes
        sorted_heroes = sorted(hero_counts.items(), key=lambda x: x[1], reverse=True)
        return [hero for hero, count in sorted_heroes[:3]]
    
    async def _analyze_temporal_patterns(self, match_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze performance patterns over time."""
        # This would analyze performance trends over time
        # For now, return placeholder data
        return {
            "morning_performance": 0.6,
            "afternoon_performance": 0.7,
            "evening_performance": 0.8,
            "night_performance": 0.5
        }
    
    async def _analyze_hero_patterns(self, match_history: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze performance patterns for different heroes."""
        hero_patterns = {}
        
        for match in match_history:
            hero = match.get("hero", "unknown")
            if hero not in hero_patterns:
                hero_patterns[hero] = {
                    "wins": 0,
                    "total_matches": 0,
                    "avg_kda": 0.0,
                    "avg_damage": 0.0
                }
            
            hero_patterns[hero]["total_matches"] += 1
            
            # Calculate KDA
            kills = match.get("kills", 0)
            deaths = max(match.get("deaths", 1), 1)
            assists = match.get("assists", 0)
            kda = (kills + assists) / deaths
            
            hero_patterns[hero]["avg_kda"] = (
                (hero_patterns[hero]["avg_kda"] * (hero_patterns[hero]["total_matches"] - 1) + kda) /
                hero_patterns[hero]["total_matches"]
            )
        
        return hero_patterns
    
    async def _calculate_confidence_score(self, metrics: BehavioralMetrics, video_insights: Dict[str, Any]) -> float:
        """Calculate confidence score for the behavioral analysis."""
        # Base confidence on amount of data and consistency
        base_confidence = 0.7
        
        # Add confidence based on video data availability
        if video_insights and any(video_insights.values()):
            base_confidence += 0.2
        
        # Adjust based on data consistency
        consistency_bonus = (metrics.gold_per_minute_consistency + 
                           metrics.death_avoidance_score + 
                           metrics.positioning_safety) / 3 * 0.1
        
        return min(base_confidence + consistency_bonus, 1.0)


# Factory function for easy integration
async def create_behavioral_analyzer(cache_manager: Optional[HybridCache] = None) -> BehavioralAnalyzer:
    """Create a behavioral analyzer instance."""
    return BehavioralAnalyzer(cache_manager)


# Export main classes
__all__ = [
    "BehavioralAnalyzer",
    "BehavioralFingerprint",
    "BehavioralMetrics",
    "PlayStyle",
    "RiskProfile",
    "GameTempo",
    "create_behavioral_analyzer"
] 