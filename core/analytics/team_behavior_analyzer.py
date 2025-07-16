"""
Team Behavior Analyzer for MLBB Coach AI
========================================

This module provides comprehensive team behavior analysis capabilities that analyze
a full 5-player squad in a single match to identify patterns of team synergy,
coordination breakdowns, role overlap, timing mismatches, and team comp dynamics.

Key Features:
- Team synergy pattern analysis
- Coordination breakdown identification  
- Role overlap detection
- Timing mismatch analysis
- Team composition dynamics
- Objective control pattern analysis
- Rotation synchronization assessment
- Positional spacing in teamfights
- Role coverage and gap analysis
- Player compatibility matrix generation
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np
from collections import defaultdict
import json

# Import existing system components
from ..behavioral_modeling import BehavioralFingerprint, BehavioralAnalyzer
from ..schemas import AnyMatch, BaseMatch
from ..cache.hybrid_cache import HybridCache
from ..events.event_bus import EventBus
from ..diagnostic_logger import diagnostic_logger

logger = logging.getLogger(__name__)


class TeamSynergyLevel(Enum):
    """Team synergy level classifications."""
    EXCEPTIONAL = "exceptional"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    DYSFUNCTIONAL = "dysfunctional"


class CoordinationPattern(Enum):
    """Team coordination patterns."""
    HIGHLY_COORDINATED = "highly_coordinated"
    MODERATELY_COORDINATED = "moderately_coordinated"
    LOOSELY_COORDINATED = "loosely_coordinated"
    UNCOORDINATED = "uncoordinated"


class RoleOverlapSeverity(Enum):
    """Role overlap severity levels."""
    NO_OVERLAP = "no_overlap"
    MINOR_OVERLAP = "minor_overlap"
    MODERATE_OVERLAP = "moderate_overlap"
    MAJOR_OVERLAP = "major_overlap"
    CRITICAL_OVERLAP = "critical_overlap"


@dataclass
class TeamfightSpacing:
    """Represents teamfight positioning and spacing analysis."""
    average_spread: float = 0.0
    frontline_backline_separation: float = 0.0
    flanking_effectiveness: float = 0.0
    positioning_discipline: float = 0.0
    spacing_score: float = 0.0


@dataclass
class ObjectiveControl:
    """Represents objective control patterns."""
    lord_control_rate: float = 0.0
    turtle_control_rate: float = 0.0
    tower_push_coordination: float = 0.0
    jungle_control_efficiency: float = 0.0
    vision_control_score: float = 0.0
    objective_timing_score: float = 0.0


@dataclass
class RotationSync:
    """Represents rotation synchronization patterns."""
    lane_rotation_timing: float = 0.0
    gank_coordination_score: float = 0.0
    recall_synchronization: float = 0.0
    team_movement_cohesion: float = 0.0
    rotation_efficiency: float = 0.0


@dataclass
class RoleAnalysis:
    """Represents role coverage and gap analysis."""
    role_distribution: Dict[str, int] = field(default_factory=dict)
    coverage_gaps: List[str] = field(default_factory=list)
    overlap_conflicts: List[str] = field(default_factory=list)
    role_effectiveness: Dict[str, float] = field(default_factory=dict)
    synergy_score: float = 0.0


@dataclass
class PlayerCompatibility:
    """Represents player compatibility analysis."""
    player_a: str
    player_b: str
    compatibility_score: float
    synergy_factors: List[str] = field(default_factory=list)
    conflict_factors: List[str] = field(default_factory=list)
    playstyle_alignment: float = 0.0
    coordination_rating: float = 0.0


@dataclass
class TeamBehaviorInsight:
    """Represents a team behavior insight."""
    category: str
    insight: str
    confidence: float
    severity: str
    suggestions: List[str] = field(default_factory=list)
    affected_players: List[str] = field(default_factory=list)
    timestamp_range: Optional[Tuple[float, float]] = None


@dataclass
class TeamBehaviorAnalysis:
    """Complete team behavior analysis result."""
    match_id: str
    team_id: str
    analysis_timestamp: datetime
    
    # Core analysis results
    synergy_level: TeamSynergyLevel
    coordination_pattern: CoordinationPattern
    role_overlap_severity: RoleOverlapSeverity
    
    # Detailed analysis
    teamfight_spacing: TeamfightSpacing
    objective_control: ObjectiveControl
    rotation_sync: RotationSync
    role_analysis: RoleAnalysis
    
    # Player compatibility matrix
    compatibility_matrix: List[PlayerCompatibility]
    
    # Insights and feedback
    team_insights: List[TeamBehaviorInsight]
    collective_feedback: str
    
    # Performance metrics
    team_coordination_score: float
    overall_synergy_score: float
    team_effectiveness_rating: float
    
    # Confidence and metadata
    confidence_score: float
    players_analyzed: int
    analysis_duration: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "match_id": self.match_id,
            "team_id": self.team_id,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "synergy_level": self.synergy_level.value,
            "coordination_pattern": self.coordination_pattern.value,
            "role_overlap_severity": self.role_overlap_severity.value,
            "teamfight_spacing": {
                "average_spread": self.teamfight_spacing.average_spread,
                "frontline_backline_separation": self.teamfight_spacing.frontline_backline_separation,
                "flanking_effectiveness": self.teamfight_spacing.flanking_effectiveness,
                "positioning_discipline": self.teamfight_spacing.positioning_discipline,
                "spacing_score": self.teamfight_spacing.spacing_score
            },
            "objective_control": {
                "lord_control_rate": self.objective_control.lord_control_rate,
                "turtle_control_rate": self.objective_control.turtle_control_rate,
                "tower_push_coordination": self.objective_control.tower_push_coordination,
                "jungle_control_efficiency": self.objective_control.jungle_control_efficiency,
                "vision_control_score": self.objective_control.vision_control_score,
                "objective_timing_score": self.objective_control.objective_timing_score
            },
            "rotation_sync": {
                "lane_rotation_timing": self.rotation_sync.lane_rotation_timing,
                "gank_coordination_score": self.rotation_sync.gank_coordination_score,
                "recall_synchronization": self.rotation_sync.recall_synchronization,
                "team_movement_cohesion": self.rotation_sync.team_movement_cohesion,
                "rotation_efficiency": self.rotation_sync.rotation_efficiency
            },
            "role_analysis": {
                "role_distribution": self.role_analysis.role_distribution,
                "coverage_gaps": self.role_analysis.coverage_gaps,
                "overlap_conflicts": self.role_analysis.overlap_conflicts,
                "role_effectiveness": self.role_analysis.role_effectiveness,
                "synergy_score": self.role_analysis.synergy_score
            },
            "compatibility_matrix": [
                {
                    "player_a": comp.player_a,
                    "player_b": comp.player_b,
                    "compatibility_score": comp.compatibility_score,
                    "synergy_factors": comp.synergy_factors,
                    "conflict_factors": comp.conflict_factors,
                    "playstyle_alignment": comp.playstyle_alignment,
                    "coordination_rating": comp.coordination_rating
                }
                for comp in self.compatibility_matrix
            ],
            "team_insights": [
                {
                    "category": insight.category,
                    "insight": insight.insight,
                    "confidence": insight.confidence,
                    "severity": insight.severity,
                    "suggestions": insight.suggestions,
                    "affected_players": insight.affected_players,
                    "timestamp_range": insight.timestamp_range
                }
                for insight in self.team_insights
            ],
            "collective_feedback": self.collective_feedback,
            "team_coordination_score": self.team_coordination_score,
            "overall_synergy_score": self.overall_synergy_score,
            "team_effectiveness_rating": self.team_effectiveness_rating,
            "confidence_score": self.confidence_score,
            "players_analyzed": self.players_analyzed,
            "analysis_duration": self.analysis_duration
        }


class TeamBehaviorAnalyzer:
    """
    Analyzes team behavior patterns for a full 5-player squad.
    
    This analyzer provides comprehensive team-level insights including:
    - Team synergy and coordination patterns
    - Role distribution and overlap analysis
    - Timing and synchronization assessment
    - Team composition dynamics
    - Player compatibility matrix
    """
    
    def __init__(self, cache_manager: Optional[HybridCache] = None):
        """
        Initialize the team behavior analyzer.
        
        Args:
            cache_manager: Optional cache manager for performance optimization
        """
        self.cache_manager = cache_manager
        self.behavioral_analyzer = None
        
        # Load configuration and thresholds
        self.synergy_thresholds = self._load_synergy_thresholds()
        self.coordination_thresholds = self._load_coordination_thresholds()
        self.role_definitions = self._load_role_definitions()
        
        logger.info("Team Behavior Analyzer initialized")
    
    def _load_synergy_thresholds(self) -> Dict[str, float]:
        """Load synergy analysis thresholds."""
        return {
            "exceptional_synergy": 0.85,
            "strong_synergy": 0.70,
            "moderate_synergy": 0.50,
            "weak_synergy": 0.30,
            "coordination_threshold": 0.60,
            "role_overlap_threshold": 0.40,
            "timing_sync_threshold": 0.55
        }
    
    def _load_coordination_thresholds(self) -> Dict[str, float]:
        """Load coordination analysis thresholds."""
        return {
            "highly_coordinated": 0.80,
            "moderately_coordinated": 0.60,
            "loosely_coordinated": 0.40,
            "rotation_sync_threshold": 0.65,
            "objective_control_threshold": 0.70,
            "teamfight_spacing_threshold": 0.60
        }
    
    def _load_role_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Load role definitions and synergy patterns."""
        return {
            "tank": {
                "primary_stats": ["teamfight_participation", "damage_taken", "positioning_rating"],
                "synergy_roles": ["support", "marksman"],
                "conflict_roles": ["tank"],
                "positioning": "frontline"
            },
            "fighter": {
                "primary_stats": ["kills", "turret_damage", "positioning_rating"],
                "synergy_roles": ["tank", "assassin"],
                "conflict_roles": ["fighter"],
                "positioning": "frontline"
            },
            "assassin": {
                "primary_stats": ["kills", "hero_damage", "positioning_rating"],
                "synergy_roles": ["support", "tank"],
                "conflict_roles": ["assassin"],
                "positioning": "flanking"
            },
            "mage": {
                "primary_stats": ["hero_damage", "assists", "positioning_rating"],
                "synergy_roles": ["tank", "support"],
                "conflict_roles": ["mage"],
                "positioning": "backline"
            },
            "marksman": {
                "primary_stats": ["hero_damage", "gold_per_min", "positioning_rating"],
                "synergy_roles": ["tank", "support"],
                "conflict_roles": ["marksman"],
                "positioning": "backline"
            },
            "support": {
                "primary_stats": ["assists", "teamfight_participation", "positioning_rating"],
                "synergy_roles": ["marksman", "mage"],
                "conflict_roles": ["support"],
                "positioning": "midline"
            }
        }
    
    async def analyze_team_behavior(
        self,
        match_data: List[Dict[str, Any]],
        behavioral_profiles: Optional[List[BehavioralFingerprint]] = None,
        match_id: str = "unknown",
        team_id: str = "team_1"
    ) -> TeamBehaviorAnalysis:
        """
        Analyze team behavior patterns for a 5-player squad.
        
        Args:
            match_data: List of match data for all 5 players
            behavioral_profiles: Optional pre-computed behavioral profiles
            match_id: Match identifier
            team_id: Team identifier
            
        Returns:
            TeamBehaviorAnalysis with comprehensive team insights
        """
        start_time = datetime.now()
        
        try:
            # Validate input data
            if len(match_data) != 5:
                raise ValueError(f"Expected 5 players, got {len(match_data)}")
            
            # Generate behavioral profiles if not provided
            if not behavioral_profiles:
                behavioral_profiles = await self._generate_behavioral_profiles(match_data)
            
            # Analyze team synergy patterns
            synergy_analysis = await self._analyze_team_synergy(match_data, behavioral_profiles)
            
            # Analyze coordination patterns
            coordination_analysis = await self._analyze_coordination_patterns(match_data, behavioral_profiles)
            
            # Analyze role distribution and overlap
            role_analysis = await self._analyze_role_patterns(match_data, behavioral_profiles)
            
            # Analyze teamfight spacing
            spacing_analysis = await self._analyze_teamfight_spacing(match_data, behavioral_profiles)
            
            # Analyze objective control
            objective_analysis = await self._analyze_objective_control(match_data, behavioral_profiles)
            
            # Analyze rotation synchronization
            rotation_analysis = await self._analyze_rotation_sync(match_data, behavioral_profiles)
            
            # Build player compatibility matrix
            compatibility_matrix = await self._build_compatibility_matrix(behavioral_profiles)
            
            # Generate team insights
            team_insights = await self._generate_team_insights(
                match_data, behavioral_profiles, synergy_analysis, coordination_analysis, role_analysis
            )
            
            # Generate collective feedback
            collective_feedback = await self._generate_collective_feedback(
                team_insights, synergy_analysis, coordination_analysis
            )
            
            # Calculate overall scores
            team_coordination_score = self._calculate_team_coordination_score(coordination_analysis)
            overall_synergy_score = self._calculate_overall_synergy_score(synergy_analysis)
            team_effectiveness_rating = self._calculate_team_effectiveness_rating(
                team_coordination_score, overall_synergy_score, role_analysis
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(match_data, behavioral_profiles)
            
            # Determine classifications
            synergy_level = self._classify_synergy_level(overall_synergy_score)
            coordination_pattern = self._classify_coordination_pattern(team_coordination_score)
            role_overlap_severity = self._classify_role_overlap_severity(role_analysis)
            
            # Create final analysis result
            analysis_duration = (datetime.now() - start_time).total_seconds()
            
            return TeamBehaviorAnalysis(
                match_id=match_id,
                team_id=team_id,
                analysis_timestamp=datetime.now(),
                synergy_level=synergy_level,
                coordination_pattern=coordination_pattern,
                role_overlap_severity=role_overlap_severity,
                teamfight_spacing=spacing_analysis,
                objective_control=objective_analysis,
                rotation_sync=rotation_analysis,
                role_analysis=role_analysis,
                compatibility_matrix=compatibility_matrix,
                team_insights=team_insights,
                collective_feedback=collective_feedback,
                team_coordination_score=team_coordination_score,
                overall_synergy_score=overall_synergy_score,
                team_effectiveness_rating=team_effectiveness_rating,
                confidence_score=confidence_score,
                players_analyzed=len(match_data),
                analysis_duration=analysis_duration
            )
            
        except Exception as e:
            logger.error(f"Team behavior analysis failed: {str(e)}")
            raise
    
    async def _generate_behavioral_profiles(self, match_data: List[Dict[str, Any]]) -> List[BehavioralFingerprint]:
        """Generate behavioral profiles for all players."""
        if not self.behavioral_analyzer:
            from ..behavioral_modeling import create_behavioral_analyzer
            self.behavioral_analyzer = await create_behavioral_analyzer(self.cache_manager)
        
        profiles = []
        for i, player_data in enumerate(match_data):
            player_id = player_data.get('player_id', f'player_{i}')
            profile = await self.behavioral_analyzer.analyze_player_behavior(
                player_id=player_id,
                match_history=[player_data]
            )
            profiles.append(profile)
        
        return profiles
    
    async def _analyze_team_synergy(
        self, match_data: List[Dict[str, Any]], profiles: List[BehavioralFingerprint]
    ) -> Dict[str, Any]:
        """Analyze team synergy patterns."""
        synergy_factors = []
        
        # Analyze playstyle compatibility
        playstyle_synergy = self._analyze_playstyle_synergy(profiles)
        synergy_factors.append(("playstyle_compatibility", playstyle_synergy))
        
        # Analyze tempo alignment
        tempo_alignment = self._analyze_tempo_alignment(profiles)
        synergy_factors.append(("tempo_alignment", tempo_alignment))
        
        # Analyze risk profile balance
        risk_balance = self._analyze_risk_profile_balance(profiles)
        synergy_factors.append(("risk_balance", risk_balance))
        
        # Analyze performance consistency
        performance_consistency = self._analyze_performance_consistency(match_data)
        synergy_factors.append(("performance_consistency", performance_consistency))
        
        return {
            "synergy_factors": synergy_factors,
            "overall_synergy": np.mean([score for _, score in synergy_factors]),
            "synergy_breakdown": {factor: score for factor, score in synergy_factors}
        }
    
    async def _analyze_coordination_patterns(
        self, match_data: List[Dict[str, Any]], profiles: List[BehavioralFingerprint]
    ) -> Dict[str, Any]:
        """Analyze team coordination patterns."""
        coordination_metrics = []
        
        # Analyze team fight participation synchronization
        teamfight_sync = self._analyze_teamfight_synchronization(match_data)
        coordination_metrics.append(("teamfight_sync", teamfight_sync))
        
        # Analyze objective timing coordination
        objective_timing = self._analyze_objective_timing_coordination(match_data)
        coordination_metrics.append(("objective_timing", objective_timing))
        
        # Analyze rotation coordination
        rotation_coordination = self._analyze_rotation_coordination(match_data, profiles)
        coordination_metrics.append(("rotation_coordination", rotation_coordination))
        
        # Analyze decision-making alignment
        decision_alignment = self._analyze_decision_alignment(profiles)
        coordination_metrics.append(("decision_alignment", decision_alignment))
        
        return {
            "coordination_metrics": coordination_metrics,
            "overall_coordination": np.mean([score for _, score in coordination_metrics]),
            "coordination_breakdown": {metric: score for metric, score in coordination_metrics}
        }
    
    async def _analyze_role_patterns(
        self, match_data: List[Dict[str, Any]], profiles: List[BehavioralFingerprint]
    ) -> RoleAnalysis:
        """Analyze role distribution and overlap patterns."""
        # Extract role information
        roles = [self._determine_player_role(data, profile) for data, profile in zip(match_data, profiles)]
        
        # Calculate role distribution
        role_distribution = {}
        for role in roles:
            role_distribution[role] = role_distribution.get(role, 0) + 1
        
        # Identify coverage gaps
        expected_roles = {"tank", "fighter", "assassin", "mage", "marksman", "support"}
        present_roles = set(roles)
        coverage_gaps = list(expected_roles - present_roles)
        
        # Identify overlap conflicts
        overlap_conflicts = []
        for role, count in role_distribution.items():
            if count > 1 and role in ["tank", "marksman", "support"]:
                overlap_conflicts.append(f"Multiple {role}s detected ({count})")
        
        # Calculate role effectiveness
        role_effectiveness = {}
        for i, (role, data) in enumerate(zip(roles, match_data)):
            effectiveness = self._calculate_role_effectiveness(role, data, profiles[i])
            role_effectiveness[f"{role}_{i}"] = effectiveness
        
        # Calculate synergy score
        synergy_score = self._calculate_role_synergy_score(roles, profiles)
        
        return RoleAnalysis(
            role_distribution=role_distribution,
            coverage_gaps=coverage_gaps,
            overlap_conflicts=overlap_conflicts,
            role_effectiveness=role_effectiveness,
            synergy_score=synergy_score
        )
    
    async def _analyze_teamfight_spacing(
        self, match_data: List[Dict[str, Any]], profiles: List[BehavioralFingerprint]
    ) -> TeamfightSpacing:
        """Analyze teamfight positioning and spacing."""
        positioning_ratings = [data.get('positioning_rating', 'average') for data in match_data]
        
        # Calculate average spread (simulated based on positioning ratings)
        positioning_scores = [self._position_rating_to_score(rating) for rating in positioning_ratings]
        average_spread = np.std(positioning_scores) * 0.6  # Normalized spread
        
        # Analyze frontline-backline separation
        frontline_players = sum(1 for p in profiles if p.preferred_role in ["tank", "fighter"])
        backline_players = sum(1 for p in profiles if p.preferred_role in ["mage", "marksman"])
        frontline_backline_separation = min(frontline_players, backline_players) / 2.5
        
        # Calculate flanking effectiveness
        flanking_players = sum(1 for p in profiles if p.preferred_role == "assassin")
        flanking_effectiveness = min(flanking_players * 0.8, 1.0)
        
        # Calculate positioning discipline
        positioning_discipline = np.mean(positioning_scores)
        
        # Overall spacing score
        spacing_score = (average_spread + frontline_backline_separation + 
                        flanking_effectiveness + positioning_discipline) / 4
        
        return TeamfightSpacing(
            average_spread=average_spread,
            frontline_backline_separation=frontline_backline_separation,
            flanking_effectiveness=flanking_effectiveness,
            positioning_discipline=positioning_discipline,
            spacing_score=spacing_score
        )
    
    async def _analyze_objective_control(
        self, match_data: List[Dict[str, Any]], profiles: List[BehavioralFingerprint]
    ) -> ObjectiveControl:
        """Analyze objective control patterns."""
        # Calculate objective control metrics based on team stats
        total_teamfight_participation = sum(data.get('teamfight_participation', 0) for data in match_data)
        avg_teamfight_participation = total_teamfight_participation / len(match_data)
        
        # Simulate objective control rates based on team performance
        lord_control_rate = min(avg_teamfight_participation / 100.0, 1.0)
        turtle_control_rate = min((avg_teamfight_participation + 10) / 100.0, 1.0)
        
        # Calculate tower push coordination
        total_turret_damage = sum(data.get('turret_damage', 0) for data in match_data)
        tower_push_coordination = min(total_turret_damage / 50000, 1.0)
        
        # Calculate jungle control efficiency
        total_gold_per_min = sum(data.get('gold_per_min', 0) for data in match_data)
        jungle_control_efficiency = min(total_gold_per_min / 20000, 1.0)
        
        # Calculate vision control score (simulated)
        vision_control_score = min(avg_teamfight_participation / 90.0, 1.0)
        
        # Calculate objective timing score
        objective_timing_score = (lord_control_rate + turtle_control_rate + 
                                tower_push_coordination) / 3
        
        return ObjectiveControl(
            lord_control_rate=lord_control_rate,
            turtle_control_rate=turtle_control_rate,
            tower_push_coordination=tower_push_coordination,
            jungle_control_efficiency=jungle_control_efficiency,
            vision_control_score=vision_control_score,
            objective_timing_score=objective_timing_score
        )
    
    async def _analyze_rotation_sync(
        self, match_data: List[Dict[str, Any]], profiles: List[BehavioralFingerprint]
    ) -> RotationSync:
        """Analyze rotation synchronization patterns."""
        # Calculate rotation metrics based on behavioral patterns
        roaming_players = sum(1 for p in profiles if "roaming" in p.behavior_tags)
        
        # Lane rotation timing (based on roaming behavior)
        lane_rotation_timing = min(roaming_players / 3.0, 1.0)
        
        # Gank coordination score
        avg_assists = sum(data.get('assists', 0) for data in match_data) / len(match_data)
        gank_coordination_score = min(avg_assists / 15.0, 1.0)
        
        # Recall synchronization (simulated based on team coordination)
        recall_synchronization = min(lane_rotation_timing + 0.2, 1.0)
        
        # Team movement cohesion
        team_movement_cohesion = (lane_rotation_timing + gank_coordination_score) / 2
        
        # Overall rotation efficiency
        rotation_efficiency = (lane_rotation_timing + gank_coordination_score + 
                             recall_synchronization + team_movement_cohesion) / 4
        
        return RotationSync(
            lane_rotation_timing=lane_rotation_timing,
            gank_coordination_score=gank_coordination_score,
            recall_synchronization=recall_synchronization,
            team_movement_cohesion=team_movement_cohesion,
            rotation_efficiency=rotation_efficiency
        )
    
    async def _build_compatibility_matrix(
        self, profiles: List[BehavioralFingerprint]
    ) -> List[PlayerCompatibility]:
        """Build player compatibility matrix."""
        compatibility_matrix = []
        
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                compatibility = self._calculate_player_compatibility(profiles[i], profiles[j])
                compatibility_matrix.append(compatibility)
        
        return compatibility_matrix
    
    async def _generate_team_insights(
        self, match_data: List[Dict[str, Any]], profiles: List[BehavioralFingerprint],
        synergy_analysis: Dict[str, Any], coordination_analysis: Dict[str, Any],
        role_analysis: RoleAnalysis
    ) -> List[TeamBehaviorInsight]:
        """Generate team behavior insights."""
        insights = []
        
        # Analyze synergy insights
        if synergy_analysis["overall_synergy"] < self.synergy_thresholds["weak_synergy"]:
            insights.append(TeamBehaviorInsight(
                category="Team Synergy",
                insight="Team shows poor synergy with misaligned playstyles and conflicting approaches",
                confidence=0.85,
                severity="high",
                suggestions=[
                    "Focus on unified team objectives",
                    "Improve communication during team fights",
                    "Consider role swaps for better synergy"
                ],
                affected_players=[p.player_id for p in profiles]
            ))
        
        # Analyze coordination insights
        if coordination_analysis["overall_coordination"] < self.coordination_thresholds["loosely_coordinated"]:
            insights.append(TeamBehaviorInsight(
                category="Team Coordination",
                insight="Team coordination is lacking with poor timing and decision-making alignment",
                confidence=0.80,
                severity="high",
                suggestions=[
                    "Practice team rotation drills",
                    "Improve objective timing communication",
                    "Establish clear shot-calling hierarchy"
                ],
                affected_players=[p.player_id for p in profiles]
            ))
        
        # Analyze role overlap insights
        if role_analysis.overlap_conflicts:
            insights.append(TeamBehaviorInsight(
                category="Role Distribution",
                insight=f"Role overlap detected: {', '.join(role_analysis.overlap_conflicts)}",
                confidence=0.90,
                severity="medium",
                suggestions=[
                    "Redistribute roles for better coverage",
                    "Consider flex picks for better adaptation",
                    "Clarify role responsibilities"
                ],
                affected_players=[p.player_id for p in profiles if p.preferred_role in role_analysis.overlap_conflicts]
            ))
        
        # Analyze individual player impacts
        for i, (profile, data) in enumerate(zip(profiles, match_data)):
            if profile.confidence_score < 0.6:
                insights.append(TeamBehaviorInsight(
                    category="Individual Performance",
                    insight=f"Player {profile.player_id} shows inconsistent performance patterns",
                    confidence=0.75,
                    severity="medium",
                    suggestions=[
                        "Focus on consistent farming patterns",
                        "Improve positioning discipline",
                        "Work on decision-making timing"
                    ],
                    affected_players=[profile.player_id]
                ))
        
        return insights
    
    async def _generate_collective_feedback(
        self, insights: List[TeamBehaviorInsight], synergy_analysis: Dict[str, Any],
        coordination_analysis: Dict[str, Any]
    ) -> str:
        """Generate collective team feedback."""
        feedback_parts = []
        
        # Team overview
        if synergy_analysis["overall_synergy"] > self.synergy_thresholds["strong_synergy"]:
            feedback_parts.append("🌟 **Team Overview**: Your team demonstrates excellent synergy with complementary playstyles and strong coordination.")
        elif synergy_analysis["overall_synergy"] > self.synergy_thresholds["moderate_synergy"]:
            feedback_parts.append("⚖️ **Team Overview**: Your team shows moderate synergy with room for improvement in coordination and timing.")
        else:
            feedback_parts.append("⚠️ **Team Overview**: Your team needs significant work on synergy and coordination. Focus on unified objectives and better communication.")
        
        # Key strengths
        strengths = []
        if coordination_analysis["coordination_breakdown"]["teamfight_sync"] > 0.7:
            strengths.append("excellent teamfight coordination")
        if coordination_analysis["coordination_breakdown"]["objective_timing"] > 0.7:
            strengths.append("strong objective control")
        if coordination_analysis["coordination_breakdown"]["rotation_coordination"] > 0.7:
            strengths.append("effective rotation patterns")
        
        if strengths:
            feedback_parts.append(f"✅ **Key Strengths**: Your team excels in {', '.join(strengths)}.")
        
        # Priority improvements
        priority_issues = [insight for insight in insights if insight.severity in ["high", "critical"]]
        if priority_issues:
            feedback_parts.append(f"🎯 **Priority Focus**: {priority_issues[0].insight}")
            feedback_parts.append(f"💡 **Immediate Actions**: {', '.join(priority_issues[0].suggestions[:2])}")
        
        # Collective recommendations
        feedback_parts.append("📈 **Collective Recommendations**: Practice team rotations, improve communication timing, and focus on unified objective control.")
        
        return "\n\n".join(feedback_parts)
    
    # Helper methods for analysis calculations
    def _analyze_playstyle_synergy(self, profiles: List[BehavioralFingerprint]) -> float:
        """Analyze playstyle compatibility."""
        play_styles = [p.play_style for p in profiles]
        
        # Define synergy matrix for play styles
        synergy_matrix = {
            ("aggressive-roamer", "support-oriented"): 0.8,
            ("objective-focused", "team-fighter"): 0.9,
            ("carry-focused", "support-oriented"): 0.85,
            ("assassin-style", "team-fighter"): 0.7,
            ("split-pusher", "objective-focused"): 0.6,
        }
        
        total_synergy = 0
        pairs = 0
        
        for i in range(len(play_styles)):
            for j in range(i + 1, len(play_styles)):
                key = (play_styles[i].value, play_styles[j].value)
                reverse_key = (play_styles[j].value, play_styles[i].value)
                
                synergy = synergy_matrix.get(key, synergy_matrix.get(reverse_key, 0.5))
                total_synergy += synergy
                pairs += 1
        
        return total_synergy / pairs if pairs > 0 else 0.5
    
    def _analyze_tempo_alignment(self, profiles: List[BehavioralFingerprint]) -> float:
        """Analyze game tempo alignment."""
        tempo_preferences = [p.game_tempo for p in profiles]
        
        # Calculate tempo alignment score
        tempo_counts = {}
        for tempo in tempo_preferences:
            tempo_counts[tempo] = tempo_counts.get(tempo, 0) + 1
        
        # Higher alignment score if players have similar tempo preferences
        max_alignment = max(tempo_counts.values())
        alignment_score = max_alignment / len(profiles)
        
        return alignment_score
    
    def _analyze_risk_profile_balance(self, profiles: List[BehavioralFingerprint]) -> float:
        """Analyze risk profile balance."""
        risk_profiles = [p.risk_profile for p in profiles]
        
        # Ideal balance: mix of calculated risk and conservative players
        risk_counts = {}
        for risk in risk_profiles:
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        # Balance score based on having mix of risk profiles
        balance_score = 0.5
        if "calculated-risk" in risk_counts and risk_counts["calculated-risk"] >= 2:
            balance_score += 0.3
        if "conservative" in risk_counts and risk_counts["conservative"] >= 1:
            balance_score += 0.2
        
        return min(balance_score, 1.0)
    
    def _analyze_performance_consistency(self, match_data: List[Dict[str, Any]]) -> float:
        """Analyze performance consistency across players."""
        kda_scores = []
        for data in match_data:
            kills = data.get('kills', 0)
            deaths = data.get('deaths', 1)
            assists = data.get('assists', 0)
            kda = (kills + assists) / deaths
            kda_scores.append(kda)
        
        # Lower standard deviation indicates better consistency
        std_dev = np.std(kda_scores)
        consistency_score = max(0, 1 - (std_dev / 5))  # Normalize by expected range
        
        return consistency_score
    
    def _analyze_teamfight_synchronization(self, match_data: List[Dict[str, Any]]) -> float:
        """Analyze teamfight participation synchronization."""
        teamfight_participations = [data.get('teamfight_participation', 0) for data in match_data]
        
        # Higher synchronization if all players have similar participation rates
        avg_participation = np.mean(teamfight_participations)
        std_participation = np.std(teamfight_participations)
        
        # Normalize synchronization score
        sync_score = max(0, 1 - (std_participation / 30))  # 30% std dev threshold
        
        return sync_score
    
    def _analyze_objective_timing_coordination(self, match_data: List[Dict[str, Any]]) -> float:
        """Analyze objective timing coordination."""
        # Use turret damage as proxy for objective coordination
        turret_damages = [data.get('turret_damage', 0) for data in match_data]
        
        # Good coordination if multiple players contribute to objectives
        contributing_players = sum(1 for damage in turret_damages if damage > 5000)
        coordination_score = min(contributing_players / 3, 1.0)
        
        return coordination_score
    
    def _analyze_rotation_coordination(
        self, match_data: List[Dict[str, Any]], profiles: List[BehavioralFingerprint]
    ) -> float:
        """Analyze rotation coordination patterns."""
        # Use assists as proxy for rotation coordination
        assists = [data.get('assists', 0) for data in match_data]
        avg_assists = np.mean(assists)
        
        # Higher average assists indicate better rotation coordination
        rotation_score = min(avg_assists / 12, 1.0)
        
        return rotation_score
    
    def _analyze_decision_alignment(self, profiles: List[BehavioralFingerprint]) -> float:
        """Analyze decision-making alignment."""
        decision_scores = [p.decision_making_score for p in profiles]
        
        # Better alignment if decision-making scores are similar
        std_dev = np.std(decision_scores)
        alignment_score = max(0, 1 - (std_dev / 0.3))  # 0.3 threshold
        
        return alignment_score
    
    def _determine_player_role(self, match_data: Dict[str, Any], profile: BehavioralFingerprint) -> str:
        """Determine player's role based on match data and profile."""
        hero = match_data.get('hero', 'unknown')
        
        # Role mapping based on hero and performance patterns
        role_indicators = {
            'tank': ['franco', 'tigreal', 'fredrinn'],
            'fighter': ['chou'],
            'assassin': ['hayabusa', 'lancelot'],
            'mage': ['kagura'],
            'marksman': ['miya'],
            'support': ['estes', 'angela']
        }
        
        # Check hero-based role first
        for role, heroes in role_indicators.items():
            if hero in heroes:
                return role
        
        # Fallback to profile-based role determination
        return profile.preferred_role if profile.preferred_role else 'unknown'
    
    def _calculate_role_effectiveness(
        self, role: str, match_data: Dict[str, Any], profile: BehavioralFingerprint
    ) -> float:
        """Calculate role effectiveness score."""
        if role not in self.role_definitions:
            return 0.5
        
        role_def = self.role_definitions[role]
        primary_stats = role_def['primary_stats']
        
        # Calculate effectiveness based on primary stats
        effectiveness_scores = []
        for stat in primary_stats:
            if stat in match_data:
                value = match_data[stat]
                # Normalize based on stat type
                if stat == 'teamfight_participation':
                    normalized = min(value / 80, 1.0)
                elif stat == 'positioning_rating':
                    normalized = self._position_rating_to_score(value)
                elif stat in ['kills', 'assists']:
                    normalized = min(value / 10, 1.0)
                elif stat in ['hero_damage', 'turret_damage']:
                    normalized = min(value / 50000, 1.0)
                elif stat == 'damage_taken':
                    normalized = min(value / 30000, 1.0)
                elif stat == 'gold_per_min':
                    normalized = min(value / 5000, 1.0)
                else:
                    normalized = 0.5
                
                effectiveness_scores.append(normalized)
        
        return np.mean(effectiveness_scores) if effectiveness_scores else 0.5
    
    def _calculate_role_synergy_score(
        self, roles: List[str], profiles: List[BehavioralFingerprint]
    ) -> float:
        """Calculate role synergy score."""
        synergy_score = 0
        
        # Check for ideal team composition
        ideal_composition = {'tank': 1, 'fighter': 1, 'assassin': 1, 'mage': 1, 'marksman': 1}
        role_counts = {role: roles.count(role) for role in set(roles)}
        
        # Calculate composition score
        composition_score = 0
        for role, ideal_count in ideal_composition.items():
            actual_count = role_counts.get(role, 0)
            if actual_count == ideal_count:
                composition_score += 0.2
            elif actual_count > 0:
                composition_score += 0.1
        
        # Add synergy bonuses for good role combinations
        synergy_bonuses = 0
        if 'tank' in roles and 'marksman' in roles:
            synergy_bonuses += 0.2
        if 'support' in roles and ('mage' in roles or 'marksman' in roles):
            synergy_bonuses += 0.15
        if 'assassin' in roles and 'tank' in roles:
            synergy_bonuses += 0.1
        
        synergy_score = composition_score + synergy_bonuses
        return min(synergy_score, 1.0)
    
    def _calculate_player_compatibility(
        self, profile_a: BehavioralFingerprint, profile_b: BehavioralFingerprint
    ) -> PlayerCompatibility:
        """Calculate compatibility between two players."""
        synergy_factors = []
        conflict_factors = []
        
        # Analyze playstyle compatibility
        playstyle_compatibility = self._calculate_playstyle_compatibility(
            profile_a.play_style, profile_b.play_style
        )
        
        if playstyle_compatibility > 0.7:
            synergy_factors.append("Compatible playstyles")
        elif playstyle_compatibility < 0.3:
            conflict_factors.append("Conflicting playstyles")
        
        # Analyze tempo alignment
        tempo_alignment = self._calculate_tempo_alignment(profile_a.game_tempo, profile_b.game_tempo)
        
        if tempo_alignment > 0.7:
            synergy_factors.append("Similar game tempo preferences")
        elif tempo_alignment < 0.3:
            conflict_factors.append("Different tempo preferences")
        
        # Analyze role synergy
        role_synergy = self._calculate_role_compatibility(profile_a.preferred_role, profile_b.preferred_role)
        
        if role_synergy > 0.7:
            synergy_factors.append("Complementary roles")
        elif role_synergy < 0.3:
            conflict_factors.append("Role overlap or conflicts")
        
        # Calculate overall compatibility
        compatibility_score = (playstyle_compatibility + tempo_alignment + role_synergy) / 3
        
        # Calculate coordination rating
        coordination_rating = (profile_a.synergy_with_team + profile_b.synergy_with_team) / 2
        
        return PlayerCompatibility(
            player_a=profile_a.player_id,
            player_b=profile_b.player_id,
            compatibility_score=compatibility_score,
            synergy_factors=synergy_factors,
            conflict_factors=conflict_factors,
            playstyle_alignment=playstyle_compatibility,
            coordination_rating=coordination_rating
        )
    
    def _calculate_playstyle_compatibility(self, style_a, style_b) -> float:
        """Calculate playstyle compatibility score."""
        # Define compatibility matrix
        compatibility_matrix = {
            ("aggressive-roamer", "support-oriented"): 0.9,
            ("objective-focused", "team-fighter"): 0.85,
            ("carry-focused", "support-oriented"): 0.8,
            ("assassin-style", "team-fighter"): 0.75,
            ("split-pusher", "objective-focused"): 0.6,
            ("passive-farmer", "aggressive-roamer"): 0.4,
        }
        
        key = (style_a.value, style_b.value)
        reverse_key = (style_b.value, style_a.value)
        
        return compatibility_matrix.get(key, compatibility_matrix.get(reverse_key, 0.5))
    
    def _calculate_tempo_alignment(self, tempo_a, tempo_b) -> float:
        """Calculate tempo alignment score."""
        if tempo_a == tempo_b:
            return 1.0
        
        # Define tempo compatibility
        tempo_compatibility = {
            ("early-aggressive", "mid-game-focused"): 0.7,
            ("mid-game-focused", "late-game-scaling"): 0.6,
            ("adaptive", "early-aggressive"): 0.8,
            ("adaptive", "mid-game-focused"): 0.8,
            ("adaptive", "late-game-scaling"): 0.8,
        }
        
        key = (tempo_a.value, tempo_b.value)
        reverse_key = (tempo_b.value, tempo_a.value)
        
        return tempo_compatibility.get(key, tempo_compatibility.get(reverse_key, 0.4))
    
    def _calculate_role_compatibility(self, role_a: str, role_b: str) -> float:
        """Calculate role compatibility score."""
        if role_a == role_b:
            return 0.2  # Role overlap is generally bad
        
        # Define role synergy
        role_synergy = {
            ("tank", "marksman"): 0.9,
            ("tank", "mage"): 0.8,
            ("support", "marksman"): 0.9,
            ("support", "mage"): 0.8,
            ("assassin", "tank"): 0.7,
            ("fighter", "tank"): 0.6,
            ("assassin", "support"): 0.7,
        }
        
        key = (role_a, role_b)
        reverse_key = (role_b, role_a)
        
        return role_synergy.get(key, role_synergy.get(reverse_key, 0.5))
    
    def _position_rating_to_score(self, rating: str) -> float:
        """Convert position rating to numerical score."""
        rating_map = {
            'low': 0.3,
            'average': 0.6,
            'good': 0.9
        }
        return rating_map.get(rating, 0.5)
    
    def _calculate_team_coordination_score(self, coordination_analysis: Dict[str, Any]) -> float:
        """Calculate overall team coordination score."""
        return coordination_analysis["overall_coordination"]
    
    def _calculate_overall_synergy_score(self, synergy_analysis: Dict[str, Any]) -> float:
        """Calculate overall synergy score."""
        return synergy_analysis["overall_synergy"]
    
    def _calculate_team_effectiveness_rating(
        self, coordination_score: float, synergy_score: float, role_analysis: RoleAnalysis
    ) -> float:
        """Calculate team effectiveness rating."""
        return (coordination_score + synergy_score + role_analysis.synergy_score) / 3
    
    def _calculate_confidence_score(
        self, match_data: List[Dict[str, Any]], profiles: List[BehavioralFingerprint]
    ) -> float:
        """Calculate analysis confidence score."""
        # Base confidence on data quality and profile confidence
        profile_confidences = [p.confidence_score for p in profiles]
        avg_profile_confidence = np.mean(profile_confidences)
        
        # Data completeness factor
        data_completeness = sum(1 for data in match_data if len(data) >= 8) / len(match_data)
        
        # Combined confidence
        confidence = (avg_profile_confidence + data_completeness) / 2
        
        return confidence
    
    def _classify_synergy_level(self, synergy_score: float) -> TeamSynergyLevel:
        """Classify team synergy level."""
        if synergy_score >= self.synergy_thresholds["exceptional_synergy"]:
            return TeamSynergyLevel.EXCEPTIONAL
        elif synergy_score >= self.synergy_thresholds["strong_synergy"]:
            return TeamSynergyLevel.STRONG
        elif synergy_score >= self.synergy_thresholds["moderate_synergy"]:
            return TeamSynergyLevel.MODERATE
        elif synergy_score >= self.synergy_thresholds["weak_synergy"]:
            return TeamSynergyLevel.WEAK
        else:
            return TeamSynergyLevel.DYSFUNCTIONAL
    
    def _classify_coordination_pattern(self, coordination_score: float) -> CoordinationPattern:
        """Classify coordination pattern."""
        if coordination_score >= self.coordination_thresholds["highly_coordinated"]:
            return CoordinationPattern.HIGHLY_COORDINATED
        elif coordination_score >= self.coordination_thresholds["moderately_coordinated"]:
            return CoordinationPattern.MODERATELY_COORDINATED
        elif coordination_score >= self.coordination_thresholds["loosely_coordinated"]:
            return CoordinationPattern.LOOSELY_COORDINATED
        else:
            return CoordinationPattern.UNCOORDINATED
    
    def _classify_role_overlap_severity(self, role_analysis: RoleAnalysis) -> RoleOverlapSeverity:
        """Classify role overlap severity."""
        overlap_count = len(role_analysis.overlap_conflicts)
        gap_count = len(role_analysis.coverage_gaps)
        
        if overlap_count == 0 and gap_count == 0:
            return RoleOverlapSeverity.NO_OVERLAP
        elif overlap_count <= 1 and gap_count <= 1:
            return RoleOverlapSeverity.MINOR_OVERLAP
        elif overlap_count <= 2 and gap_count <= 2:
            return RoleOverlapSeverity.MODERATE_OVERLAP
        elif overlap_count <= 3 and gap_count <= 3:
            return RoleOverlapSeverity.MAJOR_OVERLAP
        else:
            return RoleOverlapSeverity.CRITICAL_OVERLAP


async def create_team_behavior_analyzer(
    cache_manager: Optional[HybridCache] = None
) -> TeamBehaviorAnalyzer:
    """Create and initialize a team behavior analyzer."""
    analyzer = TeamBehaviorAnalyzer(cache_manager)
    return analyzer 