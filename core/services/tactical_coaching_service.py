"""
Tactical Coaching Service for MLBB Coach AI
==========================================

This service provides AI-powered tactical coaching by analyzing timestamped 
gameplay data and generating comprehensive coaching reports with natural 
language insights, visual overlays, and strategic recommendations.

Features:
- Natural language tactical reports
- Timestamp-aligned visual annotations
- Decision-making quality inference
- Corrective strategy suggestions
- Game phase breakdown analysis
- Missed opportunity identification
- Gamified feedback and achievements
"""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np

# Import existing system components
from .base_service import BaseService, ServiceResult
from ..behavioral_modeling import (
    BehavioralAnalyzer, BehavioralFingerprint, PlayStyle, RiskProfile
)
from ..event_detector import GameEvent, GameEventType
from ..advanced_performance_analyzer import AdvancedPerformanceAnalyzer
from ..cache.hybrid_cache import HybridCache
from ..events.event_bus import EventBus

logger = logging.getLogger(__name__)


class GamePhase(Enum):
    """Game phases for tactical analysis."""
    EARLY_GAME = "early_game"    # 0-10 minutes
    MID_GAME = "mid_game"        # 10-20 minutes
    LATE_GAME = "late_game"      # 20+ minutes


class TacticalSeverity(Enum):
    """Severity levels for tactical findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    POSITIVE = "positive"


@dataclass
class TacticalFinding:
    """Represents a tactical finding with coaching insight."""
    timestamp: float
    event: str
    finding: str
    suggestion: str
    severity: TacticalSeverity
    confidence: float
    game_phase: GamePhase
    event_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "timestamp": self.timestamp,
            "event": self.event,
            "finding": self.finding,
            "suggestion": self.suggestion,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "game_phase": self.game_phase.value,
            "event_type": self.event_type,
            "metadata": self.metadata
        }


@dataclass
class VisualOverlay:
    """Visual overlay annotation for frames."""
    frame_path: str
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_annotation(self, annotation_type: str, **kwargs):
        """Add an annotation to the overlay."""
        annotation = {"type": annotation_type, **kwargs}
        self.annotations.append(annotation)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "frame_path": self.frame_path,
            "annotations": self.annotations
        }


@dataclass
class OpportunityAnalysis:
    """Analysis of missed opportunities."""
    timestamp: float
    event: str
    missed_action: str
    alternative: str
    impact_score: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "timestamp": self.timestamp,
            "event": self.event,
            "missed_action": self.missed_action,
            "alternative": self.alternative,
            "impact_score": self.impact_score,
            "reasoning": self.reasoning,
            "metadata": self.metadata
        }


@dataclass
class TacticalCoachingReport:
    """Complete tactical coaching report."""
    player_ign: str
    video_path: str
    analysis_timestamp: datetime
    
    # Core sections
    post_game_summary: str
    tactical_findings: List[TacticalFinding]
    visual_overlays: List[VisualOverlay]
    opportunity_analysis: List[OpportunityAnalysis]
    gamified_feedback: List[str]
    
    # Game phase breakdown
    game_phase_breakdown: Dict[str, List[TacticalFinding]]
    
    # Confidence and metadata
    overall_confidence: float
    processing_time: float
    insights_generated: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "player_ign": self.player_ign,
            "video_path": self.video_path,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "post_game_summary": self.post_game_summary,
            "tactical_findings": [finding.to_dict() for finding in self.tactical_findings],
            "visual_overlays": [overlay.to_dict() for overlay in self.visual_overlays],
            "game_phase_breakdown": {
                phase: [finding.to_dict() for finding in findings]
                for phase, findings in self.game_phase_breakdown.items()
            },
            "opportunity_analysis": [opp.to_dict() for opp in self.opportunity_analysis],
            "gamified_feedback": self.gamified_feedback,
            "overall_confidence": self.overall_confidence,
            "processing_time": self.processing_time,
            "insights_generated": self.insights_generated
        }


class TacticalAnalysisEngine:
    """Core tactical analysis engine."""
    
    def __init__(self):
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.performance_analyzer = AdvancedPerformanceAnalyzer()
        
        # Analysis thresholds
        self.phase_thresholds = {
            GamePhase.EARLY_GAME: (0, 600),    # 0-10 minutes
            GamePhase.MID_GAME: (600, 1200),   # 10-20 minutes
            GamePhase.LATE_GAME: (1200, 3600)  # 20+ minutes
        }
        
        # Tactical patterns and rules
        self.tactical_patterns = self._load_tactical_patterns()
        self.coaching_templates = self._load_coaching_templates()
        
    def _load_tactical_patterns(self) -> Dict[str, Any]:
        """Load tactical analysis patterns."""
        return {
                         "rotation_patterns": {
                 "late_rotation": {
                     "description": "Player rotates too late to objectives",
                     "indicators": ["missed_teamfight", "solo_farming", 
                                    "late_arrival"],
                     "severity": TacticalSeverity.HIGH,
                     "coaching_tip": ("Prioritize map awareness and anticipate "
                                      "objective timings")
                 },
                 "overextension": {
                     "description": "Player extends too far without vision",
                     "indicators": ["death_in_enemy_jungle", "no_vision_wards", 
                                    "solo_push"],
                     "severity": TacticalSeverity.CRITICAL,
                     "coaching_tip": ("Maintain vision control and communicate "
                                      "with team before extending")
                 }
             },
             "farming_patterns": {
                 "inefficient_farming": {
                     "description": "Player farms inefficiently during key moments",
                     "indicators": ["low_gpm", "missed_objectives", "overfarming"],
                     "severity": TacticalSeverity.MEDIUM,
                     "coaching_tip": ("Balance farming with team objectives and "
                                    "map pressure")
                 }
             },
             "teamfight_patterns": {
                 "poor_positioning": {
                     "description": "Player positions poorly in teamfights",
                     "indicators": ["early_death", "low_damage", "caught_out"],
                     "severity": TacticalSeverity.HIGH,
                     "coaching_tip": ("Focus on positioning behind tanks and "
                                    "identifying threats")
                 }
             }
        }
    
    def _load_coaching_templates(self) -> Dict[str, str]:
        """Load coaching message templates."""
        return {
            "rotation_late": "Late rotation to {location} after enemy initiated {action}. Missed opportunity to {counter_action}.",
            "overextension": "Overextended in {location} without vision coverage. High risk of getting caught by enemy rotations.",
            "farming_inefficient": "Continued farming while team engaged at {objective}. Missing critical team moments.",
            "positioning_poor": "Poor positioning in teamfight led to early elimination. Consider staying behind {tank_hero}.",
            "vision_lack": "No vision control in {area} during key moment. Blind rotations are extremely risky.",
            "macro_error": "Macro decision error: {decision} when team needed {alternative}. Focus on team coordination.",
            "opportunity_missed": "Missed opportunity at {timestamp}: {opportunity}. Could have resulted in {potential_outcome}."
        }
    
    def analyze_decision_making(self, events: List[GameEvent], 
                              behavioral_profile: BehavioralFingerprint) -> List[TacticalFinding]:
        """Analyze decision-making quality from events."""
        findings = []
        
        for i, event in enumerate(events):
            game_phase = self._determine_game_phase(event.timestamp)
            
            # Analyze based on event type and behavioral profile
            if event.event_type == GameEventType.DEATH:
                finding = self._analyze_death_event(event, behavioral_profile, game_phase)
                if finding:
                    findings.append(finding)
            
            elif event.event_type == GameEventType.HERO_ROTATION:
                finding = self._analyze_rotation_event(event, behavioral_profile, game_phase, events)
                if finding:
                    findings.append(finding)
            
            elif event.event_type == GameEventType.TEAMFIGHT_START:
                finding = self._analyze_teamfight_event(event, behavioral_profile, game_phase, events)
                if finding:
                    findings.append(finding)
        
        return findings
    
    def _determine_game_phase(self, timestamp: float) -> GamePhase:
        """Determine game phase from timestamp."""
        if timestamp <= 600:
            return GamePhase.EARLY_GAME
        elif timestamp <= 1200:
            return GamePhase.MID_GAME
        else:
            return GamePhase.LATE_GAME
    
    def _analyze_death_event(self, event: GameEvent, profile: BehavioralFingerprint, 
                           phase: GamePhase) -> Optional[TacticalFinding]:
        """Analyze death events for tactical insights."""
        # Get context from event metadata
        location = event.metadata.get("location", "unknown")
        cause = event.metadata.get("cause", "unknown")
        
        # Analyze based on behavioral profile
        if profile.risk_profile == RiskProfile.HIGH_RISK_HIGH_REWARD:
            if location in ["enemy_jungle", "deep_ward"]:
                return TacticalFinding(
                    timestamp=event.timestamp,
                    event="death_overextension",
                    finding=f"Overextended in {location} without vision coverage. High risk of getting caught by enemy rotations.",
                    suggestion="Prioritize vision control and communicate with team before extending to high-risk areas.",
                    severity=TacticalSeverity.HIGH,
                    confidence=0.85,
                    game_phase=phase,
                    event_type="death_analysis",
                    metadata={"location": location, "cause": cause}
                )
        
        elif profile.positioning_safety < 0.5:
            return TacticalFinding(
                timestamp=event.timestamp,
                event="death_positioning",
                finding=f"Poor positioning led to elimination at {event.timestamp:.1f}s. Pattern indicates positioning issues.",
                suggestion="Focus on staying behind tanks and identifying threats before engaging.",
                severity=TacticalSeverity.MEDIUM,
                confidence=0.75,
                game_phase=phase,
                event_type="positioning_analysis",
                metadata={"positioning_score": profile.positioning_safety}
            )
        
        return None
    
    def _analyze_rotation_event(self, event: GameEvent, profile: BehavioralFingerprint, 
                              phase: GamePhase, all_events: List[GameEvent]) -> Optional[TacticalFinding]:
        """Analyze rotation events for tactical insights."""
        # Check if rotation was timely
        nearby_events = [e for e in all_events if abs(e.timestamp - event.timestamp) < 30]
        teamfight_events = [e for e in nearby_events if e.event_type == GameEventType.TEAMFIGHT_START]
        
        if teamfight_events:
            # Check if rotation was after teamfight started
            teamfight_start = min(e.timestamp for e in teamfight_events)
            if event.timestamp > teamfight_start + 5:  # 5 seconds late
                return TacticalFinding(
                    timestamp=event.timestamp,
                    event="late_rotation",
                    finding=f"Late rotation to teamfight area. Arrived {event.timestamp - teamfight_start:.1f}s after engagement started.",
                    suggestion="Improve map awareness and anticipate teamfight locations. Watch for enemy positioning cues.",
                    severity=TacticalSeverity.HIGH,
                    confidence=0.80,
                    game_phase=phase,
                    event_type="rotation_analysis",
                    metadata={"delay": event.timestamp - teamfight_start}
                )
        
        return None
    
    def _analyze_teamfight_event(self, event: GameEvent, profile: BehavioralFingerprint, 
                               phase: GamePhase, all_events: List[GameEvent]) -> Optional[TacticalFinding]:
        """Analyze teamfight events for tactical insights."""
        # Look for death shortly after teamfight start
        nearby_deaths = [e for e in all_events 
                        if e.event_type == GameEventType.DEATH 
                        and e.timestamp > event.timestamp 
                        and e.timestamp < event.timestamp + 10]
        
        if nearby_deaths and profile.decision_making_score < 0.6:
            return TacticalFinding(
                timestamp=event.timestamp,
                event="teamfight_positioning",
                finding="Poor teamfight positioning led to early elimination. Consider engagement timing and position.",
                suggestion="Wait for tank initiation and focus on staying at maximum effective range.",
                severity=TacticalSeverity.HIGH,
                confidence=0.70,
                game_phase=phase,
                event_type="teamfight_analysis",
                metadata={"deaths_in_fight": len(nearby_deaths)}
            )
        
        return None
    
    def identify_missed_opportunities(self, events: List[GameEvent], 
                                   behavioral_profile: BehavioralFingerprint) -> List[OpportunityAnalysis]:
        """Identify missed opportunities from event patterns."""
        opportunities = []
        
        # Look for patterns that indicate missed opportunities
        for i, event in enumerate(events):
            if event.event_type == GameEventType.TOWER_DESTROYED:
                # Check if player was present for tower push
                nearby_events = [e for e in events if abs(e.timestamp - event.timestamp) < 30]
                player_actions = [e for e in nearby_events if e.player_ign == behavioral_profile.player_id]
                
                if not player_actions:
                    opportunities.append(OpportunityAnalysis(
                        timestamp=event.timestamp,
                        event="tower_destroyed",
                        missed_action="Not present for tower push after favorable team fight",
                        alternative=f"Rotate to tower immediately after team fight advantage at {event.timestamp - 15:.1f}s",
                        impact_score=0.8,
                        reasoning="Tower gold and map control are crucial for maintaining momentum",
                        metadata={"tower_gold": 320, "map_control_value": "high"}
                    ))
        
        return opportunities
    
    def generate_visual_overlays(self, events: List[GameEvent], 
                               findings: List[TacticalFinding]) -> List[VisualOverlay]:
        """Generate visual overlay annotations."""
        overlays = []
        
        for finding in findings:
            if finding.event_type == "rotation_analysis":
                frame_path = finding.metadata.get("frame_path", "")
                if frame_path:
                    overlay = VisualOverlay(frame_path=frame_path)
                    overlay.add_annotation(
                        "arrow",
                        from_region="current_position",
                        to_region="objective_location",
                        label="Missed Rotation",
                        color="red"
                    )
                    overlay.add_annotation(
                        "zone",
                        region="objective_area",
                        label="No Vision",
                        color="orange"
                    )
                    overlays.append(overlay)
            
            elif finding.event_type == "positioning_analysis":
                frame_path = finding.metadata.get("frame_path", "")
                if frame_path:
                    overlay = VisualOverlay(frame_path=frame_path)
                    overlay.add_annotation(
                        "zone",
                        region="danger_zone",
                        label="High Risk Position",
                        color="red"
                    )
                    overlay.add_annotation(
                        "arrow",
                        from_region="current_position",
                        to_region="safe_position",
                        label="Suggested Position",
                        color="green"
                    )
                    overlays.append(overlay)
        
        return overlays
    
    def generate_gamified_feedback(self, findings: List[TacticalFinding], 
                                 behavioral_profile: BehavioralFingerprint) -> List[str]:
        """Generate gamified feedback and achievements."""
        feedback = []
        
        # Count different types of issues
        positioning_issues = len([f for f in findings if "positioning" in f.event])
        rotation_issues = len([f for f in findings if "rotation" in f.event])
        
        # Generate achievements based on performance
        if positioning_issues >= 3:
            feedback.append("🏆 Map Awareness Rookie: Missed 3+ positioning opportunities – focus on threat identification")
        elif positioning_issues <= 1:
            feedback.append("🎯 Positioning Pro: Excellent positioning awareness – keep it up!")
        
        if rotation_issues >= 2:
            feedback.append("🗺️ Rotation Trainee: Late to 2+ key rotations – improve map awareness and timing")
        elif rotation_issues == 0:
            feedback.append("⚡ Rotation Master: Perfect rotation timing – excellent macro awareness!")
        
        # Behavioral profile based feedback
        if behavioral_profile.risk_profile == RiskProfile.HIGH_RISK_HIGH_REWARD:
            feedback.append("🎲 Risk Taker: High-risk plays detected – balance aggression with safety")
        elif behavioral_profile.risk_profile == RiskProfile.CONSERVATIVE:
            feedback.append("🛡️ Calculated Player: Conservative approach – consider more aggressive opportunities")
        
        # Positive reinforcement
        high_confidence_findings = [f for f in findings if f.confidence > 0.8]
        if len(high_confidence_findings) < 3:
            feedback.append("🌟 Consistent Performer: Few major tactical issues identified – solid gameplay!")
        
        return feedback


class TacticalCoachingService(BaseService):
    """Tactical coaching service for AI-powered gameplay analysis."""
    
    def __init__(self, cache_manager: Optional[HybridCache] = None):
        super().__init__("tactical_coaching")
        self.cache_manager = cache_manager or HybridCache()
        self.event_bus = EventBus()
        self.analysis_engine = TacticalAnalysisEngine()
        
        # Service configuration
        self.min_confidence_threshold = 0.6
        self.max_findings_per_phase = 10
        
    async def process(self, request: Dict[str, Any]) -> ServiceResult:
        """
        Process tactical coaching request.
        
        Args:
            request: Dictionary containing:
                - temporal_analysis: TemporalAnalysisResult or dict
                - behavioral_profile: BehavioralFingerprint or dict (optional)
                - coaching_focus: List of focus areas (optional)
                
        Returns:
            ServiceResult with tactical coaching report
        """
        try:
            start_time = time.time()
            
            # Extract request data
            temporal_data = request.get("temporal_analysis")
            behavioral_data = request.get("behavioral_profile")
            coaching_focus = request.get("coaching_focus", [])
            
            if not temporal_data:
                return ServiceResult(
                    success=False,
                    error="Missing temporal analysis data",
                    data={}
                )
            
            # Parse temporal analysis data
            if isinstance(temporal_data, dict):
                player_ign = temporal_data.get("player_ign", "Unknown")
                video_path = temporal_data.get("video_path", "")
                game_events = temporal_data.get("game_events", [])
                # Convert dict events to GameEvent objects if needed
                if game_events and isinstance(game_events[0], dict):
                    # This would need proper conversion logic
                    pass
            else:
                player_ign = temporal_data.player_ign
                video_path = temporal_data.video_path
                game_events = temporal_data.game_events
            
            # Get or create behavioral profile
            behavioral_profile = await self._get_behavioral_profile(
                player_ign, behavioral_data, game_events
            )
            
            # Perform tactical analysis
            tactical_findings = self.analysis_engine.analyze_decision_making(
                game_events, behavioral_profile
            )
            
            # Filter by confidence threshold
            high_confidence_findings = [
                f for f in tactical_findings 
                if f.confidence >= self.min_confidence_threshold
            ]
            
            # Group findings by game phase
            game_phase_breakdown = self._group_by_game_phase(high_confidence_findings)
            
            # Identify missed opportunities
            opportunities = self.analysis_engine.identify_missed_opportunities(
                game_events, behavioral_profile
            )
            
            # Generate visual overlays
            visual_overlays = self.analysis_engine.generate_visual_overlays(
                game_events, high_confidence_findings
            )
            
            # Generate gamified feedback
            gamified_feedback = self.analysis_engine.generate_gamified_feedback(
                high_confidence_findings, behavioral_profile
            )
            
            # Generate post-game summary
            post_game_summary = self._generate_post_game_summary(
                behavioral_profile, high_confidence_findings, opportunities
            )
            
            # Create comprehensive report
            report = TacticalCoachingReport(
                player_ign=player_ign,
                video_path=video_path,
                analysis_timestamp=datetime.now(),
                post_game_summary=post_game_summary,
                tactical_findings=high_confidence_findings,
                visual_overlays=visual_overlays,
                game_phase_breakdown=game_phase_breakdown,
                opportunity_analysis=opportunities,
                gamified_feedback=gamified_feedback,
                overall_confidence=np.mean([f.confidence for f in high_confidence_findings]) if high_confidence_findings else 0.0,
                processing_time=time.time() - start_time,
                insights_generated=len(high_confidence_findings) + len(opportunities)
            )
            
            # Cache the report
            cache_key = f"tactical_report:{player_ign}:{hash(video_path)}"
            await self.cache_manager.set(cache_key, report.to_dict(), ttl=1800)
            
            # Emit completion event
            await self.event_bus.emit("tactical_coaching_completed", {
                "player_ign": player_ign,
                "insights_generated": report.insights_generated,
                "processing_time": report.processing_time,
                "confidence": report.overall_confidence
            })
            
            return ServiceResult(
                success=True,
                data=report.to_dict(),
                metadata={
                    "processing_time": report.processing_time,
                    "insights_count": report.insights_generated,
                    "confidence": report.overall_confidence
                }
            )
            
        except Exception as e:
            logger.error(f"Error in tactical coaching service: {str(e)}")
            return ServiceResult(
                success=False,
                error=str(e),
                data={}
            )
    
    async def _get_behavioral_profile(self, player_ign: str, 
                                    behavioral_data: Optional[Dict[str, Any]], 
                                    game_events: List[GameEvent]) -> BehavioralFingerprint:
        """Get or create behavioral profile for player."""
        if behavioral_data:
            # Use provided behavioral data
            if isinstance(behavioral_data, dict):
                # Convert dict to BehavioralFingerprint
                # This would need proper conversion logic
                pass
            else:
                return behavioral_data
        
        # Create basic profile from game events
        return BehavioralFingerprint(
            player_id=player_ign,
            play_style=PlayStyle.CARRY_FOCUSED,  # Default
            risk_profile=RiskProfile.CALCULATED_RISK,  # Default
            game_tempo=GameTempo.ADAPTIVE,  # Default
            map_awareness_score=0.5,
            synergy_with_team=0.5,
            adaptability_score=0.5,
            mechanical_skill_score=0.5,
            decision_making_score=0.5,
            preferred_lane="mid",
            preferred_role="carry",
            preferred_heroes=[],
            behavior_tags=[],
            identified_flaws=[],
            strength_areas=[],
            behavioral_metrics=None,
            confidence_score=0.5,
            analysis_date=datetime.now(),
            matches_analyzed=1
        )
    
    def _group_by_game_phase(self, findings: List[TacticalFinding]) -> Dict[str, List[TacticalFinding]]:
        """Group findings by game phase."""
        phase_groups = {
            "early_game": [],
            "mid_game": [],
            "late_game": []
        }
        
        for finding in findings:
            phase_key = finding.game_phase.value
            if phase_key in phase_groups:
                phase_groups[phase_key].append(finding)
        
        return phase_groups
    
    def _generate_post_game_summary(self, behavioral_profile: BehavioralFingerprint, 
                                  findings: List[TacticalFinding], 
                                  opportunities: List[OpportunityAnalysis]) -> str:
        """Generate natural language post-game summary."""
        critical_issues = len([f for f in findings if f.severity == TacticalSeverity.CRITICAL])
        high_issues = len([f for f in findings if f.severity == TacticalSeverity.HIGH])
        missed_opportunities = len(opportunities)
        
        if critical_issues > 0:
            severity_assessment = f"Critical tactical issues identified ({critical_issues})"
        elif high_issues > 2:
            severity_assessment = f"Multiple high-priority areas for improvement ({high_issues})"
        elif high_issues > 0:
            severity_assessment = f"Some tactical adjustments needed ({high_issues})"
        else:
            severity_assessment = "Strong tactical gameplay overall"
        
        play_style_insight = f"Your {behavioral_profile.play_style.value.replace('_', ' ')} approach"
        
        if missed_opportunities > 0:
            opportunity_insight = f"Analysis detected {missed_opportunities} missed opportunities for greater impact"
        else:
            opportunity_insight = "Good opportunity recognition and execution"
        
        return (f"{severity_assessment}. {play_style_insight} shows "
                f"{'strong' if behavioral_profile.decision_making_score > 0.7 else 'developing'} "
                f"decision-making patterns. {opportunity_insight}. "
                f"Focus areas: positioning, map awareness, and objective timing.")


def create_tactical_coaching_service(cache_manager: Optional[HybridCache] = None) -> TacticalCoachingService:
    """Create and configure tactical coaching service."""
    return TacticalCoachingService(cache_manager) 