"""
Real-time Event Detection System for MLBB Coach AI
==================================================

This module provides comprehensive event detection capabilities for analyzing
gameplay videos and identifying key events with precise timestamps.

Features:
- Kill/Death event detection
- Objective completion detection (towers, lord, turtle)
- Gold spike detection
- Teamfight event detection
- Hero rotation detection
- Timestamp correlation for all events
"""

import logging
import time
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re

# Import existing components
from .video_reader import TimestampedFrame
from .enhanced_data_collector import EnhancedDataCollector
from .events.event_types import Event, EventType

logger = logging.getLogger(__name__)


class GameEventType(Enum):
    """Types of game events that can be detected."""
    KILL = "kill"
    DEATH = "death"
    ASSIST = "assist"
    TOWER_DESTROYED = "tower_destroyed"
    TOWER_DAMAGE = "tower_damage"
    LORD_KILLED = "lord_killed"
    TURTLE_KILLED = "turtle_killed"
    GOLD_SPIKE = "gold_spike"
    TEAMFIGHT_START = "teamfight_start"
    TEAMFIGHT_END = "teamfight_end"
    HERO_ROTATION = "hero_rotation"
    FIRST_BLOOD = "first_blood"
    MULTI_KILL = "multi_kill"
    SAVAGE = "savage"
    MANIAC = "maniac"
    MATCH_START = "match_start"
    MATCH_END = "match_end"


@dataclass
class GameEvent:
    """Represents a detected game event with timestamp and metadata."""
    event_type: GameEventType
    timestamp: float  # Video timestamp in seconds
    frame_number: int
    confidence: float
    player_ign: str = ""
    hero: str = ""
    location: str = ""
    participants: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary format."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "confidence": self.confidence,
            "player_ign": self.player_ign,
            "hero": self.hero,
            "location": self.location,
            "participants": self.participants,
            "metadata": self.metadata
        }


class EventPatternMatcher:
    """Pattern matching for different types of game events."""
    
    def __init__(self):
        self.patterns = {
            GameEventType.KILL: {
                "text_patterns": [
                    r"(\w+)\s+killed\s+(\w+)",
                    r"(\w+)\s+slain\s+(\w+)",
                    r"(\w+)\s+eliminated\s+(\w+)",
                    r"killed",
                    r"slain"
                ],
                "visual_indicators": ["red_cross", "kill_notification"]
            },
            GameEventType.TOWER_DESTROYED: {
                "text_patterns": [
                    r"tower\s+destroyed",
                    r"turret\s+destroyed",
                    r"destroyed.*tower",
                    r"tower.*fallen"
                ],
                "visual_indicators": ["tower_destruction_effect"]
            },
            GameEventType.LORD_KILLED: {
                "text_patterns": [
                    r"lord\s+killed",
                    r"lord\s+slain",
                    r"lord.*defeated",
                    r"team.*killed.*lord"
                ],
                "visual_indicators": ["lord_buff_icon"]
            },
            GameEventType.TURTLE_KILLED: {
                "text_patterns": [
                    r"turtle\s+killed",
                    r"turtle\s+slain",
                    r"turtle.*defeated"
                ],
                "visual_indicators": ["turtle_buff_icon"]
            },
            GameEventType.FIRST_BLOOD: {
                "text_patterns": [
                    r"first\s+blood",
                    r"firstblood"
                ],
                "visual_indicators": ["first_blood_announcement"]
            },
            GameEventType.MULTI_KILL: {
                "text_patterns": [
                    r"double\s+kill",
                    r"triple\s+kill",
                    r"quadra\s+kill",
                    r"penta\s+kill"
                ],
                "visual_indicators": ["multi_kill_announcement"]
            },
            GameEventType.SAVAGE: {
                "text_patterns": [
                    r"savage",
                    r"legendary"
                ],
                "visual_indicators": ["savage_announcement"]
            }
        }
    
    def match_event(self, text_content: str, visual_features: Dict[str, Any]) -> List[Tuple[GameEventType, float]]:
        """
        Match events based on text content and visual features.
        
        Args:
            text_content: OCR text content from the frame
            visual_features: Extracted visual features
            
        Returns:
            List of (event_type, confidence) tuples
        """
        matches = []
        text_lower = text_content.lower()
        
        for event_type, patterns in self.patterns.items():
            confidence = 0.0
            
            # Check text patterns
            for pattern in patterns.get("text_patterns", []):
                if re.search(pattern, text_lower):
                    confidence = max(confidence, 0.8)
                    break
            
            # Check visual indicators
            for indicator in patterns.get("visual_indicators", []):
                if visual_features.get(indicator, False):
                    confidence = max(confidence, 0.9)
            
            if confidence > 0.5:
                matches.append((event_type, confidence))
        
        return matches


class GoldSpikeDetector:
    """Detects significant gold changes that indicate events."""
    
    def __init__(self, spike_threshold: int = 200):
        self.spike_threshold = spike_threshold
        self.gold_history = []
        self.smoothing_window = 3
    
    def detect_gold_spike(self, current_gold: int, timestamp: float) -> Optional[GameEvent]:
        """
        Detect gold spikes that indicate kills or objectives.
        
        Args:
            current_gold: Current gold amount
            timestamp: Current timestamp
            
        Returns:
            GameEvent if spike detected, None otherwise
        """
        if len(self.gold_history) < 2:
            self.gold_history.append((current_gold, timestamp))
            return None
        
        # Calculate recent gold change
        recent_gold = [g for g, t in self.gold_history[-self.smoothing_window:]]
        avg_recent_gold = sum(recent_gold) / len(recent_gold)
        
        gold_change = current_gold - avg_recent_gold
        
        # Detect significant increase
        if gold_change >= self.spike_threshold:
            # Determine likely cause based on gold amount
            if gold_change >= 500:
                event_type = GameEventType.LORD_KILLED
            elif gold_change >= 300:
                event_type = GameEventType.TOWER_DESTROYED
            elif gold_change >= 150:
                event_type = GameEventType.TURTLE_KILLED
            else:
                event_type = GameEventType.KILL
            
            confidence = min(0.9, gold_change / 500)  # Scale confidence
            
            event = GameEvent(
                event_type=event_type,
                timestamp=timestamp,
                frame_number=0,  # Will be set by caller
                confidence=confidence,
                metadata={
                    "gold_change": gold_change,
                    "current_gold": current_gold,
                    "detection_method": "gold_spike"
                }
            )
            
            # Add to history
            self.gold_history.append((current_gold, timestamp))
            
            # Keep history size manageable
            if len(self.gold_history) > 10:
                self.gold_history.pop(0)
            
            return event
        
        # Add to history
        self.gold_history.append((current_gold, timestamp))
        
        # Keep history size manageable
        if len(self.gold_history) > 10:
            self.gold_history.pop(0)
        
        return None


class TeamfightDetector:
    """Detects teamfight events based on multiple indicators."""
    
    def __init__(self):
        self.activity_threshold = 0.7
        self.duration_threshold = 5.0  # Minimum teamfight duration
        self.current_teamfight = None
        self.teamfight_indicators = [
            "multiple_abilities",
            "rapid_health_changes",
            "multiple_players_visible",
            "damage_numbers",
            "crowd_control_effects"
        ]
    
    def detect_teamfight(self, frame_analysis: Dict[str, Any], 
                        timestamp: float) -> Optional[GameEvent]:
        """
        Detect teamfight events based on frame analysis.
        
        Args:
            frame_analysis: Analysis results from the frame
            timestamp: Current timestamp
            
        Returns:
            GameEvent if teamfight start/end detected
        """
        # Calculate activity score
        activity_score = self._calculate_activity_score(frame_analysis)
        
        if activity_score >= self.activity_threshold:
            # High activity - potential teamfight
            if self.current_teamfight is None:
                # Start new teamfight
                self.current_teamfight = {
                    "start_time": timestamp,
                    "participants": frame_analysis.get("visible_heroes", []),
                    "location": frame_analysis.get("location", "unknown")
                }
                
                return GameEvent(
                    event_type=GameEventType.TEAMFIGHT_START,
                    timestamp=timestamp,
                    frame_number=0,
                    confidence=activity_score,
                    location=self.current_teamfight["location"],
                    participants=self.current_teamfight["participants"],
                    metadata={
                        "activity_score": activity_score,
                        "detection_method": "activity_analysis"
                    }
                )
        
        elif self.current_teamfight is not None:
            # Low activity - potential teamfight end
            duration = timestamp - self.current_teamfight["start_time"]
            
            if duration >= self.duration_threshold:
                # End teamfight
                teamfight_data = self.current_teamfight
                self.current_teamfight = None
                
                return GameEvent(
                    event_type=GameEventType.TEAMFIGHT_END,
                    timestamp=timestamp,
                    frame_number=0,
                    confidence=0.8,
                    location=teamfight_data["location"],
                    participants=teamfight_data["participants"],
                    metadata={
                        "duration": duration,
                        "start_time": teamfight_data["start_time"],
                        "detection_method": "activity_analysis"
                    }
                )
        
        return None
    
    def _calculate_activity_score(self, frame_analysis: Dict[str, Any]) -> float:
        """Calculate activity score based on frame analysis."""
        score = 0.0
        indicators_found = 0
        
        # Check for various activity indicators
        if frame_analysis.get("ability_effects", 0) > 2:
            score += 0.3
            indicators_found += 1
        
        if frame_analysis.get("visible_heroes", 0) >= 3:
            score += 0.2
            indicators_found += 1
        
        if frame_analysis.get("damage_numbers", 0) > 5:
            score += 0.2
            indicators_found += 1
        
        if frame_analysis.get("health_changes", 0) > 3:
            score += 0.2
            indicators_found += 1
        
        if frame_analysis.get("crowd_control_effects", 0) > 0:
            score += 0.1
            indicators_found += 1
        
        # Normalize based on indicators found
        if indicators_found > 0:
            score = min(1.0, score * (indicators_found / len(self.teamfight_indicators)))
        
        return score


class EventDetector:
    """Main event detection system for MLBB gameplay analysis."""
    
    def __init__(self, player_ign: str):
        self.player_ign = player_ign
        self.data_collector = EnhancedDataCollector()
        self.pattern_matcher = EventPatternMatcher()
        self.gold_detector = GoldSpikeDetector()
        self.teamfight_detector = TeamfightDetector()
        
        # Event detection parameters
        self.detection_threshold = 0.6
        self.event_history = []
        self.last_gold_value = 0
        self.frame_analysis_cache = {}
        
        # Performance tracking
        self.detection_stats = {
            "total_frames_analyzed": 0,
            "events_detected": 0,
            "processing_time": 0.0
        }
    
    def detect_events_from_frame(self, timestamped_frame: TimestampedFrame) -> List[GameEvent]:
        """
        Detect events from a single timestamped frame.
        
        Args:
            timestamped_frame: Frame with timestamp information
            
        Returns:
            List of detected GameEvent objects
        """
        start_time = time.time()
        events = []
        
        try:
            # Analyze frame for event indicators
            frame_analysis = self._analyze_frame_for_events(timestamped_frame)
            
            # Text-based event detection
            text_events = self._detect_text_events(frame_analysis, timestamped_frame)
            events.extend(text_events)
            
            # Gold spike detection
            gold_event = self._detect_gold_events(frame_analysis, timestamped_frame)
            if gold_event:
                events.append(gold_event)
            
            # Teamfight detection
            teamfight_event = self._detect_teamfight_events(frame_analysis, timestamped_frame)
            if teamfight_event:
                events.append(teamfight_event)
            
            # Visual event detection
            visual_events = self._detect_visual_events(frame_analysis, timestamped_frame)
            events.extend(visual_events)
            
            # Add frame metadata to all events
            for event in events:
                event.frame_number = timestamped_frame.frame_number
                event.player_ign = self.player_ign
                event.metadata.update({
                    "frame_path": timestamped_frame.frame_path,
                    "frame_metadata": timestamped_frame.metadata
                })
            
            # Update statistics
            self.detection_stats["total_frames_analyzed"] += 1
            self.detection_stats["events_detected"] += len(events)
            self.detection_stats["processing_time"] += time.time() - start_time
            
            # Add to event history
            self.event_history.extend(events)
            
            # Keep history size manageable
            if len(self.event_history) > 100:
                self.event_history = self.event_history[-100:]
            
        except Exception as e:
            logger.error(f"Error detecting events from frame at {timestamped_frame.timestamp:.2f}s: {str(e)}")
        
        return events
    
    def _analyze_frame_for_events(self, timestamped_frame: TimestampedFrame) -> Dict[str, Any]:
        """
        Analyze frame for event indicators.
        
        Args:
            timestamped_frame: Frame to analyze
            
        Returns:
            Dictionary containing frame analysis results
        """
        # Check cache first
        cache_key = f"{timestamped_frame.timestamp:.2f}"
        if cache_key in self.frame_analysis_cache:
            return self.frame_analysis_cache[cache_key]
        
        # Perform OCR analysis
        ocr_result = self.data_collector.analyze_screenshot_with_timestamp(
            image_path=timestamped_frame.frame_path,
            ign=self.player_ign,
            video_timestamp=timestamped_frame.timestamp,
            frame_number=timestamped_frame.frame_number,
            frame_metadata=timestamped_frame.metadata
        )
        
        # Extract relevant information
        analysis = {
            "text_content": self._extract_text_content(ocr_result),
            "current_gold": ocr_result.get("data", {}).get("gold", 0),
            "visible_heroes": self._count_visible_heroes(ocr_result),
            "ability_effects": self._count_ability_effects(timestamped_frame.frame_path),
            "damage_numbers": self._count_damage_numbers(ocr_result),
            "health_changes": self._detect_health_changes(ocr_result),
            "crowd_control_effects": self._detect_cc_effects(timestamped_frame.frame_path),
            "location": self._detect_location(ocr_result),
            "visual_features": self._extract_visual_features(timestamped_frame.frame_path),
            "ocr_result": ocr_result
        }
        
        # Cache the result
        self.frame_analysis_cache[cache_key] = analysis
        
        # Keep cache size manageable
        if len(self.frame_analysis_cache) > 50:
            oldest_key = min(self.frame_analysis_cache.keys())
            del self.frame_analysis_cache[oldest_key]
        
        return analysis
    
    def _detect_text_events(self, frame_analysis: Dict[str, Any], 
                           timestamped_frame: TimestampedFrame) -> List[GameEvent]:
        """Detect events based on text content."""
        events = []
        text_content = frame_analysis.get("text_content", "")
        visual_features = frame_analysis.get("visual_features", {})
        
        # Use pattern matcher to find events
        matches = self.pattern_matcher.match_event(text_content, visual_features)
        
        for event_type, confidence in matches:
            if confidence >= self.detection_threshold:
                event = GameEvent(
                    event_type=event_type,
                    timestamp=timestamped_frame.timestamp,
                    frame_number=timestamped_frame.frame_number,
                    confidence=confidence,
                    metadata={
                        "detection_method": "text_pattern",
                        "matched_text": text_content,
                        "visual_features": visual_features
                    }
                )
                events.append(event)
        
        return events
    
    def _detect_gold_events(self, frame_analysis: Dict[str, Any], 
                           timestamped_frame: TimestampedFrame) -> Optional[GameEvent]:
        """Detect events based on gold changes."""
        current_gold = frame_analysis.get("current_gold", 0)
        
        if current_gold > 0:
            event = self.gold_detector.detect_gold_spike(
                current_gold, 
                timestamped_frame.timestamp
            )
            if event:
                event.frame_number = timestamped_frame.frame_number
                return event
        
        return None
    
    def _detect_teamfight_events(self, frame_analysis: Dict[str, Any], 
                                timestamped_frame: TimestampedFrame) -> Optional[GameEvent]:
        """Detect teamfight events."""
        event = self.teamfight_detector.detect_teamfight(
            frame_analysis, 
            timestamped_frame.timestamp
        )
        if event:
            event.frame_number = timestamped_frame.frame_number
        return event
    
    def _detect_visual_events(self, frame_analysis: Dict[str, Any], 
                             timestamped_frame: TimestampedFrame) -> List[GameEvent]:
        """Detect events based on visual features."""
        events = []
        visual_features = frame_analysis.get("visual_features", {})
        
        # Example: Detect hero rotation based on minimap changes
        if visual_features.get("minimap_movement", False):
            event = GameEvent(
                event_type=GameEventType.HERO_ROTATION,
                timestamp=timestamped_frame.timestamp,
                frame_number=timestamped_frame.frame_number,
                confidence=0.7,
                metadata={
                    "detection_method": "visual_analysis",
                    "visual_features": visual_features
                }
            )
            events.append(event)
        
        return events
    
    # Helper methods for frame analysis
    def _extract_text_content(self, ocr_result: Dict[str, Any]) -> str:
        """Extract text content from OCR result."""
        if ocr_result.get("success") and ocr_result.get("data"):
            # Extract text from various fields
            text_parts = []
            data = ocr_result["data"]
            
            # Add any text-based information
            for key, value in data.items():
                if isinstance(value, str):
                    text_parts.append(value)
            
            return " ".join(text_parts).lower()
        
        return ""
    
    def _count_visible_heroes(self, ocr_result: Dict[str, Any]) -> int:
        """Count visible heroes in the frame."""
        # Placeholder implementation
        return 0
    
    def _count_ability_effects(self, frame_path: str) -> int:
        """Count ability effects in the frame."""
        # Placeholder implementation
        return 0
    
    def _count_damage_numbers(self, ocr_result: Dict[str, Any]) -> int:
        """Count damage numbers in the frame."""
        # Placeholder implementation
        return 0
    
    def _detect_health_changes(self, ocr_result: Dict[str, Any]) -> int:
        """Detect health changes in the frame."""
        # Placeholder implementation
        return 0
    
    def _detect_cc_effects(self, frame_path: str) -> int:
        """Detect crowd control effects."""
        # Placeholder implementation
        return 0
    
    def _detect_location(self, ocr_result: Dict[str, Any]) -> str:
        """Detect game location/area."""
        # Placeholder implementation
        return "unknown"
    
    def _extract_visual_features(self, frame_path: str) -> Dict[str, Any]:
        """Extract visual features from frame."""
        # Placeholder implementation
        return {}
    
    def generate_event_summary(self, time_range: Tuple[float, float] = None) -> Dict[str, Any]:
        """
        Generate a summary of detected events.
        
        Args:
            time_range: Optional time range (start, end) in seconds
            
        Returns:
            Dictionary containing event summary
        """
        events_to_analyze = self.event_history
        
        if time_range:
            start_time, end_time = time_range
            events_to_analyze = [
                event for event in self.event_history
                if start_time <= event.timestamp <= end_time
            ]
        
        # Count events by type
        event_counts = {}
        for event in events_to_analyze:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Calculate statistics
        total_events = len(events_to_analyze)
        avg_confidence = (
            sum(event.confidence for event in events_to_analyze) / total_events
            if total_events > 0 else 0
        )
        
        return {
            "total_events": total_events,
            "event_counts": event_counts,
            "average_confidence": avg_confidence,
            "time_range": time_range,
            "detection_stats": self.detection_stats.copy(),
            "events": [event.to_dict() for event in events_to_analyze]
        }


def create_event_detector(player_ign: str) -> EventDetector:
    """
    Create and configure an event detector for the specified player.
    
    Args:
        player_ign: Player's in-game name
        
    Returns:
        Configured EventDetector instance
    """
    return EventDetector(player_ign)


# Example usage
if __name__ == "__main__":
    # Example of using the event detector
    detector = create_event_detector("Lesz XVII")
    
    # Example timestamped frame (would come from VideoReader)
    example_frame = TimestampedFrame(
        frame_path="example_frame.jpg",
        timestamp=245.5,
        frame_number=7365,
        confidence=0.9,
        metadata={"fps": 30, "sample_rate": 1}
    )
    
    # Detect events from frame
    events = detector.detect_events_from_frame(example_frame)
    
    print(f"Detected {len(events)} events:")
    for event in events:
        print(f"  - {event.event_type.value} at {event.timestamp:.2f}s (confidence: {event.confidence:.2f})")
    
    # Generate summary
    summary = detector.generate_event_summary()
    print(f"\nEvent Summary:")
    print(f"  Total events: {summary['total_events']}")
    print(f"  Average confidence: {summary['average_confidence']:.2f}")
    print(f"  Event types: {list(summary['event_counts'].keys())}") 