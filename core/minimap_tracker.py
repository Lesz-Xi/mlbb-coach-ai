"""
Minimap Movement Tracking System for MLBB Coach AI
=================================================

This module provides comprehensive minimap analysis capabilities for tracking
player movements, rotations, and positioning patterns with precise timestamps.

Features:
- Player position tracking on minimap
- Movement pattern detection
- Rotation timing analysis
- Lane presence tracking
- Objective area monitoring
- Team coordination analysis
- Timestamp correlation for all movement events
"""

import logging
import time
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import math

# Import existing components
from .video_reader import TimestampedFrame
from .event_detector import GameEvent, GameEventType

logger = logging.getLogger(__name__)


class LaneType(Enum):
    """Types of lanes in MLBB."""
    TOP = "top"
    MID = "mid"
    BOT = "bot"
    JUNGLE = "jungle"
    RIVER = "river"
    BASE = "base"
    UNKNOWN = "unknown"


class MinimapRegion(Enum):
    """Regions on the minimap."""
    TOP_LANE = "top_lane"
    MID_LANE = "mid_lane"
    BOT_LANE = "bot_lane"
    TOP_JUNGLE = "top_jungle"
    BOT_JUNGLE = "bot_jungle"
    RIVER = "river"
    LORD_PIT = "lord_pit"
    TURTLE_PIT = "turtle_pit"
    BLUE_BASE = "blue_base"
    RED_BASE = "red_base"
    UNKNOWN = "unknown"


@dataclass
class Position:
    """2D position with optional metadata."""
    x: float
    y: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class MovementEvent:
    """Represents a movement event with timestamp and trajectory."""
    timestamp: float
    frame_number: int
    from_position: Position
    to_position: Position
    from_region: MinimapRegion
    to_region: MinimapRegion
    movement_type: str  # "rotation", "recall", "teleport", "normal"
    distance: float
    duration: float
    confidence: float
    player_ign: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert movement event to dictionary."""
        return {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "from_position": self.from_position.to_dict(),
            "to_position": self.to_position.to_dict(),
            "from_region": self.from_region.value,
            "to_region": self.to_region.value,
            "movement_type": self.movement_type,
            "distance": self.distance,
            "duration": self.duration,
            "confidence": self.confidence,
            "player_ign": self.player_ign,
            "metadata": self.metadata
        }


class MinimapExtractor:
    """Extracts and preprocesses minimap from game frames."""
    
    def __init__(self):
        # Typical minimap location (bottom-right corner)
        self.minimap_region = (0.75, 0.75, 0.98, 0.98)  # (x1, y1, x2, y2)
        self.minimap_size = (200, 200)  # Standard minimap size
        self.color_ranges = {
            "blue_team": [(100, 50, 50), (130, 255, 255)],  # Blue range in HSV
            "red_team": [(0, 50, 50), (10, 255, 255)],      # Red range in HSV
            "neutral": [(40, 50, 50), (80, 255, 255)]       # Green range in HSV
        }
    
    def extract_minimap(self, frame_path: str) -> Optional[np.ndarray]:
        """
        Extract minimap from game frame.
        
        Args:
            frame_path: Path to the game frame
            
        Returns:
            Extracted minimap as numpy array, or None if extraction fails
        """
        try:
            # Load frame
            frame = cv2.imread(frame_path)
            if frame is None:
                return None
            
            height, width = frame.shape[:2]
            
            # Extract minimap region
            x1 = int(self.minimap_region[0] * width)
            y1 = int(self.minimap_region[1] * height)
            x2 = int(self.minimap_region[2] * width)
            y2 = int(self.minimap_region[3] * height)
            
            minimap = frame[y1:y2, x1:x2]
            
            # Resize to standard size
            minimap_resized = cv2.resize(minimap, self.minimap_size)
            
            return minimap_resized
            
        except Exception as e:
            logger.error(f"Error extracting minimap: {str(e)}")
            return None
    
    def detect_player_positions(self, minimap: np.ndarray) -> Dict[str, List[Position]]:
        """
        Detect player positions on the minimap.
        
        Args:
            minimap: Extracted minimap image
            
        Returns:
            Dictionary mapping team colors to list of positions
        """
        positions = {"blue_team": [], "red_team": [], "neutral": []}
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        
        for team, (lower, upper) in self.color_ranges.items():
            # Create mask for team color
            lower_bound = np.array(lower, dtype=np.uint8)
            upper_bound = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter by area (hero icons should be reasonably sized)
                area = cv2.contourArea(contour)
                if 5 <= area <= 100:  # Reasonable size for hero icons
                    # Get centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Normalize coordinates to 0-1 range
                        x_norm = cx / self.minimap_size[0]
                        y_norm = cy / self.minimap_size[1]
                        
                        position = Position(
                            x=x_norm,
                            y=y_norm,
                            confidence=0.8,
                            metadata={"area": area, "team": team}
                        )
                        positions[team].append(position)
        
        return positions
    
    def classify_region(self, position: Position) -> MinimapRegion:
        """
        Classify which region of the map a position belongs to.
        
        Args:
            position: Position to classify
            
        Returns:
            MinimapRegion enum value
        """
        x, y = position.x, position.y
        
        # Define region boundaries (normalized coordinates)
        if y < 0.25:  # Top area
            if x < 0.3:
                return MinimapRegion.TOP_JUNGLE
            elif x > 0.7:
                return MinimapRegion.TOP_LANE
            else:
                return MinimapRegion.TOP_LANE
        elif y > 0.75:  # Bottom area
            if x < 0.3:
                return MinimapRegion.BOT_LANE
            elif x > 0.7:
                return MinimapRegion.BOT_JUNGLE
            else:
                return MinimapRegion.BOT_LANE
        elif 0.25 <= y <= 0.75:  # Middle area
            if x < 0.2:
                return MinimapRegion.BLUE_BASE
            elif x > 0.8:
                return MinimapRegion.RED_BASE
            elif 0.4 <= x <= 0.6:
                if 0.3 <= y <= 0.4:
                    return MinimapRegion.TURTLE_PIT
                elif 0.6 <= y <= 0.7:
                    return MinimapRegion.LORD_PIT
                else:
                    return MinimapRegion.RIVER
            else:
                return MinimapRegion.MID_LANE
        
        return MinimapRegion.UNKNOWN


class MovementAnalyzer:
    """Analyzes movement patterns and detects rotations."""
    
    def __init__(self, player_ign: str):
        self.player_ign = player_ign
        self.position_history = []
        self.movement_threshold = 0.1  # Minimum movement distance
        self.rotation_threshold = 0.3  # Minimum distance for rotation
        self.max_history_size = 50
        
    def analyze_movement(self, positions: Dict[str, List[Position]], 
                        timestamp: float, frame_number: int) -> List[MovementEvent]:
        """
        Analyze movement patterns from detected positions.
        
        Args:
            positions: Detected positions by team
            timestamp: Current timestamp
            frame_number: Current frame number
            
        Returns:
            List of detected movement events
        """
        movement_events = []
        
        # For now, focus on blue team movements (can be extended)
        current_positions = positions.get("blue_team", [])
        
        if not current_positions:
            return movement_events
        
        # If we have previous positions, analyze movement
        if len(self.position_history) > 0:
            previous_entry = self.position_history[-1]
            previous_positions = previous_entry["positions"]
            previous_timestamp = previous_entry["timestamp"]
            
            # Match positions between frames (simplified)
            for i, current_pos in enumerate(current_positions):
                if i < len(previous_positions):
                    previous_pos = previous_positions[i]
                    distance = current_pos.distance_to(previous_pos)
                    
                    if distance >= self.movement_threshold:
                        # Significant movement detected
                        duration = timestamp - previous_timestamp
                        
                        # Classify movement type
                        movement_type = self._classify_movement(
                            previous_pos, current_pos, distance, duration
                        )
                        
                        # Determine regions
                        minimap_extractor = MinimapExtractor()
                        from_region = minimap_extractor.classify_region(previous_pos)
                        to_region = minimap_extractor.classify_region(current_pos)
                        
                        # Create movement event
                        movement_event = MovementEvent(
                            timestamp=timestamp,
                            frame_number=frame_number,
                            from_position=previous_pos,
                            to_position=current_pos,
                            from_region=from_region,
                            to_region=to_region,
                            movement_type=movement_type,
                            distance=distance,
                            duration=duration,
                            confidence=min(previous_pos.confidence, current_pos.confidence),
                            player_ign=self.player_ign,
                            metadata={
                                "speed": distance / duration if duration > 0 else 0,
                                "previous_timestamp": previous_timestamp
                            }
                        )
                        
                        movement_events.append(movement_event)
        
        # Update position history
        self.position_history.append({
            "timestamp": timestamp,
            "frame_number": frame_number,
            "positions": current_positions
        })
        
        # Keep history size manageable
        if len(self.position_history) > self.max_history_size:
            self.position_history.pop(0)
        
        return movement_events
    
    def _classify_movement(self, from_pos: Position, to_pos: Position, 
                          distance: float, duration: float) -> str:
        """Classify the type of movement."""
        speed = distance / duration if duration > 0 else 0
        
        if distance >= self.rotation_threshold:
            return "rotation"
        elif speed > 0.5:  # Very fast movement
            return "teleport"
        elif distance < 0.05:  # Very small movement
            return "adjustment"
        else:
            return "normal"


class MinimapTracker:
    """Main minimap tracking system."""
    
    def __init__(self, player_ign: str):
        self.player_ign = player_ign
        self.minimap_extractor = MinimapExtractor()
        self.movement_analyzer = MovementAnalyzer(player_ign)
        
        # Tracking parameters
        self.tracking_enabled = True
        self.min_confidence = 0.5
        
        # Statistics
        self.tracking_stats = {
            "total_frames_processed": 0,
            "successful_extractions": 0,
            "movement_events_detected": 0,
            "average_processing_time": 0.0
        }
        
        # Event storage
        self.movement_history = []
        self.position_heatmap = {}
    
    def track_movement_from_frame(self, timestamped_frame: TimestampedFrame) -> List[MovementEvent]:
        """
        Track player movement from a timestamped frame.
        
        Args:
            timestamped_frame: Frame with timestamp information
            
        Returns:
            List of detected movement events
        """
        start_time = time.time()
        movement_events = []
        
        try:
            # Extract minimap
            minimap = self.minimap_extractor.extract_minimap(timestamped_frame.frame_path)
            
            if minimap is not None:
                self.tracking_stats["successful_extractions"] += 1
                
                # Detect player positions
                positions = self.minimap_extractor.detect_player_positions(minimap)
                
                # Analyze movement
                movement_events = self.movement_analyzer.analyze_movement(
                    positions, timestamped_frame.timestamp, timestamped_frame.frame_number
                )
                
                # Update heatmap
                self._update_position_heatmap(positions, timestamped_frame.timestamp)
                
                # Store movement events
                self.movement_history.extend(movement_events)
                
                # Keep history size manageable
                if len(self.movement_history) > 200:
                    self.movement_history = self.movement_history[-200:]
                
                self.tracking_stats["movement_events_detected"] += len(movement_events)
            
            # Update statistics
            self.tracking_stats["total_frames_processed"] += 1
            processing_time = time.time() - start_time
            self.tracking_stats["average_processing_time"] = (
                (self.tracking_stats["average_processing_time"] * 
                 (self.tracking_stats["total_frames_processed"] - 1) + processing_time) /
                self.tracking_stats["total_frames_processed"]
            )
            
        except Exception as e:
            logger.error(f"Error tracking movement from frame at {timestamped_frame.timestamp:.2f}s: {str(e)}")
        
        return movement_events
    
    def _update_position_heatmap(self, positions: Dict[str, List[Position]], timestamp: float):
        """Update position heatmap for analysis."""
        for team, team_positions in positions.items():
            if team not in self.position_heatmap:
                self.position_heatmap[team] = {}
            
            for pos in team_positions:
                # Create grid key (simplified heatmap)
                grid_x = int(pos.x * 10)  # 10x10 grid
                grid_y = int(pos.y * 10)
                grid_key = f"{grid_x},{grid_y}"
                
                if grid_key not in self.position_heatmap[team]:
                    self.position_heatmap[team][grid_key] = []
                
                self.position_heatmap[team][grid_key].append(timestamp)
    
    def get_movement_summary(self, time_range: Tuple[float, float] = None) -> Dict[str, Any]:
        """
        Generate a summary of movement patterns.
        
        Args:
            time_range: Optional time range (start, end) in seconds
            
        Returns:
            Dictionary containing movement analysis
        """
        events_to_analyze = self.movement_history
        
        if time_range:
            start_time, end_time = time_range
            events_to_analyze = [
                event for event in self.movement_history
                if start_time <= event.timestamp <= end_time
            ]
        
        # Analyze movement patterns
        movement_types = {}
        region_transitions = {}
        total_distance = 0
        
        for event in events_to_analyze:
            # Count movement types
            movement_type = event.movement_type
            movement_types[movement_type] = movement_types.get(movement_type, 0) + 1
            
            # Count region transitions
            transition = f"{event.from_region.value} -> {event.to_region.value}"
            region_transitions[transition] = region_transitions.get(transition, 0) + 1
            
            # Sum total distance
            total_distance += event.distance
        
        # Calculate average values
        total_events = len(events_to_analyze)
        avg_distance = total_distance / total_events if total_events > 0 else 0
        avg_confidence = (
            sum(event.confidence for event in events_to_analyze) / total_events
            if total_events > 0 else 0
        )
        
        return {
            "total_movement_events": total_events,
            "movement_types": movement_types,
            "region_transitions": region_transitions,
            "total_distance_traveled": total_distance,
            "average_distance_per_move": avg_distance,
            "average_confidence": avg_confidence,
            "tracking_stats": self.tracking_stats.copy(),
            "position_heatmap": self.position_heatmap,
            "time_range": time_range,
            "events": [event.to_dict() for event in events_to_analyze]
        }
    
    def detect_rotation_patterns(self) -> List[Dict[str, Any]]:
        """
        Detect common rotation patterns.
        
        Returns:
            List of detected rotation patterns
        """
        rotation_events = [
            event for event in self.movement_history
            if event.movement_type == "rotation"
        ]
        
        # Group rotations by common patterns
        patterns = {}
        
        for event in rotation_events:
            pattern_key = f"{event.from_region.value}_to_{event.to_region.value}"
            if pattern_key not in patterns:
                patterns[pattern_key] = []
            patterns[pattern_key].append(event)
        
        # Analyze patterns
        pattern_analysis = []
        for pattern_key, events in patterns.items():
            if len(events) >= 2:  # At least 2 occurrences
                avg_duration = sum(event.duration for event in events) / len(events)
                avg_distance = sum(event.distance for event in events) / len(events)
                
                pattern_analysis.append({
                    "pattern": pattern_key,
                    "frequency": len(events),
                    "average_duration": avg_duration,
                    "average_distance": avg_distance,
                    "timestamps": [event.timestamp for event in events],
                    "confidence": sum(event.confidence for event in events) / len(events)
                })
        
        # Sort by frequency
        pattern_analysis.sort(key=lambda x: x["frequency"], reverse=True)
        
        return pattern_analysis


def create_minimap_tracker(player_ign: str) -> MinimapTracker:
    """
    Create and configure a minimap tracker for the specified player.
    
    Args:
        player_ign: Player's in-game name
        
    Returns:
        Configured MinimapTracker instance
    """
    return MinimapTracker(player_ign)


# Example usage
if __name__ == "__main__":
    # Example of using the minimap tracker
    tracker = create_minimap_tracker("Lesz XVII")
    
    # Example timestamped frame
    example_frame = TimestampedFrame(
        frame_path="example_frame.jpg",
        timestamp=180.0,
        frame_number=5400,
        confidence=0.9,
        metadata={"fps": 30, "sample_rate": 1}
    )
    
    # Track movement from frame
    movement_events = tracker.track_movement_from_frame(example_frame)
    
    print(f"Detected {len(movement_events)} movement events:")
    for event in movement_events:
        print(f"  - {event.movement_type} from {event.from_region.value} to {event.to_region.value}")
        print(f"    at {event.timestamp:.2f}s (distance: {event.distance:.3f})")
    
    # Generate movement summary
    summary = tracker.get_movement_summary()
    print(f"\nMovement Summary:")
    print(f"  Total movements: {summary['total_movement_events']}")
    print(f"  Distance traveled: {summary['total_distance_traveled']:.3f}")
    print(f"  Movement types: {summary['movement_types']}")
    
    # Detect rotation patterns
    patterns = tracker.detect_rotation_patterns()
    print(f"\nRotation Patterns:")
    for pattern in patterns[:3]:  # Top 3 patterns
        print(f"  - {pattern['pattern']}: {pattern['frequency']} times")
        print(f"    Average duration: {pattern['average_duration']:.2f}s") 