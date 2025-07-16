"""
Enhanced Minimap Tracker with YOLOv8 Integration
===============================================

This module extends the existing minimap tracker with YOLOv8 object detection
capabilities for improved UI element detection and minimap region identification.

Features:
- YOLOv8-based minimap detection
- Enhanced UI element tracking
- Fallback to traditional CV methods
- Performance monitoring
- Integration with existing pipeline
"""

import logging
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from .minimap_tracker import (
    MinimapTracker, 
    MinimapExtractor, 
    MovementAnalyzer,
    Position,
    MinimapRegion,
    MovementEvent
)
from .services.yolo_detection_service import get_yolo_detection_service
from .yolo_fallback import should_use_yolo
from .video_reader import TimestampedFrame

logger = logging.getLogger(__name__)


class EnhancedMinimapExtractor(MinimapExtractor):
    """Enhanced minimap extractor with YOLOv8 detection capabilities."""
    
    def __init__(self, use_yolo: bool = True):
        """
        Initialize enhanced minimap extractor.
        
        Args:
            use_yolo: Whether to use YOLOv8 for minimap detection
        """
        super().__init__()
        self.use_yolo = use_yolo
        self.yolo_service = get_yolo_detection_service() if use_yolo else None
        
        # Performance tracking
        self.detection_stats = {
            'yolo_detections': 0,
            'cv_fallbacks': 0,
            'detection_times': []
        }
        
        logger.info(f"🎯 Enhanced Minimap Extractor initialized (YOLOv8: {'enabled' if use_yolo else 'disabled'})")
    
    def extract_minimap(self, frame_path: str, ocr_confidence: float = 1.0) -> Optional[np.ndarray]:
        """
        Extract minimap using YOLOv8 detection with CV fallback.
        
        Args:
            frame_path: Path to the frame image
            ocr_confidence: OCR confidence for fallback decision
            
        Returns:
            Extracted minimap image or None if not found
        """
        try:
            # Try YOLOv8 detection first
            if self.use_yolo and self.yolo_service and should_use_yolo(ocr_confidence, 0.6):
                minimap_region = self.yolo_service.get_minimap_region(frame_path)
                
                if minimap_region and minimap_region['confidence'] > 0.5:
                    logger.info(f"🎯 YOLO minimap detected (confidence: {minimap_region['confidence']:.3f})")
                    
                    # Extract minimap using YOLO bounding box
                    image = cv2.imread(frame_path)
                    if image is not None:
                        x1, y1, x2, y2 = [int(coord) for coord in minimap_region['bbox']]
                        minimap = image[y1:y2, x1:x2]
                        
                        self.detection_stats['yolo_detections'] += 1
                        return minimap
            
            # Fallback to traditional CV method
            logger.info("🔄 Falling back to traditional CV minimap detection")
            self.detection_stats['cv_fallbacks'] += 1
            return super().extract_minimap(frame_path)
            
        except Exception as e:
            logger.error(f"❌ Enhanced minimap extraction failed: {e}")
            return super().extract_minimap(frame_path)
    
    def get_ui_elements(self, frame_path: str) -> Dict[str, Any]:
        """
        Get UI elements using YOLOv8 detection.
        
        Args:
            frame_path: Path to the frame image
            
        Returns:
            Dictionary of detected UI elements
        """
        if not self.use_yolo or not self.yolo_service:
            return {}
        
        try:
            ui_elements = self.yolo_service.detect_ui_elements(frame_path)
            
            # Organize by element type
            organized_elements = {
                'timer': [],
                'score_indicator': [],
                'minimap': [],
                'other': []
            }
            
            for element in ui_elements:
                element_type = element['class_name']
                if element_type in organized_elements:
                    organized_elements[element_type].append(element)
                else:
                    organized_elements['other'].append(element)
            
            logger.info(f"🎮 Detected {len(ui_elements)} UI elements")
            return organized_elements
            
        except Exception as e:
            logger.error(f"❌ UI element detection failed: {e}")
            return {}
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        total_detections = self.detection_stats['yolo_detections'] + self.detection_stats['cv_fallbacks']
        
        return {
            'total_detections': total_detections,
            'yolo_detections': self.detection_stats['yolo_detections'],
            'cv_fallbacks': self.detection_stats['cv_fallbacks'],
            'yolo_success_rate': (self.detection_stats['yolo_detections'] / total_detections) if total_detections > 0 else 0,
            'avg_detection_time': np.mean(self.detection_stats['detection_times']) if self.detection_stats['detection_times'] else 0
        }


class EnhancedMinimapTracker(MinimapTracker):
    """Enhanced minimap tracker with YOLOv8 integration."""
    
    def __init__(self, player_ign: str, use_yolo: bool = True):
        """
        Initialize enhanced minimap tracker.
        
        Args:
            player_ign: Player's in-game name
            use_yolo: Whether to use YOLOv8 enhancements
        """
        # Initialize with enhanced extractor
        self.player_ign = player_ign
        self.use_yolo = use_yolo
        
        # Initialize components
        self.minimap_extractor = EnhancedMinimapExtractor(use_yolo)
        self.movement_analyzer = MovementAnalyzer(player_ign)
        
        # Initialize parent class attributes
        self.movement_history = []
        self.position_heatmap = {}
        self.last_positions = {}
        
        # Enhanced features
        self.ui_element_history = []
        self.detection_confidence_history = []
        
        logger.info(f"🎯 Enhanced Minimap Tracker initialized for {player_ign}")
    
    def track_movement_from_frame(self, timestamped_frame: TimestampedFrame) -> List[MovementEvent]:
        """
        Enhanced movement tracking with YOLOv8 integration.
        
        Args:
            timestamped_frame: Frame with timestamp information
            
        Returns:
            List of movement events detected
        """
        try:
            # Extract minimap using enhanced method
            minimap = self.minimap_extractor.extract_minimap(
                timestamped_frame.frame_path,
                ocr_confidence=0.8  # Assume good OCR confidence for now
            )
            
            if minimap is None:
                logger.warning(f"⚠️ No minimap found in frame {timestamped_frame.frame_number}")
                return []
            
            # Get UI elements for additional context
            ui_elements = self.minimap_extractor.get_ui_elements(timestamped_frame.frame_path)
            
            # Detect player positions in minimap
            positions = self.minimap_extractor.detect_player_positions(minimap)
            
            # Analyze movement with enhanced context
            movement_events = self.movement_analyzer.analyze_movement(
                positions,
                timestamped_frame.timestamp,
                timestamped_frame.frame_number
            )
            
            # Enhance movement events with UI context
            enhanced_events = self._enhance_movement_events(movement_events, ui_elements)
            
            # Update history
            self.movement_history.extend(enhanced_events)
            self._update_position_heatmap(positions, timestamped_frame.timestamp)
            
            # Store UI element history for analysis
            self.ui_element_history.append({
                'timestamp': timestamped_frame.timestamp,
                'frame_number': timestamped_frame.frame_number,
                'ui_elements': ui_elements
            })
            
            logger.info(f"🎯 Tracked {len(enhanced_events)} movement events in frame {timestamped_frame.frame_number}")
            return enhanced_events
            
        except Exception as e:
            logger.error(f"❌ Enhanced movement tracking failed: {e}")
            return []
    
    def _enhance_movement_events(self, 
                               movement_events: List[MovementEvent], 
                               ui_elements: Dict[str, Any]) -> List[MovementEvent]:
        """
        Enhance movement events with UI context.
        
        Args:
            movement_events: Original movement events
            ui_elements: Detected UI elements
            
        Returns:
            Enhanced movement events
        """
        enhanced_events = []
        
        for event in movement_events:
            # Add UI context to metadata
            event.metadata.update({
                'ui_context': {
                    'timer_visible': len(ui_elements.get('timer', [])) > 0,
                    'score_visible': len(ui_elements.get('score_indicator', [])) > 0,
                    'ui_confidence': self._calculate_ui_confidence(ui_elements)
                }
            })
            
            # Enhance movement classification based on UI elements
            if self._detect_teamfight_context(ui_elements):
                event.metadata['teamfight_context'] = True
                event.movement_type = 'teamfight_rotation'
            
            if self._detect_objective_context(ui_elements):
                event.metadata['objective_context'] = True
                event.movement_type = 'objective_rotation'
            
            enhanced_events.append(event)
        
        return enhanced_events
    
    def _calculate_ui_confidence(self, ui_elements: Dict[str, Any]) -> float:
        """
        Calculate overall UI detection confidence.
        
        Args:
            ui_elements: Detected UI elements
            
        Returns:
            Overall confidence score
        """
        all_confidences = []
        
        for element_type, elements in ui_elements.items():
            for element in elements:
                all_confidences.append(element.get('confidence', 0.0))
        
        return np.mean(all_confidences) if all_confidences else 0.0
    
    def _detect_teamfight_context(self, ui_elements: Dict[str, Any]) -> bool:
        """
        Detect if current frame shows teamfight context.
        
        Args:
            ui_elements: Detected UI elements
            
        Returns:
            True if teamfight context detected
        """
        # Look for multiple score indicators or specific UI patterns
        score_indicators = ui_elements.get('score_indicator', [])
        return len(score_indicators) > 2
    
    def _detect_objective_context(self, ui_elements: Dict[str, Any]) -> bool:
        """
        Detect if current frame shows objective context.
        
        Args:
            ui_elements: Detected UI elements
            
        Returns:
            True if objective context detected
        """
        # Look for timer elements or specific objective indicators
        timers = ui_elements.get('timer', [])
        return len(timers) > 0
    
    def get_enhanced_movement_summary(self, time_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Get enhanced movement summary with UI context.
        
        Args:
            time_range: Optional time range filter
            
        Returns:
            Enhanced movement summary
        """
        # Get base summary
        base_summary = self.get_movement_summary(time_range)
        
        # Add enhanced metrics
        ui_confidence_history = [
            event.metadata.get('ui_context', {}).get('ui_confidence', 0.0)
            for event in self.movement_history
        ]
        
        teamfight_rotations = [
            event for event in self.movement_history
            if event.metadata.get('teamfight_context', False)
        ]
        
        objective_rotations = [
            event for event in self.movement_history
            if event.metadata.get('objective_context', False)
        ]
        
        # Enhanced summary
        enhanced_summary = {
            **base_summary,
            'ui_detection_stats': self.minimap_extractor.get_detection_stats(),
            'avg_ui_confidence': np.mean(ui_confidence_history) if ui_confidence_history else 0.0,
            'teamfight_rotations': len(teamfight_rotations),
            'objective_rotations': len(objective_rotations),
            'ui_element_frames': len(self.ui_element_history),
            'enhancement_ratio': self._calculate_enhancement_ratio()
        }
        
        return enhanced_summary
    
    def _calculate_enhancement_ratio(self) -> float:
        """
        Calculate the enhancement ratio (YOLOv8 vs traditional methods).
        
        Returns:
            Enhancement ratio as percentage
        """
        stats = self.minimap_extractor.get_detection_stats()
        total = stats['total_detections']
        
        if total == 0:
            return 0.0
        
        return (stats['yolo_detections'] / total) * 100.0
    
    def export_enhanced_data(self, output_path: str) -> str:
        """
        Export enhanced tracking data.
        
        Args:
            output_path: Path to save exported data
            
        Returns:
            Path to exported file
        """
        export_data = {
            'player_ign': self.player_ign,
            'tracking_summary': self.get_enhanced_movement_summary(),
            'movement_events': [event.to_dict() for event in self.movement_history],
            'ui_element_history': self.ui_element_history,
            'detection_stats': self.minimap_extractor.get_detection_stats(),
            'enhancement_enabled': self.use_yolo
        }
        
        # Save to file
        import json
        export_file = Path(output_path) / f'enhanced_minimap_tracking_{self.player_ign}.json'
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"📦 Enhanced tracking data exported to: {export_file}")
        return str(export_file)


def create_enhanced_minimap_tracker(player_ign: str, use_yolo: bool = True) -> EnhancedMinimapTracker:
    """
    Create an enhanced minimap tracker with YOLOv8 integration.
    
    Args:
        player_ign: Player's in-game name
        use_yolo: Whether to use YOLOv8 enhancements
        
    Returns:
        Enhanced minimap tracker instance
    """
    return EnhancedMinimapTracker(player_ign, use_yolo)


# For backward compatibility
def create_minimap_tracker(player_ign: str) -> EnhancedMinimapTracker:
    """
    Create a minimap tracker (enhanced version by default).
    
    Args:
        player_ign: Player's in-game name
        
    Returns:
        Enhanced minimap tracker instance
    """
    return create_enhanced_minimap_tracker(player_ign, use_yolo=True) 