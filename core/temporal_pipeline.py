"""
Comprehensive Temporal Pipeline for MLBB Coach AI
================================================

This module integrates all timestamp tracking systems to provide end-to-end
temporal analysis of MLBB gameplay videos.

Components:
1. VideoReader with timestamp tracking
2. Enhanced OCR with timestamp correlation
3. Event detection with temporal analysis
4. Minimap movement tracking
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Import all components
from .video_reader import VideoReader, TimestampedFrame
from .enhanced_data_collector import EnhancedDataCollector
from .event_detector import EventDetector, GameEvent, GameEventType
from .minimap_tracker import MinimapTracker, MovementEvent

logger = logging.getLogger(__name__)


@dataclass
class TemporalAnalysisResult:
    """Complete temporal analysis result with all detected events."""
    video_path: str
    player_ign: str
    total_duration: float
    processing_time: float
    
    # Frame analysis
    total_frames: int
    processed_frames: int
    frame_timestamps: List[float]
    
    # OCR analysis results
    ocr_results: List[Dict[str, Any]]
    
    # Detected events
    game_events: List[GameEvent]
    movement_events: List[MovementEvent]
    
    # Summary statistics
    event_summary: Dict[str, Any]
    movement_summary: Dict[str, Any]
    
    # Performance metrics
    performance_metrics: Dict[str, Any]
    
    def save_to_json(self, output_path: str):
        """Save results to JSON file."""
        data = {
            "video_path": self.video_path,
            "player_ign": self.player_ign,
            "total_duration": self.total_duration,
            "processing_time": self.processing_time,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "frame_timestamps": self.frame_timestamps,
            "ocr_results": self.ocr_results,
            "game_events": [event.to_dict() for event in self.game_events],
            "movement_events": [event.to_dict() for event in self.movement_events],
            "event_summary": self.event_summary,
            "movement_summary": self.movement_summary,
            "performance_metrics": self.performance_metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Temporal analysis results saved to {output_path}")


class TemporalPipeline:
    """Main temporal analysis pipeline."""
    
    def __init__(self, player_ign: str):
        self.player_ign = player_ign
        
        # Initialize all components
        self.video_reader = VideoReader()
        self.data_collector = EnhancedDataCollector()
        self.event_detector = EventDetector(player_ign)
        self.minimap_tracker = MinimapTracker(player_ign)
        
        # Pipeline configuration
        self.sample_rate = 1  # 1 frame per second
        self.enable_minimap_tracking = True
        self.enable_event_detection = True
        
        # Performance tracking
        self.pipeline_stats = {
            "total_videos_processed": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "frames_per_second": 0.0
        }
    
    def analyze_video(self, video_path: str, 
                     output_dir: str = "temp/analysis_results") -> TemporalAnalysisResult:
        """
        Perform comprehensive temporal analysis on a video.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save analysis results
            
        Returns:
            TemporalAnalysisResult with all detected events and metadata
        """
        start_time = time.time()
        
        logger.info(f"Starting temporal analysis of video: {video_path}")
        logger.info(f"Target player: {self.player_ign}")
        
        # Step 1: Extract timestamped frames
        logger.info("Step 1: Extracting timestamped frames...")
        result = self.video_reader.analyze_video(video_path, self.player_ign)
        
        if not result.success:
            raise ValueError(f"Failed to analyze video: {result.warnings}")
        
        timestamped_frames = result.timestamped_frames
        
        # Step 2: Process each frame with timestamp correlation
        logger.info("Step 2: Processing frames with timestamp correlation...")
        ocr_results = []
        game_events = []
        movement_events = []
        
        for i, frame in enumerate(timestamped_frames):
            logger.info(f"Processing frame {i+1}/{len(timestamped_frames)} at {frame.timestamp:.2f}s")
            
            try:
                # OCR Analysis with timestamp
                ocr_result = self.data_collector.analyze_screenshot_with_timestamp(
                    image_path=frame.frame_path,
                    ign=self.player_ign,
                    video_timestamp=frame.timestamp,
                    frame_number=frame.frame_number,
                    frame_metadata=frame.metadata
                )
                ocr_results.append(ocr_result)
                
                # Event Detection
                if self.enable_event_detection:
                    frame_events = self.event_detector.detect_events_from_frame(frame)
                    game_events.extend(frame_events)
                
                # Minimap Tracking
                if self.enable_minimap_tracking:
                    frame_movements = self.minimap_tracker.track_movement_from_frame(frame)
                    movement_events.extend(frame_movements)
                
            except Exception as e:
                logger.error(f"Error processing frame at {frame.timestamp:.2f}s: {str(e)}")
                continue
        
        # Step 3: Generate comprehensive summaries
        logger.info("Step 3: Generating analysis summaries...")
        event_summary = self.event_detector.generate_event_summary()
        movement_summary = self.minimap_tracker.get_movement_summary()
        
        # Step 4: Calculate performance metrics
        processing_time = time.time() - start_time
        video_info = self.video_reader.get_video_info(video_path)
        
        performance_metrics = {
            "total_processing_time": processing_time,
            "frames_processed": len(timestamped_frames),
            "processing_fps": len(timestamped_frames) / processing_time if processing_time > 0 else 0,
            "video_duration": video_info.get("duration", 0),
            "real_time_ratio": video_info.get("duration", 0) / processing_time if processing_time > 0 else 0,
            "ocr_success_rate": sum(1 for r in ocr_results if r.get("success", False)) / len(ocr_results) if ocr_results else 0,
            "events_per_minute": len(game_events) / (video_info.get("duration", 1) / 60) if video_info.get("duration", 0) > 0 else 0,
            "movements_per_minute": len(movement_events) / (video_info.get("duration", 1) / 60) if video_info.get("duration", 0) > 0 else 0
        }
        
        # Create result object
        result = TemporalAnalysisResult(
            video_path=video_path,
            player_ign=self.player_ign,
            total_duration=video_info.get("duration", 0),
            processing_time=processing_time,
            total_frames=len(timestamped_frames),
            processed_frames=len([r for r in ocr_results if r.get("success", False)]),
            frame_timestamps=[frame.timestamp for frame in timestamped_frames],
            ocr_results=ocr_results,
            game_events=game_events,
            movement_events=movement_events,
            event_summary=event_summary,
            movement_summary=movement_summary,
            performance_metrics=performance_metrics
        )
        
        # Step 5: Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        json_path = output_path / f"temporal_analysis_{timestamp_str}.json"
        result.save_to_json(str(json_path))
        
        # Save sample event log
        self._save_sample_event_log(result, str(output_path / f"event_log_{timestamp_str}.json"))
        
        # Update pipeline statistics
        self.pipeline_stats["total_videos_processed"] += 1
        self.pipeline_stats["total_processing_time"] += processing_time
        self.pipeline_stats["average_processing_time"] = (
            self.pipeline_stats["total_processing_time"] / 
            self.pipeline_stats["total_videos_processed"]
        )
        self.pipeline_stats["frames_per_second"] = performance_metrics["processing_fps"]
        
        logger.info(f"Temporal analysis completed in {processing_time:.2f}s")
        logger.info(f"Detected {len(game_events)} game events and {len(movement_events)} movement events")
        
        return result
    
    def _save_sample_event_log(self, result: TemporalAnalysisResult, output_path: str):
        """Save a sample event log in the requested format."""
        
        # Create sample event log showing timestamped player events
        sample_events = []
        
        # Add game events
        for event in result.game_events[:10]:  # Top 10 events
            sample_events.append({
                "timestamp": event.timestamp,
                "event_type": "game_event",
                "event_subtype": event.event_type.value,
                "player": event.player_ign,
                "confidence": event.confidence,
                "metadata": event.metadata
            })
        
        # Add movement events
        for event in result.movement_events[:5]:  # Top 5 movements
            sample_events.append({
                "timestamp": event.timestamp,
                "event_type": "movement_event",
                "event_subtype": event.movement_type,
                "player": event.player_ign,
                "from_region": event.from_region.value,
                "to_region": event.to_region.value,
                "distance": event.distance,
                "confidence": event.confidence,
                "metadata": event.metadata
            })
        
        # Sort by timestamp
        sample_events.sort(key=lambda x: x["timestamp"])
        
        # Create sample log structure
        sample_log = {
            "video_analysis": {
                "video_path": result.video_path,
                "player_ign": result.player_ign,
                "total_duration": result.total_duration,
                "analysis_timestamp": time.time()
            },
            "timestamped_events": sample_events,
            "event_summary": {
                "total_events": len(sample_events),
                "game_events": len([e for e in sample_events if e["event_type"] == "game_event"]),
                "movement_events": len([e for e in sample_events if e["event_type"] == "movement_event"]),
                "time_span": {
                    "start": min(e["timestamp"] for e in sample_events) if sample_events else 0,
                    "end": max(e["timestamp"] for e in sample_events) if sample_events else 0
                }
            },
            "sample_events": {
                "hero_rotation": {
                    "timestamp": 245.5,
                    "event_type": "movement_event",
                    "event_subtype": "rotation",
                    "player": result.player_ign,
                    "from_region": "mid_lane",
                    "to_region": "bot_lane",
                    "distance": 0.45,
                    "confidence": 0.85,
                    "metadata": {
                        "speed": 0.12,
                        "duration": 3.75,
                        "frame_path": "frame_007365_t245.50s.jpg"
                    }
                },
                "tower_destruction": {
                    "timestamp": 512.3,
                    "event_type": "game_event",
                    "event_subtype": "tower_destroyed",
                    "player": result.player_ign,
                    "confidence": 0.92,
                    "metadata": {
                        "detection_method": "text_pattern",
                        "gold_change": 320,
                        "frame_path": "frame_015369_t512.30s.jpg"
                    }
                }
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(sample_log, f, indent=2)
        
        logger.info(f"Sample event log saved to {output_path}")


def create_temporal_pipeline(player_ign: str) -> TemporalPipeline:
    """
    Create and configure a temporal analysis pipeline.
    
    Args:
        player_ign: Player's in-game name
        
    Returns:
        Configured TemporalPipeline instance
    """
    return TemporalPipeline(player_ign)


# Example usage and testing
if __name__ == "__main__":
    # Create pipeline
    pipeline = create_temporal_pipeline("Lesz XVII")
    
    # Example video path (replace with actual video)
    video_path = "example_gameplay.mp4"
    
    if Path(video_path).exists():
        print(f"Analyzing video: {video_path}")
        
        # Run complete temporal analysis
        result = pipeline.analyze_video(video_path)
        
        print(f"\n=== TEMPORAL ANALYSIS RESULTS ===")
        print(f"Video Duration: {result.total_duration:.2f}s")
        print(f"Processing Time: {result.processing_time:.2f}s")
        print(f"Real-time Ratio: {result.performance_metrics['real_time_ratio']:.2f}x")
        print(f"Frames Processed: {result.processed_frames}/{result.total_frames}")
        
        print(f"\n=== DETECTED EVENTS ===")
        print(f"Game Events: {len(result.game_events)}")
        print(f"Movement Events: {len(result.movement_events)}")
        
        if result.game_events:
            print(f"\nTop Game Events:")
            for i, event in enumerate(result.game_events[:5]):
                print(f"  {i+1}. {event.event_type.value} at {event.timestamp:.2f}s "
                      f"(confidence: {event.confidence:.2f})")
        
        if result.movement_events:
            print(f"\nTop Movement Events:")
            for i, event in enumerate(result.movement_events[:5]):
                print(f"  {i+1}. {event.movement_type} from {event.from_region.value} "
                      f"to {event.to_region.value} at {event.timestamp:.2f}s")
        
        print(f"\n=== PERFORMANCE METRICS ===")
        for key, value in result.performance_metrics.items():
            print(f"{key}: {value:.3f}")
        
        print(f"\nResults saved to temp/analysis_results/")
    
    else:
        print(f"Video file not found: {video_path}")
        print("Please provide a valid MLBB gameplay video to test the temporal pipeline.")
        
        # Show example of what the system can detect
        print(f"\n=== EXAMPLE EVENT LOG ===")
        sample_log = {
            "video_analysis": {
                "video_path": "example_gameplay.mp4",
                "player_ign": "Lesz XVII",
                "total_duration": 720.5,
                "analysis_timestamp": time.time()
            },
            "timestamped_events": [
                {
                    "timestamp": 245.5,
                    "event_type": "movement_event",
                    "event_subtype": "rotation",
                    "player": "Lesz XVII",
                    "from_region": "mid_lane",
                    "to_region": "bot_lane",
                    "distance": 0.45,
                    "confidence": 0.85
                },
                {
                    "timestamp": 512.3,
                    "event_type": "game_event",
                    "event_subtype": "tower_destroyed",
                    "player": "Lesz XVII",
                    "confidence": 0.92
                }
            ]
        }
        
        print(json.dumps(sample_log, indent=2)) 