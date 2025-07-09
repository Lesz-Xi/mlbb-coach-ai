import cv2
import os
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
# from moviepy.editor import VideoFileClip  # Optional dependency
# import ffmpeg  # Optional dependency for video info

from .data_collector import DataCollector
from .ign_validator import IGNValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VideoAnalysisResult:
    """Result of video analysis with extracted match data."""
    success: bool
    match_data: Dict[str, Any]
    frame_count: int
    processed_frames: int
    confidence_score: float
    warnings: List[str]
    processing_time: float


class VideoReader:
    """
    Extracts match statistics from gameplay video recordings.
    
    Uses frame sampling and OCR to analyze video content and extract
    player statistics at key moments during gameplay.
    """
    
    def __init__(self, temp_dir: str = "temp/video_frames"):
        """
        Initialize the video reader.
        
        Args:
            temp_dir: Directory to store temporary frame extractions
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data collector for OCR processing
        self.data_collector = DataCollector()
        
        # Video processing parameters
        self.target_fps = 1  # Extract 1 frame per second
        self.min_video_duration = 5  # Minimum video duration in seconds
        self.max_video_duration = 1800  # Maximum video duration (30 minutes)
        
        # Frame analysis parameters
        self.score_screen_keywords = [
            "victory", "defeat", "duration", "kills", "deaths", "assists",
            "gold", "damage", "kda", "match", "result"
        ]
        
        # Video format support
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    def validate_video_file(self, video_path: str) -> Tuple[bool, str]:
        """
        Validate if video file is suitable for analysis.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        video_path = Path(video_path)
        
        # Check if file exists
        if not video_path.exists():
            return False, f"Video file not found: {video_path}"
        
        # Check file extension
        if video_path.suffix.lower() not in self.supported_formats:
            return False, f"Unsupported video format: {video_path.suffix}"
        
        # Check file size (max 500MB)
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 500:
            return False, f"Video file too large: {file_size_mb:.1f}MB (max 500MB)"
        
        try:
            # Basic validation using OpenCV
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return False, "Cannot open video file"
            
            # Get basic properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            if duration < self.min_video_duration:
                return False, f"Video too short: {duration:.1f}s (min {self.min_video_duration}s)"
            
            if duration > self.max_video_duration:
                return False, f"Video too long: {duration:.1f}s (max {self.max_video_duration}s)"
            
            return True, ""
            
        except Exception as e:
            return False, f"Error reading video properties: {str(e)}"
    
    def extract_frames(self, video_path: str, sample_rate: int = None) -> List[str]:
        """
        Extract frames from video at specified intervals.
        
        Args:
            video_path: Path to video file
            sample_rate: Frames per second to extract (default: self.target_fps)
            
        Returns:
            List of paths to extracted frame images
        """
        if sample_rate is None:
            sample_rate = self.target_fps
        
        frame_paths = []
        
        try:
            # Use OpenCV for frame extraction
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            logger.info(f"Video info: {duration:.1f}s, {fps:.1f}fps, {total_frames} frames")
            
            # Calculate frame interval
            frame_interval = int(fps / sample_rate)
            
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at specified intervals
                if frame_count % frame_interval == 0:
                    # Create frame filename
                    frame_filename = f"frame_{extracted_count:06d}.jpg"
                    frame_path = self.temp_dir / frame_filename
                    
                    # Save frame
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(str(frame_path))
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"Extracted {extracted_count} frames from {total_frames} total frames")
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise
        
        return frame_paths
    
    def identify_score_screen_frames(self, frame_paths: List[str]) -> List[str]:
        """
        Identify frames that likely contain score/statistics screens.
        
        Args:
            frame_paths: List of frame image paths
            
        Returns:
            List of frame paths that contain score screens
        """
        score_frames = []
        
        for frame_path in frame_paths:
            try:
                # Load image
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue
                
                # Convert to grayscale for text detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Use simple OCR to detect score screen keywords
                from .data_collector import get_ocr_reader
                reader = get_ocr_reader()
                
                # Quick OCR scan for keywords
                results = reader.readtext(gray, detail=0)
                text_content = " ".join(results).lower()
                
                # Check for score screen indicators
                keyword_count = sum(1 for keyword in self.score_screen_keywords 
                                  if keyword in text_content)
                
                # If we find multiple keywords, likely a score screen
                if keyword_count >= 3:
                    score_frames.append(frame_path)
                    logger.info(f"Score screen detected: {frame_path} (keywords: {keyword_count})")
                
            except Exception as e:
                logger.warning(f"Error analyzing frame {frame_path}: {str(e)}")
                continue
        
        return score_frames
    
    def analyze_video(self, video_path: str, ign: str, 
                     known_igns: List[str] = None,
                     hero_override: str = None) -> VideoAnalysisResult:
        """
        Analyze video file and extract match statistics.
        
        Args:
            video_path: Path to video file
            ign: Player's in-game name to look for
            known_igns: List of known IGNs for validation
            hero_override: Manually specified hero name
            
        Returns:
            VideoAnalysisResult with extracted match data
        """
        import time
        start_time = time.time()
        
        warnings = []
        
        try:
            # Validate video file
            is_valid, error_msg = self.validate_video_file(video_path)
            if not is_valid:
                return VideoAnalysisResult(
                    success=False,
                    match_data={},
                    frame_count=0,
                    processed_frames=0,
                    confidence_score=0.0,
                    warnings=[error_msg],
                    processing_time=time.time() - start_time
                )
            
            # Clean up previous frames
            self._cleanup_temp_frames()
            
            # Extract frames from video
            logger.info(f"Extracting frames from video: {video_path}")
            frame_paths = self.extract_frames(video_path)
            
            if not frame_paths:
                return VideoAnalysisResult(
                    success=False,
                    match_data={},
                    frame_count=0,
                    processed_frames=0,
                    confidence_score=0.0,
                    warnings=["No frames could be extracted from video"],
                    processing_time=time.time() - start_time
                )
            
            # Identify score screen frames
            logger.info("Identifying score screen frames...")
            score_frames = self.identify_score_screen_frames(frame_paths)
            
            if not score_frames:
                # If no score screens found, analyze last few frames
                logger.info("No score screens detected, analyzing last frames...")
                score_frames = frame_paths[-5:] if len(frame_paths) >= 5 else frame_paths
                warnings.append("No clear score screens detected, analyzing end-game frames")
            
            # Analyze score screen frames
            best_result = None
            best_confidence = 0.0
            
            for frame_path in score_frames:
                try:
                    logger.info(f"Analyzing frame: {frame_path}")
                    
                    # Use existing OCR pipeline
                    result = self.data_collector.from_screenshot(
                        ign=ign,
                        image_path=frame_path,
                        hero_override=hero_override,
                        known_igns=known_igns
                    )
                    
                    if result and result.get("data"):
                        # Calculate confidence based on how much data we extracted
                        data = result["data"]
                        confidence = self._calculate_confidence(data)
                        
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_result = result
                            logger.info(f"Better result found with confidence: {confidence:.3f}")
                    
                    # Collect warnings
                    if result and result.get("warnings"):
                        warnings.extend(result["warnings"])
                
                except Exception as e:
                    logger.error(f"Error analyzing frame {frame_path}: {str(e)}")
                    warnings.append(f"Frame analysis error: {str(e)}")
                    continue
            
            # Return best result
            if best_result:
                return VideoAnalysisResult(
                    success=True,
                    match_data=best_result["data"],
                    frame_count=len(frame_paths),
                    processed_frames=len(score_frames),
                    confidence_score=best_confidence,
                    warnings=warnings,
                    processing_time=time.time() - start_time
                )
            else:
                return VideoAnalysisResult(
                    success=False,
                    match_data={},
                    frame_count=len(frame_paths),
                    processed_frames=len(score_frames),
                    confidence_score=0.0,
                    warnings=warnings + ["No valid match data could be extracted"],
                    processing_time=time.time() - start_time
                )
        
        except Exception as e:
            logger.error(f"Video analysis failed: {str(e)}")
            return VideoAnalysisResult(
                success=False,
                match_data={},
                frame_count=0,
                processed_frames=0,
                confidence_score=0.0,
                warnings=[f"Analysis failed: {str(e)}"],
                processing_time=time.time() - start_time
            )
        
        finally:
            # Clean up temporary frames
            self._cleanup_temp_frames()
    
    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence score based on extracted data completeness."""
        score = 0.0
        total_fields = 0
        
        # Core stats (higher weight)
        core_fields = ["kills", "deaths", "assists", "gold", "hero"]
        for field in core_fields:
            total_fields += 2  # Higher weight
            if field in data and data[field] is not None:
                if field == "hero" and data[field] != "unknown":
                    score += 2
                elif field != "hero" and data[field] > 0:
                    score += 2
        
        # Additional stats (lower weight)
        additional_fields = ["hero_damage", "turret_damage", "damage_taken", "teamfight_participation"]
        for field in additional_fields:
            total_fields += 1
            if field in data and data[field] is not None and data[field] > 0:
                score += 1
        
        return score / total_fields if total_fields > 0 else 0.0
    
    def _cleanup_temp_frames(self):
        """Clean up temporary frame files."""
        try:
            for frame_file in self.temp_dir.glob("frame_*.jpg"):
                frame_file.unlink()
            logger.info("Cleaned up temporary frame files")
        except Exception as e:
            logger.warning(f"Error cleaning up temp frames: {str(e)}")
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get video file information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video properties
        """
        try:
            # Use OpenCV for basic video info
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {}
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'duration': duration,
                'width': width,
                'height': height,
                'fps': fps,
                'frame_count': frame_count,
                'file_size': Path(video_path).stat().st_size,
            }
        
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            return {}


# Example usage
if __name__ == "__main__":
    # Initialize video reader
    reader = VideoReader()
    
    # Example video analysis
    video_path = "gameplay_video.mp4"  # Replace with actual video path
    ign = "Lesz XVII"
    known_igns = ["Lesz XVII", "Player1", "Enemy1"]
    
    if Path(video_path).exists():
        print(f"Analyzing video: {video_path}")
        
        # Get video info
        video_info = reader.get_video_info(video_path)
        print(f"Video info: {video_info}")
        
        # Analyze video
        result = reader.analyze_video(video_path, ign, known_igns)
        
        print(f"\nAnalysis Result:")
        print(f"Success: {result.success}")
        print(f"Confidence: {result.confidence_score:.3f}")
        print(f"Frames processed: {result.processed_frames}/{result.frame_count}")
        print(f"Processing time: {result.processing_time:.2f}s")
        
        if result.success:
            print(f"\nExtracted Data:")
            for key, value in result.match_data.items():
                print(f"  {key}: {value}")
        
        if result.warnings:
            print(f"\nWarnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
    
    else:
        print(f"Video file not found: {video_path}")
        print("Please place a gameplay video in the project directory to test.")