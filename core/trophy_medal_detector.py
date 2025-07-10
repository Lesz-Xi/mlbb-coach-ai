import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TrophyType(Enum):
    """Types of trophies/medals that can be detected."""
    MVP_CROWN = "mvp_crown"
    GOLD_MEDAL = "gold_medal"
    SILVER_MEDAL = "silver_medal"
    BRONZE_MEDAL = "bronze_medal"
    NONE = "none"


class PerformanceLabel(Enum):
    """Performance labels based on trophy detection."""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    AVERAGE = "Average"
    POOR = "Poor"
    UNKNOWN = "Unknown"


@dataclass
class TrophyDetectionResult:
    """Result of trophy/medal detection analysis."""
    trophy_type: TrophyType
    confidence: float
    performance_label: PerformanceLabel
    detection_method: str
    bounding_box: Optional[Tuple[int, int, int, int]]
    color_analysis: Dict[str, Any]
    shape_analysis: Dict[str, Any]
    debug_info: Dict[str, Any]


class TrophyMedalDetector:
    """
    Advanced trophy and medal detection system for MLBB screenshots.
    
    Detects MVP crowns and Bronze/Silver/Gold medals to determine
    accurate performance labels based on visual trophy indicators.
    """
    
    def __init__(self):
        # MVP Crown detection parameters (gold color)
        self.mvp_crown_hsv_range = {
            'lower': np.array([15, 50, 50]),   # Lower bound for gold color
            'upper': np.array([35, 255, 255]) # Upper bound for gold color
        }
        
        # Medal color ranges in HSV
        self.medal_colors = {
            'gold': {
                'lower': np.array([15, 50, 100]),
                'upper': np.array([35, 255, 255])
            },
            'silver': {
                'lower': np.array([0, 0, 150]),
                'upper': np.array([180, 50, 255])
            },
            'bronze': {
                'lower': np.array([8, 50, 80]),
                'upper': np.array([25, 255, 200])
            }
        }
        
        # Trophy detection regions relative to player row
        self.trophy_search_regions = {
            'left_of_name': (-80, -20, 60, 40),    # Left side of player name
            'right_of_name': (150, -20, 80, 40),   # Right side of player name
            'above_name': (-40, -60, 120, 40),     # Above player name
            'full_row': (-100, -60, 300, 80)      # Extended search area
        }
        
        # Minimum contour areas for trophy detection
        self.min_trophy_area = 100
        self.max_trophy_area = 5000
        
        # Crown shape detection parameters
        self.crown_aspect_ratio_range = (0.7, 1.5)  # Width/height ratio
        self.crown_circularity_threshold = 0.3
        
    def detect_trophy_in_player_row(
        self, 
        image_path: str, 
        player_row_y: float,
        player_name_x: Optional[float] = None,
        debug_mode: bool = False
    ) -> TrophyDetectionResult:
        """
        Detect trophy/medal in the vicinity of a specific player row.
        
        Args:
            image_path: Path to the screenshot
            player_row_y: Y coordinate of the player row
            player_name_x: X coordinate of player name (optional)
            debug_mode: Whether to save debug images
            
        Returns:
            TrophyDetectionResult with detected trophy information
        """
        debug_info = {
            "regions_searched": [],
            "color_masks_tried": [],
            "contours_found": 0,
            "shape_analysis_results": [],
            "mvp_indicators": []
        }
        
        try:
            # Load and validate image
            image = cv2.imread(image_path)
            if image is None:
                return self._create_failed_result("Could not load image", debug_info)
            
            # Convert to HSV for better color detection
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define search region around player row
            search_region = self._get_player_trophy_region(
                image, player_row_y, player_name_x
            )
            debug_info["search_region"] = search_region
            
            # Extract region of interest
            roi_bgr = image[search_region[1]:search_region[3], 
                           search_region[0]:search_region[2]]
            roi_hsv = hsv_image[search_region[1]:search_region[3], 
                               search_region[0]:search_region[2]]
            
            # Strategy 1: MVP Crown Detection (highest priority)
            mvp_result = self._detect_mvp_crown(roi_bgr, roi_hsv, debug_info)
            if mvp_result.confidence > 0.7:
                mvp_result.bounding_box = self._adjust_bounding_box(
                    mvp_result.bounding_box, search_region
                )
                if debug_mode:
                    self._save_debug_image(roi_bgr, mvp_result, "mvp_detection")
                return mvp_result
            
            # Strategy 2: Medal Detection (gold > silver > bronze)
            medal_result = self._detect_medal(roi_bgr, roi_hsv, debug_info)
            if medal_result.confidence > 0.6:
                medal_result.bounding_box = self._adjust_bounding_box(
                    medal_result.bounding_box, search_region
                )
                if debug_mode:
                    self._save_debug_image(roi_bgr, medal_result, "medal_detection")
                return medal_result
            
            # Strategy 3: Text-based MVP detection
            text_mvp_result = self._detect_mvp_by_text(roi_bgr, debug_info)
            if text_mvp_result.confidence > 0.5:
                if debug_mode:
                    self._save_debug_image(roi_bgr, text_mvp_result, "text_mvp")
                return text_mvp_result
            
            # No trophy detected
            return TrophyDetectionResult(
                trophy_type=TrophyType.NONE,
                confidence=0.0,
                performance_label=PerformanceLabel.UNKNOWN,
                detection_method="no_trophy_detected",
                bounding_box=None,
                color_analysis={},
                shape_analysis={},
                debug_info=debug_info
            )
            
        except Exception as e:
            logger.error(f"Error in trophy detection: {str(e)}")
            debug_info["error"] = str(e)
            return self._create_failed_result(f"Detection failed: {str(e)}", debug_info)
    
    def _detect_mvp_crown(
        self, 
        roi_bgr: np.ndarray, 
        roi_hsv: np.ndarray, 
        debug_info: Dict[str, Any]
    ) -> TrophyDetectionResult:
        """Detect MVP crown using gold color and crown shape analysis."""
        
        # Create mask for gold color (MVP crown)
        gold_mask = cv2.inRange(
            roi_hsv, 
            self.mvp_crown_hsv_range['lower'], 
            self.mvp_crown_hsv_range['upper']
        )
        debug_info["color_masks_tried"].append("mvp_gold")
        
        # Morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_CLOSE, kernel)
        gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            gold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        debug_info["contours_found"] += len(contours)
        
        best_crown_confidence = 0.0
        best_contour = None
        best_bbox = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_trophy_area or area > self.max_trophy_area:
                continue
            
            # Analyze shape characteristics for crown detection
            shape_confidence = self._analyze_crown_shape(contour)
            debug_info["shape_analysis_results"].append({
                "area": area,
                "shape_confidence": shape_confidence,
                "contour_points": len(contour)
            })
            
            if shape_confidence > best_crown_confidence:
                best_crown_confidence = shape_confidence
                best_contour = contour
                best_bbox = cv2.boundingRect(contour)
        
        # Additional MVP text detection near crown
        text_mvp_boost = self._detect_mvp_text_nearby(roi_bgr, best_bbox)
        debug_info["mvp_indicators"].append(f"text_boost: {text_mvp_boost}")
        
        # Calculate final confidence
        final_confidence = best_crown_confidence * 0.7 + text_mvp_boost * 0.3
        
        if final_confidence > 0.6:
            return TrophyDetectionResult(
                trophy_type=TrophyType.MVP_CROWN,
                confidence=final_confidence,
                performance_label=PerformanceLabel.EXCELLENT,
                detection_method="crown_shape_color_analysis",
                bounding_box=best_bbox,
                color_analysis={"gold_pixels": cv2.countNonZero(gold_mask)},
                shape_analysis={"crown_confidence": best_crown_confidence},
                debug_info=debug_info
            )
        
        return self._create_failed_result("MVP crown not detected", debug_info)
    
    def _detect_medal(
        self, 
        roi_bgr: np.ndarray, 
        roi_hsv: np.ndarray, 
        debug_info: Dict[str, Any]
    ) -> TrophyDetectionResult:
        """Detect Bronze/Silver/Gold medals using color analysis."""
        
        best_medal_type = TrophyType.NONE
        best_confidence = 0.0
        best_bbox = None
        best_color_analysis = {}
        
        # Check each medal type (gold first for priority)
        medal_priority = ['gold', 'silver', 'bronze']
        
        for medal_color in medal_priority:
            color_range = self.medal_colors[medal_color]
            
            # Create color mask
            mask = cv2.inRange(roi_hsv, color_range['lower'], color_range['upper'])
            debug_info["color_masks_tried"].append(f"medal_{medal_color}")
            
            # Morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (medals are typically circular)
                if area < self.min_trophy_area or area > self.max_trophy_area:
                    continue
                
                # Analyze circularity for medal detection
                circularity = self._calculate_circularity(contour)
                
                # Medal confidence based on circularity and area
                medal_confidence = min(circularity * 0.8 + (area / 1000) * 0.2, 1.0)
                
                if medal_confidence > best_confidence:
                    best_confidence = medal_confidence
                    best_bbox = cv2.boundingRect(contour)
                    best_color_analysis = {
                        "medal_color": medal_color,
                        "color_pixels": cv2.countNonZero(mask),
                        "circularity": circularity,
                        "area": area
                    }
                    
                    # Map medal color to trophy type
                    if medal_color == 'gold':
                        best_medal_type = TrophyType.GOLD_MEDAL
                    elif medal_color == 'silver':
                        best_medal_type = TrophyType.SILVER_MEDAL
                    else:
                        best_medal_type = TrophyType.BRONZE_MEDAL
        
        if best_confidence > 0.5:
            # Map trophy type to performance label
            performance_map = {
                TrophyType.GOLD_MEDAL: PerformanceLabel.GOOD,
                TrophyType.SILVER_MEDAL: PerformanceLabel.AVERAGE,
                TrophyType.BRONZE_MEDAL: PerformanceLabel.POOR
            }
            
            return TrophyDetectionResult(
                trophy_type=best_medal_type,
                confidence=best_confidence,
                performance_label=performance_map[best_medal_type],
                detection_method="medal_color_shape_analysis",
                bounding_box=best_bbox,
                color_analysis=best_color_analysis,
                shape_analysis={"circularity": best_color_analysis.get("circularity", 0)},
                debug_info=debug_info
            )
        
        return self._create_failed_result("No medal detected", debug_info)
    
    def _detect_mvp_by_text(
        self, 
        roi_bgr: np.ndarray, 
        debug_info: Dict[str, Any]
    ) -> TrophyDetectionResult:
        """Detect MVP by looking for 'MVP' text in the region."""
        
        try:
            from .data_collector import get_ocr_reader
            
            # Convert to grayscale for OCR
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            
            # Get OCR reader
            reader = get_ocr_reader()
            ocr_results = reader.readtext(gray, detail=1)
            
            mvp_confidence = 0.0
            mvp_bbox = None
            
            for (bbox, text, confidence) in ocr_results:
                text_lower = text.lower().strip()
                
                # Look for MVP indicators
                if any(indicator in text_lower for indicator in ['mvp', 'most valuable', 'crown']):
                    mvp_confidence = max(mvp_confidence, confidence * 0.8)
                    mvp_bbox = (int(min(bbox, key=lambda x: x[0])[0]), 
                               int(min(bbox, key=lambda x: x[1])[1]),
                               int(max(bbox, key=lambda x: x[0])[0]) - int(min(bbox, key=lambda x: x[0])[0]),
                               int(max(bbox, key=lambda x: x[1])[1]) - int(min(bbox, key=lambda x: x[1])[1]))
                    
                debug_info["mvp_indicators"].append(f"text_found: {text} (conf: {confidence})")
            
            if mvp_confidence > 0.4:
                return TrophyDetectionResult(
                    trophy_type=TrophyType.MVP_CROWN,
                    confidence=mvp_confidence,
                    performance_label=PerformanceLabel.EXCELLENT,
                    detection_method="text_based_mvp_detection",
                    bounding_box=mvp_bbox,
                    color_analysis={},
                    shape_analysis={},
                    debug_info=debug_info
                )
            
        except Exception as e:
            debug_info["text_detection_error"] = str(e)
        
        return self._create_failed_result("MVP text not detected", debug_info)
    
    def _get_player_trophy_region(
        self, 
        image: np.ndarray, 
        player_row_y: float, 
        player_name_x: Optional[float] = None
    ) -> Tuple[int, int, int, int]:
        """Calculate the search region for trophy detection around player row."""
        
        height, width = image.shape[:2]
        
        # Default search region if player_name_x not provided
        if player_name_x is None:
            # Search across the full width around the player row
            x1 = max(0, int(width * 0.1))
            x2 = min(width, int(width * 0.9))
        else:
            # Search around the player name position
            search_width = 200  # Pixels around name
            x1 = max(0, int(player_name_x - search_width))
            x2 = min(width, int(player_name_x + search_width))
        
        # Vertical search region around player row
        search_height = 60  # Pixels above and below
        y1 = max(0, int(player_row_y - search_height))
        y2 = min(height, int(player_row_y + search_height))
        
        return (x1, y1, x2, y2)
    
    def _analyze_crown_shape(self, contour: np.ndarray) -> float:
        """Analyze contour shape to determine if it looks like a crown."""
        
        # Calculate basic shape properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        # Circularity (4π * area / perimeter²)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Bounding rectangle aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Crown should be wider than tall but not perfectly circular
        aspect_ratio_score = 1.0 if self.crown_aspect_ratio_range[0] <= aspect_ratio <= self.crown_aspect_ratio_range[1] else 0.5
        
        # Crown should not be perfectly circular (unlike medals)
        circularity_score = 1.0 - abs(circularity - 0.5)  # Prefer moderate circularity
        
        # Convexity (crown might have jagged edges)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        
        # Combine scores
        shape_confidence = (aspect_ratio_score * 0.4 + 
                           circularity_score * 0.3 + 
                           convexity * 0.3)
        
        return min(shape_confidence, 1.0)
    
    def _calculate_circularity(self, contour: np.ndarray) -> float:
        """Calculate how circular a contour is (1.0 = perfect circle)."""
        
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return min(circularity, 1.0)
    
    def _detect_mvp_text_nearby(
        self, 
        roi_bgr: np.ndarray, 
        bbox: Optional[Tuple[int, int, int, int]]
    ) -> float:
        """Look for MVP-related text near a detected shape."""
        
        if bbox is None:
            return 0.0
        
        try:
            from .data_collector import get_ocr_reader
            
            # Expand search area around the bounding box
            x, y, w, h = bbox
            expand = 20
            x1 = max(0, x - expand)
            y1 = max(0, y - expand)
            x2 = min(roi_bgr.shape[1], x + w + expand)
            y2 = min(roi_bgr.shape[0], y + h + expand)
            
            text_region = roi_bgr[y1:y2, x1:x2]
            gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
            
            reader = get_ocr_reader()
            ocr_results = reader.readtext(gray, detail=1)
            
            for (_, text, confidence) in ocr_results:
                text_lower = text.lower().strip()
                if 'mvp' in text_lower:
                    return confidence * 0.8
                elif any(word in text_lower for word in ['most', 'valuable', 'crown']):
                    return confidence * 0.6
            
        except Exception:
            pass
        
        return 0.0
    
    def _adjust_bounding_box(
        self, 
        bbox: Optional[Tuple[int, int, int, int]], 
        search_region: Tuple[int, int, int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Adjust bounding box coordinates from ROI to full image coordinates."""
        
        if bbox is None:
            return None
        
        x, y, w, h = bbox
        roi_x1, roi_y1, _, _ = search_region
        
        return (x + roi_x1, y + roi_y1, w, h)
    
    def _save_debug_image(
        self, 
        roi: np.ndarray, 
        result: TrophyDetectionResult, 
        prefix: str
    ):
        """Save debug image with detection results."""
        
        try:
            debug_img = roi.copy()
            
            if result.bounding_box:
                x, y, w, h = result.bounding_box
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(debug_img, f"{result.trophy_type.value} ({result.confidence:.2f})",
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            debug_path = f"temp/debug_{prefix}_{result.trophy_type.value}.png"
            cv2.imwrite(debug_path, debug_img)
            logger.info(f"Debug image saved: {debug_path}")
            
        except Exception as e:
            logger.warning(f"Could not save debug image: {str(e)}")
    
    def _create_failed_result(self, reason: str, debug_info: Dict[str, Any]) -> TrophyDetectionResult:
        """Create a failed detection result."""
        
        debug_info["failure_reason"] = reason
        
        return TrophyDetectionResult(
            trophy_type=TrophyType.NONE,
            confidence=0.0,
            performance_label=PerformanceLabel.UNKNOWN,
            detection_method="failed",
            bounding_box=None,
            color_analysis={},
            shape_analysis={},
            debug_info=debug_info
        )
    
    def _validate_hero_override(self, hero_override: str) -> str:
        """Validate and normalize hero override input."""
        
        # Simple validation - in real implementation, would check against hero database
        return hero_override.lower().strip()


# Global instance
trophy_medal_detector = TrophyMedalDetector() 