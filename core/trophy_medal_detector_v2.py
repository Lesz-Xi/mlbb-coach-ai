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


class ImprovedTrophyMedalDetector:
    """
    IMPROVED Trophy and Medal Detection System for MLBB Screenshots.
    
    Based on test results analysis, this version has:
    - Enhanced MVP crown detection with multiple color ranges
    - Improved medal color differentiation
    - Better search region logic
    - Text-based MVP detection as fallback
    """
    
    def __init__(self):
        # IMPROVED: Multiple MVP Crown detection parameters
        self.mvp_crown_hsv_ranges = [
            # Primary gold range (bright yellow-gold)
            {'lower': np.array([20, 80, 120]), 'upper': np.array([30, 255, 255])},
            # Secondary gold range (orange-gold)
            {'lower': np.array([15, 60, 100]), 'upper': np.array([25, 255, 255])},
            # Dark gold range
            {'lower': np.array([10, 50, 80]), 'upper': np.array([35, 255, 200])},
        ]
        
        # ENHANCED: Precise medal color differentiation with non-overlapping ranges
        self.medal_colors = {
            'gold': [
                # Bright gold medals - distinct from bronze
                {'lower': np.array([20, 100, 140]), 
                 'upper': np.array([30, 255, 255])},
                # Golden yellow range
                {'lower': np.array([15, 80, 120]), 
                 'upper': np.array([25, 255, 255])},
                # Rich gold range
                {'lower': np.array([18, 120, 160]), 
                 'upper': np.array([28, 255, 255])},
            ],
                          'silver': [
                  # Light silver/gray range - distinct from white and other colors
                  {'lower': np.array([0, 0, 160]), 
                   'upper': np.array([180, 40, 240])},
                  # Metallic silver range
                  {'lower': np.array([0, 0, 140]), 
                   'upper': np.array([180, 50, 200])},
                  # Dark silver range
                  {'lower': np.array([0, 0, 120]), 
                   'upper': np.array([180, 60, 180])},
              ],
              'bronze': [
                  # Bronze/copper range - distinct from gold and brown
                  {'lower': np.array([5, 60, 80]), 
                   'upper': np.array([15, 255, 160])},
                  # Dark bronze range
                  {'lower': np.array([8, 80, 60]), 
                   'upper': np.array([18, 255, 140])},
                  # Copper-bronze range
                  {'lower': np.array([3, 100, 70]), 
                   'upper': np.array([12, 255, 150])},
              ]
        }
        
        # ENHANCED: More targeted search regions
        self.trophy_search_offsets = [
            (-120, -35, 70, 45),   # Left of name (reduced overlap)
            (-70, -25, 50, 35),    # Near left of name  
            (100, -25, 80, 35),    # Right of name
            (180, -25, 70, 35),    # Far right of name
            (-30, -60, 100, 45),   # Above name (reduced height)
            (-80, -70, 200, 80)    # Extended search area (optimized)
        ]
        
        # ENHANCED: More restrictive trophy size constraints for better accuracy
        self.min_trophy_area = 80      # Increased from 50
        self.max_trophy_area = 6000    # Decreased from 8000
        
        # Medal-specific size constraints
        self.medal_size_ranges = {
            'gold': {'min': 120, 'max': 4000},
            'silver': {'min': 100, 'max': 3500}, 
            'bronze': {'min': 80, 'max': 3000}
        }
        
        # MVP crown specific patterns
        self.mvp_text_patterns = ['mvp', 'most valuable', 'crown', '👑']
        
    def detect_trophy_in_player_row(
        self, 
        image_path: str, 
        player_row_y: float,
        player_name_x: Optional[float] = None,
        debug_mode: bool = False
    ) -> TrophyDetectionResult:
        """
        IMPROVED trophy detection with multiple strategies and better parameters.
        """
        debug_info = {
            "strategies_attempted": [],
            "color_ranges_tested": [],
            "contours_analyzed": 0,
            "mvp_text_indicators": [],
            "confidence_breakdown": {}
        }
        
        try:
            # Load and validate image
            image = cv2.imread(image_path)
            if image is None:
                return self._create_failed_result("Could not load image", debug_info)
            
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # STRATEGY 1: Enhanced MVP Crown Detection (Priority)
            mvp_result = self._detect_mvp_crown_enhanced(
                image, hsv_image, player_row_y, player_name_x, debug_info
            )
            debug_info["strategies_attempted"].append("enhanced_mvp_crown")
            
            if mvp_result.confidence > 0.6:
                logger.info(f"🏆 MVP crown detected with confidence {mvp_result.confidence:.1%}")
                if debug_mode:
                    self._save_debug_image(image, mvp_result, "enhanced_mvp")
                return mvp_result
            
            # STRATEGY 2: Text-based MVP detection (High priority fallback)
            text_mvp_result = self._detect_mvp_by_text_enhanced(
                image, player_row_y, player_name_x, debug_info
            )
            debug_info["strategies_attempted"].append("enhanced_mvp_text")
            
            if text_mvp_result.confidence > 0.5:
                logger.info(f"🏆 MVP detected via text with confidence {text_mvp_result.confidence:.1%}")
                if debug_mode:
                    self._save_debug_image(image, text_mvp_result, "text_mvp")
                return text_mvp_result
            
            # STRATEGY 3: Improved Medal Detection
            medal_result = self._detect_medal_enhanced(
                image, hsv_image, player_row_y, player_name_x, debug_info
            )
            debug_info["strategies_attempted"].append("enhanced_medal")
            
            if medal_result.confidence > 0.5:
                logger.info(f"🥇 Medal detected: {medal_result.trophy_type.value} with confidence {medal_result.confidence:.1%}")
                if debug_mode:
                    self._save_debug_image(image, medal_result, "enhanced_medal")
                return medal_result
            
            # No trophy detected
            logger.warning("No trophy detected with any strategy")
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
            logger.error(f"Enhanced trophy detection failed: {str(e)}")
            debug_info["error"] = str(e)
            return self._create_failed_result(f"Detection failed: {str(e)}", debug_info)
    
    def _detect_mvp_crown_enhanced(
        self, 
        image: np.ndarray,
        hsv_image: np.ndarray,
        player_row_y: float,
        player_name_x: Optional[float],
        debug_info: Dict[str, Any]
    ) -> TrophyDetectionResult:
        """Enhanced MVP crown detection with multiple color ranges and improved shape analysis."""
        
        best_confidence = 0.0
        best_result = None
        
        # Test multiple search regions around player row
        for i, (dx, dy, w, h) in enumerate(self.trophy_search_offsets):
            search_region = self._get_search_region(image, player_row_y, player_name_x, dx, dy, w, h)
            
            if search_region is None:
                continue
                
            x1, y1, x2, y2 = search_region
            roi_bgr = image[y1:y2, x1:x2]
            roi_hsv = hsv_image[y1:y2, x1:x2]
            
            # Test multiple MVP crown color ranges
            for j, color_range in enumerate(self.mvp_crown_hsv_ranges):
                debug_info["color_ranges_tested"].append(f"mvp_range_{j}_region_{i}")
                
                # Create mask for this color range
                mask = cv2.inRange(roi_hsv, color_range['lower'], color_range['upper'])
                
                # Morphological operations
                kernel = np.ones((2, 2), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                debug_info["contours_analyzed"] += len(contours)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    # Filter by area
                    if area < self.min_trophy_area or area > self.max_trophy_area:
                        continue
                    
                    # Analyze crown-like characteristics
                    crown_confidence = self._analyze_crown_characteristics(contour, mask)
                    
                    # Boost confidence for typical crown positions (above/beside name)
                    position_boost = self._get_position_boost(i, area)
                    total_confidence = crown_confidence * 0.7 + position_boost * 0.3
                    
                    if total_confidence > best_confidence:
                        best_confidence = total_confidence
                        bbox = cv2.boundingRect(contour)
                        # Adjust bbox to full image coordinates
                        adjusted_bbox = (bbox[0] + x1, bbox[1] + y1, bbox[2], bbox[3])
                        
                        best_result = TrophyDetectionResult(
                            trophy_type=TrophyType.MVP_CROWN,
                            confidence=total_confidence,
                            performance_label=PerformanceLabel.EXCELLENT,
                            detection_method=f"enhanced_crown_color_{j}_region_{i}",
                            bounding_box=adjusted_bbox,
                            color_analysis={
                                "color_range_index": j,
                                "region_index": i,
                                "mask_pixels": cv2.countNonZero(mask),
                                "contour_area": area
                            },
                            shape_analysis={"crown_confidence": crown_confidence},
                            debug_info=debug_info.copy()
                        )
        
        if best_result and best_confidence > 0.3:
            debug_info["confidence_breakdown"]["mvp_crown"] = best_confidence
            return best_result
        
        return self._create_failed_result("MVP crown not detected", debug_info)
    
    def _detect_mvp_by_text_enhanced(
        self,
        image: np.ndarray,
        player_row_y: float,
        player_name_x: Optional[float],
        debug_info: Dict[str, Any]
    ) -> TrophyDetectionResult:
        """Enhanced text-based MVP detection with expanded search and better patterns."""
        
        try:
            from .data_collector import get_ocr_reader
            
            # Expanded search region for text
            search_height = 100
            search_width = 300
            
            height, width = image.shape[:2]
            y1 = max(0, int(player_row_y - search_height))
            y2 = min(height, int(player_row_y + search_height))
            
            if player_name_x:
                x1 = max(0, int(player_name_x - search_width))
                x2 = min(width, int(player_name_x + search_width))
            else:
                x1 = max(0, int(width * 0.1))
                x2 = min(width, int(width * 0.9))
            
            text_region = image[y1:y2, x1:x2]
            gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
            
            # Apply text enhancement
            enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            reader = get_ocr_reader()
            ocr_results = reader.readtext(enhanced, detail=1)
            
            mvp_confidence = 0.0
            mvp_bbox = None
            
            for (bbox, text, confidence) in ocr_results:
                text_lower = text.lower().strip()
                debug_info["mvp_text_indicators"].append(f"text: '{text}' conf: {confidence:.3f}")
                
                # Check for MVP indicators with weighted scoring
                mvp_score = 0.0
                if 'mvp' in text_lower:
                    mvp_score = confidence * 1.0  # Full weight for direct MVP
                elif any(word in text_lower for word in ['most', 'valuable']):
                    mvp_score = confidence * 0.8
                elif any(word in text_lower for word in ['crown', 'gold', 'champion']):
                    mvp_score = confidence * 0.6
                elif any(symbol in text for symbol in ['★', '👑', '🏆']):
                    mvp_score = confidence * 0.9
                
                if mvp_score > mvp_confidence:
                    mvp_confidence = mvp_score
                    # Convert bbox coordinates
                    points = np.array(bbox)
                    x_min, y_min = np.min(points, axis=0)
                    x_max, y_max = np.max(points, axis=0)
                    mvp_bbox = (int(x_min + x1), int(y_min + y1), int(x_max - x_min), int(y_max - y_min))
            
            debug_info["confidence_breakdown"]["mvp_text"] = mvp_confidence
            
            if mvp_confidence > 0.4:
                return TrophyDetectionResult(
                    trophy_type=TrophyType.MVP_CROWN,
                    confidence=mvp_confidence,
                    performance_label=PerformanceLabel.EXCELLENT,
                    detection_method="enhanced_text_mvp_detection",
                    bounding_box=mvp_bbox,
                    color_analysis={},
                    shape_analysis={},
                    debug_info=debug_info.copy()
                )
            
        except Exception as e:
            debug_info["mvp_text_error"] = str(e)
        
        return self._create_failed_result("Enhanced MVP text not detected", debug_info)
    
    def _detect_medal_enhanced(
        self,
        image: np.ndarray,
        hsv_image: np.ndarray,
        player_row_y: float,
        player_name_x: Optional[float],
        debug_info: Dict[str, Any]
    ) -> TrophyDetectionResult:
        """Enhanced medal detection with improved color differentiation."""
        
        best_medal_type = TrophyType.NONE
        best_confidence = 0.0
        best_bbox = None
        best_color_analysis = {}
        
        # Search multiple regions
        for i, (dx, dy, w, h) in enumerate(self.trophy_search_offsets):
            search_region = self._get_search_region(image, player_row_y, player_name_x, dx, dy, w, h)
            
            if search_region is None:
                continue
                
            x1, y1, x2, y2 = search_region
            roi_bgr = image[y1:y2, x1:x2]
            roi_hsv = hsv_image[y1:y2, x1:x2]
            
            # Test each medal type with priority order (gold > silver > bronze)
            for medal_color in ['gold', 'silver', 'bronze']:
                for j, color_range in enumerate(self.medal_colors[medal_color]):
                    debug_info["color_ranges_tested"].append(f"{medal_color}_range_{j}_region_{i}")
                    
                    # Create color mask
                    mask = cv2.inRange(roi_hsv, color_range['lower'], color_range['upper'])
                    
                    # Morphological operations for cleaner detection
                    kernel = np.ones((3, 3), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    
                    # Find contours
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        
                        # ENHANCED: Medal-specific size filtering
                        size_ranges = self.medal_size_ranges[medal_color]
                        if area < size_ranges['min'] or area > size_ranges['max']:
                            continue
                        
                        # ENHANCED: Advanced shape analysis
                        circularity = self._calculate_circularity(contour)
                        aspect_ratio = self._calculate_aspect_ratio(contour)
                        solidity = self._calculate_solidity(contour)
                        
                        # ENHANCED: Weighted confidence calculation
                        medal_confidence = self._calculate_enhanced_medal_confidence(
                            medal_color, circularity, aspect_ratio, solidity, area, i, j
                        )
                        
                        if medal_confidence > best_confidence:
                            best_confidence = medal_confidence
                            bbox = cv2.boundingRect(contour)
                            best_bbox = (bbox[0] + x1, bbox[1] + y1, bbox[2], bbox[3])
                            best_color_analysis = {
                                "medal_color": medal_color,
                                "color_range_index": j,
                                "region_index": i,
                                "circularity": circularity,
                                "aspect_ratio": aspect_ratio,
                                "solidity": solidity,
                                "area": area,
                                "mask_pixels": cv2.countNonZero(mask)
                            }
                            
                            # Map medal color to trophy type
                            if medal_color == 'gold':
                                best_medal_type = TrophyType.GOLD_MEDAL
                            elif medal_color == 'silver':
                                best_medal_type = TrophyType.SILVER_MEDAL
                            else:
                                best_medal_type = TrophyType.BRONZE_MEDAL
        
        # ENHANCED: Higher confidence threshold for better accuracy
        if best_confidence > 0.7:
            # Map trophy type to performance label
            performance_map = {
                TrophyType.GOLD_MEDAL: PerformanceLabel.GOOD,
                TrophyType.SILVER_MEDAL: PerformanceLabel.AVERAGE,
                TrophyType.BRONZE_MEDAL: PerformanceLabel.POOR
            }
            
            debug_info["confidence_breakdown"]["medal"] = best_confidence
            
            return TrophyDetectionResult(
                trophy_type=best_medal_type,
                confidence=best_confidence,
                performance_label=performance_map[best_medal_type],
                detection_method="enhanced_medal_detection",
                bounding_box=best_bbox,
                color_analysis=best_color_analysis,
                shape_analysis={"circularity": best_color_analysis.get("circularity", 0)},
                debug_info=debug_info.copy()
            )
        
        return self._create_failed_result("Enhanced medal not detected", debug_info)
    
    def _get_search_region(
        self, 
        image: np.ndarray, 
        player_row_y: float, 
        player_name_x: Optional[float],
        dx: int, dy: int, w: int, h: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """Calculate search region with bounds checking."""
        
        height, width = image.shape[:2]
        
        if player_name_x is None:
            # Use center of image as reference
            center_x = width // 2
        else:
            center_x = int(player_name_x)
        
        x1 = max(0, center_x + dx)
        y1 = max(0, int(player_row_y + dy))
        x2 = min(width, x1 + w)
        y2 = min(height, y1 + h)
        
        # Ensure minimum region size
        if x2 - x1 < 20 or y2 - y1 < 20:
            return None
            
        return (x1, y1, x2, y2)
    
    def _analyze_crown_characteristics(self, contour: np.ndarray, mask: np.ndarray) -> float:
        """Analyze contour characteristics for crown-like features."""
        
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Bounding rectangle aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Crown characteristics scoring
        # Crowns are typically wider than tall but not perfectly circular
        aspect_score = 1.0 if 0.8 <= aspect_ratio <= 2.0 else 0.5
        circularity_score = 1.0 - abs(circularity - 0.4)  # Prefer moderate circularity
        size_score = min(1.0, area / 1000)  # Prefer larger areas
        
        # Convexity analysis
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        convexity_score = 1.0 - abs(convexity - 0.8)  # Prefer slightly non-convex
        
        # Combine scores
        crown_confidence = (aspect_score * 0.3 + 
                           circularity_score * 0.25 + 
                           size_score * 0.25 + 
                           convexity_score * 0.2)
        
        return min(crown_confidence, 1.0)
    
    def _get_position_boost(self, region_index: int, area: float) -> float:
        """Give confidence boost based on trophy position and size."""
        
        # Position-based boost (regions closer to typical trophy positions get higher scores)
        position_boosts = [0.3, 0.8, 0.9, 0.7, 0.6, 0.4]  # Corresponding to trophy_search_offsets
        position_boost = position_boosts[min(region_index, len(position_boosts) - 1)]
        
        # Size-based boost
        size_boost = min(1.0, area / 500)  # Larger areas get higher boost
        
        return position_boost * 0.7 + size_boost * 0.3
    
    def _calculate_medal_confidence(self, medal_color: str, circularity: float, area: float, region_index: int) -> float:
        """Calculate medal detection confidence with color-specific weighting."""
        
        # Base confidence from circularity (medals should be circular)
        circularity_score = circularity
        
        # Area-based confidence (reasonable medal sizes)
        area_score = min(1.0, area / 800)
        
        # Position boost
        position_boost = self._get_position_boost(region_index, area)
        
        # Color-specific adjustments
        color_weights = {'gold': 1.0, 'silver': 0.9, 'bronze': 0.8}
        color_weight = color_weights.get(medal_color, 0.7)
        
        # Combine scores
        medal_confidence = (circularity_score * 0.4 + 
                           area_score * 0.3 + 
                           position_boost * 0.2 + 
                           color_weight * 0.1)
        
        return min(medal_confidence, 1.0)
    
    def _calculate_circularity(self, contour: np.ndarray) -> float:
        """Calculate how circular a contour is (1.0 = perfect circle)."""
        
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return min(circularity, 1.0)
    
    def _save_debug_image(self, image: np.ndarray, result: TrophyDetectionResult, prefix: str):
        """Save debug image with detection results."""
        
        try:
            debug_img = image.copy()
            
            if result.bounding_box:
                x, y, w, h = result.bounding_box
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                label = f"{result.trophy_type.value} ({result.confidence:.1%})"
                cv2.putText(debug_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            debug_path = f"temp/debug_{prefix}_{result.trophy_type.value}_v2.png"
            cv2.imwrite(debug_path, debug_img)
            logger.info(f"Enhanced debug image saved: {debug_path}")
            
        except Exception as e:
            logger.warning(f"Could not save enhanced debug image: {str(e)}")
    
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

    def _calculate_aspect_ratio(self, contour: np.ndarray) -> float:
        """Calculate aspect ratio of contour bounding rectangle."""
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            return 0.0
        return w / h

    def _calculate_solidity(self, contour: np.ndarray) -> float:
        """Calculate solidity (ratio of contour area to convex hull area)."""
        area = cv2.contourArea(contour)
        if area == 0:
            return 0.0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return 0.0
        return area / hull_area

    def _calculate_enhanced_medal_confidence(
        self, 
        medal_color: str, 
        circularity: float, 
        aspect_ratio: float,
        solidity: float,
        area: float, 
        region_index: int,
        color_range_index: int
    ) -> float:
        """Enhanced medal confidence calculation with weighted scoring."""
        
        # Base confidence from shape analysis
        shape_score = 0.0
        
        # Circularity scoring (medals should be circular)
        if circularity > 0.7:
            shape_score += 0.4
        elif circularity > 0.5:
            shape_score += 0.2
        
        # Aspect ratio scoring (medals should be roughly square)
        if 0.8 <= aspect_ratio <= 1.2:
            shape_score += 0.3
        elif 0.7 <= aspect_ratio <= 1.4:
            shape_score += 0.15
        
        # Solidity scoring (medals should be solid shapes)
        if solidity > 0.8:
            shape_score += 0.3
        elif solidity > 0.6:
            shape_score += 0.15
        
        # Size scoring based on medal type
        size_score = 0.0
        size_ranges = self.medal_size_ranges[medal_color]
        optimal_size = (size_ranges['min'] + size_ranges['max']) / 2
        size_diff = abs(area - optimal_size) / optimal_size
        
        if size_diff < 0.3:
            size_score = 0.25
        elif size_diff < 0.5:
            size_score = 0.15
        elif size_diff < 0.7:
            size_score = 0.1
        
        # Position scoring (medals typically appear in specific regions)
        position_score = self._get_position_boost(region_index, area)
        
        # Color range preference (primary ranges get higher scores)
        color_range_score = 0.1 if color_range_index == 0 else 0.05
        
        # Medal type confidence (gold > silver > bronze in typical scenarios)
        type_bonus = {'gold': 0.05, 'silver': 0.03, 'bronze': 0.01}[medal_color]
        
        total_confidence = (shape_score + size_score + 
                          position_score + color_range_score + type_bonus)
        
        return min(total_confidence, 1.0)


# Global instance
improved_trophy_medal_detector = ImprovedTrophyMedalDetector() 