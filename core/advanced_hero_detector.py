import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Optional
import re

from .hero_identifier import hero_identifier
from .data_collector import get_ocr_reader

logger = logging.getLogger(__name__)


class AdvancedHeroDetector:
    """Advanced hero detection with multiple strategies and fallbacks."""
    
    def __init__(self):
        self.hero_identifier = hero_identifier
        
        # Region of interest for hero names (relative coordinates)
        self.hero_name_regions = [
            (0.0, 0.3, 0.5, 0.8),    # Left side player names
            (0.5, 0.3, 1.0, 0.8),    # Right side player names
            (0.0, 0.1, 1.0, 0.9),    # Full screen fallback
        ]
    
    def detect_hero_comprehensive(
        self, 
        image_path: str, 
        player_ign: str,
        hero_override: Optional[str] = None
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Comprehensive hero detection using multiple strategies.
        
        Returns:
            Tuple of (hero_name, confidence, debug_info)
        """
        debug_info = {
            "strategies_tried": [],
            "ocr_text_found": [],
            "hero_suggestions": [],
            "manual_override": hero_override is not None
        }
        
        if hero_override:
            # Validate manual override
            canonical_hero = self._validate_hero_override(hero_override)
            debug_info["strategies_tried"].append("manual_override")
            return canonical_hero, 1.0, debug_info
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                return "unknown", 0.0, debug_info
            
            # Strategy 1: Full image OCR analysis
            hero, confidence, strategy_debug = self._detect_from_full_image(image, player_ign)
            debug_info["strategies_tried"].append("full_image_ocr")
            debug_info.update(strategy_debug)
            
            if confidence >= 0.7:
                return hero, confidence, debug_info
            
            # Strategy 2: Region-based OCR analysis
            hero2, confidence2, strategy_debug2 = self._detect_from_regions(image, player_ign)
            debug_info["strategies_tried"].append("region_based_ocr")
            debug_info.update(strategy_debug2)
            
            if confidence2 > confidence:
                hero, confidence = hero2, confidence2
            
            if confidence >= 0.6:
                return hero, confidence, debug_info
            
            # Strategy 3: Enhanced preprocessing and OCR
            hero3, confidence3, strategy_debug3 = self._detect_with_enhanced_preprocessing(image, player_ign)
            debug_info["strategies_tried"].append("enhanced_preprocessing")
            debug_info.update(strategy_debug3)
            
            if confidence3 > confidence:
                hero, confidence = hero3, confidence3
            
            # Strategy 4: Fuzzy matching on all detected text
            if confidence < 0.5:
                hero4, confidence4, strategy_debug4 = self._detect_with_fuzzy_matching(image)
                debug_info["strategies_tried"].append("fuzzy_matching")
                debug_info.update(strategy_debug4)
                
                if confidence4 > confidence:
                    hero, confidence = hero4, confidence4
            
            # Generate suggestions for debugging
            debug_info["hero_suggestions"] = self.hero_identifier.get_hero_suggestions(
                " ".join(debug_info.get("ocr_text_found", []))
            )
            
            return hero, confidence, debug_info
            
        except Exception as e:
            logger.error(f"Hero detection failed: {str(e)}")
            debug_info["error"] = str(e)
            return "unknown", 0.0, debug_info
    
    def _validate_hero_override(self, hero_override: str) -> str:
        """Validate and normalize manual hero override."""
        hero, confidence = self.hero_identifier.identify_hero(hero_override, confidence_threshold=0.5)
        return hero if confidence >= 0.5 else hero_override.lower()
    
    def _detect_from_full_image(self, image: np.ndarray, player_ign: str) -> Tuple[str, float, Dict]:
        """Detect hero from full image OCR."""
        debug_info = {"method": "full_image"}
        
        try:
            reader = get_ocr_reader()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Standard OCR
            ocr_results = reader.readtext(gray, detail=1)
            all_text = " ".join([result[1] for result in ocr_results])
            debug_info["ocr_text_found"] = [result[1] for result in ocr_results]
            
            # Look for hero names near the player IGN
            hero, confidence = self._find_hero_near_ign(ocr_results, player_ign)
            
            if confidence < 0.5:
                # Fallback to general hero identification
                hero, confidence = self.hero_identifier.identify_from_ocr_results(ocr_results)
            
            return hero, confidence, debug_info
            
        except Exception as e:
            debug_info["error"] = str(e)
            return "unknown", 0.0, debug_info
    
    def _detect_from_regions(self, image: np.ndarray, player_ign: str) -> Tuple[str, float, Dict]:
        """Detect hero from specific regions of the image."""
        debug_info = {"method": "region_based", "regions_analyzed": []}
        
        height, width = image.shape[:2]
        best_hero = "unknown"
        best_confidence = 0.0
        
        try:
            reader = get_ocr_reader()
            
            for i, (x1, y1, x2, y2) in enumerate(self.hero_name_regions):
                # Convert relative coordinates to absolute
                abs_x1, abs_y1 = int(x1 * width), int(y1 * height)
                abs_x2, abs_y2 = int(x2 * width), int(y2 * height)
                
                # Extract region
                region = image[abs_y1:abs_y2, abs_x1:abs_x2]
                gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                
                # OCR on region
                ocr_results = reader.readtext(gray_region, detail=1)
                region_text = " ".join([result[1] for result in ocr_results])
                
                debug_info["regions_analyzed"].append({
                    "region": f"({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})",
                    "text_found": [result[1] for result in ocr_results][:5],  # Limit for debug
                    "contains_ign": player_ign.lower() in region_text.lower()
                })
                
                # Check if this region contains the player IGN
                if player_ign.lower() in region_text.lower():
                    hero, confidence = self._find_hero_near_ign(ocr_results, player_ign)
                    if confidence > best_confidence:
                        best_hero = hero
                        best_confidence = confidence
                
                # Also try general hero identification on this region
                hero, confidence = self.hero_identifier.identify_from_ocr_results(ocr_results)
                if confidence > best_confidence:
                    best_hero = hero
                    best_confidence = confidence
            
            return best_hero, best_confidence, debug_info
            
        except Exception as e:
            debug_info["error"] = str(e)
            return "unknown", 0.0, debug_info
    
    def _detect_with_enhanced_preprocessing(self, image: np.ndarray, player_ign: str) -> Tuple[str, float, Dict]:
        """Detect hero with enhanced image preprocessing."""
        debug_info = {"method": "enhanced_preprocessing", "preprocessing_steps": []}
        
        try:
            # Multiple preprocessing strategies
            preprocessed_images = []
            
            # Strategy 1: High contrast
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            contrast = cv2.convertScaleAbs(gray, alpha=2.0, beta=50)
            preprocessed_images.append(("high_contrast", contrast))
            
            # Strategy 2: Gaussian blur + threshold
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed_images.append(("threshold", thresh))
            
            # Strategy 3: Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            preprocessed_images.append(("morphological", morph))
            
            reader = get_ocr_reader()
            best_hero = "unknown"
            best_confidence = 0.0
            
            for name, processed_img in preprocessed_images:
                debug_info["preprocessing_steps"].append(name)
                
                try:
                    ocr_results = reader.readtext(processed_img, detail=1)
                    hero, confidence = self.hero_identifier.identify_from_ocr_results(ocr_results)
                    
                    if confidence > best_confidence:
                        best_hero = hero
                        best_confidence = confidence
                        debug_info["best_preprocessing"] = name
                    
                except Exception as step_error:
                    debug_info[f"{name}_error"] = str(step_error)
                    continue
            
            return best_hero, best_confidence, debug_info
            
        except Exception as e:
            debug_info["error"] = str(e)
            return "unknown", 0.0, debug_info
    
    def _detect_with_fuzzy_matching(self, image: np.ndarray) -> Tuple[str, float, Dict]:
        """Detect hero using aggressive fuzzy matching on all text."""
        debug_info = {"method": "fuzzy_matching"}
        
        try:
            reader = get_ocr_reader()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Get all text with lower confidence threshold
            ocr_results = reader.readtext(gray, detail=0, width_ths=0.7, height_ths=0.7)
            all_text = " ".join(ocr_results)
            debug_info["all_text"] = all_text[:200]  # Limit for debug
            
            # Try fuzzy matching on the entire text blob
            suggestions = self.hero_identifier.get_hero_suggestions(all_text, top_n=5)
            debug_info["top_suggestions"] = suggestions
            
            if suggestions and suggestions[0][1] >= 0.6:  # Lower threshold for fuzzy matching
                return suggestions[0][0], suggestions[0][1], debug_info
            
            return "unknown", 0.0, debug_info
            
        except Exception as e:
            debug_info["error"] = str(e)
            return "unknown", 0.0, debug_info
    
    def _find_hero_near_ign(self, ocr_results: List, player_ign: str) -> Tuple[str, float]:
        """Find hero name that appears near the player IGN."""
        ign_position = None
        
        # Find IGN position
        for bbox, text, conf in ocr_results:
            if player_ign.lower() in text.lower():
                ign_position = bbox
                break
        
        if not ign_position:
            return "unknown", 0.0
        
        # Find nearby text that might be hero names
        ign_center_y = sum(point[1] for point in ign_position) / 4
        nearby_texts = []
        
        for bbox, text, conf in ocr_results:
            text_center_y = sum(point[1] for point in bbox) / 4
            
            # Check if text is on the same horizontal line (within 30 pixels)
            if abs(text_center_y - ign_center_y) < 30:
                nearby_texts.append(text)
        
        # Try to identify hero from nearby text
        combined_text = " ".join(nearby_texts)
        return self.hero_identifier.identify_hero(combined_text)


# Global instance
advanced_hero_detector = AdvancedHeroDetector()