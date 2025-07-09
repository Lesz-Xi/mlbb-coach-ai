import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any, List
import re

from .session_manager import ScreenshotType
from .data_collector import get_ocr_reader

logger = logging.getLogger(__name__)


class ScreenshotAnalyzer:
    """Analyzes screenshots to determine their type and extract relevant information."""
    
    def __init__(self):
        # Keywords that help identify different screenshot types
        self.scoreboard_keywords = [
            "victory", "defeat", "duration", "kills", "deaths", "assists",
            "gold", "kda", "match", "result", "mvp", "legendary"
        ]
        
        self.stats_keywords = [
            "hero damage", "turret damage", "damage taken", "healing",
            "teamfight participation", "gold earned", "items", "equipment"
        ]
        
        self.general_ui_keywords = [
            "mobile legends", "bang bang", "mlbb", "battle", "classic", "ranked"
        ]
    
    def analyze_screenshot_type(self, image_path: str) -> Tuple[ScreenshotType, float, List[str]]:
        """
        Analyze a screenshot to determine its type.
        
        Returns:
            Tuple of (screenshot_type, confidence, detected_keywords)
        """
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return ScreenshotType.UNKNOWN, 0.0, []
            
            # Get OCR results
            reader = get_ocr_reader()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ocr_results = reader.readtext(gray, detail=0)
            
            if not ocr_results:
                logger.warning("No OCR results from screenshot")
                return ScreenshotType.UNKNOWN, 0.0, []
            
            # Combine all text for analysis
            all_text = " ".join(ocr_results).lower()
            logger.info(f"OCR text preview: {all_text[:200]}...")
            
            # Count keyword matches
            scoreboard_matches = self._count_keyword_matches(all_text, self.scoreboard_keywords)
            stats_matches = self._count_keyword_matches(all_text, self.stats_keywords)
            
            detected_keywords = []
            
            # Determine screenshot type based on keyword density
            if scoreboard_matches >= 3:  # Need at least 3 scoreboard keywords
                screenshot_type = ScreenshotType.SCOREBOARD
                confidence = min(0.9, 0.5 + (scoreboard_matches * 0.1))
                detected_keywords = [kw for kw in self.scoreboard_keywords if kw in all_text]
                
            elif stats_matches >= 2:  # Need at least 2 stats keywords
                screenshot_type = ScreenshotType.STATS_PAGE
                confidence = min(0.9, 0.4 + (stats_matches * 0.15))
                detected_keywords = [kw for kw in self.stats_keywords if kw in all_text]
                
            else:
                # Try additional heuristics
                screenshot_type, confidence = self._apply_heuristics(image, all_text)
                detected_keywords = [kw for kw in self.general_ui_keywords if kw in all_text]
            
            logger.info(
                f"Screenshot type: {screenshot_type.value}, "
                f"confidence: {confidence:.3f}, "
                f"keywords: {detected_keywords[:5]}"
            )
            
            return screenshot_type, confidence, detected_keywords
            
        except Exception as e:
            logger.error(f"Error analyzing screenshot type: {str(e)}")
            return ScreenshotType.UNKNOWN, 0.0, []
    
    def _count_keyword_matches(self, text: str, keywords: List[str]) -> int:
        """Count how many keywords are found in the text."""
        return sum(1 for keyword in keywords if keyword in text)
    
    def _apply_heuristics(self, image: np.ndarray, text: str) -> Tuple[ScreenshotType, float]:
        """Apply additional heuristics when keyword matching is insufficient."""
        # Image dimension analysis
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        # Check for KDA pattern (common in scoreboards)
        kda_pattern = r"\b\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{1,2}\b"
        kda_matches = len(re.findall(kda_pattern, text))
        
        # Check for gold amounts (typically in scoreboards)
        gold_pattern = r"\b\d{4,6}\b"  # 4-6 digit numbers (gold amounts)
        gold_matches = len(re.findall(gold_pattern, text))
        
        # Check for percentage values (common in stats pages)
        percentage_pattern = r"\b\d{1,3}%\b"
        percentage_matches = len(re.findall(percentage_pattern, text))
        
        # Heuristic scoring
        scoreboard_score = (kda_matches * 2) + (gold_matches * 1.5)
        stats_score = percentage_matches * 2
        
        if scoreboard_score > stats_score and scoreboard_score >= 2:
            return ScreenshotType.SCOREBOARD, min(0.8, 0.3 + (scoreboard_score * 0.1))
        elif stats_score >= 1:
            return ScreenshotType.STATS_PAGE, min(0.7, 0.3 + (stats_score * 0.15))
        else:
            return ScreenshotType.UNKNOWN, 0.1
    
    def extract_basic_info(self, image_path: str, screenshot_type: ScreenshotType) -> Dict[str, Any]:
        """Extract basic information based on screenshot type."""
        try:
            reader = get_ocr_reader()
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ocr_results = reader.readtext(gray, detail=1)
            
            info = {
                "ocr_results": ocr_results,
                "text_content": " ".join([result[1] for result in ocr_results]),
                "screenshot_type": screenshot_type.value
            }
            
            if screenshot_type == ScreenshotType.SCOREBOARD:
                info.update(self._extract_scoreboard_info(ocr_results))
            elif screenshot_type == ScreenshotType.STATS_PAGE:
                info.update(self._extract_stats_info(ocr_results))
            
            return info
            
        except Exception as e:
            logger.error(f"Error extracting basic info: {str(e)}")
            return {"error": str(e)}
    
    def _extract_scoreboard_info(self, ocr_results: List) -> Dict[str, Any]:
        """Extract information specific to scoreboard screenshots."""
        info = {}
        
        # Look for match duration
        text_content = " ".join([result[1] for result in ocr_results])
        duration_match = re.search(r"duration\s+(\d{2}):(\d{2})", text_content.lower())
        if duration_match:
            minutes = int(duration_match.group(1))
            seconds = int(duration_match.group(2))
            info["match_duration"] = minutes + seconds / 60
        
        # Look for victory/defeat
        if "victory" in text_content.lower():
            info["match_result"] = "victory"
        elif "defeat" in text_content.lower():
            info["match_result"] = "defeat"
        
        return info
    
    def _extract_stats_info(self, ocr_results: List) -> Dict[str, Any]:
        """Extract information specific to stats page screenshots."""
        info = {}
        
        text_content = " ".join([result[1] for result in ocr_results])
        
        # Look for damage values
        damage_matches = re.findall(r"(\d{4,7})", text_content)
        if damage_matches:
            info["potential_damage_values"] = [int(match) for match in damage_matches]
        
        # Look for percentage values
        percentage_matches = re.findall(r"(\d{1,3})%", text_content)
        if percentage_matches:
            info["percentage_values"] = [int(match) for match in percentage_matches]
        
        return info


# Global analyzer instance
screenshot_analyzer = ScreenshotAnalyzer()