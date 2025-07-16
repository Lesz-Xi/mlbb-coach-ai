"""
Row-Specific Hero Detector

Fixes the critical issue where hero detection analyzes the entire image
instead of focusing on the specific player's row that was already identified.

This addresses the user's Mathilda -> Miya misclassification issue.
"""
import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Optional
import re
from difflib import SequenceMatcher

from .hero_identifier import hero_identifier
from .data_collector import get_ocr_reader

logger = logging.getLogger(__name__)


class RowSpecificHeroDetector:
    """
    Hero detector that focuses on the specific player's row instead of full image.
    
    This fixes the issue where the system detects heroes from other players
    instead of the target player's row.
    """
    
    def __init__(self):
        self.hero_identifier = hero_identifier
        self.ocr_reader = None
        
        # Support hero patterns that might be misclassified
        self.support_heroes = {
            'mathilda': ['mathilda', 'matilda', 'mathylda', 'matthilda'],
            'estes': ['estes', 'estés', 'este'],
            'angela': ['angela', 'anjela', 'anggela'],
            'diggie': ['diggie', 'digie', 'diggi'],
            'rafaela': ['rafaela', 'rafaella', 'raffaela']
        }
        
        # Common OCR misreads for hero names
        self.hero_ocr_fixes = {
            'miya': ['mathilda'],  # Common misread
            'roger': ['mathilda', 'angela'],  # Another common misread
            'layla': ['mathilda']
        }
    
    def detect_hero_in_player_row(
        self,
        image_path: str,
        player_ign: str,
        player_row_y: float,
        ocr_results: List,
        hero_override: Optional[str] = None
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Detect hero specifically in the player's row.
        
        Args:
            image_path: Path to screenshot
            player_ign: Player's IGN
            player_row_y: Y-coordinate of player's row
            ocr_results: Full OCR results
            hero_override: Manual hero override
            
        Returns:
            Tuple of (hero_name, confidence, debug_info)
        """
        debug_info = {
            "method": "row_specific",
            "player_row_y": player_row_y,
            "strategies_tried": [],
            "row_texts": [],
            "hero_candidates": [],
            "ocr_corrections_applied": []
        }
        
        if hero_override:
            validated_hero = self._validate_hero_name(hero_override)
            debug_info["strategies_tried"].append("manual_override")
            return validated_hero, 1.0, debug_info
        
        # Extract texts from player's row only
        row_texts = self._extract_row_texts(ocr_results, player_row_y)
        debug_info["row_texts"] = row_texts
        
        if not row_texts:
            logger.warning(f"No texts found in player row at y={player_row_y}")
            return "unknown", 0.0, debug_info
        
        # Try multiple detection strategies
        strategies = [
            ("exact_match", self._exact_hero_match),
            ("fuzzy_match", self._fuzzy_hero_match),
            ("support_pattern_match", self._support_pattern_match),
            ("ocr_correction_match", self._ocr_correction_match),
            ("visual_analysis", self._visual_row_analysis)
        ]
        
        best_hero = "unknown"
        best_confidence = 0.0
        
        for strategy_name, strategy_func in strategies:
            debug_info["strategies_tried"].append(strategy_name)
            
            try:
                hero, confidence = strategy_func(row_texts, image_path, player_row_y)
                
                if confidence > best_confidence:
                    best_hero = hero
                    best_confidence = confidence
                    debug_info["best_strategy"] = strategy_name
                
                debug_info["hero_candidates"].append({
                    "strategy": strategy_name,
                    "hero": hero,
                    "confidence": confidence
                })
                
                # If we have high confidence, we can stop
                if confidence >= 0.9:
                    break
                    
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {str(e)}")
        
        # Apply contextual validation
        if best_hero != "unknown":
            best_hero, best_confidence = self._validate_hero_context(
                best_hero, best_confidence, row_texts, player_ign
            )
        
        logger.info(f"Row-specific hero detection: {best_hero} (confidence: {best_confidence:.3f})")
        return best_hero, best_confidence, debug_info
    
    def _extract_row_texts(self, ocr_results: List, player_row_y: float, tolerance: int = 50) -> List[str]:
        """Extract all texts from the player's row."""
        row_texts = []
        
        for bbox, text, conf in ocr_results:
            # Calculate Y center of this text
            y_coords = [point[1] for point in bbox]
            text_y_center = (min(y_coords) + max(y_coords)) / 2
            
            # Check if this text is in the player's row
            if abs(text_y_center - player_row_y) <= tolerance:
                if text.strip() and conf > 0.5:  # Only high-confidence text
                    row_texts.append(text.strip())
        
        logger.debug(f"Extracted {len(row_texts)} texts from player row: {row_texts}")
        return row_texts
    
    def _exact_hero_match(self, row_texts: List[str], image_path: str, player_row_y: float) -> Tuple[str, float]:
        """Try exact hero name matching in row texts."""
        for text in row_texts:
            cleaned_text = text.lower().strip()
            hero_result = self.hero_identifier.identify_hero(cleaned_text)
            
            if hero_result and hero_result != "unknown":
                return hero_result, 0.95
        
        return "unknown", 0.0
    
    def _fuzzy_hero_match(self, row_texts: List[str], image_path: str, player_row_y: float) -> Tuple[str, float]:
        """Try fuzzy matching for hero names."""
        all_heroes = list(self.hero_identifier.hero_names)
        best_hero = "unknown"
        best_score = 0.0
        
        for text in row_texts:
            cleaned_text = text.lower().strip()
            
            for hero in all_heroes:
                similarity = SequenceMatcher(None, cleaned_text, hero.lower()).ratio()
                
                if similarity > best_score and similarity >= 0.6:
                    best_hero = hero
                    best_score = similarity
        
        confidence = min(0.9, best_score * 1.2)  # Scale up similarity to confidence
        return best_hero, confidence
    
    def _support_pattern_match(self, row_texts: List[str], image_path: str, player_row_y: float) -> Tuple[str, float]:
        """Special pattern matching for support heroes."""
        combined_text = " ".join(row_texts).lower()
        
        for hero, patterns in self.support_heroes.items():
            for pattern in patterns:
                if pattern in combined_text:
                    return hero, 0.85
        
        # Check for support-specific indicators
        support_indicators = [
            'heal', 'shield', 'buff', 'assist', 'support', 'helper'
        ]
        
        for indicator in support_indicators:
            if indicator in combined_text:
                # If we see support indicators, bias towards support heroes
                # Look for partial matches with support heroes
                for hero in ['mathilda', 'estes', 'angela', 'diggie']:
                    for text in row_texts:
                        if any(char in text.lower() for char in hero[:3]):
                            return hero, 0.7
        
        return "unknown", 0.0
    
    def _ocr_correction_match(self, row_texts: List[str], image_path: str, player_row_y: float) -> Tuple[str, float]:
        """Apply OCR corrections and try matching again."""
        for text in row_texts:
            cleaned_text = self._apply_ocr_corrections(text.lower())
            
            # Try hero identification on corrected text
            hero_result = self.hero_identifier.identify_hero(cleaned_text)
            if hero_result and hero_result != "unknown":
                return hero_result, 0.8
            
            # Check against known OCR misreads
            for misread_hero, possible_heroes in self.hero_ocr_fixes.items():
                if misread_hero in cleaned_text:
                    # This might be a misread - check context
                    for possible_hero in possible_heroes:
                        return possible_hero, 0.75
        
        return "unknown", 0.0
    
    def _visual_row_analysis(self, row_texts: List[str], image_path: str, player_row_y: float) -> Tuple[str, float]:
        """
        Analyze the visual region around the player's row for hero portraits/indicators.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return "unknown", 0.0
            
            # Extract a strip around the player's row
            strip_height = 100  # 50 pixels above and below
            y_start = max(0, int(player_row_y - strip_height // 2))
            y_end = min(image.shape[0], int(player_row_y + strip_height // 2))
            
            row_strip = image[y_start:y_end, :]
            
            # Save debug strip
            debug_path = f"temp/player_row_strip_{int(player_row_y)}.png"
            cv2.imwrite(debug_path, row_strip)
            logger.debug(f"Saved player row strip to: {debug_path}")
            
            # Perform OCR on just this strip
            if self.ocr_reader is None:
                self.ocr_reader = get_ocr_reader()
            
            gray_strip = cv2.cvtColor(row_strip, cv2.COLOR_BGR2GRAY)
            strip_results = self.ocr_reader.readtext(gray_strip, detail=1)
            
            # Analyze strip-specific results
            for bbox, text, conf in strip_results:
                if conf > 0.7:  # High confidence text only
                    hero_result = self.hero_identifier.identify_hero(text.lower())
                    if hero_result and hero_result != "unknown":
                        return hero_result, 0.85
            
        except Exception as e:
            logger.warning(f"Visual row analysis failed: {str(e)}")
        
        return "unknown", 0.0
    
    def _apply_ocr_corrections(self, text: str) -> str:
        """Apply common OCR corrections."""
        corrections = {
            '0': 'o', '1': 'i', '5': 's', '8': 'b',
            'rn': 'm', 'vv': 'w', 'ii': 'n'
        }
        
        corrected = text
        for wrong, right in corrections.items():
            corrected = corrected.replace(wrong, right)
        
        return corrected
    
    def _validate_hero_context(self, hero: str, confidence: float, row_texts: List[str], player_ign: str) -> Tuple[str, float]:
        """Validate hero detection against context."""
        # Check if this makes sense given the game context
        combined_text = " ".join(row_texts).lower()
        
        # If we detected a marksman hero but see support indicators, reduce confidence
        marksman_heroes = ['miya', 'layla', 'bruno', 'clint', 'moskov']
        support_indicators = ['heal', 'shield', 'assist', 'support']
        
        if hero in marksman_heroes and any(indicator in combined_text for indicator in support_indicators):
            logger.warning(f"Detected {hero} but found support indicators, reducing confidence")
            confidence *= 0.6
        
        # Boost confidence for support heroes with support indicators
        support_heroes = ['mathilda', 'estes', 'angela', 'diggie']
        if hero in support_heroes and any(indicator in combined_text for indicator in support_indicators):
            confidence = min(0.95, confidence * 1.2)
        
        return hero, confidence
    
    def _validate_hero_name(self, hero_name: str) -> str:
        """Validate and normalize hero name."""
        if not hero_name:
            return "unknown"
        
        # Check if it's a valid hero
        normalized = hero_name.lower().strip()
        if normalized in self.hero_identifier.hero_names:
            return normalized
        
        # Try fuzzy matching against known heroes
        all_heroes = list(self.hero_identifier.hero_names)
        best_match = None
        best_score = 0.0
        
        for hero in all_heroes:
            score = SequenceMatcher(None, normalized, hero.lower()).ratio()
            if score > best_score and score >= 0.8:
                best_match = hero
                best_score = score
        
        return best_match if best_match else "unknown"


# Create global instance
row_specific_hero_detector = RowSpecificHeroDetector() 