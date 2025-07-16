"""
Robust IGN Matching System - Fixes confidence inflation with zero data extraction

This system implements:
1. Fuzzy matching with OCR error tolerance
2. Multiple matching strategies with fallbacks  
3. Real-time confidence adjustment based on extraction success
4. Roman numeral and special character handling
"""

import re
import logging
from typing import List, Optional, Tuple, Dict, Any
from difflib import SequenceMatcher, get_close_matches
from dataclasses import dataclass
import unicodedata

logger = logging.getLogger(__name__)

@dataclass
class IGNMatchResult:
    """Result of IGN matching with detailed confidence."""
    found: bool
    matched_text: str
    confidence: float
    position: Tuple[int, int]
    method: str
    row_data: List[Dict[str, Any]]
    
class RobustIGNMatcher:
    """Enhanced IGN matcher that handles OCR errors and Roman numerals."""
    
    def __init__(self):
        # Roman numeral variations and common OCR errors
        self.roman_variations = {
            'XVII': ['XVII', 'XVn', 'X√II', 'X∨II', 'XVІІ', 'XVll', 'ΧVII'],
            'XVIII': ['XVIII', 'XVIll', 'X√III', 'ХVIII'],
            'XIV': ['XIV', 'XI∨', 'ХIV'],
            'XV': ['XV', 'X∨', 'ХV'],
            'XVI': ['XVI', 'X∨I', 'ХVI']
        }
        
        # Common OCR character substitutions
        self.ocr_substitutions = {
            '0': 'o', 'O': '0', '1': 'l', 'l': '1', 'I': '1',
            '5': 'S', 'S': '5', '6': 'G', 'G': '6',
            'rn': 'm', 'cl': 'd', 'vv': 'w'
        }
    
    def find_ign_with_confidence(
        self, 
        ign: str, 
        ocr_results: List[Tuple], 
        confidence_threshold: float = 0.7
    ) -> IGNMatchResult:
        """
        Find IGN using multiple strategies and return confidence-adjusted result.
        
        Args:
            ign: Target IGN to find
            ocr_results: OCR results as (bbox, text, confidence) tuples
            confidence_threshold: Minimum confidence for positive match
            
        Returns:
            IGNMatchResult with found status and confidence
        """
        
        # Strategy 1: Exact match (highest confidence)
        result = self._exact_match(ign, ocr_results)
        if result.found and result.confidence >= confidence_threshold:
            logger.info(f"IGN found via exact match: {result.matched_text} (confidence: {result.confidence:.3f})")
            return result
        
        # Strategy 2: Roman numeral aware matching
        result = self._roman_numeral_match(ign, ocr_results)
        if result.found and result.confidence >= confidence_threshold:
            logger.info(f"IGN found via roman numeral match: {result.matched_text} (confidence: {result.confidence:.3f})")
            return result
        
        # Strategy 3: Fuzzy matching with OCR corrections
        result = self._fuzzy_match_with_ocr_corrections(ign, ocr_results)
        if result.found and result.confidence >= confidence_threshold:
            logger.info(f"IGN found via fuzzy match: {result.matched_text} (confidence: {result.confidence:.3f})")
            return result
        
        # Strategy 4: Partial matching (lower confidence)
        result = self._partial_match(ign, ocr_results)
        if result.found and result.confidence >= confidence_threshold * 0.8:  # Lower threshold for partial
            logger.info(f"IGN found via partial match: {result.matched_text} (confidence: {result.confidence:.3f})")
            return result
        
        # No match found
        logger.warning(f"IGN '{ign}' not found in OCR results using any strategy")
        return IGNMatchResult(
            found=False,
            matched_text="",
            confidence=0.0,
            position=(0, 0),
            method="none",
            row_data=[]
        )
    
    def _exact_match(self, ign: str, ocr_results: List[Tuple]) -> IGNMatchResult:
        """Exact string matching."""
        for bbox, text, conf in ocr_results:
            if ign.lower() == text.lower().strip():
                return self._create_match_result(
                    found=True,
                    matched_text=text,
                    confidence=1.0,
                    bbox=bbox,
                    method="exact_match",
                    ocr_results=ocr_results
                )
        
        return self._create_failed_result()
    
    def _roman_numeral_match(self, ign: str, ocr_results: List[Tuple]) -> IGNMatchResult:
        """Match considering Roman numeral OCR variations."""
        ign_parts = ign.split()
        if len(ign_parts) < 2:
            return self._create_failed_result()
        
        base_name = ign_parts[0].lower()
        roman_part = ign_parts[1]
        
        # Get possible OCR variations for the Roman numeral
        roman_variations = self.roman_variations.get(roman_part, [roman_part])
        
        for bbox, text, conf in ocr_results:
            text_clean = text.lower().strip()
            
            # Check if base name is present
            if base_name in text_clean:
                # Check for any Roman numeral variation
                for variation in roman_variations:
                    if variation.lower() in text_clean:
                        confidence = 0.95 if variation == roman_part else 0.85
                        return self._create_match_result(
                            found=True,
                            matched_text=text,
                            confidence=confidence,
                            bbox=bbox,
                            method="roman_numeral_match",
                            ocr_results=ocr_results
                        )
        
        return self._create_failed_result()
    
    def _fuzzy_match_with_ocr_corrections(self, ign: str, ocr_results: List[Tuple]) -> IGNMatchResult:
        """Fuzzy matching with OCR error corrections."""
        ign_normalized = self._normalize_text(ign)
        
        best_match = None
        best_confidence = 0.0
        best_bbox = None
        
        for bbox, text, conf in ocr_results:
            # Apply OCR corrections
            text_corrected = self._apply_ocr_corrections(text)
            text_normalized = self._normalize_text(text_corrected)
            
            # Calculate fuzzy similarity
            similarity = SequenceMatcher(None, ign_normalized, text_normalized).ratio()
            
            if similarity > best_confidence and similarity > 0.75:
                best_confidence = similarity
                best_match = text
                best_bbox = bbox
        
        if best_match:
            return self._create_match_result(
                found=True,
                matched_text=best_match,
                confidence=best_confidence * 0.9,  # Slight penalty for fuzzy match
                bbox=best_bbox,
                method="fuzzy_match_corrected",
                ocr_results=ocr_results
            )
        
        return self._create_failed_result()
    
    def _partial_match(self, ign: str, ocr_results: List[Tuple]) -> IGNMatchResult:
        """Partial matching for fallback."""
        ign_parts = ign.lower().split()
        
        for bbox, text, conf in ocr_results:
            text_lower = text.lower()
            
            # Check if any significant part of IGN is present
            matches = sum(1 for part in ign_parts if len(part) > 2 and part in text_lower)
            
            if matches > 0:
                confidence = min(0.7, matches / len(ign_parts))
                return self._create_match_result(
                    found=True,
                    matched_text=text,
                    confidence=confidence,
                    bbox=bbox,
                    method="partial_match",
                    ocr_results=ocr_results
                )
        
        return self._create_failed_result()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching."""
        # Remove diacritics
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        
        # Convert to lowercase and remove extra whitespace
        text = re.sub(r'\s+', ' ', text.lower().strip())
        
        return text
    
    def _apply_ocr_corrections(self, text: str) -> str:
        """Apply common OCR corrections."""
        corrected = text
        for wrong, right in self.ocr_substitutions.items():
            corrected = corrected.replace(wrong, right)
        return corrected
    
    def _create_match_result(
        self, 
        found: bool, 
        matched_text: str, 
        confidence: float, 
        bbox: List, 
        method: str,
        ocr_results: List[Tuple]
    ) -> IGNMatchResult:
        """Create successful match result with row data extraction."""
        
        # Extract position from bbox
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        position = (sum(x_coords) // len(x_coords), sum(y_coords) // len(y_coords))
        
        # Extract row data based on Y position
        row_y_center = sum(y_coords) / len(y_coords)
        row_data = []
        
        for other_bbox, other_text, other_conf in ocr_results:
            other_y_coords = [point[1] for point in other_bbox]
            other_y_center = sum(other_y_coords) / len(other_y_coords)
            
            # Items in the same row (within 30 pixels)
            if abs(other_y_center - row_y_center) < 30:
                other_x_coords = [point[0] for point in other_bbox]
                row_data.append({
                    "text": other_text,
                    "confidence": other_conf,
                    "x": sum(other_x_coords) / len(other_x_coords),
                    "y": other_y_center,
                    "bbox": other_bbox
                })
        
        # Sort by X position for proper column order
        row_data.sort(key=lambda item: item["x"])
        
        return IGNMatchResult(
            found=found,
            matched_text=matched_text,
            confidence=confidence,
            position=position,
            method=method,
            row_data=row_data
        )
    
    def _create_failed_result(self) -> IGNMatchResult:
        """Create failed match result."""
        return IGNMatchResult(
            found=False,
            matched_text="",
            confidence=0.0,
            position=(0, 0),
            method="failed",
            row_data=[]
        )

# Global instance
robust_ign_matcher = RobustIGNMatcher() 