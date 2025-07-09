import re
import logging
from typing import List, Optional, Tuple, Dict
from difflib import get_close_matches, SequenceMatcher
from dataclasses import dataclass
import unicodedata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IGNMatch:
    """Represents a matched IGN with confidence score."""
    original_text: str
    matched_ign: str
    confidence: float
    position: Tuple[int, int]  # (x, y) position in image


class IGNValidator:
    def __init__(self):
        # MLBB-specific allowed characters
        self.allowed_special = {'|', '[', ']', '.', '_', '-', ' '}
        
        # Common OCR interference patterns
        self.ocr_noise_patterns = [
            r'★\d+',           # Star ratings
            r'\b(I{1,3}|IV)\b', # Roman numeral ranks
            r'\bOffline\b',     # Status text
            r'\bOnline\b',
            r'\bAll\b',         # UI elements
            r'\bFr\b',          # Friends tab
            r'[\u2500-\u257F]+'  # Box-drawing characters (profile borders)
        ]
    
    def clean_ocr_result(self, raw_text: str) -> Tuple[str, List[str]]:
        """Remove OCR noise from MLBB screenshots."""
        cleaned = raw_text
        
        # Remove common UI elements
        for pattern in self.ocr_noise_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove standalone numbers (likely coordinates)
        cleaned = re.sub(r'\b\d+\b', '', cleaned)
        
        # Handle flag emojis separately (don't remove, just note)
        flags = re.findall(r'[\U0001F1E6-\U0001F1FF]{2}', cleaned)
        
        cleaned = cleaned.strip()
        cleaned = unicodedata.normalize('NFC', cleaned)
        return cleaned, flags
    
    def validate_mlbb_ign(self, ocr_text: str) -> Dict[str, any]:
        """MLBB-specific IGN validation."""
        cleaned_ign, flags = self.clean_ocr_result(ocr_text)
        
        issues = []
        
        # Check length (MLBB: 4-16 characters excluding spaces)
        ign_no_spaces = cleaned_ign.replace(' ', '')
        if not 4 <= len(ign_no_spaces) <= 16:
            issues.append("invalid_length")
        
        # Check for mixed scripts (common in MLBB)
        scripts = self._detect_scripts(cleaned_ign)
        if len(scripts) > 3:  # Allow up to 3 scripts
            issues.append("too_many_scripts")
        
        # MLBB-specific patterns that INCREASE confidence
        confidence_boost = 1.0
        
        # Clan tags with brackets
        if re.search(r'\[[\w\s]+\]', cleaned_ign):
            confidence_boost *= 1.2
            
        # Vertical bars (squad separator)
        if '|' in cleaned_ign:
            confidence_boost *= 1.1
            
        # Superscript/special Unicode (pro players)
        if re.search(r'[ᵃ-ᵿ]', cleaned_ign):
            confidence_boost *= 1.1
        
        return {
            'original': ocr_text,
            'cleaned_ign': cleaned_ign,
            'country_flags': flags,
            'scripts_detected': list(scripts),
            'is_valid': len(issues) == 0,
            'issues': issues,
            'confidence': min(confidence_boost, 1.5),
            'mlbb_style_score': self._calculate_style_score(cleaned_ign)
        }
    
    def _detect_scripts(self, text: str) -> set:
        """Detect Unicode script types in text."""
        scripts = set()
        for char in text:
            if re.match(r'[A-Za-z\u00C0-\u024F]', char):
                scripts.add('latin')
            elif re.match(r'[\u3040-\u309F\u30A0-\u30FF]', char):
                scripts.add('japanese')
            elif re.match(r'[\uAC00-\uD7AF]', char):
                scripts.add('korean')
            elif re.match(r'[\u4E00-\u9FFF]', char):
                scripts.add('chinese')
            elif re.match(r'[\u0400-\u04FF]', char):
                scripts.add('cyrillic')
        return scripts
    
    def _calculate_style_score(self, ign: str) -> float:
        """Score how 'MLBB-like' an IGN is."""
        score = 0.5  # Base score
        
        # Common MLBB patterns
        patterns = {
            r'\|.*\|': 0.2,              # |text|
            r'^\[[\w\s]+\]': 0.15,       # [CLAN]
            r'[ᵃ-ᵿ⁰-⁹]': 0.1,           # Superscript
            r'[★☆✦✧♦♠♥♣]': 0.1,         # Symbols
            r'\d{2,4}$': 0.1,            # Numbers at end
            r'(?i)(pro|god|king)': 0.05, # Common words
        }
        
        for pattern, boost in patterns.items():
            if re.search(pattern, ign):
                score += boost
                
        return min(score, 1.0)