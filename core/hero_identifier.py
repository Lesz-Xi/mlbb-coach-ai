import re
import logging
from typing import List, Tuple, Optional, Dict, Any
from difflib import get_close_matches
from fuzzywuzzy import fuzz, process
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class HeroIdentifier:
    """Advanced hero identification with fuzzy matching and fallback methods."""
    
    def __init__(self):
        # Comprehensive hero list with common variations and misspellings
        self.heroes = {
            # Marksmen
            'miya': ['miya', 'miyah', 'mya'],
            'layla': ['layla', 'lyla', 'layia'],
            'bruno': ['bruno', 'brno', 'bruno'],
            'clint': ['clint', 'clnt'],
            'moskov': ['moskov', 'moskow', 'moskou'],
            'karrie': ['karrie', 'karie', 'karri'],
            'irithel': ['irithel', 'irithal'],
            'hanabi': ['hanabi', 'hanab'],
            'lesley': ['lesley', 'lesly'],
            'claude': ['claude', 'claud'],
            'kimmy': ['kimmy', 'kimy'],
            'granger': ['granger', 'grangr'],
            'wanwan': ['wanwan', 'wan wan'],
            'popol and kupa': ['popol', 'kupa', 'popol and kupa'],
            'brody': ['brody', 'brodi'],
            'beatrix': ['beatrix', 'beatrx'],
            'natan': ['natan', 'nathan'],
            'melissa': ['melissa', 'melisa'],
            'ixia': ['ixia', 'ixie'],
            
            # Assassins
            'saber': ['saber', 'sabr'],
            'karina': ['karina', 'karin'],
            'fanny': ['fanny', 'fany'],
            'hayabusa': ['hayabusa', 'hayabsa', 'haya'],
            'natalia': ['natalia', 'natalie'],
            'lancelot': ['lancelot', 'lance', 'lancelott'],
            'gusion': ['gusion', 'gusio'],
            'helcurt': ['helcurt', 'helcrt'],
            'hanzo': ['hanzo', 'hanzo'],
            'selena': ['selena', 'selen'],
            'ling': ['ling', 'lng'],
            'benedetta': ['benedetta', 'bene'],
            'karina': ['karina', 'karin'],
            'aamon': ['aamon', 'amon'],
            'yin': ['yin', 'yn'],
            'julian': ['julian', 'julien'],
            'joy': ['joy', 'joi'],
            
            # Fighters
            'alucard': ['alucard', 'alu', 'alucar'],
            'balmond': ['balmond', 'balmon'],
            'alpha': ['alpha', 'alfa'],
            'chou': ['chou', 'chu'],
            'bane': ['bane', 'ban'],
            'zilong': ['zilong', 'zilon'],
            'freya': ['freya', 'freya'],
            'ruby': ['ruby', 'rubi'],
            'roger': ['roger', 'rogr'],
            'jawhead': ['jawhead', 'jawhad'],
            'martis': ['martis', 'matis'],
            'aldous': ['aldous', 'aldus'],
            'leomord': ['leomord', 'leomrd'],
            'thamuz': ['thamuz', 'thamz'],
            'minsitthar': ['minsitthar', 'minsi'],
            'badang': ['badang', 'badan'],
            'guinevere': ['guinevere', 'guin', 'guinevr'],
            'dyrroth': ['dyrroth', 'dyroth'],
            'terizla': ['terizla', 'terzia'],
            'x.borg': ['x.borg', 'xborg', 'x borg'],
            'silvanna': ['silvanna', 'silvan'],
            'yu zhong': ['yu zhong', 'yuzhong', 'yu'],
            'khaleed': ['khaleed', 'khaled'],
            'barats': ['barats', 'barat'],
            'paquito': ['paquito', 'paquit'],
            'aulus': ['aulus', 'alus'],
            'fredrinn': ['fredrinn', 'fredrin'],
            'lapu-lapu': ['lapu-lapu', 'lapu lapu', 'lapu'],
            
            # Tanks
            'tigreal': ['tigreal', 'tigral'],
            'franco': ['franco', 'franc'],
            'minotaur': ['minotaur', 'mino'],
            'lolita': ['lolita', 'lolit'],
            'johnson': ['johnson', 'johnso'],
            'gatot kaca': ['gatot kaca', 'gatot', 'gatotkaca'],
            'uranus': ['uranus', 'urans'],
            'belerick': ['belerick', 'beler'],
            'khufra': ['khufra', 'khufr'],
            'grock': ['grock', 'grck'],
            'atlas': ['atlas', 'atls'],
            'baxia': ['baxia', 'baxi'],
            'hylos': ['hylos', 'hylo'],
            'mathilda': ['mathilda', 'mathild', 'matilda'],
            'gloo': ['gloo', 'glo'],
            'floryn': ['floryn', 'florn'],
            'edith': ['edith', 'edit'],
            
            # Mages
            'nana': ['nana', 'nan'],
            'alice': ['alice', 'alic'],
            'gord': ['gord', 'grd'],
            'eudora': ['eudora', 'eudor'],
            'cyclops': ['cyclops', 'cyclop'],
            'kagura': ['kagura', 'kagur'],
            'aurora': ['aurora', 'auror'],
            'vexana': ['vexana', 'vexan'],
            'harley': ['harley', 'harle'],
            'pharsa': ['pharsa', 'phars'],
            'zhask': ['zhask', 'zask'],
            'chang\'e': ['chang\'e', 'change', 'chang e'],
            'lunox': ['lunox', 'lunx'],
            'vale': ['vale', 'val'],
            'valir': ['valir', 'valr'],
            'cecilion': ['cecilion', 'cecil'],
            'luo yi': ['luo yi', 'luoyi', 'luo'],
            'yve': ['yve', 'yv'],
            'xavier': ['xavier', 'xavir'],
            'novaria': ['novaria', 'novar'],
            
            # Support
            'estes': ['estes', 'este'],
            'rafaela': ['rafaela', 'rafa'],
            'diggie': ['diggie', 'digie'],
            'angela': ['angela', 'angel'],
            'kaja': ['kaja', 'kaj'],
            'faramis': ['faramis', 'faram'],
            'carmilla': ['carmilla', 'carmila'],
            'chip': ['chip', 'chp']
        }
        
        # Flatten hero list for quick lookup
        self.all_heroes = []
        self.hero_mapping = {}  # Maps variations to canonical name
        
        for canonical, variations in self.heroes.items():
            self.all_heroes.extend(variations)
            for variation in variations:
                self.hero_mapping[variation.lower()] = canonical
        
        # Common OCR misreadings
        self.ocr_corrections = {
            '0': 'o', '1': 'i', '5': 's', '8': 'b',
            'rn': 'm', 'vv': 'w', 'ii': 'n'
        }
    
    def identify_hero(self, text: str, confidence_threshold: float = 0.7) -> Tuple[str, float]:
        """
        Identify hero from text using multiple methods.
        
        Args:
            text: Text potentially containing hero name
            confidence_threshold: Minimum confidence for positive identification
            
        Returns:
            Tuple of (hero_name, confidence)
        """
        if not text:
            return "unknown", 0.0
        
        text_clean = self._clean_text(text)
        
        # Method 1: Exact match
        hero, confidence = self._exact_match(text_clean)
        if confidence >= confidence_threshold:
            logger.info(f"Hero identified by exact match: {hero} (confidence: {confidence:.3f})")
            return hero, confidence
        
        # Method 2: Fuzzy string matching
        hero, confidence = self._fuzzy_match(text_clean)
        if confidence >= confidence_threshold:
            logger.info(f"Hero identified by fuzzy match: {hero} (confidence: {confidence:.3f})")
            return hero, confidence
        
        # Method 3: Partial matching with OCR corrections
        hero, confidence = self._partial_match_with_corrections(text_clean)
        if confidence >= confidence_threshold:
            logger.info(f"Hero identified by corrected match: {hero} (confidence: {confidence:.3f})")
            return hero, confidence
        
        # Method 4: Substring matching
        hero, confidence = self._substring_match(text_clean)
        if confidence >= confidence_threshold:
            logger.info(f"Hero identified by substring match: {hero} (confidence: {confidence:.3f})")
            return hero, confidence
        
        logger.warning(f"Could not identify hero from text: '{text}'")
        return "unknown", 0.0
    
    def _clean_text(self, text: str) -> str:
        """Clean text for better matching."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces and dots
        text = re.sub(r'[^\w\s\.\-\']', ' ', text)
        
        # Normalize spaces
        text = ' '.join(text.split())
        
        return text
    
    def _exact_match(self, text: str) -> Tuple[str, float]:
        """Try exact matching first."""
        words = text.split()
        
        for word in words:
            if word in self.hero_mapping:
                return self.hero_mapping[word], 1.0
        
        # Try multi-word combinations
        for i in range(len(words)):
            for j in range(i + 1, min(i + 4, len(words) + 1)):  # Max 3-word hero names
                phrase = ' '.join(words[i:j])
                if phrase in self.hero_mapping:
                    return self.hero_mapping[phrase], 1.0
        
        return "unknown", 0.0
    
    def _fuzzy_match(self, text: str) -> Tuple[str, float]:
        """Use fuzzy string matching."""
        best_match = process.extractOne(
            text, 
            self.all_heroes, 
            scorer=fuzz.token_sort_ratio
        )
        
        if best_match and best_match[1] >= 70:  # 70% similarity threshold
            hero_variation = best_match[0]
            canonical_name = self.hero_mapping.get(hero_variation.lower(), hero_variation)
            confidence = best_match[1] / 100.0
            return canonical_name, confidence
        
        return "unknown", 0.0
    
    def _partial_match_with_corrections(self, text: str) -> Tuple[str, float]:
        """Try matching with OCR error corrections."""
        # Apply common OCR corrections
        corrected_text = text
        for wrong, right in self.ocr_corrections.items():
            corrected_text = corrected_text.replace(wrong, right)
        
        if corrected_text != text:
            return self._fuzzy_match(corrected_text)
        
        return "unknown", 0.0
    
    def _substring_match(self, text: str) -> Tuple[str, float]:
        """Try finding hero names as substrings."""
        text_words = set(text.split())
        
        best_hero = "unknown"
        best_score = 0.0
        
        for canonical, variations in self.heroes.items():
            for variation in variations:
                # Check if variation is a substring
                if variation in text:
                    # Calculate confidence based on length ratio
                    confidence = len(variation) / len(text)
                    if confidence > best_score:
                        best_hero = canonical
                        best_score = confidence
                
                # Check word-level matching
                variation_words = set(variation.split())
                if variation_words.issubset(text_words):
                    confidence = len(variation_words) / len(text_words)
                    if confidence > best_score:
                        best_hero = canonical
                        best_score = confidence
        
        return best_hero, min(best_score, 0.9)  # Cap at 0.9 for substring matches
    
    def identify_from_ocr_results(self, ocr_results: List) -> Tuple[str, float]:
        """Identify hero from OCR results with position-aware analysis."""
        if not ocr_results:
            return "unknown", 0.0
        
        # Extract all text
        all_text = " ".join([result[1] for result in ocr_results])
        
        # Try standard identification first
        hero, confidence = self.identify_hero(all_text)
        if confidence > 0.7:
            return hero, confidence
        
        # Try each OCR result individually (sometimes hero names are isolated)
        best_hero = "unknown"
        best_confidence = 0.0
        
        for bbox, text, ocr_conf in ocr_results:
            if len(text.strip()) > 2:  # Skip very short text
                hero, conf = self.identify_hero(text)
                # Weight by OCR confidence
                weighted_conf = conf * ocr_conf
                if weighted_conf > best_confidence:
                    best_hero = hero
                    best_confidence = weighted_conf
        
        return best_hero, best_confidence
    
    def get_hero_suggestions(self, text: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """Get top N hero suggestions for debugging."""
        if not text:
            return []
        
        text_clean = self._clean_text(text)
        matches = process.extract(
            text_clean, 
            self.all_heroes, 
            scorer=fuzz.token_sort_ratio,
            limit=top_n
        )
        
        suggestions = []
        for match, score in matches:
            canonical = self.hero_mapping.get(match.lower(), match)
            suggestions.append((canonical, score / 100.0))
        
        return suggestions


# Global hero identifier instance
hero_identifier = HeroIdentifier()