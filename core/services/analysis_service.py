"""
Analysis Service Implementation
Handles data parsing, intelligent completion, and match analysis
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import re

from .base_service import BaseService, ServiceResult

logger = logging.getLogger(__name__)


class AnalysisService(BaseService):
    """Service for analyzing OCR results and generating match data"""
    
    def __init__(self):
        super().__init__("AnalysisService")
        self.data_collector = None
        self.intelligent_completer = None
        self.pattern_matchers = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for data extraction"""
        return {
            "kda": re.compile(r"(\d+)[/-](\d+)[/-](\d+)"),
            "damage": re.compile(r"(\d{1,3}[.,]?\d{0,3})"),
            "gold": re.compile(r"(\d{1,2}[.,]?\d{0,3})"),
            "percentage": re.compile(r"(\d{1,3})%"),
            "timer": re.compile(r"(\d{1,2}):(\d{2})")
        }
    
    async def process(self, request: Dict[str, Any]) -> ServiceResult:
        """Process analysis request"""
        try:
            ocr_results = request.get("ocr_results", [])
            hero = request.get("hero", "unknown")
            player_ign = request.get("player_ign", "")
            context = request.get("context", "scoreboard")
            
            if not ocr_results:
                return ServiceResult(
                    success=False,
                    error="No OCR results provided"
                )
            
            # Initialize components if needed
            await self._initialize_components()
            
            # Parse OCR results
            parsed_data = self._parse_ocr_results(
                ocr_results, player_ign, context
            )
            
            # Complete missing data intelligently
            completed_data = await self._complete_data(
                parsed_data, hero, context
            )
            
            # Validate and score data quality
            validation_result = self._validate_data(completed_data)
            
            # Generate final match data
            match_data = self._generate_match_data(
                completed_data, 
                validation_result,
                hero
            )
            
            return ServiceResult(
                success=True,
                data=match_data,
                metadata={
                    "data_completeness": validation_result["completeness"],
                    "confidence_score": validation_result["confidence"],
                    "missing_fields": validation_result["missing_fields"],
                    "context": context
                }
            )
            
        except Exception as e:
            logger.error(f"Analysis processing failed: {str(e)}")
            return ServiceResult(
                success=False,
                error=str(e)
            )
    
    async def _initialize_components(self):
        """Initialize data processing components"""
        if self.data_collector is None:
            try:
                from ..enhanced_data_collector import EnhancedDataCollector
                self.data_collector = EnhancedDataCollector()
            except ImportError:
                from core.enhanced_data_collector import (
                    EnhancedDataCollector
                )
                self.data_collector = EnhancedDataCollector()
        
        if self.intelligent_completer is None:
            try:
                from ..intelligent_data_completer import (
                    IntelligentDataCompleter
                )
                self.intelligent_completer = IntelligentDataCompleter()
            except ImportError:
                from core.intelligent_data_completer import (
                    IntelligentDataCompleter
                )
                self.intelligent_completer = IntelligentDataCompleter()
    
    def _parse_ocr_results(
        self, 
        ocr_results: List[Tuple],
        player_ign: str,
        context: str
    ) -> Dict[str, Any]:
        """Parse OCR results to extract structured data"""
        parsed_data = {
            "kda": {"kills": None, "deaths": None, "assists": None},
            "damage": {"hero": None, "turret": None, "taken": None},
            "gold": {"earned": None, "spent": None},
            "participation": None,
            "mvp_score": None,
            "match_duration": None
        }
        
        # Find player row
        player_row_idx = self._find_player_row(ocr_results, player_ign)
        
        if player_row_idx is None:
            logger.warning(f"Could not find player row for {player_ign}")
            return parsed_data
        
        # Extract data from player row and nearby regions
        for i in range(
            max(0, player_row_idx - 2), 
            min(len(ocr_results), player_row_idx + 3)
        ):
            bbox, text, confidence = ocr_results[i]
            
            # Try to match patterns
            self._extract_kda(text, parsed_data)
            self._extract_damage(text, parsed_data)
            self._extract_gold(text, parsed_data)
            self._extract_participation(text, parsed_data)
            self._extract_duration(text, parsed_data)
        
        return parsed_data
    
    def _find_player_row(
        self, 
        ocr_results: List[Tuple],
        player_ign: str
    ) -> Optional[int]:
        """Find the row index containing the player's IGN"""
        player_ign_lower = player_ign.lower()
        
        for idx, (bbox, text, confidence) in enumerate(ocr_results):
            if player_ign_lower in text.lower():
                return idx
        
        # Try fuzzy matching if exact match fails
        from difflib import SequenceMatcher
        best_match_idx = None
        best_ratio = 0.0
        
        for idx, (bbox, text, confidence) in enumerate(ocr_results):
            ratio = SequenceMatcher(
                None, player_ign_lower, text.lower()
            ).ratio()
            if ratio > best_ratio and ratio > 0.8:
                best_ratio = ratio
                best_match_idx = idx
        
        return best_match_idx
    
    def _extract_kda(self, text: str, data: Dict[str, Any]):
        """Extract KDA from text"""
        match = self.pattern_matchers["kda"].search(text)
        if match:
            data["kda"]["kills"] = int(match.group(1))
            data["kda"]["deaths"] = int(match.group(2))
            data["kda"]["assists"] = int(match.group(3))
    
    def _extract_damage(self, text: str, data: Dict[str, Any]):
        """Extract damage values from text"""
        # Look for damage indicators
        text_lower = text.lower()
        
        if "damage" in text_lower or "dmg" in text_lower:
            numbers = self.pattern_matchers["damage"].findall(text)
            if numbers:
                # Convert to integers
                values = []
                for num in numbers:
                    num_str = num.replace(",", "").replace(".", "")
                    try:
                        values.append(int(num_str))
                    except ValueError:
                        continue
                
                if values:
                    # Assign based on context and magnitude
                    if "hero" in text_lower:
                        data["damage"]["hero"] = values[0]
                    elif "turret" in text_lower or "tower" in text_lower:
                        data["damage"]["turret"] = values[0]
                    elif "taken" in text_lower or "received" in text_lower:
                        data["damage"]["taken"] = values[0]
                    else:
                        # Assume largest is hero damage
                        if not data["damage"]["hero"]:
                            data["damage"]["hero"] = max(values)
    
    def _extract_gold(self, text: str, data: Dict[str, Any]):
        """Extract gold values from text"""
        if "gold" in text.lower() or "g" in text.lower():
            numbers = self.pattern_matchers["gold"].findall(text)
            if numbers:
                # Convert to integers
                for num in numbers:
                    num_str = num.replace(",", "").replace(".", "")
                    try:
                        value = int(num_str)
                        if value > 100:  # Likely a gold value
                            if "earned" in text.lower():
                                data["gold"]["earned"] = value
                            elif not data["gold"]["earned"]:
                                data["gold"]["earned"] = value
                    except ValueError:
                        continue
    
    def _extract_participation(self, text: str, data: Dict[str, Any]):
        """Extract participation percentage from text"""
        match = self.pattern_matchers["percentage"].search(text)
        if (match and ("participation" in text.lower() or 
                       "teamfight" in text.lower() or
                       "tf" in text.lower())):
            data["participation"] = int(match.group(1))
    
    def _extract_duration(self, text: str, data: Dict[str, Any]):
        """Extract match duration from text"""
        match = self.pattern_matchers["timer"].search(text)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            data["match_duration"] = f"{minutes:02d}:{seconds:02d}"
    
    async def _complete_data(
        self,
        parsed_data: Dict[str, Any],
        hero: str,
        context: str
    ) -> Dict[str, Any]:
        """Complete missing data using intelligent completion"""
        if self.intelligent_completer:
            completed = self.intelligent_completer.complete_match_data(
                {
                    **parsed_data,
                    "hero": hero,
                    "context": context
                }
            )
            return completed
        
        return parsed_data
    
    def _validate_data(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and score data quality"""
        total_fields = 0
        filled_fields = 0
        missing_fields = []
        
        # Check KDA
        for field in ["kills", "deaths", "assists"]:
            total_fields += 1
            if data["kda"].get(field) is not None:
                filled_fields += 1
            else:
                missing_fields.append(f"kda.{field}")
        
        # Check damage
        for field in ["hero", "turret", "taken"]:
            total_fields += 1
            if data["damage"].get(field) is not None:
                filled_fields += 1
            else:
                missing_fields.append(f"damage.{field}")
        
        # Check other fields
        for field in ["participation", "mvp_score", "match_duration"]:
            total_fields += 1
            if data.get(field) is not None:
                filled_fields += 1
            else:
                missing_fields.append(field)
        
        completeness = filled_fields / total_fields if total_fields > 0 else 0
        
        # Calculate confidence based on completeness and data patterns
        confidence = completeness
        
        # Boost confidence if key fields are present
        if all(data["kda"].get(f) is not None 
               for f in ["kills", "deaths", "assists"]):
            confidence = min(1.0, confidence + 0.2)
        
        return {
            "completeness": completeness,
            "confidence": confidence,
            "missing_fields": missing_fields,
            "filled_fields": filled_fields,
            "total_fields": total_fields
        }
    
    def _generate_match_data(
        self,
        data: Dict[str, Any],
        validation: Dict[str, Any],
        hero: str
    ) -> Dict[str, Any]:
        """Generate final match data structure"""
        return {
            "hero": hero,
            "kda": data["kda"],
            "damage": data["damage"],
            "gold": data["gold"],
            "participation": data.get("participation"),
            "mvp_score": data.get("mvp_score"),
            "match_duration": data.get("match_duration"),
            "data_quality": {
                "completeness": validation["completeness"],
                "confidence": validation["confidence"],
                "missing_fields": validation["missing_fields"]
            }
        } 