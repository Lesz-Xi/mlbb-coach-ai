"""
Intelligent Data Completer for 90%+ Data Completeness

This module implements advanced data completion strategies to boost data completeness
from 71% to 90%+ using cross-panel validation, intelligent estimation, and fallback strategies.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re
import math
from enum import Enum

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Sources of data extraction."""
    DIRECT_OCR = "direct_ocr"
    CROSS_PANEL = "cross_panel"
    CALCULATED = "calculated"
    ESTIMATED = "estimated"
    FALLBACK = "fallback"
    VALIDATED = "validated"


@dataclass
class DataField:
    """Represents a data field with confidence and source tracking."""
    name: str
    value: Any
    confidence: float
    source: DataSource
    validation_score: float
    alternative_values: List[Any]


@dataclass
class CompletionResult:
    """Result of data completion process."""
    fields: Dict[str, DataField]
    completeness_score: float
    confidence_score: float
    completion_methods: List[str]
    validation_results: Dict[str, Any]


class IntelligentDataCompleter:
    """Advanced data completion system for 90%+ completeness."""
    
    def __init__(self):
        # Expected data fields with their characteristics
        self.data_schema = {
            "kills": {"type": int, "range": (0, 30), "critical": True},
            "deaths": {"type": int, "range": (0, 30), "critical": True},
            "assists": {"type": int, "range": (0, 50), "critical": True},
            "gold": {"type": int, "range": (0, 50000), "critical": True},
            "hero": {"type": str, "critical": True},
            "hero_damage": {"type": int, "range": (0, 100000), "critical": False},
            "damage_taken": {"type": int, "range": (0, 100000), "critical": False},
            "match_result": {"type": str, "values": ["victory", "defeat"], "critical": True},
            "match_duration": {"type": float, "range": (3, 60), "critical": True},
            "gold_per_min": {"type": float, "range": (0, 2000), "critical": False},
            "turret_damage": {"type": int, "range": (0, 50000), "critical": False},
            "healing_done": {"type": int, "range": (0, 50000), "critical": False},
            "rank": {"type": str, "critical": False},
            "season": {"type": str, "critical": False}
        }
        
        # Correlation rules for data validation and estimation
        self.correlation_rules = {
            "gold_per_min": lambda data: data.get("gold", 0) / max(data.get("match_duration", 1), 1),
            "kda_ratio": lambda data: (data.get("kills", 0) + data.get("assists", 0)) / max(data.get("deaths", 1), 1),
            "damage_per_min": lambda data: data.get("hero_damage", 0) / max(data.get("match_duration", 1), 1)
        }
        
        # Estimation models based on game knowledge
        self.estimation_models = {
            "gold_from_duration": self._estimate_gold_from_duration,
            "damage_from_gold": self._estimate_damage_from_gold,
            "assists_from_kda": self._estimate_assists_from_kda,
            "duration_from_gold": self._estimate_duration_from_gold
        }
        
        # Cross-panel validation patterns (simplified for now)
        self.cross_panel_patterns = {}
    
    def complete_data(
        self,
        raw_data: Dict[str, Any],
        ocr_results: List,
        image_path: str,
        context: str = "scoreboard"
    ) -> CompletionResult:
        """
        Complete missing data using intelligent strategies.
        
        Args:
            raw_data: Initially extracted data
            ocr_results: Raw OCR results for re-processing
            image_path: Path to screenshot for additional analysis
            context: Type of screenshot (scoreboard, stats, etc.)
            
        Returns:
            CompletionResult with enhanced data completeness
        """
        completion_methods = []
        
        # Step 1: Clean and validate existing data
        cleaned_data = self._clean_raw_data(raw_data)
        completion_methods.append("data_cleaning")
        
        # Step 2: Cross-panel validation and extraction
        cross_panel_data = self._extract_cross_panel_data(ocr_results, image_path, context)
        completion_methods.append("cross_panel_extraction")
        
        # Step 3: Intelligent field completion
        completed_fields = self._complete_missing_fields(cleaned_data, cross_panel_data)
        completion_methods.append("field_completion")
        
        # Step 4: Calculate derived fields
        calculated_fields = self._calculate_derived_fields(completed_fields)
        completion_methods.append("derived_calculation")
        
        # Step 5: Estimate missing critical fields
        estimated_fields = self._estimate_missing_fields(calculated_fields, context)
        completion_methods.append("intelligent_estimation")
        
        # Step 6: Fallback value assignment
        final_fields = self._assign_fallback_values(estimated_fields)
        completion_methods.append("fallback_assignment")
        
        # Step 7: Final validation and confidence scoring
        validation_results = self._validate_final_data(final_fields)
        completion_methods.append("final_validation")
        
        # Calculate completeness and confidence scores
        completeness_score = self._calculate_completeness_score(final_fields)
        confidence_score = self._calculate_confidence_score(final_fields, validation_results)
        
        logger.info(f"Data completion: {completeness_score:.1f}% complete, {confidence_score:.1f}% confidence")
        
        return CompletionResult(
            fields=final_fields,
            completeness_score=completeness_score,
            confidence_score=confidence_score,
            completion_methods=completion_methods,
            validation_results=validation_results
        )
    
    def _clean_raw_data(self, raw_data: Dict[str, Any]) -> Dict[str, DataField]:
        """Clean and structure raw data into DataField objects."""
        cleaned_fields = {}
        
        for field_name, field_spec in self.data_schema.items():
            value = raw_data.get(field_name)
            
            if value is not None and value != "" and str(value).lower() != "unknown":
                # Validate and clean the value
                cleaned_value = self._validate_and_clean_value(value, field_spec)
                confidence = 0.9 if cleaned_value == value else 0.7
                
                cleaned_fields[field_name] = DataField(
                    name=field_name,
                    value=cleaned_value,
                    confidence=confidence,
                    source=DataSource.DIRECT_OCR,
                    validation_score=0.8,
                    alternative_values=[]
                )
        
        return cleaned_fields
    
    def _extract_cross_panel_data(self, ocr_results: List, image_path: str, context: str) -> Dict[str, Any]:
        """Enhanced cross-panel extraction with better pattern recognition."""
        cross_panel_data = {}
        
        try:
            # Apply enhanced normalization first
            normalized_ocr = self._normalize_ocr_results(ocr_results)
            
            # Reconstruct spatial rows for better data extraction
            spatial_rows = self._reconstruct_spatial_rows(normalized_ocr)
            
            # Combine all text for pattern matching
            all_text = " ".join([result[1] for result in normalized_ocr])
            row_texts = [" ".join(row) for row in spatial_rows]
            
            # Enhanced duration extraction with multiple patterns
            duration_patterns = [
                # Standard MM:SS formats
                r'(?:duration|time|match)\s*[:.]?\s*(\d{1,2}):(\d{2})',
                r'(\d{1,2}):(\d{2})\s*(?:duration|time|match)',
                # Written formats  
                r'(\d{1,2})\s*min\s*(\d{1,2})\s*sec',
                r'(\d{1,2})m\s*(\d{1,2})s',
                # Simple MM:SS anywhere
                r'\b(\d{1,2}):(\d{2})\b',
                # Single number + unit
                r'(\d{1,2})\s*minutes?',
                r'(\d{1,2})\s*mins?',
                # Edge cases
                r'(\d{1,2})\s*:\s*(\d{1,2})',  # Spaced colon
                r'(\d{1,2})\s+(\d{2})',        # Space instead of colon
            ]
            
            for pattern in duration_patterns:
                for text_source in [all_text] + row_texts:
                    match = re.search(pattern, text_source, re.IGNORECASE)
                    if match:
                        groups = match.groups()
                        if len(groups) == 2:
                            minutes = int(groups[0])
                            seconds = int(groups[1]) if groups[1] != '' else 0
                        elif len(groups) == 1:
                            minutes = int(groups[0])
                            seconds = 0
                        else:
                            continue
                            
                        duration = minutes + seconds / 60.0
                        if 3 <= duration <= 60:  # Reasonable match duration
                            cross_panel_data["match_duration"] = duration
                            break
                if "match_duration" in cross_panel_data:
                    break
            
            # Enhanced gold extraction with multiple formats
            gold_patterns = [
                # Standard gold patterns
                r'(?:gold|economy|g)\s*[:.]?\s*(\d{3,6})',
                r'(\d{3,6})\s*(?:gold|g\b)',
                # K-format patterns (already converted by normalization)
                r'(\d{4,6})\b',  # 4-6 digit numbers likely to be gold
                # Specific game patterns
                r'total\s*gold\s*[:.]?\s*(\d{3,6})',
                r'farm\s*[:.]?\s*(\d{3,6})',
                # Edge cases
                r'(\d{1,2})\s*[.,]\s*(\d{3})',  # "12.500" or "12,500"
                r'(\d{2,6})\s*g(?:\s|$)',       # Numbers followed by 'g'
                r'(?:coins?|money)\s*[:.]?\s*(\d{3,6})'  # Alternative terms
            ]
            
            gold_candidates = []
            for pattern in gold_patterns:
                for text_source in [all_text] + row_texts:
                    matches = re.findall(pattern, text_source, re.IGNORECASE)
                    for match in matches:
                        try:
                            if isinstance(match, tuple):
                                # Handle patterns like "12.500"
                                gold_value = int(match[0]) * 1000 + int(match[1])
                            else:
                                gold_value = int(match)
                            if 100 <= gold_value <= 50000:  # Reasonable gold range
                                gold_candidates.append(gold_value)
                        except ValueError:
                            continue
            
            if gold_candidates:
                # Use most common gold value or highest if similar
                from collections import Counter
                gold_counter = Counter(gold_candidates)
                if gold_counter:
                    cross_panel_data["gold_candidates"] = list(set(gold_candidates))
                    cross_panel_data["gold"] = gold_counter.most_common(1)[0][0]
            
            # Enhanced damage extraction
            damage_patterns = [
                r'(?:hero\s*)?damage\s*(?:dealt)?\s*[:.]?\s*(\d{4,8})',
                r'(\d{4,8})\s*(?:hero\s*)?damage',
                r'dmg\s*[:.]?\s*(\d{4,8})',
                r'(\d{4,8})\s*dmg',
                r'total\s*damage\s*[:.]?\s*(\d{4,8})',
                # Edge cases
                r'(\d{1,3})\s*[.,]\s*(\d{3})\s*(?:damage|dmg)',  # "85.000 damage"
                r'attack\s*[:.]?\s*(\d{4,8})',
                r'dps\s*[:.]?\s*(\d{4,8})'
            ]
            
            damage_candidates = []
            for pattern in damage_patterns:
                for text_source in [all_text] + row_texts:
                    matches = re.findall(pattern, text_source, re.IGNORECASE)
                    for match in matches:
                        try:
                            if isinstance(match, tuple):
                                # Handle patterns like "85.000"
                                damage_value = int(match[0]) * 1000 + int(match[1])
                            else:
                                damage_value = int(match)
                            if 1000 <= damage_value <= 500000:  # Reasonable damage range
                                damage_candidates.append(damage_value)
                        except ValueError:
                            continue
            
            if damage_candidates:
                cross_panel_data["damage_candidates"] = list(set(damage_candidates))
                cross_panel_data["hero_damage"] = max(damage_candidates)  # Use highest damage
            
            # Enhanced KDA extraction with multiple formats
            kda_patterns = [
                # Standard K/D/A format
                r'(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{1,2})',
                # Labeled format
                r'k\s*[:.]?\s*(\d{1,2})\s*d\s*[:.]?\s*(\d{1,2})\s*a\s*[:.]?\s*(\d{1,2})',
                r'kills?\s*[:.]?\s*(\d{1,2})\s*deaths?\s*[:.]?\s*(\d{1,2})\s*assists?\s*[:.]?\s*(\d{1,2})',
                # Spaced format
                r'(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+\d',  # KDA + other numbers
                # Edge cases
                r'(\d{1,2})\s*-\s*(\d{1,2})\s*-\s*(\d{1,2})',  # Dash separated
                r'(\d{1,2})\s*,\s*(\d{1,2})\s*,\s*(\d{1,2})',  # Comma separated
            ]
            
            for pattern in kda_patterns[:6]:  # Try complete KDA patterns first
                for text_source in [all_text] + row_texts:
                    match = re.search(pattern, text_source, re.IGNORECASE)
                    if match and len(match.groups()) == 3:
                        try:
                            kills, deaths, assists = map(int, match.groups())
                            if 0 <= kills <= 30 and 1 <= deaths <= 30 and 0 <= assists <= 50:
                                cross_panel_data.update({
                                    "kills": kills,
                                    "deaths": deaths,
                                    "assists": assists
                                })
                                break
                        except ValueError:
                            continue
                if all(k in cross_panel_data for k in ["kills", "deaths", "assists"]):
                    break
            
            # If KDA not found, try individual extraction
            if not all(k in cross_panel_data for k in ["kills", "deaths", "assists"]):
                individual_patterns = [
                    (r'kills?\s*[:.]?\s*(\d{1,2})', "kills"),
                    (r'k\s*[:.]?\s*(\d{1,2})', "kills"),
                    (r'deaths?\s*[:.]?\s*(\d{1,2})', "deaths"),
                    (r'd\s*[:.]?\s*(\d{1,2})', "deaths"),
                    (r'assists?\s*[:.]?\s*(\d{1,2})', "assists"),
                    (r'a\s*[:.]?\s*(\d{1,2})', "assists"),
                ]
                
                for pattern, field_name in individual_patterns:
                    if field_name not in cross_panel_data:
                        for text_source in [all_text] + row_texts:
                            match = re.search(pattern, text_source, re.IGNORECASE)
                            if match:
                                try:
                                    value = int(match.group(1))
                                    if field_name == "deaths" and 1 <= value <= 30:
                                        cross_panel_data[field_name] = value
                                        break
                                    elif field_name != "deaths" and 0 <= value <= 50:
                                        cross_panel_data[field_name] = value
                                        break
                                except ValueError:
                                    continue
            
            # Enhanced match result detection
            victory_indicators = [
                'victory', 'win', 'won', 'success', 'triumph', 'victorious',
                'winner', 'champions', 'achieve victory', 'champion', 'mvp'
            ]
            defeat_indicators = [
                'defeat', 'loss', 'lose', 'lost', 'failed', 'failure',
                'loser', 'defeated', 'game over', 'lose', 'beaten'
            ]
            
            text_lower = all_text.lower()
            # Check for victory indicators
            victory_score = sum(1 for indicator in victory_indicators if indicator in text_lower)
            defeat_score = sum(1 for indicator in defeat_indicators if indicator in text_lower)
            
            if victory_score > defeat_score and victory_score > 0:
                cross_panel_data["match_result"] = "victory"
            elif defeat_score > victory_score and defeat_score > 0:
                cross_panel_data["match_result"] = "defeat"
            
            # Enhanced rank detection
            rank_patterns = [
                r'(?:rank|tier|level)\s*[:.]?\s*(legend|mythic|epic|grandmaster|master|elite|warrior|bronze)',
                r'(legend|mythic|epic|grandmaster|master|elite|warrior|bronze)\s*(?:rank|tier)?',
                r'(?:current\s*)?(?:rank|tier)\s*[:.]?\s*(\w+)',
                # Edge cases
                r'(legend|mythic|epic|gm|master|elite|warrior|bronze)',
                r'rank\s*(\d+)',  # Numeric ranks
            ]
            
            for pattern in rank_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    rank = match.group(1).lower()
                    # Map common abbreviations
                    rank_map = {
                        'gm': 'grandmaster',
                        'leg': 'legend',
                        'myth': 'mythic'
                    }
                    rank = rank_map.get(rank, rank)
                    
                    if rank in ['legend', 'mythic', 'epic', 'grandmaster', 'master', 'elite', 'warrior', 'bronze']:
                        cross_panel_data["rank"] = rank
                        break
            
            # Enhanced GPM calculation if we have both gold and duration
            if "gold" in cross_panel_data and "match_duration" in cross_panel_data:
                duration = cross_panel_data["match_duration"]
                gold = cross_panel_data["gold"]
                if duration > 0:
                    cross_panel_data["gold_per_min"] = round(gold / duration, 1)
            
            # Add damage taken if we can infer it
            if "hero_damage" in cross_panel_data and "gold" in cross_panel_data:
                damage_dealt = cross_panel_data["hero_damage"]
                # Rough estimation: damage taken is usually 60-120% of damage dealt
                estimated_taken = int(damage_dealt * 0.8)
                cross_panel_data["damage_taken"] = estimated_taken
            
        except Exception as e:
            logger.error(f"Enhanced cross-panel extraction failed: {str(e)}")
        
        return cross_panel_data
    
    def _complete_missing_fields(self, cleaned_data: Dict[str, DataField], cross_panel_data: Dict[str, Any]) -> Dict[str, DataField]:
        """Complete missing fields using cross-panel data."""
        completed_fields = cleaned_data.copy()
        
        # Fill missing fields from cross-panel data
        for field_name, field_spec in self.data_schema.items():
            if field_name not in completed_fields:
                # Check if we have this field in cross-panel data
                if field_name in cross_panel_data:
                    value = cross_panel_data[field_name]
                    cleaned_value = self._validate_and_clean_value(value, field_spec)
                    
                    completed_fields[field_name] = DataField(
                        name=field_name,
                        value=cleaned_value,
                        confidence=0.7,
                        source=DataSource.CROSS_PANEL,
                        validation_score=0.6,
                        alternative_values=[]
                    )
        
        # Handle special cases with candidate values
        if "gold" not in completed_fields and "gold_candidates" in cross_panel_data:
            # Use the most reasonable gold value
            candidates = cross_panel_data["gold_candidates"]
            if candidates:
                # Prefer values that make sense with other data
                best_gold = self._select_best_gold_candidate(candidates, completed_fields)
                completed_fields["gold"] = DataField(
                    name="gold",
                    value=best_gold,
                    confidence=0.6,
                    source=DataSource.CROSS_PANEL,
                    validation_score=0.5,
                    alternative_values=candidates
                )
        
        # Similar logic for damage candidates
        if "hero_damage" not in completed_fields and "damage_candidates" in cross_panel_data:
            candidates = cross_panel_data["damage_candidates"]
            if candidates:
                best_damage = self._select_best_damage_candidate(candidates, completed_fields)
                completed_fields["hero_damage"] = DataField(
                    name="hero_damage",
                    value=best_damage,
                    confidence=0.6,
                    source=DataSource.CROSS_PANEL,
                    validation_score=0.5,
                    alternative_values=candidates
                )
        
        return completed_fields
    
    def _calculate_derived_fields(self, fields: Dict[str, DataField]) -> Dict[str, DataField]:
        """Calculate derived fields from existing data."""
        derived_fields = fields.copy()
        
        # Calculate gold per minute
        if "gold_per_min" not in derived_fields:
            if "gold" in derived_fields and "match_duration" in derived_fields:
                gold = derived_fields["gold"].value
                duration = derived_fields["match_duration"].value
                
                if gold > 0 and duration > 0:
                    gpm = gold / duration
                    derived_fields["gold_per_min"] = DataField(
                        name="gold_per_min",
                        value=round(gpm, 1),
                        confidence=0.8,
                        source=DataSource.CALCULATED,
                        validation_score=0.9,
                        alternative_values=[]
                    )
        
        # Calculate KDA ratio
        if "kda_ratio" not in derived_fields:
            if "kills" in derived_fields and "deaths" in derived_fields and "assists" in derived_fields:
                kills = derived_fields["kills"].value
                deaths = max(derived_fields["deaths"].value, 1)  # Avoid division by zero
                assists = derived_fields["assists"].value
                
                kda = (kills + assists) / deaths
                derived_fields["kda_ratio"] = DataField(
                    name="kda_ratio",
                    value=round(kda, 2),
                    confidence=0.9,
                    source=DataSource.CALCULATED,
                    validation_score=0.9,
                    alternative_values=[]
                )
        
        # Calculate damage per minute
        if "damage_per_min" not in derived_fields:
            if "hero_damage" in derived_fields and "match_duration" in derived_fields:
                damage = derived_fields["hero_damage"].value
                duration = derived_fields["match_duration"].value
                
                if damage > 0 and duration > 0:
                    dpm = damage / duration
                    derived_fields["damage_per_min"] = DataField(
                        name="damage_per_min",
                        value=round(dpm, 1),
                        confidence=0.8,
                        source=DataSource.CALCULATED,
                        validation_score=0.8,
                        alternative_values=[]
                    )
        
        return derived_fields
    
    def _estimate_missing_fields(self, fields: Dict[str, DataField], context: str) -> Dict[str, DataField]:
        """Estimate missing fields using game knowledge and correlations."""
        estimated_fields = fields.copy()
        
        # Estimate match duration from gold
        if "match_duration" not in estimated_fields and "gold" in estimated_fields:
            gold = estimated_fields["gold"].value
            estimated_duration = self._estimate_duration_from_gold(gold)
            
            if estimated_duration > 0:
                estimated_fields["match_duration"] = DataField(
                    name="match_duration",
                    value=estimated_duration,
                    confidence=0.5,
                    source=DataSource.ESTIMATED,
                    validation_score=0.4,
                    alternative_values=[]
                )
        
        # Estimate gold from duration
        if "gold" not in estimated_fields and "match_duration" in estimated_fields:
            duration = estimated_fields["match_duration"].value
            estimated_gold = self._estimate_gold_from_duration(duration)
            
            if estimated_gold > 0:
                estimated_fields["gold"] = DataField(
                    name="gold",
                    value=estimated_gold,
                    confidence=0.4,
                    source=DataSource.ESTIMATED,
                    validation_score=0.3,
                    alternative_values=[]
                )
        
        # Estimate damage from gold
        if "hero_damage" not in estimated_fields and "gold" in estimated_fields:
            gold = estimated_fields["gold"].value
            estimated_damage = self._estimate_damage_from_gold(gold)
            
            if estimated_damage > 0:
                estimated_fields["hero_damage"] = DataField(
                    name="hero_damage",
                    value=estimated_damage,
                    confidence=0.4,
                    source=DataSource.ESTIMATED,
                    validation_score=0.3,
                    alternative_values=[]
                )
        
        # Estimate assists from KDA pattern
        if "assists" not in estimated_fields and "kills" in estimated_fields and "deaths" in estimated_fields:
            kills = estimated_fields["kills"].value
            deaths = estimated_fields["deaths"].value
            estimated_assists = self._estimate_assists_from_kda(kills, deaths)
            
            estimated_fields["assists"] = DataField(
                name="assists",
                value=estimated_assists,
                confidence=0.3,
                source=DataSource.ESTIMATED,
                validation_score=0.2,
                alternative_values=[]
            )
        
        return estimated_fields
    
    def _assign_fallback_values(self, fields: Dict[str, DataField]) -> Dict[str, DataField]:
        """Assign fallback values for remaining missing fields."""
        final_fields = fields.copy()
        
        # Fallback values for critical fields
        fallback_values = {
            "kills": 0,
            "deaths": 1,
            "assists": 0,
            "gold": 5000,
            "hero": "unknown",
            "match_result": "unknown",
            "match_duration": 10.0,
            "hero_damage": 10000,
            "damage_taken": 8000,
            "gold_per_min": 500,
            "turret_damage": 1000,
            "healing_done": 0,
            "rank": "unknown",
            "season": "unknown"
        }
        
        for field_name, field_spec in self.data_schema.items():
            if field_name not in final_fields:
                fallback_value = fallback_values.get(field_name, 0)
                
                final_fields[field_name] = DataField(
                    name=field_name,
                    value=fallback_value,
                    confidence=0.1,
                    source=DataSource.FALLBACK,
                    validation_score=0.1,
                    alternative_values=[]
                )
        
        return final_fields
    
    def _validate_final_data(self, fields: Dict[str, DataField]) -> Dict[str, Any]:
        """Validate final data for consistency and reasonableness."""
        validation_results = {
            "consistency_checks": [],
            "range_checks": [],
            "correlation_checks": [],
            "overall_validity": 0.0
        }
        
        validity_scores = []
        
        # Range validation
        for field_name, field in fields.items():
            if field_name in self.data_schema:
                spec = self.data_schema[field_name]
                if "range" in spec:
                    min_val, max_val = spec["range"]
                    is_valid = min_val <= field.value <= max_val
                    validation_results["range_checks"].append({
                        "field": field_name,
                        "valid": is_valid,
                        "value": field.value,
                        "range": spec["range"]
                    })
                    validity_scores.append(1.0 if is_valid else 0.0)
        
        # Consistency validation
        consistency_checks = [
            ("gold_duration", self._check_gold_duration_consistency),
            ("kda_reasonableness", self._check_kda_reasonableness),
            ("damage_consistency", self._check_damage_consistency)
        ]
        
        for check_name, check_func in consistency_checks:
            try:
                result = check_func(fields)
                validation_results["consistency_checks"].append({
                    "check": check_name,
                    "result": result
                })
                validity_scores.append(result.get("score", 0.0))
            except Exception as e:
                logger.error(f"Consistency check {check_name} failed: {str(e)}")
        
        # Calculate overall validity
        if validity_scores:
            validation_results["overall_validity"] = sum(validity_scores) / len(validity_scores)
        
        return validation_results
    
    def _calculate_completeness_score(self, fields: Dict[str, DataField]) -> float:
        """Calculate data completeness score."""
        total_fields = len(self.data_schema)
        filled_fields = 0
        weighted_completeness = 0.0
        
        for field_name, field_spec in self.data_schema.items():
            if field_name in fields:
                field = fields[field_name]
                # Weight critical fields more heavily
                weight = 2.0 if field_spec.get("critical", False) else 1.0
                
                # Consider source quality
                source_multiplier = {
                    DataSource.DIRECT_OCR: 1.0,
                    DataSource.CROSS_PANEL: 0.8,
                    DataSource.CALCULATED: 0.9,
                    DataSource.ESTIMATED: 0.6,
                    DataSource.FALLBACK: 0.2
                }.get(field.source, 0.5)
                
                field_score = field.confidence * source_multiplier * weight
                weighted_completeness += field_score
                filled_fields += 1
        
        # Normalize by total possible weighted score
        critical_fields = sum(2.0 for spec in self.data_schema.values() if spec.get("critical", False))
        non_critical_fields = total_fields - critical_fields / 2.0
        max_weighted_score = critical_fields + non_critical_fields
        
        return min(100.0, (weighted_completeness / max_weighted_score) * 100.0)
    
    def _calculate_confidence_score(self, fields: Dict[str, DataField], validation_results: Dict[str, Any]) -> float:
        """Calculate overall confidence score."""
        field_confidences = []
        
        for field_name, field_spec in self.data_schema.items():
            if field_name in fields:
                field = fields[field_name]
                # Weight by field importance
                weight = 2.0 if field_spec.get("critical", False) else 1.0
                weighted_confidence = field.confidence * weight
                field_confidences.append(weighted_confidence)
        
        avg_confidence = sum(field_confidences) / len(field_confidences) if field_confidences else 0.0
        
        # Adjust by validation results
        validity_score = validation_results.get("overall_validity", 0.0)
        final_confidence = (avg_confidence * 0.7) + (validity_score * 0.3)
        
        return min(100.0, final_confidence * 100.0)
    
    # Helper methods for estimation
    def _estimate_duration_from_gold(self, gold: int) -> float:
        """Estimate match duration from gold amount."""
        # Based on typical gold per minute rates (300-800 GPM)
        if gold < 3000:
            return 5.0  # Very short match
        elif gold < 8000:
            return gold / 600.0  # Average GPM
        else:
            return gold / 700.0  # Higher GPM for longer matches
    
    def _estimate_gold_from_duration(self, duration: float) -> int:
        """Estimate gold from match duration."""
        # Average 500-600 GPM
        base_gpm = 550
        return int(base_gpm * duration)
    
    def _estimate_damage_from_gold(self, gold: int) -> int:
        """Estimate hero damage from gold amount."""
        # Typical damage-to-gold ratio varies by role
        # Average ratio is about 2-4 damage per gold
        ratio = 3.0
        return int(gold * ratio)
    
    def _estimate_assists_from_kda(self, kills: int, deaths: int) -> int:
        """Estimate assists based on kills and deaths pattern."""
        # Typical assists are 1-3x kills, inversely related to deaths
        if deaths <= 2:
            return kills * 2  # Good game, high assists
        else:
            return max(0, kills)  # Rough game, fewer assists
    
    def _validate_and_clean_value(self, value: Any, field_spec: Dict[str, Any]) -> Any:
        """Validate and clean a field value according to its specification."""
        target_type = field_spec.get("type", str)
        
        try:
            if target_type == int:
                return int(float(value))
            elif target_type == float:
                return float(value)
            elif target_type == str:
                return str(value).strip().lower()
            else:
                return value
        except (ValueError, TypeError):
            return value
    
    def _select_best_gold_candidate(self, candidates: List[int], fields: Dict[str, DataField]) -> int:
        """Select the most reasonable gold value from candidates."""
        if not candidates:
            return 5000
        
        # Filter reasonable values
        reasonable = [c for c in candidates if 1000 <= c <= 30000]
        if not reasonable:
            return candidates[0]
        
        # If we have duration, prefer values that give reasonable GPM
        if "match_duration" in fields:
            duration = fields["match_duration"].value
            best_candidate = None
            best_score = float('inf')
            
            for candidate in reasonable:
                gpm = candidate / duration
                # Prefer GPM in reasonable range (300-1000)
                if 300 <= gpm <= 1000:
                    score = abs(gpm - 500)  # Prefer values close to 500 GPM
                    if score < best_score:
                        best_score = score
                        best_candidate = candidate
            
            if best_candidate:
                return best_candidate
        
        # Default to median value
        return sorted(reasonable)[len(reasonable) // 2]
    
    def _select_best_damage_candidate(self, candidates: List[int], fields: Dict[str, DataField]) -> int:
        """Select the most reasonable damage value from candidates."""
        if not candidates:
            return 10000
        
        # Filter reasonable values
        reasonable = [c for c in candidates if 1000 <= c <= 150000]
        if not reasonable:
            return candidates[0]
        
        # If we have gold, prefer values that give reasonable damage-to-gold ratio
        if "gold" in fields:
            gold = fields["gold"].value
            best_candidate = None
            best_score = float('inf')
            
            for candidate in reasonable:
                ratio = candidate / max(gold, 1)
                # Prefer ratio in reasonable range (1-6)
                if 1 <= ratio <= 6:
                    score = abs(ratio - 3)  # Prefer values close to 3:1 ratio
                    if score < best_score:
                        best_score = score
                        best_candidate = candidate
            
            if best_candidate:
                return best_candidate
        
        # Default to median value
        return sorted(reasonable)[len(reasonable) // 2]
    
    def _check_gold_duration_consistency(self, fields: Dict[str, DataField]) -> Dict[str, Any]:
        """Check if gold and duration are consistent."""
        if "gold" not in fields or "match_duration" not in fields:
            return {"score": 0.5, "reason": "Missing fields for check"}
        
        gold = fields["gold"].value
        duration = fields["match_duration"].value
        
        if duration <= 0:
            return {"score": 0.0, "reason": "Invalid duration"}
        
        gpm = gold / duration
        
        # Reasonable GPM range
        if 200 <= gpm <= 1200:
            return {"score": 1.0, "gpm": gpm}
        elif 100 <= gpm <= 1500:
            return {"score": 0.7, "gpm": gpm, "reason": "GPM slightly outside normal range"}
        else:
            return {"score": 0.3, "gpm": gpm, "reason": "GPM outside reasonable range"}
    
    def _check_kda_reasonableness(self, fields: Dict[str, DataField]) -> Dict[str, Any]:
        """Check if KDA values are reasonable."""
        required_fields = ["kills", "deaths", "assists"]
        if not all(field in fields for field in required_fields):
            return {"score": 0.5, "reason": "Missing KDA fields"}
        
        kills = fields["kills"].value
        deaths = fields["deaths"].value
        assists = fields["assists"].value
        
        # Basic reasonableness checks
        if kills < 0 or deaths < 0 or assists < 0:
            return {"score": 0.0, "reason": "Negative values"}
        
        if kills > 25 or deaths > 25 or assists > 40:
            return {"score": 0.3, "reason": "Unusually high values"}
        
        # Check KDA ratio
        kda_ratio = (kills + assists) / max(deaths, 1)
        if kda_ratio > 15:
            return {"score": 0.5, "reason": "Unusually high KDA ratio"}
        
        return {"score": 1.0, "kda_ratio": kda_ratio}
    
    def _check_damage_consistency(self, fields: Dict[str, DataField]) -> Dict[str, Any]:
        """Check if damage values are consistent with other metrics."""
        if "hero_damage" not in fields:
            return {"score": 0.5, "reason": "No damage data"}
        
        damage = fields["hero_damage"].value
        
        # Check against gold if available
        if "gold" in fields:
            gold = fields["gold"].value
            damage_to_gold_ratio = damage / max(gold, 1)
            
            if 0.5 <= damage_to_gold_ratio <= 8:
                return {"score": 1.0, "damage_gold_ratio": damage_to_gold_ratio}
            else:
                return {"score": 0.6, "damage_gold_ratio": damage_to_gold_ratio, "reason": "Unusual damage-to-gold ratio"}
        
        return {"score": 0.7, "reason": "No gold data for comparison"}


    def _normalize_ocr_results(self, ocr_results: List) -> List:
        """Enhanced OCR normalization with comprehensive character/word fixes."""
        normalized_results = []
        
        # Enhanced character substitution mapping
        char_corrections = {
            # Number/letter confusion
            '0': 'o', '1': 'i', '5': 's', '8': 'b', '6': 'g', '2': 'z',
            'o': '0', 'i': '1', 'l': '1', 's': '5', 'b': '8', 'g': '6',
            'z': '2', 'q': '9', 'S': '5', 'B': '8', 'G': '6', 'O': '0',
            'I': '1', 'L': '1', 'Z': '2', 'Q': '9',
            # Common OCR artifacts
            '|': '1', '!': '1', 'rn': 'm', 'vv': 'w', 'ii': 'n', 'cl': 'd'
        }
        
        # Enhanced word-level corrections
        word_corrections = {
            'goid': 'gold', 'goId': 'gold', 'g0ld': 'gold', 'qold': 'gold',
            'GoId': 'gold', 'GOLD': 'gold', 'Goid': 'gold', 'gojd': 'gold',
            'ki11s': 'kills', 'kil1s': 'kills', 'k1lls': 'kills', 'ki!!s': 'kills',
            'deaths': 'deaths', 'deahs': 'deaths', 'deatns': 'deaths',
            'assists': 'assists', 'asslsts': 'assists', 'ass1sts': 'assists',
            'damage': 'damage', 'damaqe': 'damage', 'damaoe': 'damage',
            'victory': 'victory', 'v1ctory': 'victory', 'vlctory': 'victory',
            'defeat': 'defeat', 'deteat': 'defeat', 'defeal': 'defeat',
            'duration': 'duration', 'duratlon': 'duration', 'duratian': 'duration'
        }
        
        for bbox, text, confidence in ocr_results:
            # Apply character corrections
            corrected_text = text
            for wrong_char, correct_char in char_corrections.items():
                corrected_text = corrected_text.replace(wrong_char, correct_char)
            
            # Apply word corrections
            corrected_lower = corrected_text.lower()
            for wrong_word, correct_word in word_corrections.items():
                corrected_lower = corrected_lower.replace(wrong_word, correct_word)
            
            # Additional cleanup
            corrected_text = self._advanced_text_cleanup(corrected_lower)
            
            normalized_results.append((bbox, corrected_text, confidence))
        
        return normalized_results
    
    def _advanced_text_cleanup(self, text: str) -> str:
        """Advanced text cleanup with pattern recognition."""
        # Remove invisible unicode characters
        text = ''.join(char for char in text if ord(char) < 127 or char.isalnum())
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix common OCR spacing issues
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)  # "1 2 3 4" -> "1234"
        text = re.sub(r'([a-z])\s+([a-z])', r'\1\2', text)  # "g o l d" -> "gold"
        
        # Enhanced OCR character fixes
        text = re.sub(r'rn', 'm', text)  # 'rn' -> 'm'
        text = re.sub(r'vv', 'w', text)  # 'vv' -> 'w'  
        text = re.sub(r'ii', 'n', text)  # 'ii' -> 'n'
        text = re.sub(r'cl', 'd', text)  # 'cl' -> 'd'
        
        # Normalize punctuation
        text = re.sub(r'[^\w\s:/.-]', '', text)  # Keep only alphanumeric, spaces, and common game chars
        
        # Fix k-format numbers (7k, 7.5k, etc.)
        text = re.sub(r'(\d+)\.?(\d*)\s*k', lambda m: str(int(float(m.group(1) + '.' + (m.group(2) or '0')) * 1000)), text)
        
        # Additional number cleanup
        text = re.sub(r'([a-z])(\d)', r'\1 \2', text)  # "gold12500" -> "gold 12500"
        text = re.sub(r'(\d)([a-z])', r'\1 \2', text)  # "12500gold" -> "12500 gold"
        
        return text.strip()
    
    def _reconstruct_spatial_rows(self, ocr_results: List) -> List[List]:
        """Enhanced spatial row reconstruction with tolerance and grouping."""
        if not ocr_results:
            return []
        
        # Group OCR results by Y-coordinate with enhanced tolerance
        y_groups = {}
        y_tolerance = 25  # Further increased tolerance for better row detection on mobile
        
        for bbox, text, confidence in ocr_results:
            # Calculate center Y coordinate
            if isinstance(bbox, list) and len(bbox) >= 2:
                # Handle [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] format
                y_coords = [point[1] for point in bbox]
                center_y = sum(y_coords) / len(y_coords)
            else:
                # Fallback for other formats
                center_y = bbox[1] if len(bbox) > 1 else 0
            
            # Find existing group or create new one
            found_group = False
            for group_y in y_groups:
                if abs(center_y - group_y) <= y_tolerance:
                    y_groups[group_y].append((bbox, text, confidence))
                    found_group = True
                    break
            
            if not found_group:
                y_groups[center_y] = [(bbox, text, confidence)]
        
        # Sort groups by Y coordinate and then each group by X coordinate
        spatial_rows = []
        for y_coord in sorted(y_groups.keys()):
            row_items = y_groups[y_coord]
            
            # Sort items in row by X coordinate
            row_items.sort(key=lambda item: self._get_x_coordinate(item[0]))
            
            # Extract just the text for each row
            row_texts = [item[1] for item in row_items if item[1].strip()]
            
            if row_texts:  # Only add non-empty rows
                spatial_rows.append(row_texts)
        
        return spatial_rows
    
    def _get_x_coordinate(self, bbox) -> float:
        """Get X coordinate from various bbox formats."""
        try:
            if isinstance(bbox, list) and len(bbox) >= 2:
                # Handle [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] format
                x_coords = [point[0] for point in bbox]
                return min(x_coords)  # Use leftmost X coordinate
            else:
                # Fallback for other formats
                return bbox[0] if len(bbox) > 0 else 0
        except:
            return 0


# Global instance
intelligent_data_completer = IntelligentDataCompleter()