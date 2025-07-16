"""
Real-time Confidence Adjustment System

Fixes the core issue where confidence is calculated before data validation.
This system adjusts confidence based on actual extraction success.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ConfidenceAdjustment:
    """Result of confidence adjustment with reasoning."""
    original_confidence: float
    adjusted_confidence: float
    adjustment_factor: float
    reason: str
    critical_issues: list
    extraction_success: bool

class RealtimeConfidenceAdjuster:
    """Adjusts confidence based on actual extraction results."""
    
    def __init__(self):
        # Critical fields that must be present for high confidence
        self.critical_fields = {
            'hero': 0.4,        # 40% of confidence
            'kills': 0.15,      # 15% of confidence  
            'deaths': 0.15,     # 15% of confidence
            'assists': 0.15,    # 15% of confidence
            'gold': 0.15,       # 15% of confidence
        }
        
        # Optional fields that boost confidence
        self.optional_fields = {
            'hero_damage': 0.1,
            'damage_taken': 0.1,
            'match_duration': 0.05,
            'match_result': 0.05
        }
    
    def adjust_confidence_realtime(
        self,
        original_confidence: float,
        extracted_data: Dict[str, Any],
        ign_match_result: Optional[Any] = None,
        warnings: list = None
    ) -> ConfidenceAdjustment:
        """
        Adjust confidence based on actual extraction success.
        
        Args:
            original_confidence: Confidence from quality/OCR analysis
            extracted_data: Actually extracted match data
            ign_match_result: IGN matching result
            warnings: List of warnings from extraction
            
        Returns:
            ConfidenceAdjustment with new confidence and reasoning
        """
        warnings = warnings or []
        critical_issues = []
        
        # Start with original confidence
        adjusted_confidence = original_confidence
        
        # 1. IGN Matching Penalty/Boost
        if ign_match_result and hasattr(ign_match_result, 'found'):
            if not ign_match_result.found:
                adjusted_confidence *= 0.1  # Massive penalty for IGN failure
                critical_issues.append("IGN not found - data extraction impossible")
            elif ign_match_result.confidence < 0.8:
                adjusted_confidence *= (0.5 + ign_match_result.confidence * 0.5)
                critical_issues.append(f"Weak IGN match (confidence: {ign_match_result.confidence:.2f})")
        
        # 2. Critical Field Analysis
        critical_score = 0.0
        missing_critical = []
        
        for field, weight in self.critical_fields.items():
            value = extracted_data.get(field)
            
            if field == 'hero':
                if value and value.lower() not in ['unknown', 'n/a', '']:
                    critical_score += weight
                else:
                    missing_critical.append(field)
            else:
                if value is not None and value not in ['N/A', '', 0]:
                    critical_score += weight
                else:
                    missing_critical.append(field)
        
        # Apply critical field penalty
        critical_penalty = critical_score  # This will be 0-1 based on field presence
        adjusted_confidence *= critical_penalty
        
        if missing_critical:
            critical_issues.append(f"Missing critical fields: {', '.join(missing_critical)}")
        
        # 3. Data Consistency Check
        consistency_score = self._check_data_consistency(extracted_data)
        if consistency_score < 0.8:
            adjusted_confidence *= consistency_score
            critical_issues.append(f"Data consistency issues (score: {consistency_score:.2f})")
        
        # 4. Optional Field Bonus
        optional_bonus = 0.0
        for field, bonus in self.optional_fields.items():
            if extracted_data.get(field) not in [None, 'N/A', '', 0]:
                optional_bonus += bonus
        
        # Apply optional bonus (max 20% boost)
        adjusted_confidence = min(1.0, adjusted_confidence * (1.0 + min(0.2, optional_bonus)))
        
        # 5. Warning Penalties
        warning_penalty = min(0.3, len(warnings) * 0.05)  # Max 30% penalty
        adjusted_confidence *= (1.0 - warning_penalty)
        
        if warnings:
            critical_issues.append(f"Processing warnings: {len(warnings)} issues")
        
        # 6. Final Validation
        extraction_success = (
            critical_score > 0.5 and  # At least 50% of critical fields
            adjusted_confidence > 0.3   # Minimum viable confidence
        )
        
        if not extraction_success:
            adjusted_confidence = min(adjusted_confidence, 0.2)  # Cap low-quality extractions
            critical_issues.append("Extraction deemed unreliable")
        
        # Calculate adjustment factor
        adjustment_factor = adjusted_confidence / original_confidence if original_confidence > 0 else 0
        
        # Generate reason
        reason = self._generate_adjustment_reason(
            original_confidence, adjusted_confidence, critical_issues, extraction_success
        )
        
        logger.info(
            f"Confidence adjusted: {original_confidence:.1f}% → {adjusted_confidence:.1f}% "
            f"(factor: {adjustment_factor:.2f}) - {reason}"
        )
        
        return ConfidenceAdjustment(
            original_confidence=original_confidence,
            adjusted_confidence=adjusted_confidence,
            adjustment_factor=adjustment_factor,
            reason=reason,
            critical_issues=critical_issues,
            extraction_success=extraction_success
        )
    
    def _check_data_consistency(self, data: Dict[str, Any]) -> float:
        """Check internal consistency of extracted data."""
        consistency_score = 1.0
        
        # Check KDA consistency
        kills = data.get('kills', 0)
        deaths = data.get('deaths', 0)
        assists = data.get('assists', 0)
        
        if kills and deaths and assists:
            # Basic sanity checks
            if kills > 50 or deaths > 50 or assists > 50:
                consistency_score *= 0.5  # Extreme values are suspicious
            
            if deaths == 0 and kills > 0:
                consistency_score *= 0.9  # Perfect KDA is rare but possible
        
        # Check gold consistency
        gold = data.get('gold', 0)
        if gold and (gold < 1000 or gold > 50000):
            consistency_score *= 0.7  # Extreme gold values
        
        # Check hero name consistency
        hero = data.get('hero', '')
        if hero and len(hero) < 3:
            consistency_score *= 0.6  # Very short hero names are suspicious
        
        return consistency_score
    
    def _generate_adjustment_reason(
        self, 
        original: float, 
        adjusted: float, 
        issues: list, 
        success: bool
    ) -> str:
        """Generate human-readable reason for confidence adjustment."""
        
        if adjusted < original * 0.3:
            return f"Major extraction failure: {'; '.join(issues[:2])}"
        elif adjusted < original * 0.7:
            return f"Significant issues found: {'; '.join(issues[:2])}"
        elif adjusted < original * 0.9:
            return f"Minor issues: {issues[0] if issues else 'Data quality concerns'}"
        elif adjusted > original:
            return "High-quality extraction with bonus points"
        else:
            return "Confidence maintained - good extraction quality"

# Global instance
realtime_confidence_adjuster = RealtimeConfidenceAdjuster() 