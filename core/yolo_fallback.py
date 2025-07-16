"""
YOLOv8 Fallback Logic Module for MLBB Coach AI
=============================================

This module provides fallback logic to use YOLOv8 object detection
when OCR confidence is low, ensuring robust analysis even with
challenging screenshots.

Features:
- Confidence-based fallback decisions
- OCR-YOLO hybrid analysis
- Dynamic threshold adjustment
- Performance monitoring
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class FallbackReason(Enum):
    """Reasons for triggering YOLOv8 fallback."""
    LOW_OCR_CONFIDENCE = "low_ocr_confidence"
    MISSING_HERO_DATA = "missing_hero_data"
    POOR_TEXT_QUALITY = "poor_text_quality"
    INCOMPLETE_STATS = "incomplete_stats"
    GLARE_DETECTION = "glare_detection"
    BLUR_DETECTION = "blur_detection"


@dataclass
class FallbackDecision:
    """Represents a fallback decision with reasoning."""
    should_use_yolo: bool
    reason: Optional[FallbackReason]
    ocr_confidence: float
    yolo_confidence: float
    metadata: Dict[str, Any]


class YOLOFallbackManager:
    """Manages fallback decisions between OCR and YOLOv8."""
    
    def __init__(self,
                 ocr_threshold: float = 0.7,
                 yolo_threshold: float = 0.5,
                 adaptive_threshold: bool = True):
        """
        Initialize the fallback manager.
        
        Args:
            ocr_threshold: Minimum OCR confidence to avoid fallback
            yolo_threshold: Minimum YOLO confidence for fallback
            adaptive_threshold: Whether to adjust thresholds based on performance
        """
        self.ocr_threshold = ocr_threshold
        self.yolo_threshold = yolo_threshold
        self.adaptive_threshold = adaptive_threshold
        
        # Performance tracking
        self.fallback_history = []
        self.success_rates = {"ocr": 0.0, "yolo": 0.0, "hybrid": 0.0}
        
        logger.info("🔄 YOLOFallbackManager initialized")
        logger.info(f"📊 OCR threshold: {ocr_threshold}, "
                    f"YOLO threshold: {yolo_threshold}")
    
    def should_use_yolo(self,
                        ocr_confidence: float,
                        threshold: Optional[float] = None) -> bool:
        """
        Determine if YOLOv8 should be used based on OCR confidence.
        
        Args:
            ocr_confidence: OCR confidence score (0.0-1.0)
            threshold: Optional custom threshold
            
        Returns:
            True if YOLOv8 should be used
        """
        effective_threshold = threshold or self.ocr_threshold
        return ocr_confidence < effective_threshold
    
    def make_fallback_decision(self,
                               ocr_confidence: float,
                               yolo_confidence: float = 0.0,
                               quality_metrics: Optional[Dict[str, Any]] = None,
                               missing_data_count: int = 0) -> FallbackDecision:
        """
        Make a comprehensive fallback decision.
        
        Args:
            ocr_confidence: OCR confidence score
            yolo_confidence: YOLO confidence score (if available)
            quality_metrics: Image quality metrics
            missing_data_count: Number of missing data fields
            
        Returns:
            FallbackDecision object
        """
        quality_metrics = quality_metrics or {}
        
        # Primary decision: OCR confidence
        if ocr_confidence < self.ocr_threshold:
            reason = FallbackReason.LOW_OCR_CONFIDENCE
            should_fallback = True
        
        # Secondary checks
        elif missing_data_count > 3:
            reason = FallbackReason.INCOMPLETE_STATS
            should_fallback = True
        
        # Quality-based fallbacks
        elif quality_metrics.get("has_glare", False):
            reason = FallbackReason.GLARE_DETECTION
            should_fallback = True
        
        elif quality_metrics.get("blur_score", 0) > 0.3:
            reason = FallbackReason.BLUR_DETECTION
            should_fallback = True
        
        else:
            reason = None
            should_fallback = False
        
        # Override if YOLO confidence is too low
        if should_fallback and yolo_confidence < self.yolo_threshold:
            logger.warning(f"⚠️ YOLO confidence too low ({yolo_confidence:.2f}), skipping fallback")
            should_fallback = False
            reason = None
        
        decision = FallbackDecision(
            should_use_yolo=should_fallback,
            reason=reason,
            ocr_confidence=ocr_confidence,
            yolo_confidence=yolo_confidence,
            metadata={
                "quality_metrics": quality_metrics,
                "missing_data_count": missing_data_count,
                "threshold_used": self.ocr_threshold
            }
        )
        
        # Track decision
        self.fallback_history.append(decision)
        
        if should_fallback:
            logger.info(f"🎯 YOLOv8 fallback triggered: {reason.value if reason else 'unknown'}")
        
        return decision
    
    def get_hybrid_confidence(self, 
                            ocr_confidence: float,
                            yolo_confidence: float,
                            weights: Tuple[float, float] = (0.6, 0.4)) -> float:
        """
        Calculate hybrid confidence score combining OCR and YOLO.
        
        Args:
            ocr_confidence: OCR confidence score
            yolo_confidence: YOLO confidence score
            weights: (OCR weight, YOLO weight)
            
        Returns:
            Weighted hybrid confidence score
        """
        ocr_weight, yolo_weight = weights
        hybrid_score = (ocr_confidence * ocr_weight) + (yolo_confidence * yolo_weight)
        
        logger.debug(f"📊 Hybrid confidence: {hybrid_score:.3f} "
                    f"(OCR: {ocr_confidence:.3f}, YOLO: {yolo_confidence:.3f})")
        
        return hybrid_score
    
    def update_performance_metrics(self, 
                                 method: str,
                                 success: bool,
                                 confidence: float):
        """
        Update performance metrics for adaptive thresholding.
        
        Args:
            method: "ocr", "yolo", or "hybrid"
            success: Whether the analysis was successful
            confidence: Confidence score achieved
        """
        if method in self.success_rates:
            # Simple exponential moving average
            alpha = 0.1
            current_rate = self.success_rates[method]
            new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate
            self.success_rates[method] = new_rate
            
            logger.debug(f"📈 {method.upper()} success rate updated: {new_rate:.3f}")
        
        # Adaptive threshold adjustment
        if self.adaptive_threshold:
            self._adjust_thresholds()
    
    def _adjust_thresholds(self):
        """Adjust thresholds based on performance history."""
        if len(self.fallback_history) < 10:
            return
        
        # Analyze recent performance
        recent_decisions = self.fallback_history[-10:]
        yolo_successes = sum(1 for d in recent_decisions if d.should_use_yolo)
        
        # Adjust OCR threshold based on YOLO usage
        if yolo_successes > 7:  # Too many fallbacks
            self.ocr_threshold = max(0.5, self.ocr_threshold - 0.05)
            logger.info(f"🔄 Lowered OCR threshold to {self.ocr_threshold:.2f}")
        
        elif yolo_successes < 2:  # Too few fallbacks
            self.ocr_threshold = min(0.9, self.ocr_threshold + 0.05)
            logger.info(f"🔄 Raised OCR threshold to {self.ocr_threshold:.2f}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get fallback statistics."""
        if not self.fallback_history:
            return {"total_decisions": 0}
        
        total_decisions = len(self.fallback_history)
        yolo_decisions = sum(1 for d in self.fallback_history if d.should_use_yolo)
        
        reason_counts = {}
        for decision in self.fallback_history:
            if decision.reason:
                reason = decision.reason.value
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        return {
            "total_decisions": total_decisions,
            "yolo_fallback_rate": yolo_decisions / total_decisions,
            "ocr_threshold": self.ocr_threshold,
            "yolo_threshold": self.yolo_threshold,
            "success_rates": self.success_rates,
            "fallback_reasons": reason_counts
        }


# Global fallback manager instance
fallback_manager = YOLOFallbackManager()


def should_use_yolo(ocr_confidence: float, threshold: float = 0.7) -> bool:
    """
    Quick function to determine if YOLOv8 should be used.
    
    Args:
        ocr_confidence: OCR confidence score (0.0-1.0)
        threshold: Confidence threshold (default: 0.7)
        
    Returns:
        True if YOLOv8 should be used as fallback
    """
    return fallback_manager.should_use_yolo(ocr_confidence, threshold)


def make_fallback_decision(ocr_confidence: float,
                          yolo_confidence: float = 0.0,
                          quality_metrics: Optional[Dict[str, Any]] = None) -> FallbackDecision:
    """
    Make a comprehensive fallback decision.
    
    Args:
        ocr_confidence: OCR confidence score
        yolo_confidence: YOLO confidence score
        quality_metrics: Image quality metrics
        
    Returns:
        FallbackDecision object
    """
    return fallback_manager.make_fallback_decision(
        ocr_confidence=ocr_confidence,
        yolo_confidence=yolo_confidence,
        quality_metrics=quality_metrics
    )


def get_fallback_stats() -> Dict[str, Any]:
    """Get fallback statistics."""
    return fallback_manager.get_statistics() 