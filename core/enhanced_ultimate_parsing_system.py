"""
Enhanced Ultimate Parsing System with Real-time Confidence Adjustment

This fixes the core issues:
1. IGN matching failure causing zero data extraction
2. Confidence calculated before data validation  
3. Silent failures with misleading success indicators
"""

import logging
import time
from typing import Dict, Any, Optional

from .ultimate_parsing_system import UltimateParsingSystem, UltimateAnalysisResult
from .robust_ign_matcher import robust_ign_matcher
from .realtime_confidence_adjuster import realtime_confidence_adjuster

logger = logging.getLogger(__name__)

class EnhancedUltimateParsingSystem(UltimateParsingSystem):
    """Enhanced version with robust IGN matching and real-time confidence adjustment."""
    
    def analyze_screenshot_ultimate(
        self,
        image_path: str,
        ign: str,
        session_id: Optional[str] = None,
        hero_override: Optional[str] = None,
        context: str = "scoreboard",
        quality_threshold: float = 85.0
    ) -> UltimateAnalysisResult:
        """
        Enhanced analysis with robust IGN matching and confidence adjustment.
        """
        start_time = time.time()
        
        logger.info(f"🎯 Starting Enhanced Ultimate Analysis for IGN: {ign}")
        
        # Get initial analysis from parent class
        base_result = super().analyze_screenshot_ultimate(
            image_path, ign, session_id, hero_override, context, quality_threshold
        )
        
        # Extract OCR results for IGN analysis
        ocr_results = []
        try:
            # Get OCR results from enhanced data collector
            from .enhanced_data_collector import EnhancedDataCollector
            collector = EnhancedDataCollector()
            
            # Quick OCR for IGN analysis
            import cv2
            image = cv2.imread(image_path)
            if image is not None:
                from .data_collector import get_ocr_reader
                ocr_reader = get_ocr_reader()
                ocr_results = ocr_reader.readtext(image)
                
        except Exception as e:
            logger.error(f"Error getting OCR results for IGN analysis: {str(e)}")
        
        # Enhanced IGN matching
        logger.info("🔍 Performing enhanced IGN matching...")
        ign_match_result = robust_ign_matcher.find_ign_with_confidence(
            ign=ign,
            ocr_results=ocr_results,
            confidence_threshold=0.6
        )
        
        # Real-time confidence adjustment
        logger.info("⚡ Adjusting confidence based on extraction success...")
        confidence_adjustment = realtime_confidence_adjuster.adjust_confidence_realtime(
            original_confidence=base_result.overall_confidence,
            extracted_data=base_result.parsed_data,
            ign_match_result=ign_match_result,
            warnings=getattr(base_result, 'warnings', [])
        )
        
        # Create enhanced result
        enhanced_result = UltimateAnalysisResult(
            # Core Results (updated)
            parsed_data=base_result.parsed_data,
            confidence_breakdown=base_result.confidence_breakdown,
            
            # Component Results
            quality_assessment=base_result.quality_assessment,
            hero_detection=base_result.hero_detection,
            data_completion=base_result.data_completion,
            
            # Performance Metrics (updated)
            processing_time=time.time() - start_time,
            analysis_stage=base_result.analysis_stage,
            session_id=base_result.session_id,
            
            # Debugging and Improvement (enhanced)
            diagnostic_info={
                **base_result.diagnostic_info,
                "ign_matching": {
                    "found": ign_match_result.found,
                    "confidence": ign_match_result.confidence,
                    "method": ign_match_result.method,
                    "matched_text": ign_match_result.matched_text,
                    "row_data_count": len(ign_match_result.row_data)
                },
                "confidence_adjustment": {
                    "original": confidence_adjustment.original_confidence,
                    "adjusted": confidence_adjustment.adjusted_confidence,
                    "factor": confidence_adjustment.adjustment_factor,
                    "reason": confidence_adjustment.reason,
                    "critical_issues": confidence_adjustment.critical_issues
                }
            },
            improvement_roadmap=base_result.improvement_roadmap + [
                f"IGN matching: {ign_match_result.method}",
                f"Confidence adjustment: {confidence_adjustment.reason}"
            ],
            success_factors=base_result.success_factors + (
                ["robust_ign_matching"] if ign_match_result.found else []
            ),
            
            # Legacy Compatibility (updated)
            overall_confidence=confidence_adjustment.adjusted_confidence,
            completeness_score=base_result.completeness_score,
            warnings=getattr(base_result, 'warnings', []) + (
                confidence_adjustment.critical_issues if confidence_adjustment.critical_issues else []
            )
        )
        
        # Log final result
        if confidence_adjustment.extraction_success:
            logger.info(f"✅ Enhanced analysis successful - Final confidence: {enhanced_result.overall_confidence:.1f}%")
        else:
            logger.warning(f"⚠️ Enhanced analysis found issues - Final confidence: {enhanced_result.overall_confidence:.1f}%")
            logger.warning(f"Issues: {'; '.join(confidence_adjustment.critical_issues)}")
        
        return enhanced_result

# Global instance
enhanced_ultimate_parsing_system = EnhancedUltimateParsingSystem() 