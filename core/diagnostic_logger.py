import logging
import json
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path
import base64
from dataclasses import dataclass, asdict
from enum import Enum


class DiagnosticLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    DEBUG = "debug"


@dataclass
class OCRDetection:
    """Represents a single OCR detection with diagnostic info."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    category: str  # "hero", "gold", "kda", "unknown"
    processing_step: str


@dataclass
class DiagnosticStep:
    """Represents a single step in the analysis pipeline."""
    step_name: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence_score: float
    warnings: List[str]
    errors: List[str]
    processing_time_ms: float
    ocr_detections: List[OCRDetection]


@dataclass
class AnalysisDiagnostics:
    """Complete diagnostic information for an analysis session."""
    session_id: str
    image_path: str
    analysis_mode: str  # "basic" or "enhanced"
    timestamp: str
    steps: List[DiagnosticStep]
    final_confidence: float
    final_warnings: List[str]
    final_errors: List[str]
    overlay_image_base64: Optional[str] = None


class DiagnosticLogger:
    """Enhanced diagnostic logging system with visual overlays."""
    
    def __init__(self, enable_overlays: bool = True, debug_mode: bool = False):
        self.enable_overlays = enable_overlays
        self.debug_mode = debug_mode
        self.current_diagnostics: Optional[AnalysisDiagnostics] = None
        self.logger = logging.getLogger(f"{__name__}.diagnostic")
        
        # Create temp directory for diagnostic outputs
        self.debug_dir = Path("temp/diagnostics")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
    
    def start_analysis(self, image_path: str, analysis_mode: str, 
                       session_id: str = None) -> str:
        """Start a new analysis session and return session ID."""
        if not session_id:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            session_id = f"analysis_{timestamp}"
        
        self.current_diagnostics = AnalysisDiagnostics(
            session_id=session_id,
            image_path=image_path,
            analysis_mode=analysis_mode,
            timestamp=datetime.now().isoformat(),
            steps=[],
            final_confidence=0.0,
            final_warnings=[],
            final_errors=[]
        )
        
        self.logger.info(f"🔍 Started {analysis_mode} analysis session: {session_id}")
        return session_id
    
    def log_step(self, step_name: str, input_data: Dict[str, Any], 
                 output_data: Dict[str, Any], confidence_score: float,
                 ocr_results: List = None, processing_time_ms: float = 0,
                 warnings: List[str] = None, errors: List[str] = None) -> None:
        """Log a single analysis step with detailed information."""
        if not self.current_diagnostics:
            self.logger.warning("No active diagnostics session")
            return
        
        warnings = warnings or []
        errors = errors or []
        
        # Process OCR detections if provided
        ocr_detections = []
        if ocr_results:
            for result in ocr_results:
                if len(result) >= 3:  # EasyOCR format: (bbox, text, confidence)
                    bbox, text, conf = result
                    # Convert bbox to x1,y1,x2,y2 format
                    if len(bbox) == 4 and len(bbox[0]) == 2:  # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                        x_coords = [p[0] for p in bbox]
                        y_coords = [p[1] for p in bbox]
                        x1, x2 = min(x_coords), max(x_coords)
                        y1, y2 = min(y_coords), max(y_coords)
                    else:
                        x1, y1, x2, y2 = 0, 0, 0, 0
                    
                    # Categorize the detection
                    category = self._categorize_detection(text, output_data)
                    
                    ocr_detections.append(OCRDetection(
                        text=text,
                        confidence=conf,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        category=category,
                        processing_step=step_name
                    ))
        
        step = DiagnosticStep(
            step_name=step_name,
            input_data=input_data,
            output_data=output_data,
            confidence_score=confidence_score,
            warnings=warnings,
            errors=errors,
            processing_time_ms=processing_time_ms,
            ocr_detections=ocr_detections
        )
        
        self.current_diagnostics.steps.append(step)
        
        # Enhanced logging with color coding
        level_icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "debug": "🔧"}
        if errors:
            icon = level_icon["error"]
            log_level = logging.ERROR
        elif warnings:
            icon = level_icon["warning"]
            log_level = logging.WARNING
        else:
            icon = level_icon["info"]
            log_level = logging.INFO
        
        self.logger.log(log_level, 
            f"{icon} {step_name}: confidence={confidence_score:.3f}, "
            f"ocr_detections={len(ocr_detections)}, "
            f"warnings={len(warnings)}, errors={len(errors)}")
        
        if self.debug_mode:
            self.logger.debug(f"  📥 Input: {input_data}")
            self.logger.debug(f"  📤 Output: {output_data}")
            if warnings:
                self.logger.debug(f"  ⚠️  Warnings: {warnings}")
            if errors:
                self.logger.debug(f"  ❌ Errors: {errors}")
    
    def finish_analysis(self, final_confidence: float, final_warnings: List[str] = None, 
                       final_errors: List[str] = None) -> AnalysisDiagnostics:
        """Complete the analysis session and generate diagnostic overlay."""
        if not self.current_diagnostics:
            raise ValueError("No active diagnostics session")
        
        self.current_diagnostics.final_confidence = final_confidence
        self.current_diagnostics.final_warnings = final_warnings or []
        self.current_diagnostics.final_errors = final_errors or []
        
        # Generate visual overlay
        if self.enable_overlays:
            overlay_image = self._create_diagnostic_overlay()
            if overlay_image is not None:
                # Convert to base64 for web display
                _, buffer = cv2.imencode('.png', overlay_image)
                overlay_b64 = base64.b64encode(buffer).decode()
                self.current_diagnostics.overlay_image_base64 = overlay_b64
        
        # Save diagnostic report
        self._save_diagnostic_report()
        
        diagnostics = self.current_diagnostics
        self.current_diagnostics = None
        
        self.logger.info(f"✅ Analysis complete: final_confidence={final_confidence:.3f}")
        return diagnostics
    
    def _categorize_detection(self, text: str, output_data: Dict[str, Any]) -> str:
        """Categorize OCR detection based on content and context."""
        text_lower = text.lower().strip()
        
        # Check if it's a hero name
        if any(hero in text_lower for hero in ['miya', 'layla', 'bruno', 'franco', 'tigreal', 'estes']):
            return "hero"
        
        # Check if it's a number that could be gold
        if text.isdigit() and int(text) > 1000:
            return "gold"
        
        # Check if it's KDA-related
        if any(kda in text_lower for kda in ['k', 'd', 'a', '/', 'kda']):
            return "kda"
        
        # Check if it's match result
        if any(result in text_lower for result in ['victory', 'defeat', 'win', 'loss']):
            return "match_result"
        
        # Check if it's a small number (could be kills/deaths/assists)
        if text.isdigit() and 0 <= int(text) <= 50:
            return "stats"
        
        return "unknown"
    
    def _create_diagnostic_overlay(self) -> Optional[np.ndarray]:
        """Create visual overlay showing OCR detections and confidence."""
        try:
            image = cv2.imread(self.current_diagnostics.image_path)
            if image is None:
                return None
            
            overlay = image.copy()
            
            # Color scheme for different categories
            colors = {
                "hero": (0, 255, 0),      # Green
                "gold": (0, 255, 255),    # Yellow
                "kda": (255, 0, 0),       # Blue
                "stats": (255, 255, 0),   # Cyan
                "match_result": (255, 0, 255),  # Magenta
                "unknown": (128, 128, 128)  # Gray
            }
            
            # Draw all OCR detections
            for step in self.current_diagnostics.steps:
                for detection in step.ocr_detections:
                    x1, y1, x2, y2 = detection.bbox
                    color = colors.get(detection.category, colors["unknown"])
                    
                    # Draw bounding box
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw confidence score
                    conf_text = f"{detection.confidence:.2f}"
                    cv2.putText(overlay, conf_text, (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Draw category label
                    cv2.putText(overlay, detection.category, (x1, y2+15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Add legend
            legend_y = 30
            for category, color in colors.items():
                cv2.putText(overlay, f"{category.upper()}", (10, legend_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                legend_y += 25
            
            # Add overall confidence
            conf_text = f"Overall Confidence: {self.current_diagnostics.final_confidence:.1%}"
            cv2.putText(overlay, conf_text, (10, image.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Save debug overlay
            debug_path = self.debug_dir / f"{self.current_diagnostics.session_id}_overlay.png"
            cv2.imwrite(str(debug_path), overlay)
            
            return overlay
            
        except Exception as e:
            self.logger.error(f"Failed to create diagnostic overlay: {e}")
            return None
    
    def _save_diagnostic_report(self) -> None:
        """Save detailed diagnostic report as JSON."""
        try:
            report_path = self.debug_dir / f"{self.current_diagnostics.session_id}_report.json"
            
            # Convert to serializable format
            report_data = asdict(self.current_diagnostics)
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            self.logger.info(f"Diagnostic report saved: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save diagnostic report: {e}")
    
    def get_failure_summary(self) -> Dict[str, Any]:
        """Generate a summary of what went wrong in the analysis."""
        if not self.current_diagnostics:
            return {}
        
        summary = {
            "hero_detection_failed": False,
            "gold_parsing_failed": False,
            "low_ocr_confidence": False,
            "insufficient_data": False,
            "recommendations": []
        }
        
        # Analyze steps for common failure patterns
        for step in self.current_diagnostics.steps:
            if "hero" in step.step_name.lower():
                if step.confidence_score < 0.7:
                    summary["hero_detection_failed"] = True
                    summary["recommendations"].append("Try a clearer screenshot with visible hero portraits")
            
            if "gold" in step.step_name.lower() or "economy" in step.step_name.lower():
                if step.confidence_score < 0.5:
                    summary["gold_parsing_failed"] = True
                    summary["recommendations"].append("Ensure gold values are clearly visible in the screenshot")
            
            # Check OCR confidence
            if step.ocr_detections:
                avg_conf = sum(d.confidence for d in step.ocr_detections) / len(step.ocr_detections)
                if avg_conf < 0.8:
                    summary["low_ocr_confidence"] = True
                    summary["recommendations"].append("Upload higher resolution screenshots")
        
        return summary

# Global diagnostic logger instance
diagnostic_logger = DiagnosticLogger(enable_overlays=True, debug_mode=True) 