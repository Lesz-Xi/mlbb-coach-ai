"""
Comprehensive error handling and validation system for the MLBB Coach AI.
Provides standardized error responses and validation for screenshots and user inputs.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import cv2
import numpy as np
import os
from PIL import Image

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur."""
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    DATA_ERROR = "data_error"
    SYSTEM_ERROR = "system_error"
    USER_ERROR = "user_error"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    error_type: Optional[ErrorType] = None
    error_message: str = ""
    warnings: List[str] = None
    suggestions: List[str] = None
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.suggestions is None:
            self.suggestions = []


@dataclass
class ErrorResponse:
    """Standardized error response."""
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any] = None
    suggestions: List[str] = None
    retry_possible: bool = True
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.suggestions is None:
            self.suggestions = []


class ScreenshotValidator:
    """Validator for screenshot uploads and quality assessment."""
    
    def __init__(self):
        """Initialize with validation parameters."""
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.min_image_size = (200, 200)
        self.max_image_size = (4000, 4000)
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        self.min_quality_score = 0.3
    
    def validate_screenshot(self, file_path: str) -> ValidationResult:
        """Comprehensive screenshot validation."""
        try:
            # Check file existence
            if not os.path.exists(file_path):
                return ValidationResult(
                    is_valid=False,
                    error_type=ErrorType.VALIDATION_ERROR,
                    error_message="Screenshot file not found",
                    suggestions=["Please upload a valid screenshot file"]
                )
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                return ValidationResult(
                    is_valid=False,
                    error_type=ErrorType.VALIDATION_ERROR,
                    error_message=f"File too large: {file_size / (1024*1024):.1f}MB",
                    suggestions=["Please compress the image or use a smaller file"]
                )
            
            # Check file format
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.supported_formats:
                return ValidationResult(
                    is_valid=False,
                    error_type=ErrorType.VALIDATION_ERROR,
                    error_message=f"Unsupported file format: {file_ext}",
                    suggestions=[f"Please use one of: {', '.join(self.supported_formats)}"]
                )
            
            # Validate image content
            image_validation = self._validate_image_content(file_path)
            if not image_validation.is_valid:
                return image_validation
            
            # Assess image quality
            quality_assessment = self._assess_image_quality(file_path)
            
            return ValidationResult(
                is_valid=True,
                warnings=quality_assessment.get("warnings", []),
                suggestions=quality_assessment.get("suggestions", []),
                confidence=quality_assessment.get("confidence", 1.0)
            )
            
        except Exception as e:
            logger.error(f"Screenshot validation error: {str(e)}")
            return ValidationResult(
                is_valid=False,
                error_type=ErrorType.SYSTEM_ERROR,
                error_message=f"Validation system error: {str(e)}",
                suggestions=["Please try again or contact support"]
            )
    
    def _validate_image_content(self, file_path: str) -> ValidationResult:
        """Validate image content and format."""
        try:
            # Try to open with PIL
            with Image.open(file_path) as img:
                width, height = img.size
                
                # Check dimensions
                if width < self.min_image_size[0] or height < self.min_image_size[1]:
                    return ValidationResult(
                        is_valid=False,
                        error_type=ErrorType.VALIDATION_ERROR,
                        error_message=f"Image too small: {width}x{height}",
                        suggestions=["Please use a larger screenshot (minimum 200x200)"]
                    )
                
                if width > self.max_image_size[0] or height > self.max_image_size[1]:
                    return ValidationResult(
                        is_valid=False,
                        error_type=ErrorType.VALIDATION_ERROR,
                        error_message=f"Image too large: {width}x{height}",
                        suggestions=["Please use a smaller screenshot (maximum 4000x4000)"]
                    )
                
                # Check if image is corrupted
                img.verify()
            
            # Try to open with OpenCV
            cv_image = cv2.imread(file_path)
            if cv_image is None:
                return ValidationResult(
                    is_valid=False,
                    error_type=ErrorType.VALIDATION_ERROR,
                    error_message="Image file appears to be corrupted",
                    suggestions=["Please try a different screenshot"]
                )
            
            return ValidationResult(is_valid=True)
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_type=ErrorType.VALIDATION_ERROR,
                error_message=f"Invalid image file: {str(e)}",
                suggestions=["Please ensure the file is a valid image"]
            )
    
    def _assess_image_quality(self, file_path: str) -> Dict[str, Any]:
        """Assess image quality for OCR processing."""
        try:
            image = cv2.imread(file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            quality_metrics = {}
            warnings = []
            suggestions = []
            
            # Blur detection
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())  # Ensure Python float
            quality_metrics["blur_score"] = blur_score
            
            if blur_score < 100:
                warnings.append("Image appears blurry")
                suggestions.append("Try taking a clearer screenshot")
            
            # Brightness assessment
            brightness = float(np.mean(gray))  # Ensure Python float
            quality_metrics["brightness"] = brightness
            
            if brightness < 50:
                warnings.append("Image is too dark")
                suggestions.append("Increase screen brightness before taking screenshot")
            elif brightness > 200:
                warnings.append("Image is too bright")
                suggestions.append("Reduce screen brightness or avoid glare")
            
            # Contrast assessment
            contrast = float(np.std(gray))  # Ensure Python float
            quality_metrics["contrast"] = contrast
            
            if contrast < 30:
                warnings.append("Low contrast image")
                suggestions.append("Ensure good contrast between text and background")
            
            # Overall quality score
            quality_score = self._calculate_quality_score(quality_metrics)
            
            return {
                "quality_score": quality_score,
                "metrics": quality_metrics,
                "warnings": warnings,
                "suggestions": suggestions,
                "confidence": quality_score
            }
            
        except Exception as e:
            logger.error(f"Quality assessment error: {str(e)}")
            return {
                "quality_score": 0.5,
                "warnings": ["Could not assess image quality"],
                "suggestions": ["Please try a different screenshot"],
                "confidence": 0.5
            }
    
    def _calculate_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score."""
        score = 1.0
        
        # Blur penalty
        blur_score = metrics.get("blur_score", 0)
        if blur_score < 100:
            score *= 0.5
        elif blur_score < 200:
            score *= 0.7
        
        # Brightness penalty
        brightness = metrics.get("brightness", 128)
        if brightness < 50 or brightness > 200:
            score *= 0.6
        elif brightness < 80 or brightness > 180:
            score *= 0.8
        
        # Contrast penalty
        contrast = metrics.get("contrast", 50)
        if contrast < 30:
            score *= 0.5
        elif contrast < 50:
            score *= 0.7
        
        return max(0.1, min(1.0, score))
    
    def detect_screenshot_type(self, file_path: str) -> Tuple[str, float]:
        """Detect if screenshot is from MLBB game."""
        try:
            image = cv2.imread(file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Look for MLBB UI elements
            mlbb_indicators = [
                "kill", "death", "assist", "gold", "duration",
                "victory", "defeat", "mobile legends", "mlbb"
            ]
            
            # Simple text detection (would be enhanced with actual OCR)
            # For now, return moderate confidence
            confidence = 0.7
            screenshot_type = "mlbb_match"
            
            return screenshot_type, confidence
            
        except Exception as e:
            logger.error(f"Screenshot type detection error: {str(e)}")
            return "unknown", 0.3


class InputValidator:
    """Validator for user inputs and parameters."""
    
    def validate_ign(self, ign: str) -> ValidationResult:
        """Validate in-game name."""
        if not ign or not ign.strip():
            return ValidationResult(
                is_valid=False,
                error_type=ErrorType.VALIDATION_ERROR,
                error_message="IGN cannot be empty",
                suggestions=["Please enter your in-game name"]
            )
        
        # Check length
        if len(ign) > 20:
            return ValidationResult(
                is_valid=False,
                error_type=ErrorType.VALIDATION_ERROR,
                error_message="IGN too long",
                suggestions=["IGN must be 20 characters or less"]
            )
        
        # Check for valid characters
        if not ign.replace(" ", "").replace("_", "").replace("-", "").isalnum():
            return ValidationResult(
                is_valid=False,
                error_type=ErrorType.VALIDATION_ERROR,
                error_message="IGN contains invalid characters",
                suggestions=["Use only letters, numbers, spaces, underscores, and hyphens"]
            )
        
        return ValidationResult(is_valid=True)
    
    def validate_hero_names(self, hero_names: List[str]) -> ValidationResult:
        """Validate hero names against database."""
        if not hero_names:
            return ValidationResult(
                is_valid=False,
                error_type=ErrorType.VALIDATION_ERROR,
                error_message="Hero list cannot be empty",
                suggestions=["Please provide at least one hero name"]
            )
        
        # Import here to avoid circular imports
        from .hero_database import hero_database
        
        invalid_heroes = []
        suggestions = []
        
        for hero in hero_names:
            if not hero_database.get_hero_info(hero):
                invalid_heroes.append(hero)
                # Try to find similar heroes
                search_results = hero_database.search_heroes(hero, limit=1)
                if search_results:
                    suggestions.append(f"Did you mean '{search_results[0].hero}' instead of '{hero}'?")
        
        if invalid_heroes:
            return ValidationResult(
                is_valid=False,
                error_type=ErrorType.VALIDATION_ERROR,
                error_message=f"Invalid hero names: {', '.join(invalid_heroes)}",
                suggestions=suggestions
            )
        
        return ValidationResult(is_valid=True)


class ErrorHandler:
    """Central error handler for the application."""
    
    def __init__(self):
        """Initialize error handler."""
        self.screenshot_validator = ScreenshotValidator()
        self.input_validator = InputValidator()
    
    def handle_screenshot_error(self, error: Exception, file_path: str) -> ErrorResponse:
        """Handle screenshot processing errors."""
        error_message = str(error)
        
        # Categorize error
        if "permission" in error_message.lower():
            return ErrorResponse(
                error_type=ErrorType.SYSTEM_ERROR,
                severity=ErrorSeverity.HIGH,
                message="Permission denied accessing screenshot",
                suggestions=["Check file permissions", "Try uploading again"],
                retry_possible=True
            )
        
        elif "memory" in error_message.lower() or "size" in error_message.lower():
            return ErrorResponse(
                error_type=ErrorType.PROCESSING_ERROR,
                severity=ErrorSeverity.MEDIUM,
                message="Image too large to process",
                suggestions=["Compress the image", "Use a smaller screenshot"],
                retry_possible=True
            )
        
        elif "format" in error_message.lower() or "decode" in error_message.lower():
            return ErrorResponse(
                error_type=ErrorType.VALIDATION_ERROR,
                severity=ErrorSeverity.MEDIUM,
                message="Invalid image format",
                suggestions=["Use PNG, JPG, or BMP format", "Ensure file is not corrupted"],
                retry_possible=True
            )
        
        else:
            return ErrorResponse(
                error_type=ErrorType.SYSTEM_ERROR,
                severity=ErrorSeverity.HIGH,
                message=f"Screenshot processing failed: {error_message}",
                suggestions=["Try again", "Use a different screenshot"],
                retry_possible=True
            )
    
    def handle_ocr_error(self, error: Exception, confidence: float = 0.0) -> ErrorResponse:
        """Handle OCR processing errors."""
        error_message = str(error)
        
        if confidence < 0.3:
            return ErrorResponse(
                error_type=ErrorType.PROCESSING_ERROR,
                severity=ErrorSeverity.MEDIUM,
                message="Low OCR confidence - text may be unclear",
                suggestions=[
                    "Use a clearer screenshot",
                    "Ensure text is readable",
                    "Try manual hero override"
                ],
                retry_possible=True
            )
        
        elif "gpu" in error_message.lower() or "cuda" in error_message.lower():
            return ErrorResponse(
                error_type=ErrorType.SYSTEM_ERROR,
                severity=ErrorSeverity.LOW,
                message="GPU acceleration unavailable, using CPU",
                suggestions=["Processing may be slower but should work"],
                retry_possible=False
            )
        
        else:
            return ErrorResponse(
                error_type=ErrorType.PROCESSING_ERROR,
                severity=ErrorSeverity.HIGH,
                message=f"OCR processing failed: {error_message}",
                suggestions=["Try a different screenshot", "Check image quality"],
                retry_possible=True
            )
    
    def handle_analysis_error(self, error: Exception, match_data: Dict[str, Any]) -> ErrorResponse:
        """Handle analysis processing errors."""
        error_message = str(error)
        
        # Check if it's a data completeness issue
        if not match_data or len(match_data) < 3:
            return ErrorResponse(
                error_type=ErrorType.DATA_ERROR,
                severity=ErrorSeverity.MEDIUM,
                message="Insufficient data for analysis",
                suggestions=[
                    "Upload both scoreboard and stats screenshots",
                    "Ensure screenshots are from post-game screen",
                    "Try manual hero override"
                ],
                retry_possible=True
            )
        
        elif "hero" in error_message.lower() or "unknown" in error_message.lower():
            return ErrorResponse(
                error_type=ErrorType.DATA_ERROR,
                severity=ErrorSeverity.MEDIUM,
                message="Hero identification failed",
                suggestions=[
                    "Use manual hero override",
                    "Ensure hero name is visible in screenshot",
                    "Try a clearer screenshot"
                ],
                retry_possible=True
            )
        
        else:
            return ErrorResponse(
                error_type=ErrorType.PROCESSING_ERROR,
                severity=ErrorSeverity.HIGH,
                message=f"Analysis failed: {error_message}",
                suggestions=["Try again", "Contact support if problem persists"],
                retry_possible=True
            )
    
    def create_user_friendly_response(self, error_response: ErrorResponse) -> Dict[str, Any]:
        """Create user-friendly error response."""
        return {
            "success": False,
            "error": {
                "type": error_response.error_type.value,
                "severity": error_response.severity.value,
                "message": error_response.message,
                "suggestions": error_response.suggestions,
                "retry_possible": error_response.retry_possible
            },
            "details": error_response.details
        }
    
    def validate_request(self, **kwargs) -> ValidationResult:
        """Validate incoming request parameters."""
        validation_results = []
        
        # Validate IGN if provided
        if "ign" in kwargs:
            ign_validation = self.input_validator.validate_ign(kwargs["ign"])
            validation_results.append(ign_validation)
        
        # Validate hero names if provided
        if "hero_names" in kwargs:
            hero_validation = self.input_validator.validate_hero_names(kwargs["hero_names"])
            validation_results.append(hero_validation)
        
        # Validate file if provided
        if "file_path" in kwargs:
            file_validation = self.screenshot_validator.validate_screenshot(kwargs["file_path"])
            validation_results.append(file_validation)
        
        # Combine results
        all_valid = all(result.is_valid for result in validation_results)
        all_warnings = []
        all_suggestions = []
        
        for result in validation_results:
            if not result.is_valid:
                return result  # Return first invalid result
            all_warnings.extend(result.warnings)
            all_suggestions.extend(result.suggestions)
        
        return ValidationResult(
            is_valid=all_valid,
            warnings=all_warnings,
            suggestions=all_suggestions
        )


# Global error handler instance
error_handler = ErrorHandler()