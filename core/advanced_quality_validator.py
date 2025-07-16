"""
Advanced Screenshot Quality Validator for 95-100% Confidence AI Coaching

This module provides comprehensive image quality assessment to push parsing confidence
from 87.8% to 95-100% by catching and correcting quality issues before OCR processing.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QualityIssue(Enum):
    """Enumeration of possible quality issues."""
    GLARE = "glare_detected"
    MOTION_BLUR = "motion_blur"
    ROTATION = "rotation_detected"
    LOW_RESOLUTION = "low_resolution"
    OVEREXPOSURE = "overexposure"
    UNDEREXPOSURE = "underexposure"
    POOR_CONTRAST = "poor_contrast"
    EXCESSIVE_NOISE = "excessive_noise"
    UI_CROPPING = "ui_elements_cropped"
    COMPRESSION_ARTIFACTS = "compression_artifacts"


@dataclass
class QualityResult:
    """Result of quality assessment."""
    overall_score: float  # 0.0 to 1.0
    is_acceptable: bool   # Whether image meets minimum standards
    issues: List[QualityIssue]
    recommendations: List[str]
    metrics: Dict[str, float]


class AdvancedQualityValidator:
    """Advanced image quality validator with ML-coaching level precision."""
    
    def __init__(self):
        # Quality thresholds for 95%+ confidence
        self.thresholds = {
            "min_resolution": (1080, 1920),  # Minimum HD resolution
            "max_rotation_angle": 2.0,       # Max rotation in degrees
            "min_sharpness": 100.0,          # Laplacian variance threshold
            "min_contrast": 50.0,            # Standard deviation threshold
            "max_noise_level": 15.0,         # Maximum acceptable noise
            "brightness_range": (80, 180),   # Acceptable brightness range
            "glare_threshold": 240,          # Pixel intensity for glare detection
            "blur_threshold": 50.0,          # Motion blur detection threshold
        }
        
        # Weights for overall quality score
        self.weights = {
            "resolution": 0.15,
            "sharpness": 0.25,
            "contrast": 0.20,
            "brightness": 0.15,
            "noise": 0.10,
            "rotation": 0.10,
            "glare": 0.05
        }
    
    def validate_screenshot(self, image_path: str) -> QualityResult:
        """
        Comprehensive quality validation for 95%+ confidence parsing.
        
        Args:
            image_path: Path to screenshot image
            
        Returns:
            QualityResult with detailed assessment
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return QualityResult(
                    overall_score=0.0,
                    is_acceptable=False,
                    issues=[],
                    recommendations=["Cannot load image - check file path and format"],
                    metrics={}
                )
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Run all quality checks
            metrics = {}
            issues = []
            recommendations = []
            
            # 1. Resolution check
            resolution_score, res_issues, res_recs = self._check_resolution(image)
            metrics["resolution"] = resolution_score
            issues.extend(res_issues)
            recommendations.extend(res_recs)
            
            # 2. Sharpness/blur detection
            sharpness_score, sharp_issues, sharp_recs = self._check_sharpness(gray)
            metrics["sharpness"] = sharpness_score
            issues.extend(sharp_issues)
            recommendations.extend(sharp_recs)
            
            # 3. Contrast assessment
            contrast_score, contrast_issues, contrast_recs = self._check_contrast(gray)
            metrics["contrast"] = contrast_score
            issues.extend(contrast_issues)
            recommendations.extend(contrast_recs)
            
            # 4. Brightness evaluation
            brightness_score, bright_issues, bright_recs = self._check_brightness(gray)
            metrics["brightness"] = brightness_score
            issues.extend(bright_issues)
            recommendations.extend(bright_recs)
            
            # 5. Noise level assessment
            noise_score, noise_issues, noise_recs = self._check_noise(gray)
            metrics["noise"] = noise_score
            issues.extend(noise_issues)
            recommendations.extend(noise_recs)
            
            # 6. Rotation detection
            rotation_score, rot_issues, rot_recs = self._check_rotation(gray)
            metrics["rotation"] = rotation_score
            issues.extend(rot_issues)
            recommendations.extend(rot_recs)
            
            # 7. Glare detection
            glare_score, glare_issues, glare_recs = self._check_glare(gray)
            metrics["glare"] = glare_score
            issues.extend(glare_issues)
            recommendations.extend(glare_recs)
            
            # 8. UI completeness check
            ui_score, ui_issues, ui_recs = self._check_ui_completeness(image)
            metrics["ui_completeness"] = ui_score
            issues.extend(ui_issues)
            recommendations.extend(ui_recs)
            
            # Calculate overall score
            overall_score = sum(
                self.weights.get(metric, 0.0) * score 
                for metric, score in metrics.items()
            )
            
            # Determine if acceptable for 95%+ confidence
            is_acceptable = (
                overall_score >= 0.85 and 
                QualityIssue.LOW_RESOLUTION not in issues and
                QualityIssue.GLARE not in issues and
                QualityIssue.MOTION_BLUR not in issues
            )
            
            logger.info(f"Quality assessment: {overall_score:.3f}, acceptable: {is_acceptable}")
            
            return QualityResult(
                overall_score=overall_score,
                is_acceptable=is_acceptable,
                issues=issues,
                recommendations=recommendations,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Quality validation error: {str(e)}")
            return QualityResult(
                overall_score=0.0,
                is_acceptable=False,
                issues=[],
                recommendations=[f"Validation error: {str(e)}"],
                metrics={}
            )
    
    def _check_resolution(self, image: np.ndarray) -> Tuple[float, List[QualityIssue], List[str]]:
        """Check if image resolution is sufficient for accurate OCR."""
        height, width = image.shape[:2]
        min_width, min_height = self.thresholds["min_resolution"]
        
        issues = []
        recommendations = []
        
        if width < min_width or height < min_height:
            issues.append(QualityIssue.LOW_RESOLUTION)
            recommendations.append(
                f"Resolution {width}x{height} is below minimum {min_width}x{min_height}. "
                "Use higher resolution screenshots for better OCR accuracy."
            )
            score = min(1.0, (width * height) / (min_width * min_height))
        else:
            score = 1.0
        
        return score, issues, recommendations
    
    def _check_sharpness(self, gray: np.ndarray) -> Tuple[float, List[QualityIssue], List[str]]:
        """Detect motion blur and focus issues."""
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        issues = []
        recommendations = []
        
        if laplacian_var < self.thresholds["blur_threshold"]:
            issues.append(QualityIssue.MOTION_BLUR)
            recommendations.append(
                "Image appears blurry or out of focus. Hold device steady and ensure "
                "good lighting when taking screenshots."
            )
        
        score = min(1.0, laplacian_var / self.thresholds["min_sharpness"])
        return score, issues, recommendations
    
    def _check_contrast(self, gray: np.ndarray) -> Tuple[float, List[QualityIssue], List[str]]:
        """Assess image contrast for text readability."""
        contrast = np.std(gray)
        
        issues = []
        recommendations = []
        
        if contrast < self.thresholds["min_contrast"]:
            issues.append(QualityIssue.POOR_CONTRAST)
            recommendations.append(
                "Low contrast detected. Increase screen brightness or avoid "
                "taking screenshots in bright ambient light."
            )
        
        score = min(1.0, contrast / 100.0)
        return score, issues, recommendations
    
    def _check_brightness(self, gray: np.ndarray) -> Tuple[float, List[QualityIssue], List[str]]:
        """Check for over/under exposure."""
        mean_brightness = np.mean(gray)
        min_bright, max_bright = self.thresholds["brightness_range"]
        
        issues = []
        recommendations = []
        
        if mean_brightness < min_bright:
            issues.append(QualityIssue.UNDEREXPOSURE)
            recommendations.append(
                f"Image too dark (brightness: {mean_brightness:.1f}). "
                "Increase screen brightness or use better lighting."
            )
        elif mean_brightness > max_bright:
            issues.append(QualityIssue.OVEREXPOSURE)
            recommendations.append(
                f"Image too bright (brightness: {mean_brightness:.1f}). "
                "Reduce screen brightness or avoid direct light sources."
            )
        
        if min_bright <= mean_brightness <= max_bright:
            score = 1.0
        else:
            score = max(0, 1.0 - abs(mean_brightness - 130) / 130)
        
        return score, issues, recommendations
    
    def _check_noise(self, gray: np.ndarray) -> Tuple[float, List[QualityIssue], List[str]]:
        """Detect image noise that could interfere with OCR."""
        # Use local variance to estimate noise
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        noise_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
        noise_level = np.mean(np.sqrt(noise_variance))
        
        issues = []
        recommendations = []
        
        if noise_level > self.thresholds["max_noise_level"]:
            issues.append(QualityIssue.EXCESSIVE_NOISE)
            recommendations.append(
                "High noise level detected. Use better lighting and avoid "
                "ISO boost in camera settings if possible."
            )
        
        score = max(0, 1.0 - noise_level / 20.0)
        return score, issues, recommendations
    
    def _check_rotation(self, gray: np.ndarray) -> Tuple[float, List[QualityIssue], List[str]]:
        """Detect if image is rotated, which affects OCR accuracy."""
        # Use Hough line detection to find predominant angles
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        issues = []
        recommendations = []
        score = 1.0
        
        if lines is not None and len(lines) > 10:
            angles = []
            for line in lines[:50]:  # Sample first 50 lines
                rho, theta = line[0]
                angle = np.degrees(theta)
                # Convert to rotation from horizontal
                if angle > 90:
                    angle -= 180
                angles.append(abs(angle))
            
            if angles:
                avg_rotation = np.median(angles)
                if avg_rotation > self.thresholds["max_rotation_angle"]:
                    issues.append(QualityIssue.ROTATION)
                    recommendations.append(
                        f"Image appears rotated by ~{avg_rotation:.1f}°. "
                        "Hold device straight when taking screenshots."
                    )
                    score = max(0, 1.0 - avg_rotation / 10.0)
        
        return score, issues, recommendations
    
    def _check_glare(self, gray: np.ndarray) -> Tuple[float, List[QualityIssue], List[str]]:
        """Detect screen glare that obscures text."""
        # Find very bright regions that might be glare
        bright_pixels = np.sum(gray > self.thresholds["glare_threshold"])
        total_pixels = gray.shape[0] * gray.shape[1]
        glare_percentage = bright_pixels / total_pixels
        
        issues = []
        recommendations = []
        
        if glare_percentage > 0.05:  # More than 5% very bright pixels
            issues.append(QualityIssue.GLARE)
            recommendations.append(
                f"Screen glare detected in {glare_percentage*100:.1f}% of image. "
                "Adjust screen angle or lighting to reduce reflections."
            )
        
        score = max(0, 1.0 - glare_percentage * 10)
        return score, issues, recommendations
    
    def _check_ui_completeness(self, image: np.ndarray) -> Tuple[float, List[QualityIssue], List[str]]:
        """Check if important UI elements are properly visible."""
        height, width = image.shape[:2]
        
        issues = []
        recommendations = []
        
        # Check aspect ratio - MLBB screenshots should be roughly 16:9 or 18:9
        aspect_ratio = width / height
        expected_ratios = [16/9, 18/9, 19.5/9]  # Common mobile aspect ratios
        
        if not any(abs(aspect_ratio - ratio) < 0.2 for ratio in expected_ratios):
            issues.append(QualityIssue.UI_CROPPING)
            recommendations.append(
                f"Aspect ratio {aspect_ratio:.2f} suggests UI elements may be cropped. "
                "Capture the full game screen without cropping."
            )
        
        # Score based on aspect ratio match
        score = 1.0
        for ratio in expected_ratios:
            if abs(aspect_ratio - ratio) < 0.1:
                score = 1.0
                break
            elif abs(aspect_ratio - ratio) < 0.3:
                score = 0.8
                break
        else:
            score = 0.6
        
        return score, issues, recommendations
    
    def suggest_corrections(self, quality_result: QualityResult) -> List[str]:
        """Provide specific suggestions to improve screenshot quality."""
        if quality_result.is_acceptable:
            return ["Screenshot quality is excellent for high-confidence parsing!"]
        
        corrections = []
        
        # Priority corrections for critical issues
        if QualityIssue.LOW_RESOLUTION in quality_result.issues:
            corrections.append("🎯 CRITICAL: Use a higher resolution - enable high-quality screenshots in camera settings")
        
        if QualityIssue.GLARE in quality_result.issues:
            corrections.append("🎯 CRITICAL: Eliminate screen glare - tilt device or change lighting angle")
        
        if QualityIssue.MOTION_BLUR in quality_result.issues:
            corrections.append("🎯 CRITICAL: Hold device steady - use both hands and brace against stable surface")
        
        # Secondary improvements
        if QualityIssue.ROTATION in quality_result.issues:
            corrections.append("📐 Hold device straight - use on-screen guides or level apps")
        
        if QualityIssue.POOR_CONTRAST in quality_result.issues:
            corrections.append("🌟 Increase screen brightness and ensure good ambient lighting")
        
        if QualityIssue.UI_CROPPING in quality_result.issues:
            corrections.append("📱 Capture full screen - don't crop important UI elements")
        
        # Add general optimization tips
        corrections.extend([
            "💡 Pro tip: Clean screen before screenshot",
            "💡 Pro tip: Use highest quality screenshot format available",
            "💡 Pro tip: Take multiple shots and pick the clearest one"
        ])
        
        return corrections


# Global instance
advanced_quality_validator = AdvancedQualityValidator()