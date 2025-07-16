"""
Edge Case Testing Framework for MLBB Coach AI

Automatically tests the system against various challenging scenarios:
- Low resolution images
- Different game locales (EN, ID, TH, VN, etc.)
- Device-specific variations (iPhone, Android, iPad)
- Image quality issues (compression, watermarks, cropping)
- UI overlay variations
- Unusual aspect ratios
- Edge lighting conditions
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import tempfile

from .ultimate_parsing_system import ultimate_parsing_system
from .validation_schemas import EdgeCaseTest, DeviceType, LocaleType

logger = logging.getLogger(__name__)


@dataclass
class EdgeCaseResult:
    """Result of an edge case test."""
    test_name: str
    test_category: str
    image_path: str
    original_confidence: float
    modified_confidence: float
    confidence_drop: float
    processing_time: float
    warnings: List[str]
    errors: List[str]
    success: bool
    metadata: Dict[str, Any]


class EdgeCaseTester:
    """Comprehensive edge case testing framework."""
    
    def __init__(self, output_dir: str = "temp/edge_case_tests"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.results_history = []
    
    def run_comprehensive_test_suite(
        self, 
        test_images: List[str],
        ign: str = "Test Player"
    ) -> Dict[str, List[EdgeCaseResult]]:
        """Run all edge case tests on provided images."""
        
        results = {
            'resolution': [],
            'compression': [],
            'locale_simulation': [],
            'device_simulation': [],
            'image_quality': [],
            'ui_overlay': [],
            'lighting': [],
            'aspect_ratio': []
        }
        
        for image_path in test_images:
            self.logger.info(f"Running edge case tests on {image_path}")
            
            # Get baseline confidence
            baseline = self._get_baseline_result(image_path, ign)
            
            # Resolution tests
            results['resolution'].extend(
                self._test_resolution_variations(image_path, ign, baseline)
            )
            
            # Compression tests
            results['compression'].extend(
                self._test_compression_levels(image_path, ign, baseline)
            )
            
            # Locale simulation tests
            results['locale_simulation'].extend(
                self._test_locale_variations(image_path, ign, baseline)
            )
            
            # Device simulation tests
            results['device_simulation'].extend(
                self._test_device_variations(image_path, ign, baseline)
            )
            
            # Image quality tests
            results['image_quality'].extend(
                self._test_image_quality_variations(image_path, ign, baseline)
            )
            
            # UI overlay tests
            results['ui_overlay'].extend(
                self._test_ui_overlay_variations(image_path, ign, baseline)
            )
            
            # Lighting condition tests
            results['lighting'].extend(
                self._test_lighting_variations(image_path, ign, baseline)
            )
            
            # Aspect ratio tests
            results['aspect_ratio'].extend(
                self._test_aspect_ratio_variations(image_path, ign, baseline)
            )
        
        # Save comprehensive results
        self._save_test_results(results)
        
        return results
    
    def _get_baseline_result(self, image_path: str, ign: str) -> Dict[str, Any]:
        """Get baseline analysis result for comparison."""
        try:
            result = ultimate_parsing_system.analyze_screenshot_ultimate(
                image_path=image_path,
                ign=ign,
                hero_override=None,
                context="edge_case_baseline",
                quality_threshold=50.0
            )
            
            return {
                "confidence": result.overall_confidence,
                "processing_time": result.processing_time,
                "warnings": result.warnings,
                "parsed_data": result.parsed_data
            }
        except Exception as e:
            self.logger.error(f"Failed to get baseline for {image_path}: {e}")
            return {
                "confidence": 0.0,
                "processing_time": 0.0,
                "warnings": [f"Baseline failed: {str(e)}"],
                "parsed_data": {}
            }
    
    def _test_resolution_variations(
        self, 
        image_path: str, 
        ign: str, 
        baseline: Dict[str, Any]
    ) -> List[EdgeCaseResult]:
        """Test different resolution scenarios."""
        results = []
        
        # Test various resolutions
        resolutions = [
            (320, 240, "Very Low (320x240)"),
            (640, 480, "Low (640x480)"),
            (1024, 768, "Medium (1024x768)"),
            (1920, 1080, "High (1920x1080)"),
            (2560, 1440, "Very High (2560x1440)")
        ]
        
        for width, height, description in resolutions:
            try:
                # Resize image
                modified_path = self._resize_image(image_path, width, height)
                
                # Test analysis
                result = self._analyze_modified_image(
                    modified_path, ign, baseline, 
                    f"Resolution Test - {description}", "resolution"
                )
                
                result.metadata.update({
                    "target_resolution": f"{width}x{height}",
                    "description": description
                })
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Resolution test failed for {description}: {e}")
        
        return results
    
    def _test_compression_levels(
        self, 
        image_path: str, 
        ign: str, 
        baseline: Dict[str, Any]
    ) -> List[EdgeCaseResult]:
        """Test different JPEG compression levels."""
        results = []
        
        # Test various compression qualities
        qualities = [10, 25, 50, 75, 90, 95]
        
        for quality in qualities:
            try:
                # Apply compression
                modified_path = self._compress_image(image_path, quality)
                
                # Test analysis
                result = self._analyze_modified_image(
                    modified_path, ign, baseline,
                    f"Compression Test - Quality {quality}%", "compression"
                )
                
                result.metadata.update({
                    "compression_quality": quality,
                    "expected_degradation": 100 - quality
                })
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Compression test failed for quality {quality}: {e}")
        
        return results
    
    def _test_locale_variations(
        self, 
        image_path: str, 
        ign: str, 
        baseline: Dict[str, Any]
    ) -> List[EdgeCaseResult]:
        """Simulate different game locale scenarios."""
        results = []
        
        # Test locale-specific variations
        locale_tests = [
            ("font_blur", "Simulate blurry non-English fonts"),
            ("character_overlay", "Simulate non-Latin character overlays"),
            ("ui_shift", "Simulate UI position shifts in different locales"),
            ("text_size_variation", "Simulate different text sizes")
        ]
        
        for test_type, description in locale_tests:
            try:
                # Apply locale-specific modification
                modified_path = self._apply_locale_simulation(image_path, test_type)
                
                # Test analysis
                result = self._analyze_modified_image(
                    modified_path, ign, baseline,
                    f"Locale Test - {description}", "locale_simulation"
                )
                
                result.metadata.update({
                    "locale_test_type": test_type,
                    "description": description
                })
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Locale test failed for {test_type}: {e}")
        
        return results
    
    def _test_device_variations(
        self, 
        image_path: str, 
        ign: str, 
        baseline: Dict[str, Any]
    ) -> List[EdgeCaseResult]:
        """Simulate different device characteristics."""
        results = []
        
        # Device simulation tests
        device_tests = [
            ("iphone_notch", "Simulate iPhone notch interference"),
            ("android_nav", "Simulate Android navigation bar"),
            ("tablet_scaling", "Simulate tablet UI scaling"),
            ("old_device", "Simulate older device limitations")
        ]
        
        for test_type, description in device_tests:
            try:
                # Apply device-specific modification
                modified_path = self._apply_device_simulation(image_path, test_type)
                
                # Test analysis
                result = self._analyze_modified_image(
                    modified_path, ign, baseline,
                    f"Device Test - {description}", "device_simulation"
                )
                
                result.metadata.update({
                    "device_test_type": test_type,
                    "description": description
                })
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Device test failed for {test_type}: {e}")
        
        return results
    
    def _test_image_quality_variations(
        self, 
        image_path: str, 
        ign: str, 
        baseline: Dict[str, Any]
    ) -> List[EdgeCaseResult]:
        """Test various image quality degradations."""
        results = []
        
        # Quality degradation tests
        quality_tests = [
            ("noise", "Add random noise"),
            ("blur", "Apply blur filter"),
            ("brightness_low", "Reduce brightness"),
            ("brightness_high", "Increase brightness"),
            ("contrast_low", "Reduce contrast"),
            ("contrast_high", "Increase contrast"),
            ("saturation_low", "Desaturate colors"),
            ("pixelation", "Apply pixelation effect")
        ]
        
        for test_type, description in quality_tests:
            try:
                # Apply quality degradation
                modified_path = self._apply_quality_degradation(image_path, test_type)
                
                # Test analysis
                result = self._analyze_modified_image(
                    modified_path, ign, baseline,
                    f"Quality Test - {description}", "image_quality"
                )
                
                result.metadata.update({
                    "quality_test_type": test_type,
                    "description": description
                })
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Quality test failed for {test_type}: {e}")
        
        return results
    
    def _test_ui_overlay_variations(
        self, 
        image_path: str, 
        ign: str, 
        baseline: Dict[str, Any]
    ) -> List[EdgeCaseResult]:
        """Test various UI overlay scenarios."""
        results = []
        
        # UI overlay tests
        overlay_tests = [
            ("notification", "Add notification overlay"),
            ("popup", "Add popup dialog overlay"),
            ("watermark", "Add watermark overlay"),
            ("recording_indicator", "Add recording indicator"),
            ("system_ui", "Add system UI elements")
        ]
        
        for test_type, description in overlay_tests:
            try:
                # Apply UI overlay
                modified_path = self._apply_ui_overlay(image_path, test_type)
                
                # Test analysis
                result = self._analyze_modified_image(
                    modified_path, ign, baseline,
                    f"UI Overlay Test - {description}", "ui_overlay"
                )
                
                result.metadata.update({
                    "overlay_type": test_type,
                    "description": description
                })
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"UI overlay test failed for {test_type}: {e}")
        
        return results
    
    def _test_lighting_variations(
        self, 
        image_path: str, 
        ign: str, 
        baseline: Dict[str, Any]
    ) -> List[EdgeCaseResult]:
        """Test different lighting conditions."""
        results = []
        
        # Lighting condition tests
        lighting_tests = [
            ("dark", "Very dark conditions"),
            ("bright", "Very bright conditions"),
            ("uneven", "Uneven lighting"),
            ("glare", "Screen glare simulation"),
            ("shadow", "Shadow overlay")
        ]
        
        for test_type, description in lighting_tests:
            try:
                # Apply lighting effect
                modified_path = self._apply_lighting_effect(image_path, test_type)
                
                # Test analysis
                result = self._analyze_modified_image(
                    modified_path, ign, baseline,
                    f"Lighting Test - {description}", "lighting"
                )
                
                result.metadata.update({
                    "lighting_type": test_type,
                    "description": description
                })
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Lighting test failed for {test_type}: {e}")
        
        return results
    
    def _test_aspect_ratio_variations(
        self, 
        image_path: str, 
        ign: str, 
        baseline: Dict[str, Any]
    ) -> List[EdgeCaseResult]:
        """Test different aspect ratios."""
        results = []
        
        # Aspect ratio tests
        ratios = [
            (1.33, "4:3 (Old tablets)"),
            (1.6, "16:10 (Some tablets)"),
            (1.77, "16:9 (Standard)"),
            (1.9, "19:10 (Modern phones)"),
            (2.1, "21:9 (Ultra-wide)"),
            (2.2, "22:10 (Tall phones)")
        ]
        
        for ratio, description in ratios:
            try:
                # Apply aspect ratio change
                modified_path = self._change_aspect_ratio(image_path, ratio)
                
                # Test analysis
                result = self._analyze_modified_image(
                    modified_path, ign, baseline,
                    f"Aspect Ratio Test - {description}", "aspect_ratio"
                )
                
                result.metadata.update({
                    "aspect_ratio": ratio,
                    "description": description
                })
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Aspect ratio test failed for {description}: {e}")
        
        return results
    
    # ============ IMAGE MODIFICATION METHODS ============
    
    def _resize_image(self, image_path: str, width: int, height: int) -> str:
        """Resize image to specified dimensions."""
        with Image.open(image_path) as img:
            resized = img.resize((width, height), Image.Resampling.LANCZOS)
            
            output_path = self.output_dir / f"resized_{width}x{height}_{Path(image_path).name}"
            resized.save(output_path)
            return str(output_path)
    
    def _compress_image(self, image_path: str, quality: int) -> str:
        """Apply JPEG compression to image."""
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            output_path = self.output_dir / f"compressed_q{quality}_{Path(image_path).name}"
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
            return str(output_path)
    
    def _apply_locale_simulation(self, image_path: str, test_type: str) -> str:
        """Apply locale-specific modifications."""
        with Image.open(image_path) as img:
            if test_type == "font_blur":
                # Simulate blurry non-English fonts
                img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
            elif test_type == "character_overlay":
                # Add some text overlays to simulate non-Latin characters
                img = self._add_text_noise(img)
            elif test_type == "ui_shift":
                # Slightly shift the image to simulate UI position differences
                img = self._shift_image(img, 5, 5)
            elif test_type == "text_size_variation":
                # Simulate different text sizes by slight scaling
                img = img.resize((int(img.width * 0.95), int(img.height * 0.95)), Image.Resampling.LANCZOS)
            
            output_path = self.output_dir / f"locale_{test_type}_{Path(image_path).name}"
            img.save(output_path)
            return str(output_path)
    
    def _apply_device_simulation(self, image_path: str, test_type: str) -> str:
        """Apply device-specific modifications."""
        with Image.open(image_path) as img:
            if test_type == "iphone_notch":
                # Add black area at top to simulate notch
                img = self._add_notch_overlay(img)
            elif test_type == "android_nav":
                # Add navigation bar at bottom
                img = self._add_nav_bar(img)
            elif test_type == "tablet_scaling":
                # Scale UI elements differently
                img = img.resize((int(img.width * 1.2), int(img.height * 1.1)), Image.Resampling.LANCZOS)
            elif test_type == "old_device":
                # Simulate older device limitations with reduced quality
                img = img.convert('P', palette=Image.Palette.ADAPTIVE, colors=64).convert('RGB')
            
            output_path = self.output_dir / f"device_{test_type}_{Path(image_path).name}"
            img.save(output_path)
            return str(output_path)
    
    def _apply_quality_degradation(self, image_path: str, test_type: str) -> str:
        """Apply various quality degradations."""
        with Image.open(image_path) as img:
            if test_type == "noise":
                img = self._add_noise(img)
            elif test_type == "blur":
                img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
            elif test_type == "brightness_low":
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(0.5)
            elif test_type == "brightness_high":
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(1.8)
            elif test_type == "contrast_low":
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(0.5)
            elif test_type == "contrast_high":
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(2.0)
            elif test_type == "saturation_low":
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(0.3)
            elif test_type == "pixelation":
                # Pixelate by downscaling and upscaling
                small = img.resize((img.width // 4, img.height // 4), Image.Resampling.NEAREST)
                img = small.resize((img.width, img.height), Image.Resampling.NEAREST)
            
            output_path = self.output_dir / f"quality_{test_type}_{Path(image_path).name}"
            img.save(output_path)
            return str(output_path)
    
    def _apply_ui_overlay(self, image_path: str, test_type: str) -> str:
        """Apply UI overlay effects."""
        with Image.open(image_path) as img:
            # These would be more sophisticated in a real implementation
            if test_type == "notification":
                img = self._add_notification_overlay(img)
            elif test_type == "popup":
                img = self._add_popup_overlay(img)
            elif test_type == "watermark":
                img = self._add_watermark(img)
            elif test_type == "recording_indicator":
                img = self._add_recording_indicator(img)
            elif test_type == "system_ui":
                img = self._add_system_ui(img)
            
            output_path = self.output_dir / f"overlay_{test_type}_{Path(image_path).name}"
            img.save(output_path)
            return str(output_path)
    
    def _apply_lighting_effect(self, image_path: str, test_type: str) -> str:
        """Apply lighting effects."""
        with Image.open(image_path) as img:
            if test_type == "dark":
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(0.3)
            elif test_type == "bright":
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(2.5)
            elif test_type == "uneven":
                img = self._add_uneven_lighting(img)
            elif test_type == "glare":
                img = self._add_glare_effect(img)
            elif test_type == "shadow":
                img = self._add_shadow_effect(img)
            
            output_path = self.output_dir / f"lighting_{test_type}_{Path(image_path).name}"
            img.save(output_path)
            return str(output_path)
    
    def _change_aspect_ratio(self, image_path: str, target_ratio: float) -> str:
        """Change image aspect ratio."""
        with Image.open(image_path) as img:
            current_ratio = img.width / img.height
            
            if target_ratio > current_ratio:
                # Make wider
                new_width = int(img.height * target_ratio)
                new_img = Image.new('RGB', (new_width, img.height), (0, 0, 0))
                offset = (new_width - img.width) // 2
                new_img.paste(img, (offset, 0))
                img = new_img
            else:
                # Make taller
                new_height = int(img.width / target_ratio)
                new_img = Image.new('RGB', (img.width, new_height), (0, 0, 0))
                offset = (new_height - img.height) // 2
                new_img.paste(img, (0, offset))
                img = new_img
            
            output_path = self.output_dir / f"aspect_{target_ratio:.2f}_{Path(image_path).name}"
            img.save(output_path)
            return str(output_path)
    
    # ============ HELPER METHODS ============
    
    def _analyze_modified_image(
        self, 
        image_path: str, 
        ign: str, 
        baseline: Dict[str, Any],
        test_name: str,
        test_category: str
    ) -> EdgeCaseResult:
        """Analyze modified image and compare to baseline."""
        start_time = time.time()
        
        try:
            result = ultimate_parsing_system.analyze_screenshot_ultimate(
                image_path=image_path,
                ign=ign,
                hero_override=None,
                context="edge_case_test",
                quality_threshold=30.0  # Lower threshold for edge case testing
            )
            
            confidence = result.overall_confidence
            processing_time = time.time() - start_time
            warnings = result.warnings
            errors = []
            success = True
            
        except Exception as e:
            confidence = 0.0
            processing_time = time.time() - start_time
            warnings = []
            errors = [str(e)]
            success = False
        
        confidence_drop = baseline["confidence"] - confidence
        
        return EdgeCaseResult(
            test_name=test_name,
            test_category=test_category,
            image_path=image_path,
            original_confidence=baseline["confidence"],
            modified_confidence=confidence,
            confidence_drop=confidence_drop,
            processing_time=processing_time,
            warnings=warnings,
            errors=errors,
            success=success,
            metadata={}
        )
    
    def _save_test_results(self, results: Dict[str, List[EdgeCaseResult]]):
        """Save test results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"edge_case_results_{timestamp}.json"
        
        serializable_results = {}
        for category, result_list in results.items():
            serializable_results[category] = []
            for result in result_list:
                serializable_results[category].append({
                    "test_name": result.test_name,
                    "test_category": result.test_category,
                    "original_confidence": result.original_confidence,
                    "modified_confidence": result.modified_confidence,
                    "confidence_drop": result.confidence_drop,
                    "processing_time": result.processing_time,
                    "warnings": result.warnings,
                    "errors": result.errors,
                    "success": result.success,
                    "metadata": result.metadata
                })
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Edge case test results saved to {output_file}")
    
    # Placeholder methods for specific image modifications
    # These would be implemented with more sophisticated image processing
    
    def _add_noise(self, img):
        """Add random noise to image."""
        return img  # Simplified - would add actual noise
    
    def _add_text_noise(self, img):
        """Add text overlays to simulate character interference."""
        return img  # Simplified
    
    def _shift_image(self, img, x_offset, y_offset):
        """Shift image position."""
        return img  # Simplified
    
    def _add_notch_overlay(self, img):
        """Add iPhone notch simulation."""
        return img  # Simplified
    
    def _add_nav_bar(self, img):
        """Add Android navigation bar."""
        return img  # Simplified
    
    def _add_notification_overlay(self, img):
        """Add notification overlay."""
        return img  # Simplified
    
    def _add_popup_overlay(self, img):
        """Add popup dialog overlay."""
        return img  # Simplified
    
    def _add_watermark(self, img):
        """Add watermark overlay."""
        return img  # Simplified
    
    def _add_recording_indicator(self, img):
        """Add recording indicator."""
        return img  # Simplified
    
    def _add_system_ui(self, img):
        """Add system UI elements."""
        return img  # Simplified
    
    def _add_uneven_lighting(self, img):
        """Add uneven lighting effect."""
        return img  # Simplified
    
    def _add_glare_effect(self, img):
        """Add screen glare effect."""
        return img  # Simplified
    
    def _add_shadow_effect(self, img):
        """Add shadow effect."""
        return img  # Simplified


# Global instance
edge_case_tester = EdgeCaseTester() 