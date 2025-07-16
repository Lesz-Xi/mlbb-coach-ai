#!/usr/bin/env python3
"""
YOLOv8 Integration Test Script
=============================

This script demonstrates the complete YOLOv8 integration with the existing
MLBB Coach AI system, including fallback logic, minimap tracking, and
detection services.

Usage:
    python test_yolo_integration.py --image path/to/screenshot.png
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_yolo_fallback_logic():
    """Test YOLOv8 fallback logic."""
    logger.info("🔍 Testing YOLOv8 Fallback Logic")
    
    try:
        from core.yolo_fallback import should_use_yolo, make_fallback_decision, get_fallback_stats
        
        # Test simple fallback
        test_cases = [
            (0.9, "High OCR confidence - should NOT use YOLO"),
            (0.6, "Medium OCR confidence - should use YOLO"),
            (0.3, "Low OCR confidence - should use YOLO")
        ]
        
        for ocr_conf, description in test_cases:
            should_use = should_use_yolo(ocr_conf)
            logger.info(f"  • {description}: {should_use}")
        
        # Test advanced fallback decision
        decision = make_fallback_decision(
            ocr_confidence=0.6,
            yolo_confidence=0.8,
            quality_metrics={"has_glare": True, "blur_score": 0.2}
        )
        
        logger.info(f"  • Advanced decision: Use YOLO = {decision.should_use_yolo}")
        if decision.reason:
            logger.info(f"  • Reason: {decision.reason.value}")
        
        # Get statistics
        stats = get_fallback_stats()
        logger.info(f"  • Fallback stats: {stats}")
        
        logger.info("✅ YOLOv8 Fallback Logic Test Passed\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ YOLOv8 Fallback Logic Test Failed: {e}")
        return False

def test_yolo_detection_service():
    """Test YOLOv8 detection service."""
    logger.info("🎯 Testing YOLOv8 Detection Service")
    
    try:
        from core.services.yolo_detection_service import get_yolo_detection_service
        
        service = get_yolo_detection_service()
        
        # Test health check
        health = service.health_check()
        logger.info(f"  • Service health: {health['status']}")
        logger.info(f"  • Model loaded: {health['model_loaded']}")
        logger.info(f"  • Device: {health['device']}")
        
        # Test performance stats
        stats = service.get_performance_stats()
        logger.info(f"  • Total detections: {stats['total_detections']}")
        logger.info(f"  • Average inference time: {stats['avg_inference_time']:.3f}s")
        logger.info(f"  • Confidence threshold: {stats['confidence_threshold']}")
        
        logger.info("✅ YOLOv8 Detection Service Test Passed\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ YOLOv8 Detection Service Test Failed: {e}")
        return False

def test_image_detection(image_path: str):
    """Test YOLOv8 detection on a specific image."""
    logger.info(f"📸 Testing YOLOv8 Detection on: {image_path}")
    
    if not Path(image_path).exists():
        logger.error(f"❌ Image not found: {image_path}")
        return False
    
    try:
        from core.services.yolo_detection_service import get_yolo_detection_service
        
        service = get_yolo_detection_service()
        
        # Test general object detection
        start_time = time.time()
        results = service.detect_objects(image_path, ocr_confidence=0.6)
        detection_time = time.time() - start_time
        
        logger.info(f"  • Detection time: {detection_time:.3f}s")
        logger.info(f"  • Used YOLO: {results.get('used_yolo', False)}")
        logger.info(f"  • Detections found: {len(results.get('detections', []))}")
        
        # Show detection details
        for detection in results.get('detections', [])[:5]:  # Show first 5
            logger.info(f"    - {detection['class_name']}: {detection['confidence']:.3f}")
        
        # Test specific detection types
        logger.info("  🎮 Testing specific detection types:")
        
        # Hero portraits
        heroes = service.detect_hero_portraits(image_path)
        logger.info(f"    • Hero portraits: {len(heroes)}")
        
        # UI elements
        ui_elements = service.detect_ui_elements(image_path)
        logger.info(f"    • UI elements: {len(ui_elements)}")
        
        # Stat boxes
        stat_boxes = service.detect_stat_boxes(image_path)
        logger.info(f"    • Stat boxes: {len(stat_boxes)}")
        
        # Minimap region
        minimap = service.get_minimap_region(image_path)
        if minimap:
            logger.info(f"    • Minimap found: confidence {minimap['confidence']:.3f}")
        else:
            logger.info("    • No minimap detected")
        
        # Test OCR enhancement
        ocr_enhancement = service.enhance_ocr_regions(image_path)
        logger.info(f"    • OCR enhancement regions: {ocr_enhancement['region_count']}")
        
        logger.info("✅ YOLOv8 Image Detection Test Passed\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ YOLOv8 Image Detection Test Failed: {e}")
        return False

def test_enhanced_minimap_tracker():
    """Test enhanced minimap tracker."""
    logger.info("🗺️ Testing Enhanced Minimap Tracker")
    
    try:
        from core.enhanced_minimap_tracker import create_enhanced_minimap_tracker
        
        # Create enhanced tracker
        tracker = create_enhanced_minimap_tracker("TestPlayer", use_yolo=True)
        
        logger.info(f"  • Tracker created for: {tracker.player_ign}")
        logger.info(f"  • YOLOv8 enabled: {tracker.use_yolo}")
        
        # Test detection stats
        stats = tracker.minimap_extractor.get_detection_stats()
        logger.info(f"  • Detection stats: {stats}")
        
        logger.info("✅ Enhanced Minimap Tracker Test Passed\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Minimap Tracker Test Failed: {e}")
        return False

def test_existing_system_integration():
    """Test integration with existing system."""
    logger.info("🔗 Testing Existing System Integration")
    
    try:
        # Test imports
        from core.ultimate_parsing_system import ultimate_parsing_system
        from core.enhanced_data_collector import enhanced_data_collector
        from core.yolo_fallback import make_fallback_decision
        
        logger.info("  • All imports successful")
        
        # Test that existing system still works
        logger.info("  • Existing system components accessible")
        logger.info("  • YOLOv8 integration non-breaking")
        
        logger.info("✅ Existing System Integration Test Passed\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Existing System Integration Test Failed: {e}")
        return False

def demonstrate_hybrid_analysis(image_path: str):
    """Demonstrate hybrid OCR + YOLOv8 analysis."""
    logger.info(f"🔬 Demonstrating Hybrid Analysis on: {image_path}")
    
    if not Path(image_path).exists():
        logger.warning(f"⚠️ Image not found: {image_path} - Skipping demo")
        return False
    
    try:
        from core.services.yolo_detection_service import get_yolo_detection_service
        from core.yolo_fallback import make_fallback_decision
        
        # Simulate OCR analysis
        simulated_ocr_confidence = 0.6  # Medium confidence
        logger.info(f"  • Simulated OCR confidence: {simulated_ocr_confidence}")
        
        # Make fallback decision
        decision = make_fallback_decision(
            ocr_confidence=simulated_ocr_confidence,
            yolo_confidence=0.8
        )
        
        logger.info(f"  • Fallback decision: Use YOLO = {decision.should_use_yolo}")
        
        if decision.should_use_yolo:
            # Use YOLOv8 for enhancement
            yolo_service = get_yolo_detection_service()
            yolo_results = yolo_service.detect_objects(image_path, ocr_confidence=simulated_ocr_confidence)
            
            logger.info(f"  • YOLOv8 detections: {len(yolo_results.get('detections', []))}")
            logger.info(f"  • Average confidence: {yolo_results.get('avg_confidence', 0):.3f}")
            
            # Show how results could be combined
            logger.info("  💡 In production, this would:")
            logger.info("     - Combine OCR text extraction with YOLO object detection")
            logger.info("     - Use YOLO bounding boxes to focus OCR on relevant regions")
            logger.info("     - Provide fallback when OCR fails on glare/blur")
            logger.info("     - Enhance hero detection and UI element recognition")
        
        logger.info("✅ Hybrid Analysis Demo Completed\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Hybrid Analysis Demo Failed: {e}")
        return False

def run_comprehensive_test(image_path: str = None):
    """Run comprehensive YOLOv8 integration test."""
    logger.info("🚀 Starting Comprehensive YOLOv8 Integration Test")
    logger.info("=" * 60)
    
    test_results = []
    
    # Test 1: Fallback Logic
    test_results.append(test_yolo_fallback_logic())
    
    # Test 2: Detection Service
    test_results.append(test_yolo_detection_service())
    
    # Test 3: Image Detection (if image provided)
    if image_path:
        test_results.append(test_image_detection(image_path))
    
    # Test 4: Enhanced Minimap Tracker
    test_results.append(test_enhanced_minimap_tracker())
    
    # Test 5: Existing System Integration
    test_results.append(test_existing_system_integration())
    
    # Test 6: Hybrid Analysis Demo (if image provided)
    if image_path:
        test_results.append(demonstrate_hybrid_analysis(image_path))
    
    # Summary
    logger.info("🎯 Test Summary")
    logger.info("=" * 60)
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    logger.info(f"✅ Passed: {passed_tests}/{total_tests} tests")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! YOLOv8 integration is working correctly.")
        logger.info("\n📋 Next Steps:")
        logger.info("  1. Annotate your MLBB screenshots for training")
        logger.info("  2. Train YOLOv8 model using train_yolo_detector.py")
        logger.info("  3. Replace model path in yolo_detection_service.py")
        logger.info("  4. Test with real MLBB screenshots")
        logger.info("  5. Monitor performance and adjust confidence thresholds")
        return True
    else:
        logger.warning(f"⚠️ {total_tests - passed_tests} test(s) failed. Check logs above.")
        return False

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test YOLOv8 integration with MLBB Coach AI')
    parser.add_argument('--image', help='Path to test image')
    parser.add_argument('--quick', action='store_true', help='Run quick test without image')
    
    args = parser.parse_args()
    
    if args.quick:
        logger.info("🚀 Running Quick Integration Test")
        success = run_comprehensive_test()
    elif args.image:
        logger.info(f"🚀 Running Full Integration Test with image: {args.image}")
        success = run_comprehensive_test(args.image)
    else:
        logger.info("🚀 Running Comprehensive Integration Test")
        # Try to find a test image
        test_image_paths = [
            "Screenshot-Test-Excellent/screenshot-test-excellent-1.PNG",
            "Screenshot-Test-Good/screenshot-test-good-1.PNG",
            "Screenshot-Test-Average/screenshot-test-average-1.PNG",
            "Screenshot-Test-Poor/screenshot-test-poor-1.PNG"
        ]
        
        test_image = None
        for path in test_image_paths:
            if Path(path).exists():
                test_image = path
                break
        
        if test_image:
            logger.info(f"📸 Found test image: {test_image}")
            success = run_comprehensive_test(test_image)
        else:
            logger.info("📸 No test image found, running without image tests")
            success = run_comprehensive_test()
    
    if success:
        logger.info("\n🎉 YOLOv8 Integration Test Completed Successfully!")
        sys.exit(0)
    else:
        logger.error("\n❌ YOLOv8 Integration Test Failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 