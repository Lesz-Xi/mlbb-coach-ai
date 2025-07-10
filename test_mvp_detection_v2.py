#!/usr/bin/env python3
"""
Focused MVP Badge Detection Test with Improved System

Tests the improved trophy detection system specifically for MVP badge detection
in "Excellent" screenshots to validate the Priority 1 implementation.
"""

import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.trophy_medal_detector_v2 import improved_trophy_medal_detector
from core.enhanced_data_collector import EnhancedDataCollector

def test_mvp_detection_focused():
    """Test MVP detection on excellent screenshots with improved system."""
    
    print("🏆 FOCUSED MVP BADGE DETECTION TEST")
    print("=" * 60)
    print("Testing Improved Trophy Detection System v2")
    print("Priority 1: MVP Badge Detection for 'Excellent' Screenshots")
    print()
    
    # Test only excellent screenshots (should have MVP badges)
    excellent_screenshots = [
        "Screenshot-Test-Excellent/screenshot-test-excellent-1.PNG",
        "Screenshot-Test-Excellent/screenshot-test-excellent-2.PNG"
    ]
    
    mvp_detections = 0
    total_tests = len(excellent_screenshots)
    
    for i, screenshot in enumerate(excellent_screenshots, 1):
        if not os.path.exists(screenshot):
            print(f"⚠️ Screenshot not found: {screenshot}")
            continue
        
        print(f"\n📸 TEST {i}/{total_tests}: {os.path.basename(screenshot)}")
        print("-" * 50)
        
        start_time = time.time()
        
        # Test multiple Y positions to find MVP badge
        test_positions = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
        
        best_mvp_result = None
        best_mvp_confidence = 0.0
        
        for y_pos in test_positions:
            try:
                result = improved_trophy_medal_detector.detect_trophy_in_player_row(
                    image_path=screenshot,
                    player_row_y=float(y_pos),
                    player_name_x=None,  # Search full width
                    debug_mode=True
                )
                
                print(f"   Y={y_pos:3d}: {result.trophy_type.value:12s} (conf: {result.confidence:5.1%}) [{result.detection_method}]")
                
                # Track best MVP detection
                if result.trophy_type.value == "mvp_crown" and result.confidence > best_mvp_confidence:
                    best_mvp_confidence = result.confidence
                    best_mvp_result = result
                
            except Exception as e:
                print(f"   Y={y_pos:3d}: ERROR - {str(e)}")
        
        processing_time = time.time() - start_time
        
        # Analyze results
        if best_mvp_result and best_mvp_confidence > 0.5:
            mvp_detections += 1
            print(f"\n✅ MVP BADGE DETECTED!")
            print(f"   🏆 Confidence: {best_mvp_confidence:.1%}")
            print(f"   📍 Method: {best_mvp_result.detection_method}")
            print(f"   📊 Debug info: {len(best_mvp_result.debug_info)} items")
            
            # Show debug breakdown if available
            if "confidence_breakdown" in best_mvp_result.debug_info:
                breakdown = best_mvp_result.debug_info["confidence_breakdown"]
                print(f"   🔍 Confidence breakdown: {breakdown}")
                
        else:
            print(f"\n❌ No MVP badge detected")
            print(f"   Best result: {best_mvp_result.trophy_type.value if best_mvp_result else 'none'}")
            print(f"   Best confidence: {best_mvp_confidence:.1%}")
        
        print(f"   ⏱️ Processing time: {processing_time:.2f}s")
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"🎯 MVP BADGE DETECTION SUMMARY")
    print(f"=" * 60)
    print(f"📊 Screenshots tested: {total_tests}")
    print(f"🏆 MVP badges detected: {mvp_detections}")
    print(f"📈 Detection rate: {mvp_detections/total_tests*100:.1f}%")
    
    if mvp_detections == 0:
        print(f"\n❌ CRITICAL ISSUE: Still 0% MVP detection rate!")
        print(f"📋 Next steps:")
        print(f"   • Check debug images in temp/ directory")
        print(f"   • Analyze color ranges and search regions")
        print(f"   • Consider manual parameter adjustment")
    elif mvp_detections < total_tests:
        print(f"\n⚠️ PARTIAL SUCCESS: {total_tests - mvp_detections} MVP badges missed")
        print(f"📋 Next steps:")
        print(f"   • Analyze failed cases")
        print(f"   • Fine-tune detection parameters")
    else:
        print(f"\n🎉 SUCCESS: All MVP badges detected!")
        print(f"✅ Priority 1 implementation working correctly")
    
    return mvp_detections == total_tests

def test_integrated_mvp():
    """Test MVP detection through the integrated enhanced data collector."""
    
    print(f"\n🚀 INTEGRATED MVP DETECTION TEST")
    print("=" * 60)
    
    collector = EnhancedDataCollector()
    excellent_screenshots = [
        "Screenshot-Test-Excellent/screenshot-test-excellent-1.PNG",
        "Screenshot-Test-Excellent/screenshot-test-excellent-2.PNG"
    ]
    
    mvp_detected_count = 0
    
    for i, screenshot in enumerate(excellent_screenshots, 1):
        if not os.path.exists(screenshot):
            continue
        
        print(f"\n📸 INTEGRATION TEST {i}: {os.path.basename(screenshot)}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            
            result = collector.analyze_screenshot_with_session(
                image_path=screenshot,
                ign="Lesz XVII",
                hero_override=None
            )
            
            processing_time = time.time() - start_time
            
            if result and result.get("data"):
                data = result["data"]
                mvp_detected = data.get("mvp_detected", False)
                performance_label = data.get("performance_label", "Unknown")
                trophy_confidence = data.get("trophy_confidence", 0.0)
                
                print(f"✅ Analysis completed:")
                print(f"   🏆 MVP Detected: {mvp_detected}")
                print(f"   📊 Performance Label: {performance_label}")
                print(f"   🎯 Trophy Confidence: {trophy_confidence:.1%}")
                print(f"   ⏱️ Processing Time: {processing_time:.1f}s")
                
                if mvp_detected:
                    mvp_detected_count += 1
                    print(f"   ✅ SUCCESS: MVP badge found in integrated analysis!")
                else:
                    print(f"   ❌ FAILED: No MVP badge found in integrated analysis")
            else:
                print(f"❌ Analysis failed - no data returned")
                
        except Exception as e:
            print(f"❌ Integration test failed: {str(e)}")
    
    print(f"\n🎯 INTEGRATION SUMMARY:")
    print(f"MVP badges detected: {mvp_detected_count}/{len(excellent_screenshots)}")
    
    return mvp_detected_count


def main():
    """Run focused MVP detection tests."""
    
    print("🚀 PRIORITY 1: MVP BADGE DETECTION VALIDATION")
    print("=" * 70)
    print("Goal: Validate improved trophy detection system")
    print("Expected: MVP badges detected in 'Excellent' screenshots")
    print()
    
    # Test 1: Direct MVP detection
    direct_success = test_mvp_detection_focused()
    
    # Test 2: Integrated MVP detection  
    integrated_count = test_integrated_mvp()
    
    # Final assessment
    print(f"\n" + "=" * 70)
    print(f"🎯 FINAL PRIORITY 1 ASSESSMENT")
    print(f"=" * 70)
    
    if direct_success:
        print(f"✅ Direct MVP Detection: SUCCESS")
    else:
        print(f"❌ Direct MVP Detection: FAILED")
    
    if integrated_count > 0:
        print(f"✅ Integrated MVP Detection: {integrated_count}/2 detected")
    else:
        print(f"❌ Integrated MVP Detection: FAILED")
    
    if direct_success and integrated_count > 0:
        print(f"\n🎉 PRIORITY 1 STATUS: OPERATIONAL")
        print(f"✅ MVP badge detection implemented successfully")
        print(f"📈 Ready to achieve 75% → 90%+ label accuracy goal")
    else:
        print(f"\n⚠️ PRIORITY 1 STATUS: NEEDS OPTIMIZATION")
        print(f"📋 Additional parameter tuning required")
        print(f"🔍 Check debug images for analysis guidance")
    
    print(f"\n💾 Debug images saved in temp/ directory for analysis")

if __name__ == "__main__":
    main() 