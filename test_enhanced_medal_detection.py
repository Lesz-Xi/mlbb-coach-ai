#!/usr/bin/env python3
"""
Enhanced Medal Detection Test - Direct Validation

Tests the improved medal detection system specifically to validate
the 50% → 90%+ accuracy improvement for production readiness.
"""

import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.trophy_medal_detector_v2 import ImprovedTrophyMedalDetector



def test_enhanced_medal_detection():
    """Test enhanced medal detection on all screenshot categories."""
    
    print("🏆 ENHANCED MEDAL DETECTION VALIDATION")
    print("=" * 60)
    print("Testing Medal Detection: 50% → 90%+ Accuracy Target")
    print("Enhanced Features: Precise color ranges, shape analysis, weighted scoring")
    print()
    
    # Test categories with expected medal types
    test_categories = {
        "Screenshot-Test-Excellent": "gold",      # Should detect gold medals
        "Screenshot-Test-Good": "silver",         # Should detect silver medals  
        "Screenshot-Test-Average": "silver",      # Should detect silver medals
        "Screenshot-Test-Poor": "bronze"          # Should detect bronze medals
    }
    
    detector = ImprovedTrophyMedalDetector()
    total_tests = 0
    successful_detections = 0
    category_results = {}
    
    for category, expected_medal in test_categories.items():
        if not os.path.exists(category):
            print(f"⚠️ Category directory not found: {category}")
            continue
            
        print(f"\n📂 TESTING CATEGORY: {category.upper()}")
        print(f"Expected Medal Type: {expected_medal}")
        print("-" * 50)
        
        category_detections = 0
        category_total = 0
        
        # Test all screenshots in category
        for filename in os.listdir(category):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            screenshot_path = os.path.join(category, filename)
            category_total += 1
            total_tests += 1
            
            print(f"\n📸 Testing: {filename}")
            
            start_time = time.time()
            
            # Test detection at multiple Y positions (simulating different player rows)
            best_result = None
            test_positions = [250, 350, 450, 550, 650, 750, 850]
            
            for y_pos in test_positions:
                try:
                    result = detector.detect_trophy_in_player_row(
                        screenshot_path, 
                        y_pos, 
                        debug_mode=True
                    )
                    
                    # Check if we found a medal
                    if (result.trophy_type.value in ['gold_medal', 'silver_medal', 'bronze_medal'] 
                        and result.confidence > 0.7):
                        
                        if best_result is None or result.confidence > best_result.confidence:
                            best_result = result
                            
                except Exception as e:
                    print(f"   Error at Y={y_pos}: {e}")
                    continue
            
            processing_time = time.time() - start_time
            
            # Evaluate result
            if best_result and best_result.confidence > 0.7:
                detected_medal = best_result.trophy_type.value.replace('_medal', '')
                
                print(f"   ✅ Medal Detected: {detected_medal}")
                print(f"   📊 Confidence: {best_result.confidence:.3f}")
                print(f"   ⚡ Processing Time: {processing_time:.2f}s")
                print(f"   🎯 Performance Label: {best_result.performance_label.value}")
                
                # Check if detection matches expected type
                if detected_medal == expected_medal:
                    successful_detections += 1
                    category_detections += 1
                    print(f"   🎉 CORRECT DETECTION!")
                else:
                    print(f"   ⚠️ Unexpected medal type (expected {expected_medal})")
            else:
                print(f"   ❌ No medal detected")
                print(f"   ⚡ Processing Time: {processing_time:.2f}s")
        
        # Category summary
        category_accuracy = (category_detections / category_total * 100) if category_total > 0 else 0
        category_results[category] = {
            'accuracy': category_accuracy,
            'detections': category_detections,
            'total': category_total
        }
        
        print(f"\n📈 CATEGORY RESULTS:")
        print(f"   Accuracy: {category_accuracy:.1f}% ({category_detections}/{category_total})")
    
    # Overall results
    overall_accuracy = (successful_detections / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n🎯 OVERALL MEDAL DETECTION RESULTS")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Successful Detections: {successful_detections}")
    print(f"Overall Accuracy: {overall_accuracy:.1f}%")
    print()
    
    # Detailed breakdown
    for category, results in category_results.items():
        print(f"{category}: {results['accuracy']:.1f}% ({results['detections']}/{results['total']})")
    
    print()
    
    # Success evaluation
    if overall_accuracy >= 90:
        print("🏆 SUCCESS: Medal detection meets 90%+ accuracy target!")
        print("✅ Production deployment ready")
    elif overall_accuracy >= 75:
        print("🟡 GOOD: Significant improvement, minor tuning needed")
        print("🔧 Consider adjusting confidence thresholds")
    else:
        print("🔴 NEEDS WORK: Further improvements required")
        print("🔧 Review color ranges and shape analysis")
    
    return overall_accuracy

if __name__ == "__main__":
    accuracy = test_enhanced_medal_detection()
    print(f"\nFinal Medal Detection Accuracy: {accuracy:.1f}%") 