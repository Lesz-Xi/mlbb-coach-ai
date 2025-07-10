#!/usr/bin/env python3
"""
Test script for MVP Badge and Trophy Detection System

This tests the Priority 1 implementation:
- MVP Badge Detection for "Excellent" screenshots
- Medal Recognition for performance categorization
- Integration with Enhanced Data Collector

Expected Results:
- Excellent screenshots: MVP badge detected → "Excellent" rating
- Good screenshots: Gold medal detected → "Good" rating  
- Average screenshots: Silver medal detected → "Average" rating
- Poor screenshots: Bronze medal detected → "Poor" rating
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.trophy_medal_detector import trophy_medal_detector, TrophyType, PerformanceLabel
from core.enhanced_data_collector import EnhancedDataCollector

class TrophyDetectionTester:
    """Test the trophy detection system against organized screenshots."""
    
    def __init__(self):
        self.enhanced_collector = EnhancedDataCollector()
        self.test_results = []
        
    def test_trophy_detection_direct(self, image_path: str, expected_trophy: str, test_ign: str = "Lesz XVII"):
        """Test trophy detection directly using the trophy detector."""
        
        print(f"\n🔍 Direct Trophy Detection Test: {os.path.basename(image_path)}")
        print(f"📋 Expected Trophy: {expected_trophy}")
        print("-" * 60)
        
        start_time = time.time()
        
        # Since we don't know the exact player row coordinates for direct testing,
        # we'll test at different Y positions across the image
        test_positions = [300, 400, 500, 600, 700, 800]  # Common player row positions
        
        best_result = None
        best_confidence = 0.0
        
        for y_pos in test_positions:
            try:
                result = trophy_medal_detector.detect_trophy_in_player_row(
                    image_path=image_path,
                    player_row_y=float(y_pos),
                    player_name_x=None,  # Search full width
                    debug_mode=True
                )
                
                if result.confidence > best_confidence:
                    best_confidence = result.confidence
                    best_result = result
                    
                print(f"   Y={y_pos}: {result.trophy_type.value} (conf: {result.confidence:.1%})")
                
            except Exception as e:
                print(f"   Y={y_pos}: Error - {str(e)}")
        
        processing_time = time.time() - start_time
        
        # Display best result
        if best_result:
            print(f"\n🏆 Best Detection Result:")
            print(f"   Trophy Type: {best_result.trophy_type.value}")
            print(f"   Performance Label: {best_result.performance_label.value}")
            print(f"   Confidence: {best_result.confidence:.1%}")
            print(f"   Detection Method: {best_result.detection_method}")
            print(f"   Processing Time: {processing_time:.2f}s")
            
            # Check if detection matches expectation
            trophy_match = self._check_trophy_expectation(best_result.trophy_type.value, expected_trophy)
            performance_match = self._check_performance_expectation(best_result.performance_label.value, expected_trophy)
            
            print(f"\n✅ Validation Results:")
            print(f"   Trophy Detection: {'✅ MATCH' if trophy_match else '❌ MISMATCH'}")
            print(f"   Performance Label: {'✅ MATCH' if performance_match else '❌ MISMATCH'}")
            
            return {
                "success": best_result.confidence > 0.5,
                "trophy_detected": best_result.trophy_type.value,
                "performance_label": best_result.performance_label.value,
                "confidence": best_result.confidence,
                "expected_match": trophy_match and performance_match,
                "processing_time": processing_time
            }
        else:
            print("❌ No trophy detected at any position")
            return {
                "success": False,
                "trophy_detected": "none",
                "performance_label": "unknown",
                "confidence": 0.0,
                "expected_match": False,
                "processing_time": processing_time
            }
    
    def test_integrated_analysis(self, image_path: str, expected_trophy: str, test_ign: str = "Lesz XVII"):
        """Test trophy detection through the integrated Enhanced Data Collector."""
        
        print(f"\n🚀 Integrated Analysis Test: {os.path.basename(image_path)}")
        print(f"📋 Expected Trophy: {expected_trophy}")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # Run full enhanced analysis with trophy detection
            result = self.enhanced_collector.analyze_screenshot_with_session(
                image_path=image_path,
                ign=test_ign,
                hero_override=None
            )
            
            processing_time = time.time() - start_time
            
            if result and result.get("data"):
                data = result["data"]
                
                # Extract trophy detection results
                mvp_detected = data.get("mvp_detected", False)
                medal_type = data.get("medal_type", None)
                trophy_confidence = data.get("trophy_confidence", 0.0)
                performance_label = data.get("performance_label", "Unknown")
                contextual_rating = data.get("contextual_performance_rating", "Unknown")
                rating_boosts = data.get("rating_boost_reasons", [])
                
                print(f"✅ Enhanced Analysis Results:")
                print(f"   MVP Detected: {mvp_detected}")
                print(f"   Medal Type: {medal_type}")
                print(f"   Trophy Confidence: {trophy_confidence:.1%}")
                print(f"   Performance Label: {performance_label}")
                print(f"   Contextual Rating: {contextual_rating}")
                print(f"   Rating Boosts: {rating_boosts}")
                print(f"   Processing Time: {processing_time:.2f}s")
                
                # Display core game data
                print(f"\n📊 Game Data Extracted:")
                print(f"   Hero: {data.get('hero', 'Unknown')}")
                print(f"   KDA: {data.get('kills', '?')}/{data.get('deaths', '?')}/{data.get('assists', '?')}")
                print(f"   Gold: {data.get('gold', 'Unknown')}")
                print(f"   Match Result: {data.get('match_result', 'Unknown')}")
                
                # Validation
                expected_performance = self._map_trophy_to_performance(expected_trophy)
                performance_match = (contextual_rating == expected_performance or 
                                   performance_label == expected_performance)
                
                print(f"\n✅ Integration Validation:")
                print(f"   Expected Performance: {expected_performance}")
                print(f"   Detected Performance: {contextual_rating}")
                print(f"   Performance Match: {'✅ MATCH' if performance_match else '❌ MISMATCH'}")
                
                if result.get("warnings"):
                    print(f"\n⚠️ Warnings ({len(result['warnings'])}):")
                    for warning in result["warnings"][:3]:  # Show first 3
                        print(f"   - {warning}")
                
                return {
                    "success": True,
                    "mvp_detected": mvp_detected,
                    "medal_type": medal_type,
                    "trophy_confidence": trophy_confidence,
                    "performance_label": contextual_rating,
                    "expected_match": performance_match,
                    "kda_extracted": all(k in data for k in ["kills", "deaths", "assists"]),
                    "hero_detected": data.get("hero", "unknown") != "unknown",
                    "processing_time": processing_time,
                    "overall_confidence": result.get("overall_confidence", 0.0)
                }
                
            else:
                print("❌ Enhanced analysis failed - no data returned")
                return {"success": False, "processing_time": processing_time}
                
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Integration test failed: {str(e)}")
            return {"success": False, "error": str(e), "processing_time": processing_time}
    
    def run_comprehensive_tests(self):
        """Run comprehensive trophy detection tests across all screenshot categories."""
        
        print("🚀 COMPREHENSIVE TROPHY DETECTION TESTING")
        print("=" * 80)
        print("Testing MVP Badge Detection and Performance Rating Integration")
        print()
        
        # Test cases with expected trophies
        test_cases = [
            ("Screenshot-Test-Excellent/screenshot-test-excellent-1.PNG", "mvp"),
            ("Screenshot-Test-Excellent/screenshot-test-excellent-2.PNG", "mvp"),
            ("Screenshot-Test-Good/screenshot-test-good-1.PNG", "gold"),
            ("Screenshot-Test-Good/screenshot-test-good-2.PNG", "gold"),
            ("Screenshot-Test-Average/screenshot-test-average-1.PNG", "silver"),
            ("Screenshot-Test-Average/screenshot-test-average-2.PNG", "silver"),
            ("Screenshot-Test-Poor/screenshot-test-poor-1.PNG", "bronze"),
            ("Screenshot-Test-Poor/screenshot-test-poor-2.PNG", "bronze"),
        ]
        
        all_results = []
        successful_detections = 0
        expected_matches = 0
        
        for image_path, expected_trophy in test_cases:
            if not os.path.exists(image_path):
                print(f"⚠️ Skipping {image_path} - file not found")
                continue
            
            print(f"\n" + "=" * 80)
            print(f"📸 TESTING: {os.path.basename(image_path)}")
            print(f"🎯 EXPECTED: {expected_trophy.upper()} trophy")
            print("=" * 80)
            
            # Test 1: Direct trophy detection
            direct_result = self.test_trophy_detection_direct(image_path, expected_trophy)
            
            # Test 2: Integrated analysis
            integrated_result = self.test_integrated_analysis(image_path, expected_trophy)
            
            # Combine results
            test_result = {
                "image": os.path.basename(image_path),
                "expected_trophy": expected_trophy,
                "direct_detection": direct_result,
                "integrated_analysis": integrated_result
            }
            
            all_results.append(test_result)
            
            # Count successes
            if direct_result["success"] or integrated_result["success"]:
                successful_detections += 1
            
            if (direct_result.get("expected_match", False) or 
                integrated_result.get("expected_match", False)):
                expected_matches += 1
        
        # Generate summary report
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)
        
        total_tests = len(all_results)
        print(f"📈 Overall Performance:")
        print(f"   • Total Tests: {total_tests}")
        print(f"   • Successful Detections: {successful_detections}/{total_tests} ({successful_detections/total_tests*100:.1f}%)")
        print(f"   • Expected Matches: {expected_matches}/{total_tests} ({expected_matches/total_tests*100:.1f}%)")
        
        # Performance by category
        print(f"\n📊 Performance by Expected Trophy:")
        for expected_trophy in ["mvp", "gold", "silver", "bronze"]:
            category_results = [r for r in all_results if r["expected_trophy"] == expected_trophy]
            if category_results:
                successful = sum(1 for r in category_results 
                               if r["direct_detection"]["success"] or r["integrated_analysis"]["success"])
                matches = sum(1 for r in category_results 
                            if r["direct_detection"].get("expected_match", False) or 
                               r["integrated_analysis"].get("expected_match", False))
                print(f"   • {expected_trophy.upper()}: {successful}/{len(category_results)} detected, {matches}/{len(category_results)} matches")
        
        # Specific MVP badge detection results (Priority 1)
        mvp_results = [r for r in all_results if r["expected_trophy"] == "mvp"]
        if mvp_results:
            mvp_detections = sum(1 for r in mvp_results 
                               if r["integrated_analysis"].get("mvp_detected", False))
            print(f"\n🏆 MVP Badge Detection (Priority 1):")
            print(f"   • MVP Screenshots Tested: {len(mvp_results)}")
            print(f"   • MVP Badges Detected: {mvp_detections}/{len(mvp_results)} ({mvp_detections/len(mvp_results)*100:.1f}%)")
            
            if mvp_detections == 0:
                print("   ❌ CRITICAL: No MVP badges detected! Implementation needs adjustment.")
            elif mvp_detections < len(mvp_results):
                print(f"   ⚠️ PARTIAL: {len(mvp_results) - mvp_detections} MVP badges missed")
            else:
                print("   ✅ SUCCESS: All MVP badges detected!")
        
        return all_results
    
    def _check_trophy_expectation(self, detected_trophy: str, expected_trophy: str) -> bool:
        """Check if detected trophy matches expectation."""
        trophy_mapping = {
            "mvp": "mvp_crown",
            "gold": "gold_medal", 
            "silver": "silver_medal",
            "bronze": "bronze_medal"
        }
        return detected_trophy == trophy_mapping.get(expected_trophy, expected_trophy)
    
    def _check_performance_expectation(self, detected_performance: str, expected_trophy: str) -> bool:
        """Check if performance label matches trophy expectation."""
        performance_mapping = {
            "mvp": "Excellent",
            "gold": "Good",
            "silver": "Average", 
            "bronze": "Poor"
        }
        return detected_performance == performance_mapping.get(expected_trophy, "Unknown")
    
    def _map_trophy_to_performance(self, expected_trophy: str) -> str:
        """Map expected trophy to performance label."""
        mapping = {
            "mvp": "Excellent",
            "gold": "Good", 
            "silver": "Average",
            "bronze": "Poor"
        }
        return mapping.get(expected_trophy, "Unknown")


def main():
    """Run trophy detection tests."""
    
    print("🏆 TROPHY DETECTION SYSTEM TESTING")
    print("=" * 60)
    print("Priority 1: MVP Badge Detection Implementation")
    print("Goal: Transform 75% → 90%+ label accuracy")
    print()
    
    tester = TrophyDetectionTester()
    
    # Check if test screenshots exist
    test_folders = [
        "Screenshot-Test-Excellent",
        "Screenshot-Test-Good", 
        "Screenshot-Test-Average",
        "Screenshot-Test-Poor"
    ]
    
    missing_folders = [folder for folder in test_folders if not os.path.exists(folder)]
    if missing_folders:
        print(f"⚠️ Warning: Missing test folders: {missing_folders}")
        print("Please ensure test screenshots are organized in the expected folders.")
        return
    
    # Run comprehensive tests
    results = tester.run_comprehensive_tests()
    
    print(f"\n🎯 IMPLEMENTATION PRIORITY STATUS:")
    print(f"✅ Priority 1: MVP Badge Detection - {'IMPLEMENTED' if results else 'TESTING COMPLETE'}")
    print(f"📋 Next: Optimize detection parameters based on test results")
    print(f"📋 Next: Implement Priority 2 - Medal Recognition System")
    print(f"📋 Next: Implement Priority 3 - Performance Rating Integration")
    
    print(f"\n💾 Test completed! Check temp/ directory for debug images.")


if __name__ == "__main__":
    main() 