"""
Performance Optimization Validation Script

Comprehensive testing suite to validate all implemented optimizations:
1. Early termination optimization
2. OCR reader caching
3. Smart search region optimization
4. Parallel processing
5. Confidence-based strategy selection
6. Performance monitoring
7. Image preprocessing optimization

This script measures actual performance improvements and validates functionality.
"""

import time
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import statistics
import json

# Add the skillshift-ai directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from core.performance_monitor import performance_monitor
    from core.optimized_image_processor import optimized_processor
    from core.trophy_medal_detector_v2 import ImprovedTrophyMedalDetector
    from core.enhanced_data_collector import enhanced_data_collector
    from core.data_collector import get_ocr_stats, reset_ocr_reader
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('performance_optimization_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class PerformanceOptimizationValidator:
    """
    Comprehensive validation system for performance optimizations.
    
    Tests all implemented optimizations and measures actual performance gains.
    """
    
    def __init__(self):
        self.test_results = {}
        self.baseline_times = {}
        self.optimized_times = {}
        self.test_screenshots = []
        
        # Performance targets (based on original 4-9s → <3s goal)
        self.performance_targets = {
            'overall_analysis': 3.0,     # Main target: under 3 seconds
            'trophy_detection': 2.0,     # Trophy detection optimizations
            'ocr_analysis': 1.5,         # OCR caching benefits
            'image_preprocessing': 0.5,  # Preprocessing optimization
            'hero_detection': 1.0        # Hero detection improvements
        }
        
        logger.info("🧪 Performance Optimization Validator initialized")
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests and return comprehensive results."""
        
        logger.info("🚀 Starting Comprehensive Performance Optimization Validation")
        
        # Phase 1: Setup and Discovery
        logger.info("📋 Phase 1: Test Environment Setup")
        self._setup_test_environment()
        
        # Phase 2: Individual Component Testing
        logger.info("🔧 Phase 2: Individual Component Validation")
        self._test_individual_components()
        
        # Phase 3: Integration Testing
        logger.info("🔗 Phase 3: Integration Performance Testing")
        self._test_integration_performance()
        
        # Phase 4: Optimization Impact Analysis
        logger.info("📊 Phase 4: Optimization Impact Analysis")
        self._analyze_optimization_impact()
        
        # Phase 5: Generate Report
        logger.info("📄 Phase 5: Generating Comprehensive Report")
        final_report = self._generate_final_report()
        
        logger.info("✅ Comprehensive validation completed!")
        return final_report
    
    def _setup_test_environment(self):
        """Setup test environment and discover test files."""
        
        # Discover test screenshots
        test_directories = [
            "Screenshot-Test-Excellent",
            "Screenshot-Test-Good", 
            "Screenshot-Test-Average",
            "Screenshot-Test-Poor"
        ]
        
        for test_dir in test_directories:
            if os.path.exists(test_dir):
                for file in os.listdir(test_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.test_screenshots.append(os.path.join(test_dir, file))
        
        logger.info(f"🔍 Found {len(self.test_screenshots)} test screenshots")
        
        # Reset caches and stats for clean testing
        reset_ocr_reader()
        optimized_processor.clear_cache()
        performance_monitor.clear_cache() if hasattr(performance_monitor, 'clear_cache') else None
        
        logger.info("🧹 Test environment prepared")
    
    def _test_individual_components(self):
        """Test individual optimization components."""
        
        # Test 1: OCR Reader Caching
        logger.info("🧪 Testing OCR Reader Caching...")
        self._test_ocr_caching()
        
        # Test 2: Image Preprocessing Optimization
        logger.info("🧪 Testing Image Preprocessing Optimization...")
        self._test_preprocessing_optimization()
        
        # Test 3: Trophy Detection Early Termination
        logger.info("🧪 Testing Trophy Detection Early Termination...")
        self._test_trophy_detection_optimization()
        
        # Test 4: Performance Monitor
        logger.info("🧪 Testing Performance Monitoring System...")
        self._test_performance_monitoring()
    
    def _test_ocr_caching(self):
        """Test OCR reader caching performance."""
        
                 if not self.test_screenshots:
             logger.warning("No test screenshots available for OCR caching test")
             return
        
        # First call (cache miss)
        start_time = time.time()
        from core.data_collector import get_ocr_reader
        reader1 = get_ocr_reader()
        first_call_time = time.time() - start_time
        
        # Second call (cache hit)  
        start_time = time.time()
        reader2 = get_ocr_reader()
        second_call_time = time.time() - start_time
        
        # Verify same instance
        same_instance = reader1 is reader2
        
        # Get OCR stats
        ocr_stats = get_ocr_stats()
        
        self.test_results['ocr_caching'] = {
            'first_call_time': first_call_time,
            'second_call_time': second_call_time,
            'speedup_factor': first_call_time / max(second_call_time, 0.001),
            'same_instance': same_instance,
            'ocr_stats': ocr_stats,
            'passed': same_instance and second_call_time < first_call_time * 0.1
        }
        
        logger.info(f"📈 OCR Caching: {first_call_time:.3f}s → {second_call_time:.3f}s (speedup: {self.test_results['ocr_caching']['speedup_factor']:.1f}x)")
    
    def _test_preprocessing_optimization(self):
        """Test image preprocessing optimization."""
        
        if not self.test_screenshots:
            logger.warning("No test screenshots available for preprocessing test")
            return
        
        test_screenshot = self.test_screenshots[0]
        
        # Test cache miss (first processing)
        start_time = time.time()
        result1 = optimized_processor.process_image(test_screenshot, debug_mode=True)
        first_process_time = time.time() - start_time
        
        # Test cache hit (second processing)
        start_time = time.time()
        result2 = optimized_processor.process_image(test_screenshot, debug_mode=True)
        second_process_time = time.time() - start_time
        
        # Get processor stats
        processor_stats = optimized_processor.get_performance_stats()
        
        self.test_results['preprocessing_optimization'] = {
            'first_process_time': first_process_time,
            'second_process_time': second_process_time,
            'cache_hit_rate': processor_stats['cache_stats']['hit_rate'],
            'quality_score': result1.quality_metrics.overall_score,
            'processing_method': result1.processing_method,
            'speedup_factor': first_process_time / max(second_process_time, 0.001),
            'processor_stats': processor_stats,
            'passed': second_process_time < first_process_time * 0.2
        }
        
        logger.info(f"📈 Preprocessing: {first_process_time:.3f}s → {second_process_time:.3f}s (speedup: {self.test_results['preprocessing_optimization']['speedup_factor']:.1f}x)")
    
    def _test_trophy_detection_optimization(self):
        """Test trophy detection optimizations."""
        
        if not self.test_screenshots:
            logger.warning("No test screenshots available for trophy detection test")
            return
        
        detector = ImprovedTrophyMedalDetector()
        
        # Test on multiple screenshots to measure average performance
        detection_times = []
        early_terminations = 0
        
        for screenshot in self.test_screenshots[:3]:  # Test first 3 screenshots
            try:
                start_time = time.time()
                result = detector.detect_trophy_in_player_row(
                    screenshot, 
                    player_row_y=400,  # Approximate middle of screen
                    debug_mode=True
                )
                detection_time = time.time() - start_time
                detection_times.append(detection_time)
                
                if result.debug_info.get('early_termination', False):
                    early_terminations += 1
                    
            except Exception as e:
                logger.warning(f"Trophy detection test failed for {screenshot}: {str(e)}")
        
        if detection_times:
            avg_detection_time = statistics.mean(detection_times)
            min_detection_time = min(detection_times)
            max_detection_time = max(detection_times)
            
            self.test_results['trophy_detection_optimization'] = {
                'avg_detection_time': avg_detection_time,
                'min_detection_time': min_detection_time,
                'max_detection_time': max_detection_time,
                'early_termination_rate': early_terminations / len(detection_times),
                'tests_run': len(detection_times),
                'meets_target': avg_detection_time <= self.performance_targets['trophy_detection'],
                'passed': avg_detection_time <= self.performance_targets['trophy_detection']
            }
            
            logger.info(f"📈 Trophy Detection: avg {avg_detection_time:.3f}s, early termination: {early_terminations}/{len(detection_times)}")
        else:
            self.test_results['trophy_detection_optimization'] = {'passed': False, 'error': 'No successful detections'}
    
    def _test_performance_monitoring(self):
        """Test performance monitoring system."""
        
        # Test performance monitor context manager
        operation_name = "test_operation"
        
        with performance_monitor.monitor_operation(operation_name, ["test_optimization"]) as operation_id:
            time.sleep(0.1)  # Simulate work
            test_result = "success"
        
        # Get performance summary
        summary = performance_monitor.get_performance_summary(operation_name)
        
        self.test_results['performance_monitoring'] = {
            'operation_tracked': operation_name in str(summary),
            'summary_generated': bool(summary),
            'has_metrics': 'performance_stats' in summary,
            'passed': bool(summary) and 'performance_stats' in summary
        }
        
        logger.info(f"📈 Performance Monitoring: tracked operations, generated summary")
    
    def _test_integration_performance(self):
        """Test overall integration performance with all optimizations."""
        
        if not self.test_screenshots:
            logger.warning("No test screenshots available for integration test")
            return
        
        test_ign = "TestPlayer"
        integration_times = []
        
        # Test enhanced data collector with all optimizations
        for screenshot in self.test_screenshots[:2]:  # Test first 2 screenshots
            try:
                start_time = time.time()
                
                result = enhanced_data_collector.analyze_screenshot_with_session(
                    image_path=screenshot,
                    ign=test_ign,
                    session_id=None
                )
                
                integration_time = time.time() - start_time
                integration_times.append(integration_time)
                
                logger.info(f"Integration test: {Path(screenshot).name} processed in {integration_time:.3f}s")
                
            except Exception as e:
                logger.warning(f"Integration test failed for {screenshot}: {str(e)}")
        
        if integration_times:
            avg_integration_time = statistics.mean(integration_times)
            min_integration_time = min(integration_times)
            max_integration_time = max(integration_times)
            
            self.test_results['integration_performance'] = {
                'avg_integration_time': avg_integration_time,
                'min_integration_time': min_integration_time,
                'max_integration_time': max_integration_time,
                'tests_run': len(integration_times),
                'meets_target': avg_integration_time <= self.performance_targets['overall_analysis'],
                'target_improvement': self.performance_targets['overall_analysis'] - avg_integration_time,
                'passed': avg_integration_time <= self.performance_targets['overall_analysis']
            }
            
            logger.info(f"📈 Integration Performance: avg {avg_integration_time:.3f}s (target: {self.performance_targets['overall_analysis']:.1f}s)")
        else:
            self.test_results['integration_performance'] = {'passed': False, 'error': 'No successful integrations'}
    
    def _analyze_optimization_impact(self):
        """Analyze the impact of all optimizations combined."""
        
        # Calculate overall performance improvements
        total_improvement = 0
        improvements_count = 0
        
        optimizations = [
            'ocr_caching',
            'preprocessing_optimization', 
            'trophy_detection_optimization',
            'integration_performance'
        ]
        
        for opt in optimizations:
            if opt in self.test_results and self.test_results[opt].get('passed', False):
                improvements_count += 1
                
                # Calculate specific improvements
                if opt == 'ocr_caching':
                    speedup = self.test_results[opt].get('speedup_factor', 1)
                    total_improvement += min(speedup - 1, 5)  # Cap at 5x improvement
                elif opt == 'preprocessing_optimization':
                    speedup = self.test_results[opt].get('speedup_factor', 1)
                    total_improvement += min(speedup - 1, 10)  # Cap at 10x improvement
                elif opt == 'integration_performance':
                    target_time = self.performance_targets['overall_analysis']
                    actual_time = self.test_results[opt].get('avg_integration_time', target_time)
                    if actual_time <= target_time:
                        improvement = (target_time - actual_time) / target_time
                        total_improvement += improvement
        
        self.test_results['optimization_impact'] = {
            'total_optimizations_working': improvements_count,
            'total_optimizations_tested': len(optimizations),
            'optimization_success_rate': improvements_count / len(optimizations),
            'estimated_total_improvement': total_improvement,
            'performance_target_met': self.test_results.get('integration_performance', {}).get('passed', False),
            'passed': improvements_count >= len(optimizations) * 0.75  # 75% of optimizations working
        }
        
        logger.info(f"📊 Optimization Impact: {improvements_count}/{len(optimizations)} optimizations working")
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        
        # Calculate overall success
        passed_tests = sum(1 for test_result in self.test_results.values() 
                          if isinstance(test_result, dict) and test_result.get('passed', False))
        total_tests = len(self.test_results)
        overall_success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # Performance summary
        performance_summary = {}
        if 'integration_performance' in self.test_results:
            perf_data = self.test_results['integration_performance']
            performance_summary = {
                'target_time': self.performance_targets['overall_analysis'],
                'actual_avg_time': perf_data.get('avg_integration_time', 'N/A'),
                'target_met': perf_data.get('passed', False),
                'improvement_margin': perf_data.get('target_improvement', 0)
            }
        
        # Optimization effectiveness
        optimization_effectiveness = {}
        for opt_name, opt_data in self.test_results.items():
            if isinstance(opt_data, dict):
                optimization_effectiveness[opt_name] = {
                    'working': opt_data.get('passed', False),
                    'impact': opt_data.get('speedup_factor', 'N/A') if 'speedup_factor' in opt_data else 'N/A'
                }
        
        final_report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'overall_success_rate': overall_success_rate,
                'timestamp': time.time()
            },
            'performance_summary': performance_summary,
            'optimization_effectiveness': optimization_effectiveness,
            'detailed_results': self.test_results,
            'recommendations': self._generate_recommendations(),
            'conclusion': self._generate_conclusion(overall_success_rate, performance_summary)
        }
        
        # Save report to file
        report_filename = f"performance_optimization_report_{int(time.time())}.json"
        try:
            with open(report_filename, 'w') as f:
                json.dump(final_report, f, indent=2, default=str)
            logger.info(f"📄 Report saved to {report_filename}")
        except Exception as e:
            logger.error(f"Failed to save report: {str(e)}")
        
        return final_report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        
        recommendations = []
        
        # Check individual optimizations
        if not self.test_results.get('ocr_caching', {}).get('passed', False):
            recommendations.append("OCR caching optimization needs attention")
        
        if not self.test_results.get('preprocessing_optimization', {}).get('passed', False):
            recommendations.append("Image preprocessing optimization needs improvement")
        
        if not self.test_results.get('trophy_detection_optimization', {}).get('passed', False):
            recommendations.append("Trophy detection optimization requires tuning")
        
        # Check performance targets
        integration_perf = self.test_results.get('integration_performance', {})
        if not integration_perf.get('passed', False):
            actual_time = integration_perf.get('avg_integration_time', 'unknown')
            target_time = self.performance_targets['overall_analysis']
            recommendations.append(f"Performance target not met: {actual_time}s > {target_time}s")
        
        # Success case
        if not recommendations:
            recommendations.append("All optimizations working well! Consider monitoring for regression.")
        
        return recommendations
    
    def _generate_conclusion(self, success_rate: float, perf_summary: Dict) -> str:
        """Generate overall conclusion."""
        
        if success_rate >= 0.9 and perf_summary.get('target_met', False):
            return "🎉 EXCELLENT: All optimizations working, performance targets achieved!"
        elif success_rate >= 0.75:
            return "✅ GOOD: Most optimizations working, minor issues to address."
        elif success_rate >= 0.5:
            return "⚠️ MODERATE: Some optimizations working, significant improvements needed."
        else:
            return "❌ POOR: Major optimization issues detected, comprehensive review required."


def main():
    """Main validation execution."""
    
    print("🚀 MLBB Coach AI Performance Optimization Validation")
    print("=" * 60)
    
    validator = PerformanceOptimizationValidator()
    
    try:
        final_report = validator.run_comprehensive_validation()
        
        # Print summary
        print("\n📊 VALIDATION SUMMARY")
        print("=" * 30)
        
        test_summary = final_report['test_summary']
        print(f"Tests Run: {test_summary['passed_tests']}/{test_summary['total_tests']}")
        print(f"Success Rate: {test_summary['overall_success_rate']:.1%}")
        
        perf_summary = final_report['performance_summary']
        if perf_summary:
            print(f"Performance Target: {perf_summary['target_time']:.1f}s")
            if isinstance(perf_summary['actual_avg_time'], (int, float)):
                print(f"Actual Performance: {perf_summary['actual_avg_time']:.3f}s")
                print(f"Target Met: {'✅' if perf_summary['target_met'] else '❌'}")
        
        print(f"\n{final_report['conclusion']}")
        
        print("\n📋 RECOMMENDATIONS:")
        for rec in final_report['recommendations']:
            print(f"  • {rec}")
        
        return final_report
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        print(f"\n❌ Validation failed: {str(e)}")
        return None


if __name__ == "__main__":
    main() 