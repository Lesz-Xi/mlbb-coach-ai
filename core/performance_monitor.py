"""
Performance Monitoring and Benchmarking System

This module provides comprehensive performance tracking for the MLBB Coach AI
optimization implementations, including timing analysis, memory usage tracking,
and comparative benchmarking.
"""

import time
import threading
import statistics
import psutil
import logging
from typing import Dict, List, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Data class for storing performance metrics."""
    operation_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success: bool
    confidence: float
    timestamp: float
    optimization_flags: List[str] = field(default_factory=list)
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Data class for benchmark comparison results."""
    operation: str
    baseline_time: float
    optimized_time: float
    improvement_percent: float
    baseline_memory: float
    optimized_memory: float
    memory_improvement_percent: float
    success_rate_baseline: float
    success_rate_optimized: float
    sample_size: int
    timestamp: float


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    Features:
    - Real-time performance tracking
    - Memory and CPU usage monitoring
    - Comparative benchmarking
    - Performance analytics and reporting
    - Thread-safe operation
    """
    
    def __init__(self, enable_detailed_monitoring: bool = True):
        self.enable_detailed_monitoring = enable_detailed_monitoring
        self.metrics_history: List[PerformanceMetrics] = []
        self.benchmark_results: List[BenchmarkResult] = []
        self.active_operations: Dict[str, float] = {}
        self.operation_stats: Dict[str, Dict[str, List[float]]] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance targets (can be adjusted based on requirements)
        self.performance_targets = {
            'trophy_detection': 3.0,    # seconds
            'ocr_analysis': 2.0,        # seconds
            'hero_detection': 1.5,      # seconds
            'image_preprocessing': 0.5,  # seconds
            'overall_analysis': 8.0     # seconds (down from 4-9s original)
        }
        
        logger.info("🔍 Performance Monitor initialized")
        if enable_detailed_monitoring:
            logger.info("📊 Detailed monitoring enabled")
    
    @contextmanager
    def monitor_operation(
        self, 
        operation_name: str, 
        optimization_flags: List[str] = None,
        target_confidence: float = 0.0
    ):
        """Context manager for monitoring operation performance."""
        
        optimization_flags = optimization_flags or []
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_usage()
        
        operation_id = f"{operation_name}_{int(start_time * 1000)}"
        
        try:
            with self._lock:
                self.active_operations[operation_id] = start_time
            
            yield operation_id
            
            # Operation completed successfully
            end_time = time.time()
            execution_time = end_time - start_time
            end_memory = self._get_memory_usage()
            end_cpu = self._get_cpu_usage()
            
            # Calculate metrics
            memory_usage = max(end_memory - start_memory, 0)
            avg_cpu = (start_cpu + end_cpu) / 2
            
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=avg_cpu,
                success=True,
                confidence=target_confidence,
                timestamp=end_time,
                optimization_flags=optimization_flags
            )
            
            self._record_metrics(metrics)
            
            # Check against targets
            target_time = self.performance_targets.get(operation_name)
            if target_time and execution_time > target_time:
                logger.warning(
                    f"⚠️ {operation_name} exceeded target: {execution_time:.3f}s > {target_time:.3f}s"
                )
            else:
                logger.debug(
                    f"✅ {operation_name} completed in {execution_time:.3f}s"
                )
            
        except Exception as e:
            # Operation failed
            end_time = time.time()
            execution_time = end_time - start_time
            
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                execution_time=execution_time,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                success=False,
                confidence=0.0,
                timestamp=end_time,
                optimization_flags=optimization_flags,
                additional_data={"error": str(e)}
            )
            
            self._record_metrics(metrics)
            
            logger.error(f"❌ {operation_name} failed after {execution_time:.3f}s: {str(e)}")
            raise
        
        finally:
            with self._lock:
                self.active_operations.pop(operation_id, None)
    
    def record_operation_result(
        self,
        operation_name: str,
        execution_time: float,
        success: bool,
        confidence: float = 0.0,
        optimization_flags: List[str] = None,
        additional_data: Dict[str, Any] = None
    ):
        """Manually record operation results."""
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            execution_time=execution_time,
            memory_usage_mb=0,  # Not measured for manual recording
            cpu_usage_percent=0,
            success=success,
            confidence=confidence,
            timestamp=time.time(),
            optimization_flags=optimization_flags or [],
            additional_data=additional_data or {}
        )
        
        self._record_metrics(metrics)
    
    def benchmark_operation(
        self,
        baseline_func: Callable,
        optimized_func: Callable,
        operation_name: str,
        test_iterations: int = 10,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """Compare baseline vs optimized operation performance."""
        
        logger.info(f"🏁 Starting benchmark: {operation_name} ({test_iterations} iterations)")
        
        # Run baseline tests
        baseline_times = []
        baseline_memory = []
        baseline_successes = []
        
        for i in range(test_iterations):
            try:
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                result = baseline_func(*args, **kwargs)
                
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                baseline_times.append(end_time - start_time)
                baseline_memory.append(max(end_memory - start_memory, 0))
                baseline_successes.append(self._is_successful_result(result))
                
            except Exception as e:
                logger.warning(f"Baseline iteration {i+1} failed: {str(e)}")
                baseline_successes.append(False)
        
        # Run optimized tests
        optimized_times = []
        optimized_memory = []
        optimized_successes = []
        
        for i in range(test_iterations):
            try:
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                result = optimized_func(*args, **kwargs)
                
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                optimized_times.append(end_time - start_time)
                optimized_memory.append(max(end_memory - start_memory, 0))
                optimized_successes.append(self._is_successful_result(result))
                
            except Exception as e:
                logger.warning(f"Optimized iteration {i+1} failed: {str(e)}")
                optimized_successes.append(False)
        
        # Calculate results
        baseline_avg_time = statistics.mean(baseline_times) if baseline_times else 0
        optimized_avg_time = statistics.mean(optimized_times) if optimized_times else 0
        
        baseline_avg_memory = statistics.mean(baseline_memory) if baseline_memory else 0
        optimized_avg_memory = statistics.mean(optimized_memory) if optimized_memory else 0
        
        time_improvement = ((baseline_avg_time - optimized_avg_time) / baseline_avg_time * 100) if baseline_avg_time > 0 else 0
        memory_improvement = ((baseline_avg_memory - optimized_avg_memory) / baseline_avg_memory * 100) if baseline_avg_memory > 0 else 0
        
        baseline_success_rate = sum(baseline_successes) / len(baseline_successes) if baseline_successes else 0
        optimized_success_rate = sum(optimized_successes) / len(optimized_successes) if optimized_successes else 0
        
        benchmark = BenchmarkResult(
            operation=operation_name,
            baseline_time=baseline_avg_time,
            optimized_time=optimized_avg_time,
            improvement_percent=time_improvement,
            baseline_memory=baseline_avg_memory,
            optimized_memory=optimized_avg_memory,
            memory_improvement_percent=memory_improvement,
            success_rate_baseline=baseline_success_rate,
            success_rate_optimized=optimized_success_rate,
            sample_size=test_iterations,
            timestamp=time.time()
        )
        
        with self._lock:
            self.benchmark_results.append(benchmark)
        
        logger.info(f"📈 Benchmark complete: {operation_name}")
        logger.info(f"   Time improvement: {time_improvement:+.1f}%")
        logger.info(f"   Memory improvement: {memory_improvement:+.1f}%")
        logger.info(f"   Success rate: {baseline_success_rate:.1%} → {optimized_success_rate:.1%}")
        
        return benchmark
    
    def get_performance_summary(self, operation_name: str = None) -> Dict[str, Any]:
        """Get performance summary for specific operation or all operations."""
        
        with self._lock:
            if operation_name:
                relevant_metrics = [m for m in self.metrics_history if m.operation_name == operation_name]
            else:
                relevant_metrics = self.metrics_history
        
        if not relevant_metrics:
            return {"message": "No metrics available"}
        
        # Calculate statistics
        execution_times = [m.execution_time for m in relevant_metrics if m.success]
        memory_usage = [m.memory_usage_mb for m in relevant_metrics if m.success and m.memory_usage_mb > 0]
        confidences = [m.confidence for m in relevant_metrics if m.success and m.confidence > 0]
        
        success_rate = sum(1 for m in relevant_metrics if m.success) / len(relevant_metrics)
        
        summary = {
            "operation": operation_name or "all_operations",
            "total_executions": len(relevant_metrics),
            "success_rate": success_rate,
            "performance_stats": {}
        }
        
        if execution_times:
            summary["performance_stats"]["execution_time"] = {
                "avg": statistics.mean(execution_times),
                "min": min(execution_times),
                "max": max(execution_times),
                "median": statistics.median(execution_times),
                "std_dev": statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            }
        
        if memory_usage:
            summary["performance_stats"]["memory_usage_mb"] = {
                "avg": statistics.mean(memory_usage),
                "min": min(memory_usage),
                "max": max(memory_usage)
            }
        
        if confidences:
            summary["performance_stats"]["confidence"] = {
                "avg": statistics.mean(confidences),
                "min": min(confidences),
                "max": max(confidences)
            }
        
        # Performance targets comparison
        if operation_name and operation_name in self.performance_targets:
            target = self.performance_targets[operation_name]
            if execution_times:
                avg_time = statistics.mean(execution_times)
                summary["target_comparison"] = {
                    "target_time": target,
                    "actual_avg_time": avg_time,
                    "meets_target": avg_time <= target,
                    "improvement_needed": max(0, avg_time - target)
                }
        
        return summary
    
    def get_optimization_impact(self) -> Dict[str, Any]:
        """Analyze the impact of different optimizations."""
        
        with self._lock:
            metrics_copy = self.metrics_history.copy()
        
        optimization_impact = {}
        
        # Group by optimization flags
        for metrics in metrics_copy:
            for flag in metrics.optimization_flags:
                if flag not in optimization_impact:
                    optimization_impact[flag] = {
                        "usage_count": 0,
                        "avg_time": [],
                        "success_rate": []
                    }
                
                optimization_impact[flag]["usage_count"] += 1
                if metrics.success:
                    optimization_impact[flag]["avg_time"].append(metrics.execution_time)
                    optimization_impact[flag]["success_rate"].append(1)
                else:
                    optimization_impact[flag]["success_rate"].append(0)
        
        # Calculate impact statistics
        for flag, data in optimization_impact.items():
            if data["avg_time"]:
                data["avg_execution_time"] = statistics.mean(data["avg_time"])
            if data["success_rate"]:
                data["success_rate"] = statistics.mean(data["success_rate"])
            else:
                data["success_rate"] = 0
        
        return optimization_impact
    
    def export_metrics(self, filename: str = None) -> str:
        """Export performance metrics to JSON file."""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_metrics_{timestamp}.json"
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "performance_summary": self.get_performance_summary(),
            "optimization_impact": self.get_optimization_impact(),
            "benchmark_results": [
                {
                    "operation": b.operation,
                    "baseline_time": b.baseline_time,
                    "optimized_time": b.optimized_time,
                    "improvement_percent": b.improvement_percent,
                    "success_rate_baseline": b.success_rate_baseline,
                    "success_rate_optimized": b.success_rate_optimized,
                    "sample_size": b.sample_size
                }
                for b in self.benchmark_results
            ],
            "performance_targets": self.performance_targets,
            "total_operations": len(self.metrics_history)
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"📊 Performance metrics exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {str(e)}")
            return ""
    
    def _record_metrics(self, metrics: PerformanceMetrics):
        """Thread-safe method to record metrics."""
        
        with self._lock:
            self.metrics_history.append(metrics)
            
            # Maintain operation statistics
            op_name = metrics.operation_name
            if op_name not in self.operation_stats:
                self.operation_stats[op_name] = {
                    "times": [],
                    "successes": [],
                    "confidences": []
                }
            
            self.operation_stats[op_name]["times"].append(metrics.execution_time)
            self.operation_stats[op_name]["successes"].append(metrics.success)
            if metrics.confidence > 0:
                self.operation_stats[op_name]["confidences"].append(metrics.confidence)
            
            # Limit history size to prevent memory issues
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-800:]  # Keep most recent 800
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0
    
    def _is_successful_result(self, result: Any) -> bool:
        """Determine if operation result indicates success."""
        if hasattr(result, 'confidence'):
            return result.confidence > 0.3
        elif hasattr(result, 'success'):
            return result.success
        elif isinstance(result, dict):
            return result.get('success', True) and result.get('confidence', 0) > 0.3
        else:
            return result is not None


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# Decorator for easy performance monitoring
def monitor_performance(operation_name: str, optimization_flags: List[str] = None):
    """Decorator for automatic performance monitoring."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with performance_monitor.monitor_operation(operation_name, optimization_flags):
                return func(*args, **kwargs)
        return wrapper
    return decorator 