"""
Metrics Collector
Collects and aggregates system metrics
"""

import time
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Single metric data point"""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


class MetricsCollector:
    """Collects and manages system metrics"""
    
    def __init__(
        self,
        retention_period: timedelta = timedelta(hours=24),
        max_metrics_per_name: int = 10000
    ):
        self.retention_period = retention_period
        self.max_metrics_per_name = max_metrics_per_name
        
        # Metrics storage
        self._metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_metrics_per_name)
        )
        
        # Aggregated metrics
        self._aggregates: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Custom collectors
        self._collectors: List[Callable] = []
        
        # System metrics
        self._last_cpu_check = 0
        self._last_memory_check = 0
    
    def record(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None
    ):
        """Record a metric"""
        metric = Metric(
            name=name,
            value=value,
            tags=tags or {},
            unit=unit
        )
        
        self._metrics[name].append(metric)
        self._update_aggregates(name, value)
        
        logger.debug(f"Recorded metric: {name}={value}")
    
    def increment(
        self, name: str, value: float = 1, 
        tags: Optional[Dict[str, str]] = None
    ):
        """Increment a counter metric"""
        current = self.get_current(name) or 0
        self.record(name, current + value, tags)
    
    def gauge(
        self, name: str, value: float, 
        tags: Optional[Dict[str, str]] = None
    ):
        """Set a gauge metric"""
        self.record(name, value, tags)
    
    def timing(
        self, name: str, duration: float, 
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a timing metric"""
        self.record(name, duration, tags, unit="ms")
    
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        return TimerContext(self, name, tags)
    
    def _update_aggregates(self, name: str, value: float):
        """Update aggregated metrics"""
        if name not in self._aggregates:
            self._aggregates[name] = {
                "count": 0,
                "sum": 0,
                "min": float('inf'),
                "max": float('-inf'),
                "last": 0
            }
        
        agg = self._aggregates[name]
        agg["count"] += 1
        agg["sum"] += value
        agg["min"] = min(agg["min"], value)
        agg["max"] = max(agg["max"], value)
        agg["last"] = value
        agg["avg"] = agg["sum"] / agg["count"]
    
    def get_current(self, name: str) -> Optional[float]:
        """Get current value of a metric"""
        if name in self._metrics and self._metrics[name]:
            return self._metrics[name][-1].value
        return None
    
    def get_metrics(
        self,
        name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Metric]:
        """Get metrics with optional filtering"""
        metrics = []
        
        # Get metric names to process
        names = [name] if name else list(self._metrics.keys())
        
        for metric_name in names:
            for metric in self._metrics[metric_name]:
                # Time filtering
                if start_time and metric.timestamp < start_time:
                    continue
                if end_time and metric.timestamp > end_time:
                    continue
                
                # Tag filtering
                if tags:
                    if not all(
                        metric.tags.get(k) == v for k, v in tags.items()
                    ):
                        continue
                
                metrics.append(metric)
        
        return metrics
    
    def get_aggregates(
        self, name: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Get aggregated metrics"""
        if name:
            return {name: self._aggregates.get(name, {})}
        return dict(self._aggregates)
    
    def collect_system_metrics(self):
        """Collect system-level metrics"""
        current_time = time.time()
        
        # CPU metrics (throttled to every 1 second)
        if current_time - self._last_cpu_check >= 1:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.gauge("system.cpu.percent", cpu_percent)
            
            # Per-core CPU
            for i, percent in enumerate(psutil.cpu_percent(percpu=True)):
                self.gauge(f"system.cpu.core{i}.percent", percent)
            
            self._last_cpu_check = current_time
        
        # Memory metrics
        if current_time - self._last_memory_check >= 5:
            memory = psutil.virtual_memory()
            self.gauge("system.memory.used", memory.used / (1024 ** 3), unit="GB")
            self.gauge("system.memory.percent", memory.percent)
            self.gauge("system.memory.available", memory.available / (1024 ** 3), unit="GB")
            
            # Swap memory
            swap = psutil.swap_memory()
            self.gauge("system.swap.used", swap.used / (1024 ** 3), unit="GB")
            self.gauge("system.swap.percent", swap.percent)
            
            self._last_memory_check = current_time
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.gauge("system.disk.used", disk.used / (1024 ** 3), unit="GB")
        self.gauge("system.disk.percent", disk.percent)
        
        # Network metrics
        net_io = psutil.net_io_counters()
        self.gauge("system.network.bytes_sent", net_io.bytes_sent)
        self.gauge("system.network.bytes_recv", net_io.bytes_recv)
    
    def add_collector(self, collector: Callable):
        """Add a custom metric collector function"""
        self._collectors.append(collector)
    
    def run_collectors(self):
        """Run all custom collectors"""
        for collector in self._collectors:
            try:
                collector(self)
            except Exception as e:
                logger.error(f"Error in custom collector: {str(e)}")
    
    def cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff_time = datetime.now() - self.retention_period
        
        for name, metrics in self._metrics.items():
            # Remove old metrics
            while metrics and metrics[0].timestamp < cutoff_time:
                metrics.popleft()
    
    def export_metrics(self, filepath: Path):
        """Export metrics to file"""
        data = {
            "exported_at": datetime.now().isoformat(),
            "metrics": {},
            "aggregates": self.get_aggregates()
        }
        
        # Convert metrics to serializable format
        for name, metrics in self._metrics.items():
            data["metrics"][name] = [
                {
                    "value": m.value,
                    "timestamp": m.timestamp.isoformat(),
                    "tags": m.tags,
                    "unit": m.unit
                }
                for m in metrics
            ]
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        total_metrics = sum(len(metrics) for metrics in self._metrics.values())
        
        return {
            "total_metrics": total_metrics,
            "metric_names": list(self._metrics.keys()),
            "aggregates": self.get_aggregates(),
            "oldest_metric": min(
                (m[0].timestamp for m in self._metrics.values() if m),
                default=None
            ),
            "newest_metric": max(
                (m[-1].timestamp for m in self._metrics.values() if m),
                default=None
            )
        }


class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(
        self,
        collector: MetricsCollector,
        name: str,
        tags: Optional[Dict[str, str]] = None
    ):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (time.time() - self.start_time) * 1000  # Convert to ms
        self.collector.timing(self.name, duration, self.tags)


# Global metrics collector
_global_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector"""
    return _global_collector


# Convenience functions
def record_metric(name: str, value: float, **kwargs):
    """Record a metric to the global collector"""
    _global_collector.record(name, value, **kwargs)


def increment_counter(name: str, value: float = 1, **kwargs):
    """Increment a counter in the global collector"""
    _global_collector.increment(name, value, **kwargs)


def set_gauge(name: str, value: float, **kwargs):
    """Set a gauge in the global collector"""
    _global_collector.gauge(name, value, **kwargs)


def time_operation(name: str, **kwargs):
    """Time an operation using the global collector"""
    return _global_collector.timer(name, **kwargs) 