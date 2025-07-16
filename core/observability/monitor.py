"""
System Monitor
Real-time system monitoring and alerting
"""

import asyncio
import logging
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import psutil

from .metrics_collector import get_metrics_collector

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """System alert"""
    name: str
    severity: str  # info, warning, critical
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Threshold:
    """Alert threshold configuration"""
    metric_name: str
    operator: str  # >, <, >=, <=, ==
    value: float
    severity: str
    message_template: str
    cooldown_minutes: int = 5


class SystemMonitor:
    """Monitors system health and generates alerts"""
    
    def __init__(self):
        self.thresholds: List[Threshold] = []
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_alert_times: Dict[str, datetime] = {}
        
        # Default thresholds
        self._setup_default_thresholds()
    
    def _setup_default_thresholds(self):
        """Setup default monitoring thresholds"""
        self.thresholds.extend([
            Threshold(
                metric_name="system.cpu.percent",
                operator=">=",
                value=80.0,
                severity="warning",
                message_template="High CPU usage: {value}%"
            ),
            Threshold(
                metric_name="system.cpu.percent",
                operator=">=",
                value=95.0,
                severity="critical",
                message_template="Critical CPU usage: {value}%"
            ),
            Threshold(
                metric_name="system.memory.percent",
                operator=">=",
                value=85.0,
                severity="warning",
                message_template="High memory usage: {value}%"
            ),
            Threshold(
                metric_name="system.disk.percent",
                operator=">=",
                value=90.0,
                severity="warning",
                message_template="Low disk space: {value}% used"
            ),
            Threshold(
                metric_name="analysis.duration",
                operator=">=",
                value=5000.0,
                severity="warning",
                message_template="Slow analysis: {value}ms"
            ),
            Threshold(
                metric_name="events_failed",
                operator=">=",
                value=10.0,
                severity="warning",
                message_template="High event failure rate: {value} failures"
            )
        ])
    
    def add_threshold(self, threshold: Threshold):
        """Add a custom threshold"""
        self.thresholds.append(threshold)
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add a handler to be called when alerts are generated"""
        self.alert_handlers.append(handler)
    
    async def start(self, interval_seconds: int = 30):
        """Start monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(interval_seconds)
        )
        logger.info("System monitoring started")
    
    async def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("System monitoring stopped")
    
    async def _monitor_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Collect system metrics
                collector = get_metrics_collector()
                collector.collect_system_metrics()
                
                # Check thresholds
                self._check_thresholds()
                
                # Sleep until next check
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(interval)
    
    def _check_thresholds(self):
        """Check all thresholds and generate alerts"""
        collector = get_metrics_collector()
        
        for threshold in self.thresholds:
            # Get current metric value
            current_value = collector.get_current(threshold.metric_name)
            
            if current_value is None:
                continue
            
            # Check if threshold is exceeded
            if self._evaluate_threshold(current_value, threshold):
                # Check cooldown
                last_alert = self._last_alert_times.get(
                    f"{threshold.metric_name}_{threshold.severity}"
                )
                
                if last_alert:
                    cooldown = timedelta(minutes=threshold.cooldown_minutes)
                    if datetime.now() - last_alert < cooldown:
                        continue
                
                # Generate alert
                alert = Alert(
                    name=f"{threshold.metric_name}_threshold",
                    severity=threshold.severity,
                    message=threshold.message_template.format(
                        value=current_value
                    ),
                    metadata={
                        "metric_name": threshold.metric_name,
                        "threshold": threshold.value,
                        "current_value": current_value,
                        "operator": threshold.operator
                    }
                )
                
                self._generate_alert(alert)
    
    def _evaluate_threshold(
        self, value: float, threshold: Threshold
    ) -> bool:
        """Evaluate if a threshold is exceeded"""
        if threshold.operator == ">":
            return value > threshold.value
        elif threshold.operator == ">=":
            return value >= threshold.value
        elif threshold.operator == "<":
            return value < threshold.value
        elif threshold.operator == "<=":
            return value <= threshold.value
        elif threshold.operator == "==":
            return value == threshold.value
        else:
            logger.warning(f"Unknown operator: {threshold.operator}")
            return False
    
    def _generate_alert(self, alert: Alert):
        """Generate and handle an alert"""
        # Add to alerts list
        self.alerts.append(alert)
        
        # Update last alert time
        key = f"{alert.metadata.get('metric_name')}_{alert.severity}"
        self._last_alert_times[key] = alert.timestamp
        
        # Log alert
        if alert.severity == "critical":
            logger.critical(alert.message)
        elif alert.severity == "warning":
            logger.warning(alert.message)
        else:
            logger.info(alert.message)
        
        # Call handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {str(e)}")
        
        # Limit alerts list size
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
    
    def get_active_alerts(self) -> List[Alert]:
        """Get currently active (unresolved) alerts"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_index: int):
        """Mark an alert as resolved"""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index].resolved = True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "percent": cpu_percent,
                "cores": psutil.cpu_count(),
                "frequency": (
                    psutil.cpu_freq().current if psutil.cpu_freq() else None
                )
            },
            "memory": {
                "total_gb": memory.total / (1024 ** 3),
                "used_gb": memory.used / (1024 ** 3),
                "available_gb": memory.available / (1024 ** 3),
                "percent": memory.percent
            },
            "disk": {
                "total_gb": disk.total / (1024 ** 3),
                "used_gb": disk.used / (1024 ** 3),
                "free_gb": disk.free / (1024 ** 3),
                "percent": disk.percent
            },
            "alerts": {
                "total": len(self.alerts),
                "active": len(self.get_active_alerts()),
                "by_severity": {
                    "critical": len([
                        a for a in self.get_active_alerts()
                        if a.severity == "critical"
                    ]),
                    "warning": len([
                        a for a in self.get_active_alerts()
                        if a.severity == "warning"
                    ]),
                    "info": len([
                        a for a in self.get_active_alerts()
                        if a.severity == "info"
                    ])
                }
            }
        }
    
    def export_alerts(self, filepath: str):
        """Export alerts to file"""
        import json
        
        alerts_data = [
            {
                "name": alert.name,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
                "metadata": alert.metadata
            }
            for alert in self.alerts
        ]
        
        with open(filepath, "w") as f:
            json.dump(alerts_data, f, indent=2) 