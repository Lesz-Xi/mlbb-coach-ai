"""
Observability Dashboard
API endpoints for monitoring and observability
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from fastapi import APIRouter, Query, HTTPException

from .metrics_collector import get_metrics_collector
from .tracer import get_tracer
from .monitor import SystemMonitor

router = APIRouter(prefix="/observability", tags=["observability"])


class ObservabilityDashboard:
    """Dashboard for system observability"""
    
    def __init__(self, monitor: Optional[SystemMonitor] = None):
        self.monitor = monitor or SystemMonitor()
        self.metrics = get_metrics_collector()
        self.tracer = get_tracer()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            "timestamp": datetime.now().isoformat(),
            "system": self.monitor.get_system_status(),
            "metrics": self._get_metrics_summary(),
            "traces": self._get_traces_summary(),
            "alerts": self._get_alerts_summary()
        }
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for dashboard"""
        summary = self.metrics.get_summary()
        aggregates = self.metrics.get_aggregates()
        
        # Get key metrics
        key_metrics = {}
        important_metrics = [
            "analysis.duration",
            "ocr.duration",
            "detection.duration",
            "system.cpu.percent",
            "system.memory.percent",
            "events_published",
            "events_processed",
            "events_failed"
        ]
        
        for metric in important_metrics:
            if metric in aggregates:
                key_metrics[metric] = aggregates[metric]
        
        return {
            "total_metrics": summary["total_metrics"],
            "key_metrics": key_metrics,
            "oldest_metric": summary.get("oldest_metric"),
            "newest_metric": summary.get("newest_metric")
        }
    
    def _get_traces_summary(self) -> Dict[str, Any]:
        """Get traces summary for dashboard"""
        active_spans = self.tracer.get_active_spans()
        
        # Group by operation
        operations = {}
        for span in active_spans:
            op = span.operation_name
            if op not in operations:
                operations[op] = {
                    "count": 0,
                    "total_duration": 0,
                    "errors": 0
                }
            
            operations[op]["count"] += 1
            if span.duration:
                operations[op]["total_duration"] += span.duration
            if span.error:
                operations[op]["errors"] += 1
        
        return {
            "active_spans": len(active_spans),
            "total_traces": len(set(s.trace_id for s in active_spans)),
            "operations": operations,
            "slowest_operations": self._get_slowest_operations()
        }
    
    def _get_slowest_operations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get slowest operations from finished spans"""
        finished = sorted(
            [s for s in self.tracer.finished_spans if s.duration],
            key=lambda s: s.duration,
            reverse=True
        )[:limit]
        
        return [
            {
                "operation": span.operation_name,
                "duration_ms": span.duration,
                "trace_id": span.trace_id,
                "timestamp": (
                    datetime.fromtimestamp(span.start_time).isoformat()
                )
            }
            for span in finished
        ]
    
    def _get_alerts_summary(self) -> Dict[str, Any]:
        """Get alerts summary for dashboard"""
        active_alerts = self.monitor.get_active_alerts()
        
        return {
            "active_count": len(active_alerts),
            "by_severity": {
                "critical": len([a for a in active_alerts if a.severity == "critical"]),
                "warning": len([a for a in active_alerts if a.severity == "warning"]),
                "info": len([a for a in active_alerts if a.severity == "info"])
            },
            "recent_alerts": [
                {
                    "name": alert.name,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in active_alerts[-10:]  # Last 10 alerts
            ]
        }


# FastAPI routes
dashboard = ObservabilityDashboard()


@router.get("/dashboard")
async def get_dashboard():
    """Get dashboard data"""
    return dashboard.get_dashboard_data()


@router.get("/metrics")
async def get_metrics(
    name: Optional[str] = Query(None, description="Metric name filter"),
    start_time: Optional[datetime] = Query(None, description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time")
):
    """Get metrics with optional filtering"""
    metrics = dashboard.metrics.get_metrics(
        name=name,
        start_time=start_time,
        end_time=end_time
    )
    
    return {
        "count": len(metrics),
        "metrics": [
            {
                "name": m.name,
                "value": m.value,
                "timestamp": m.timestamp.isoformat(),
                "tags": m.tags,
                "unit": m.unit
            }
            for m in metrics
        ]
    }


@router.get("/traces/{trace_id}")
async def get_trace(trace_id: str):
    """Get a specific trace"""
    spans = dashboard.tracer.get_trace(trace_id)
    
    if not spans:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    return {
        "trace_id": trace_id,
        "spans": [span.to_dict() for span in spans],
        "total_duration": max(
            s.end_time for s in spans if s.end_time
        ) - min(s.start_time for s in spans) if spans else 0
    }


@router.get("/alerts")
async def get_alerts(
    active_only: bool = Query(True, description="Show only active alerts"),
    severity: Optional[str] = Query(None, description="Filter by severity")
):
    """Get system alerts"""
    if active_only:
        alerts = dashboard.monitor.get_active_alerts()
    else:
        alerts = dashboard.monitor.alerts
    
    if severity:
        alerts = [a for a in alerts if a.severity == severity]
    
    return {
        "count": len(alerts),
        "alerts": [
            {
                "name": alert.name,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
                "metadata": alert.metadata
            }
            for alert in alerts
        ]
    }


@router.post("/alerts/{alert_index}/resolve")
async def resolve_alert(alert_index: int):
    """Resolve an alert"""
    dashboard.monitor.resolve_alert(alert_index)
    return {"message": "Alert resolved"}


@router.get("/health")
async def health_check():
    """System health check"""
    status = dashboard.monitor.get_system_status()
    
    # Determine overall health
    health = "healthy"
    if status["alerts"]["by_severity"]["critical"] > 0:
        health = "critical"
    elif status["alerts"]["by_severity"]["warning"] > 0:
        health = "warning"
    
    return {
        "status": health,
        "timestamp": status["timestamp"],
        "details": status
    } 