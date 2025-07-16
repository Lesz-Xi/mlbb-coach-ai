"""
Event Types and Base Event Class
Defines all event types in the system
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import uuid


class EventType(Enum):
    """All event types in the system"""
    
    # Analysis events
    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_PROGRESS = "analysis.progress"
    ANALYSIS_COMPLETED = "analysis.completed"
    ANALYSIS_FAILED = "analysis.failed"
    
    # OCR events
    OCR_STARTED = "ocr.started"
    OCR_COMPLETED = "ocr.completed"
    OCR_FAILED = "ocr.failed"
    
    # Detection events
    HERO_DETECTED = "detection.hero"
    TROPHY_DETECTED = "detection.trophy"
    DETECTION_FAILED = "detection.failed"
    
    # Model events
    MODEL_LOADED = "model.loaded"
    MODEL_UPDATED = "model.updated"
    MODEL_FAILED = "model.failed"
    
    # Cache events
    CACHE_HIT = "cache.hit"
    CACHE_MISS = "cache.miss"
    CACHE_EXPIRED = "cache.expired"
    
    # Performance events
    PERFORMANCE_WARNING = "performance.warning"
    PERFORMANCE_CRITICAL = "performance.critical"
    
    # User events
    USER_CONNECTED = "user.connected"
    USER_DISCONNECTED = "user.disconnected"
    USER_ACTION = "user.action"
    
    # System events
    SYSTEM_READY = "system.ready"
    SYSTEM_ERROR = "system.error"
    SYSTEM_SHUTDOWN = "system.shutdown"


@dataclass
class Event:
    """Base event class"""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary"""
        return cls(
            event_type=EventType(data["event_type"]),
            data=data["data"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_id=data["event_id"],
            source=data.get("source"),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {})
        )


# Specific event classes
@dataclass
class AnalysisEvent(Event):
    """Analysis-specific event"""
    def __init__(self, event_type: EventType, **kwargs):
        if not event_type.value.startswith("analysis."):
            raise ValueError("Invalid event type for AnalysisEvent")
        super().__init__(event_type=event_type, **kwargs)


@dataclass
class EvaluationEvent(Event):
    """Evaluation-specific event for hero evaluations"""
    def __init__(
        self,
        type: str,
        role: Optional[str] = None,
        hero: Optional[str] = None,
        player_ign: Optional[str] = None,
        score: Optional[float] = None,
        performance_rating: Optional[str] = None,
        evaluation_mode: Optional[str] = None,
        error: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
        timestamp: Optional[str] = None,
        **kwargs
    ):
        """Initialize evaluation event with specific fields"""
        data = {
            "type": type,
            "role": role,
            "hero": hero,
            "player_ign": player_ign,
            "score": score,
            "performance_rating": performance_rating,
            "evaluation_mode": evaluation_mode,
            "error": error,
            "processing_time_ms": processing_time_ms,
            "timestamp": timestamp or datetime.now().isoformat()
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        data.update(kwargs.get("data", {}))
        kwargs["data"] = data
        
        # Determine appropriate event type based on the type parameter
        if type.endswith("_error"):
            event_type = EventType.SYSTEM_ERROR
        elif type.endswith("_start"):
            event_type = EventType.ANALYSIS_STARTED
        elif type.endswith("_complete"):
            event_type = EventType.ANALYSIS_COMPLETED
        else:
            event_type = EventType.USER_ACTION
            
        super().__init__(event_type=event_type, **kwargs)


@dataclass
class ProgressEvent(Event):
    """Progress tracking event"""
    def __init__(self, progress: float, message: str, **kwargs):
        data = {
            "progress": progress,
            "message": message
        }
        data.update(kwargs.get("data", {}))
        kwargs["data"] = data
        super().__init__(
            event_type=EventType.ANALYSIS_PROGRESS,
            **kwargs
        )


@dataclass
class ErrorEvent(Event):
    """Error event"""
    def __init__(
        self,
        error_type: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        data = {
            "error_type": error_type,
            "error_message": error_message,
            "error_details": error_details or {}
        }
        kwargs["data"] = data
        super().__init__(
            event_type=EventType.SYSTEM_ERROR,
            **kwargs
        )


@dataclass
class PerformanceEvent(Event):
    """Performance monitoring event"""
    def __init__(
        self,
        metric_name: str,
        metric_value: float,
        threshold: float,
        severity: str = "warning",
        **kwargs
    ):
        event_type = (
            EventType.PERFORMANCE_CRITICAL 
            if severity == "critical" 
            else EventType.PERFORMANCE_WARNING
        )
        
        data = {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "threshold": threshold,
            "severity": severity,
            "exceeded_by": metric_value - threshold
        }
        kwargs["data"] = data
        super().__init__(event_type=event_type, **kwargs) 