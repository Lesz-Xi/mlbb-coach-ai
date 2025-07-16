"""
Distributed Tracing
Provides request tracing across services
"""

import time
import uuid
import logging
from typing import Dict, Any, Optional, Callable, List
from contextvars import ContextVar
from dataclasses import dataclass, field
import functools
import asyncio

logger = logging.getLogger(__name__)

# Context variables for trace propagation
trace_context: ContextVar[Optional['TraceContext']] = ContextVar(
    'trace_context', default=None
)


@dataclass
class TraceContext:
    """Trace context for distributed tracing"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)


@dataclass
class Span:
    """Represents a span in a trace"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "in_progress"
    error: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get span duration in milliseconds"""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None
    
    def set_tag(self, key: str, value: Any):
        """Set a tag on the span"""
        self.tags[key] = value
    
    def log(self, message: str, **kwargs):
        """Add a log entry to the span"""
        self.logs.append({
            "timestamp": time.time(),
            "message": message,
            **kwargs
        })
    
    def finish(self, error: Optional[str] = None):
        """Finish the span"""
        self.end_time = time.time()
        self.status = "error" if error else "success"
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary"""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "tags": self.tags,
            "logs": self.logs,
            "status": self.status,
            "error": self.error
        }


class Tracer:
    """Distributed tracer for tracking operations"""
    
    def __init__(self):
        self.spans: Dict[str, Span] = {}
        self.finished_spans: List[Span] = []
        self.max_finished_spans = 10000
    
    def start_span(
        self,
        operation_name: str,
        child_of: Optional[Span] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Start a new span"""
        # Get or create trace context
        ctx = trace_context.get()
        
        if child_of:
            trace_id = child_of.trace_id
            parent_span_id = child_of.span_id
        elif ctx:
            trace_id = ctx.trace_id
            parent_span_id = ctx.span_id
        else:
            trace_id = str(uuid.uuid4())
            parent_span_id = None
        
        span_id = str(uuid.uuid4())
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time(),
            tags=tags or {}
        )
        
        self.spans[span_id] = span
        
        # Update context
        new_ctx = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id
        )
        trace_context.set(new_ctx)
        
        logger.debug(f"Started span: {operation_name} ({span_id})")
        
        return span
    
    def finish_span(self, span: Span, error: Optional[str] = None):
        """Finish a span"""
        span.finish(error)
        
        # Move to finished spans
        if span.span_id in self.spans:
            del self.spans[span.span_id]
        
        self.finished_spans.append(span)
        
        # Limit finished spans
        if len(self.finished_spans) > self.max_finished_spans:
            self.finished_spans = (
                self.finished_spans[-self.max_finished_spans:]
            )
        
        logger.debug(
            f"Finished span: {span.operation_name} ({span.span_id}) "
            f"- Duration: {span.duration}ms"
        )
    
    def get_active_spans(self) -> List[Span]:
        """Get all active spans"""
        return list(self.spans.values())
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace"""
        trace_spans = []
        
        # Check active spans
        for span in self.spans.values():
            if span.trace_id == trace_id:
                trace_spans.append(span)
        
        # Check finished spans
        for span in self.finished_spans:
            if span.trace_id == trace_id:
                trace_spans.append(span)
        
        # Sort by start time
        trace_spans.sort(key=lambda s: s.start_time)
        
        return trace_spans
    
    def clear(self):
        """Clear all spans"""
        self.spans.clear()
        self.finished_spans.clear()


# Global tracer instance
_global_tracer = Tracer()


def get_tracer() -> Tracer:
    """Get the global tracer instance"""
    return _global_tracer


# Decorator for tracing functions
def trace_method(
    operation_name: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None
):
    """Decorator to trace a method"""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                span = _global_tracer.start_span(op_name, tags=tags)
                
                try:
                    # Add function arguments as tags
                    span.set_tag("args", str(args))
                    span.set_tag("kwargs", str(kwargs))
                    
                    result = await func(*args, **kwargs)
                    
                    _global_tracer.finish_span(span)
                    return result
                    
                except Exception as e:
                    span.set_tag("error", True)
                    span.log(f"Error: {str(e)}")
                    _global_tracer.finish_span(span, error=str(e))
                    raise
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                span = _global_tracer.start_span(op_name, tags=tags)
                
                try:
                    # Add function arguments as tags
                    span.set_tag("args", str(args))
                    span.set_tag("kwargs", str(kwargs))
                    
                    result = func(*args, **kwargs)
                    
                    _global_tracer.finish_span(span)
                    return result
                    
                except Exception as e:
                    span.set_tag("error", True)
                    span.log(f"Error: {str(e)}")
                    _global_tracer.finish_span(span, error=str(e))
                    raise
            
            return sync_wrapper
    
    return decorator


class SpanContext:
    """Context manager for spans"""
    
    def __init__(
        self,
        operation_name: str,
        tracer: Optional[Tracer] = None,
        tags: Optional[Dict[str, Any]] = None
    ):
        self.operation_name = operation_name
        self.tracer = tracer or _global_tracer
        self.tags = tags
        self.span = None
    
    def __enter__(self) -> Span:
        self.span = self.tracer.start_span(
            self.operation_name,
            tags=self.tags
        )
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span is None:
            return False
            
        if exc_type:
            self.tracer.finish_span(
                self.span,
                error=f"{exc_type.__name__}: {exc_val}"
            )
        else:
            self.tracer.finish_span(self.span)
        
        return False  # Don't suppress exceptions


# Convenience function for creating span context
def trace_span(
    operation_name: str,
    tags: Optional[Dict[str, Any]] = None
) -> SpanContext:
    """Create a span context"""
    return SpanContext(operation_name, tags=tags) 