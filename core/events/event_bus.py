"""
Event Bus Implementation
Central hub for event-driven communication
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional
from collections import defaultdict
import time

from .event_types import Event, EventType

logger = logging.getLogger(__name__)


class EventBus:
    """Asynchronous event bus for pub/sub communication"""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._async_subscribers: Dict[EventType, List[Callable]] = (
            defaultdict(list)
        )
        self._middleware: List[Callable] = []
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.metrics = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "processing_times": defaultdict(list)
        }
    
    def subscribe(
        self,
        event_type: EventType,
        handler: Callable,
        priority: int = 0
    ):
        """Subscribe to an event type"""
        if asyncio.iscoroutinefunction(handler):
            self._async_subscribers[event_type].append((priority, handler))
            self._async_subscribers[event_type].sort(
                key=lambda x: x[0], reverse=True
            )
        else:
            self._subscribers[event_type].append((priority, handler))
            self._subscribers[event_type].sort(
                key=lambda x: x[0], reverse=True
            )
        
        logger.debug(f"Subscribed {handler.__name__} to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Unsubscribe from an event type"""
        # Remove from sync subscribers
        self._subscribers[event_type] = [
            (p, h) for p, h in self._subscribers[event_type]
            if h != handler
        ]
        
        # Remove from async subscribers
        self._async_subscribers[event_type] = [
            (p, h) for p, h in self._async_subscribers[event_type]
            if h != handler
        ]
        
        logger.debug(
            f"Unsubscribed {handler.__name__} from {event_type.value}"
        )
    
    def on(self, event_type: EventType, priority: int = 0):
        """Decorator for subscribing to events"""
        def decorator(func: Callable) -> Callable:
            self.subscribe(event_type, func, priority)
            return func
        return decorator
    
    async def publish(self, event: Event):
        """Publish an event"""
        # Apply middleware
        for middleware in self._middleware:
            event = await self._apply_middleware(middleware, event)
            if event is None:
                return  # Event filtered out
        
        # Add to queue
        await self._event_queue.put(event)
        self.metrics["events_published"] += 1
        
        logger.debug(f"Published event: {event.event_type.value}")
    
    def publish_sync(self, event: Event):
        """Synchronous publish (creates task)"""
        asyncio.create_task(self.publish(event))
    
    async def _apply_middleware(
        self, middleware: Callable, event: Event
    ) -> Optional[Event]:
        """Apply middleware to event"""
        if asyncio.iscoroutinefunction(middleware):
            return await middleware(event)
        else:
            return middleware(event)
    
    async def start(self):
        """Start the event bus worker"""
        if self._running:
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")
    
    async def stop(self):
        """Stop the event bus worker"""
        self._running = False
        
        if self._worker_task:
            await self._event_queue.put(None)  # Sentinel
            await self._worker_task
        
        logger.info("Event bus stopped")
    
    async def _process_events(self):
        """Process events from the queue"""
        while self._running:
            try:
                event = await self._event_queue.get()
                
                if event is None:  # Sentinel
                    break
                
                start_time = time.time()
                await self._dispatch_event(event)
                
                # Track metrics
                processing_time = time.time() - start_time
                self.metrics["events_processed"] += 1
                self.metrics["processing_times"][event.event_type].append(
                    processing_time
                )
                
            except Exception as e:
                logger.error(f"Error processing event: {str(e)}")
                self.metrics["events_failed"] += 1
    
    async def _dispatch_event(self, event: Event):
        """Dispatch event to all subscribers"""
        # Get all subscribers for this event type
        sync_handlers = self._subscribers.get(event.event_type, [])
        async_handlers = self._async_subscribers.get(event.event_type, [])
        
        # Execute sync handlers
        for priority, handler in sync_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    f"Error in sync handler {handler.__name__}: {str(e)}"
                )
        
        # Execute async handlers
        if async_handlers:
            tasks = []
            for priority, handler in async_handlers:
                task = asyncio.create_task(
                    self._call_async_handler(handler, event)
                )
                tasks.append(task)
            
            # Wait for all async handlers
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _call_async_handler(self, handler: Callable, event: Event):
        """Call async handler with error handling"""
        try:
            await handler(event)
        except Exception as e:
            logger.error(
                f"Error in async handler {handler.__name__}: {str(e)}"
            )
    
    def add_middleware(self, middleware: Callable):
        """Add middleware to process events"""
        self._middleware.append(middleware)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics"""
        avg_processing_times = {}
        
        for event_type, times in self.metrics["processing_times"].items():
            if times:
                avg_processing_times[event_type.value] = {
                    "average": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "count": len(times)
                }
        
        return {
            "events_published": self.metrics["events_published"],
            "events_processed": self.metrics["events_processed"],
            "events_failed": self.metrics["events_failed"],
            "queue_size": self._event_queue.qsize(),
            "processing_times": avg_processing_times,
            "subscriber_count": sum(
                len(handlers) for handlers in self._subscribers.values()
            ) + sum(
                len(handlers) for handlers in self._async_subscribers.values()
            )
        }
    
    def clear_metrics(self):
        """Clear metrics"""
        self.metrics["events_published"] = 0
        self.metrics["events_processed"] = 0
        self.metrics["events_failed"] = 0
        self.metrics["processing_times"].clear()


# Global event bus instance
_global_event_bus = EventBus()


def get_event_bus() -> EventBus:
    """Get the global event bus instance"""
    return _global_event_bus


# Convenience decorators
def on_event(event_type: EventType, priority: int = 0):
    """Decorator to subscribe to events on global bus"""
    return _global_event_bus.on(event_type, priority)


async def publish_event(event: Event):
    """Publish event to global bus"""
    await _global_event_bus.publish(event)


# Event filtering middleware
class EventFilter:
    """Middleware for filtering events"""
    
    def __init__(self):
        self.filters: List[Callable[[Event], bool]] = []
    
    def add_filter(self, filter_func: Callable[[Event], bool]):
        """Add a filter function"""
        self.filters.append(filter_func)
    
    def __call__(self, event: Event) -> Optional[Event]:
        """Apply filters to event"""
        for filter_func in self.filters:
            if not filter_func(event):
                return None  # Filter out event
        return event


# Event logging middleware
class EventLogger:
    """Middleware for logging events"""
    
    def __init__(self, log_level: int = logging.DEBUG):
        self.log_level = log_level
    
    def __call__(self, event: Event) -> Event:
        """Log event"""
        logger.log(
            self.log_level,
            f"Event: {event.event_type.value} - {event.event_id}"
        )
        return event 