"""
Event Handler Base Class
"""

from abc import ABC, abstractmethod
from .event_types import Event


class EventHandler(ABC):
    """Base class for event handlers"""
    
    @abstractmethod
    async def handle(self, event: Event):
        """Handle an event"""
        pass 