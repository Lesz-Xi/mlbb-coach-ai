"""
WebSocket Manager
Manages WebSocket connections for real-time updates
"""

import asyncio
import json
import logging
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass
import uuid

from .event_types import Event, EventType
from .event_bus import get_event_bus

logger = logging.getLogger(__name__)


@dataclass
class WebSocketClient:
    """Represents a connected WebSocket client"""
    client_id: str
    websocket: Any  # FastAPI WebSocket or similar
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    subscriptions: Set[EventType] = None
    
    def __post_init__(self):
        if self.subscriptions is None:
            self.subscriptions = set()


class WebSocketManager:
    """Manages WebSocket connections and event broadcasting"""
    
    def __init__(self):
        self.clients: Dict[str, WebSocketClient] = {}
        self._event_bus = get_event_bus()
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Setup handlers for broadcasting events"""
        # Subscribe to all event types for broadcasting
        for event_type in EventType:
            self._event_bus.subscribe(
                event_type,
                self._broadcast_event,
                priority=10
            )
    
    async def connect(
        self,
        websocket: Any,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """Connect a new WebSocket client"""
        client_id = str(uuid.uuid4())
        
        client = WebSocketClient(
            client_id=client_id,
            websocket=websocket,
            user_id=user_id,
            session_id=session_id
        )
        
        self.clients[client_id] = client
        
        # Send welcome message
        await self.send_to_client(client_id, {
            "type": "connection",
            "client_id": client_id,
            "message": "Connected to MLBB Coach AI"
        })
        
        logger.info(f"WebSocket client connected: {client_id}")
        
        return client_id
    
    async def disconnect(self, client_id: str):
        """Disconnect a WebSocket client"""
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"WebSocket client disconnected: {client_id}")
    
    async def subscribe(
        self,
        client_id: str,
        event_types: Set[EventType]
    ):
        """Subscribe a client to specific event types"""
        if client_id in self.clients:
            self.clients[client_id].subscriptions.update(event_types)
            
            await self.send_to_client(client_id, {
                "type": "subscription",
                "subscribed": [e.value for e in event_types],
                "message": "Subscription updated"
            })
    
    async def unsubscribe(
        self,
        client_id: str,
        event_types: Set[EventType]
    ):
        """Unsubscribe a client from event types"""
        if client_id in self.clients:
            self.clients[client_id].subscriptions.difference_update(
                event_types
            )
            
            await self.send_to_client(client_id, {
                "type": "unsubscription",
                "unsubscribed": [e.value for e in event_types],
                "message": "Subscription updated"
            })
    
    async def send_to_client(
        self,
        client_id: str,
        data: Dict[str, Any]
    ):
        """Send data to a specific client"""
        if client_id in self.clients:
            client = self.clients[client_id]
            try:
                await client.websocket.send_text(json.dumps(data))
            except Exception as e:
                logger.error(
                    f"Failed to send to client {client_id}: {str(e)}"
                )
                await self.disconnect(client_id)
    
    async def broadcast(
        self,
        data: Dict[str, Any],
        event_type: Optional[EventType] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Broadcast data to multiple clients"""
        tasks = []
        
        for client_id, client in self.clients.items():
            # Filter by event subscription
            if event_type and event_type not in client.subscriptions:
                continue
            
            # Filter by user_id
            if user_id and client.user_id != user_id:
                continue
            
            # Filter by session_id
            if session_id and client.session_id != session_id:
                continue
            
            tasks.append(self.send_to_client(client_id, data))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _broadcast_event(self, event: Event):
        """Broadcast an event to subscribed clients"""
        # Convert event to WebSocket message
        message = {
            "type": "event",
            "event": event.to_dict()
        }
        
        # Broadcast to relevant clients
        await self.broadcast(
            message,
            event_type=event.event_type,
            user_id=event.user_id,
            session_id=event.session_id
        )
    
    async def handle_client_message(
        self,
        client_id: str,
        message: str
    ):
        """Handle incoming message from client"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "subscribe":
                event_types = {
                    EventType(e) for e in data.get("events", [])
                }
                await self.subscribe(client_id, event_types)
                
            elif message_type == "unsubscribe":
                event_types = {
                    EventType(e) for e in data.get("events", [])
                }
                await self.unsubscribe(client_id, event_types)
                
            elif message_type == "ping":
                await self.send_to_client(client_id, {"type": "pong"})
                
            else:
                logger.warning(
                    f"Unknown message type from {client_id}: {message_type}"
                )
                
        except Exception as e:
            logger.error(
                f"Error handling client message: {str(e)}"
            )
            await self.send_to_client(client_id, {
                "type": "error",
                "message": "Invalid message format"
            })
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        total_clients = len(self.clients)
        
        subscriptions_by_event = {}
        for event_type in EventType:
            count = sum(
                1 for client in self.clients.values()
                if event_type in client.subscriptions
            )
            if count > 0:
                subscriptions_by_event[event_type.value] = count
        
        return {
            "total_clients": total_clients,
            "clients_by_user": len(set(
                c.user_id for c in self.clients.values()
                if c.user_id
            )),
            "subscriptions_by_event": subscriptions_by_event
        } 