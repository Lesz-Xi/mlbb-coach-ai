"""
MLBB Coach AI v2.0 - Enhanced Architecture
Demonstrates all 5 strategic improvements integrated together
"""

import asyncio
from pathlib import Path
from datetime import datetime

# Import new architectural components
from core.services import AnalysisOrchestrator, AnalysisRequest
from core.cache import cache_result, get_metrics_collector
from core.ml_versioning import VersionManager, ModelVersion, ModelMetrics
from core.events import EventBus, Event, EventType, on_event
from core.observability import (
    setup_logging, get_logger, SystemMonitor,
    trace_method, get_tracer
)

# Setup logging first
setup_logging(level="INFO", structured=False, colored=True)
logger = get_logger(__name__)


class MLBBCoachAI:
    """Enhanced MLBB Coach AI with all architectural improvements"""
    
    def __init__(self):
        # Initialize components
        self.orchestrator = AnalysisOrchestrator()
        self.event_bus = EventBus()
        self.version_manager = VersionManager(
            Path("models"), Path("models/archive")
        )
        self.system_monitor = SystemMonitor()
        
        # Start event bus
        asyncio.create_task(self.event_bus.start())
        
        # Register event handlers
        self._register_event_handlers()
        
        # Load models
        self._load_models()
        
        logger.info("MLBB Coach AI v2.0 initialized successfully")
    
    def _register_event_handlers(self):
        """Register event handlers for real-time updates"""
        
        @on_event(EventType.ANALYSIS_STARTED)
        async def on_analysis_started(event: Event):
            logger.info(f"Analysis started: {event.data}")
            # Could send websocket update here
        
        @on_event(EventType.ANALYSIS_COMPLETED)
        async def on_analysis_completed(event: Event):
            logger.info(f"Analysis completed: {event.data}")
            # Record metrics
            get_metrics_collector().increment("analysis.completed")
        
        @on_event(EventType.ANALYSIS_FAILED)
        async def on_analysis_failed(event: Event):
            logger.error(f"Analysis failed: {event.data}")
            get_metrics_collector().increment("analysis.failed")
        
        @on_event(EventType.PERFORMANCE_WARNING)
        async def on_performance_warning(event: Event):
            logger.warning(f"Performance warning: {event.data}")
    
    def _load_models(self):
        """Load ML models with versioning"""
        # Example: Register a hero detection model
        hero_model = ModelVersion(
            name="hero_detector",
            version="1.0.0",
            model_type="hero_detection",
            created_at=datetime.now(),
            created_by="system",
            model_path=Path("models/hero_detector_v1.pkl"),
            description="YOLOv5-based hero detector",
            tags=["production", "stable"],
            metrics=ModelMetrics(
                accuracy=0.95,
                precision=0.93,
                recall=0.94,
                f1_score=0.935,
                inference_time_ms=45.2,
                memory_usage_mb=256
            )
        )
        
        self.version_manager.register_model(hero_model, activate=True)
        logger.info("Models loaded successfully")
    
    @trace_method()
    @cache_result(ttl=300)  # Cache for 5 minutes
    async def analyze_screenshot(
        self,
        image_path: str,
        player_ign: str,
        hero_override: str = None
    ):
        """
        Analyze a screenshot with all enhancements
        
        This demonstrates:
        1. Service-oriented architecture (orchestrator)
        2. Caching (decorator)
        3. Event-driven updates
        4. Distributed tracing
        5. Observability (metrics)
        """
        analysis_start = asyncio.get_event_loop().time()
        
        # Emit start event
        await self.event_bus.publish(Event(
            event_type=EventType.ANALYSIS_STARTED,
            data={
                "image_path": image_path,
                "player_ign": player_ign,
                "timestamp": datetime.now().isoformat()
            }
        ))
        
        try:
            # Create analysis request
            request = AnalysisRequest(
                image_path=image_path,
                ign=player_ign,
                hero_override=hero_override,
                session_id=str(get_tracer().spans.get("trace_id", "unknown"))
            )
            
            # Process through orchestrator (parallel processing)
            results = await self.orchestrator.analyze([request])
            
            if not results or not results[0].success:
                raise Exception("Analysis failed")
            
            # Record metrics
            analysis_time = (
                asyncio.get_event_loop().time() - analysis_start
            ) * 1000
            get_metrics_collector().timing(
                "analysis.duration", 
                analysis_time
            )
            
            # Check performance
            if analysis_time > 1000:  # 1 second threshold
                await self.event_bus.publish(Event(
                    event_type=EventType.PERFORMANCE_WARNING,
                    data={
                        "operation": "analyze_screenshot",
                        "duration_ms": analysis_time,
                        "threshold_ms": 1000
                    }
                ))
            
            # Emit completion event
            await self.event_bus.publish(Event(
                event_type=EventType.ANALYSIS_COMPLETED,
                data={
                    "results": results[0].data,
                    "duration_ms": analysis_time
                }
            ))
            
            return results[0]
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            
            # Emit failure event
            await self.event_bus.publish(Event(
                event_type=EventType.ANALYSIS_FAILED,
                data={
                    "error": str(e),
                    "image_path": image_path
                }
            ))
            
            raise
    
    async def get_system_health(self):
        """Get comprehensive system health metrics"""
        return {
            "services": await self.orchestrator.get_service_health(),
            "models": self.version_manager.get_metrics_summary(),
            "cache": cache_result.cache_stats(),
            "events": self.event_bus.get_metrics(),
            "traces": {
                "active_spans": len(get_tracer().get_active_spans()),
                "finished_spans": len(get_tracer().finished_spans)
            },
            "system": get_metrics_collector().get_summary()
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.event_bus.stop()
        await self.orchestrator.cleanup()


# Example usage
async def main():
    """Example usage of enhanced MLBB Coach AI"""
    coach = MLBBCoachAI()
    
    try:
        # Example analysis
        result = await coach.analyze_screenshot(
            "screenshots/example.png",
            "PlayerName",
            hero_override="Chou"
        )
        
        logger.info(f"Analysis result: {result}")
        
        # Get system health
        health = await coach.get_system_health()
        logger.info(f"System health: {health}")
        
        # Demonstrate model versioning
        # Roll back to previous version if needed
        # coach.version_manager.rollback_model("hero_detection")
        
        # Demonstrate A/B testing
        # hero_model_v2 = ModelVersion(...)
        # coach.version_manager.register_model(hero_model_v2)
        # Run A/B test between v1 and v2
        
    finally:
        await coach.cleanup()


if __name__ == "__main__":
    asyncio.run(main()) 