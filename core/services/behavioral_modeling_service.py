"""
Behavioral Modeling Service - Service Orchestrator
==================================================

This service orchestrator provides a clean interface for behavioral modeling
that integrates with the existing SOA architecture and services.

Key Features:
- Async service patterns matching existing architecture
- Integration with caching, events, and monitoring
- Batch processing capabilities
- Error handling and diagnostics
- Performance optimization with concurrent processing
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time

from .base_service import BaseService
from ..behavioral_modeling import (
    BehavioralFingerprint, 
    create_behavioral_analyzer
)
from ..cache.hybrid_cache import HybridCache
from ..events.event_bus import EventBus
from ..observability.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class BehavioralModelingService(BaseService):
    """
    Service orchestrator for behavioral modeling operations.
    
    Provides high-level interface for behavioral analysis while maintaining
    compatibility with existing SOA patterns and performance requirements.
    """
    
    def __init__(
        self,
        cache_manager: Optional[HybridCache] = None,
        event_bus: Optional[EventBus] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        max_concurrent_analyses: int = 5,
        batch_size: int = 10
    ):
        """
        Initialize the behavioral modeling service.
        
        Args:
            cache_manager: Cache manager for performance optimization
            event_bus: Event bus for monitoring and notifications
            metrics_collector: Metrics collector for performance monitoring
            max_concurrent_analyses: Maximum concurrent behavioral analyses
            batch_size: Batch size for bulk processing
        """
        super().__init__(
            cache_manager=cache_manager,
            event_bus=event_bus,
            metrics_collector=metrics_collector
        )
        
        self.max_concurrent_analyses = max_concurrent_analyses
        self.batch_size = batch_size
        self.behavioral_analyzer = None
        self._semaphore = asyncio.Semaphore(max_concurrent_analyses)
        
        # Service configuration
        self.service_name = "behavioral_modeling"
        self.version = "1.0.0"
        
        # Performance tracking
        self.analysis_metrics = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "avg_processing_time": 0.0,
            "cache_hit_rate": 0.0
        }
    
    async def initialize(self):
        """Initialize the behavioral modeling service."""
        try:
            # Initialize behavioral analyzer
            self.behavioral_analyzer = await create_behavioral_analyzer(
                self.cache_manager
            )
            
            # Register event handlers
            await self._register_event_handlers()
            
            # Initialize metrics
            await self._initialize_metrics()
            
            logger.info(f"Behavioral Modeling Service v{self.version} initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize service: {str(e)}")
            raise
    
    async def analyze_player_behavior(
        self,
        player_id: str,
        match_history: List[Dict[str, Any]],
        video_paths: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> BehavioralFingerprint:
        """
        Analyze player behavior from match history and optional video data.
        
        Args:
            player_id: Unique player identifier
            match_history: List of match data dictionaries
            video_paths: Optional list of video file paths
            force_refresh: Force refresh of cached results
            
        Returns:
            BehavioralFingerprint with comprehensive analysis
        """
        async with self._semaphore:
            start_time = time.time()
            
            try:
                # Check cache unless force refresh
                if not force_refresh:
                    cached_result = await self._get_cached_analysis(player_id)
                    if cached_result:
                        await self._emit_analysis_event(
                            "cache_hit", player_id, time.time() - start_time
                        )
                        return cached_result
                
                # Validate inputs
                await self._validate_analysis_inputs(player_id, match_history)
                
                # Perform behavioral analysis
                fingerprint = await self.behavioral_analyzer.analyze_player_behavior(
                    player_id=player_id,
                    match_history=match_history,
                    video_paths=video_paths
                )
                
                # Cache the result
                await self._cache_analysis_result(player_id, fingerprint)
                
                # Update metrics
                processing_time = time.time() - start_time
                await self._update_analysis_metrics(processing_time, success=True)
                
                # Emit success event
                await self._emit_analysis_event(
                    "completed", player_id, processing_time, fingerprint
                )
                
                return fingerprint
                
            except Exception as e:
                processing_time = time.time() - start_time
                await self._update_analysis_metrics(processing_time, success=False)
                await self._emit_analysis_event(
                    "failed", player_id, processing_time, error=str(e)
                )
                
                logger.error(f"Analysis failed for player {player_id}: {str(e)}")
                raise
    
    async def batch_analyze_players(
        self,
        analysis_requests: List[Dict[str, Any]],
        max_concurrent: Optional[int] = None
    ) -> List[Tuple[str, Optional[BehavioralFingerprint], Optional[str]]]:
        """
        Batch analyze multiple players concurrently.
        
        Args:
            analysis_requests: List of analysis request dictionaries
            max_concurrent: Override max concurrent analyses
            
        Returns:
            List of tuples (player_id, fingerprint, error_message)
        """
        max_concurrent = max_concurrent or self.max_concurrent_analyses
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_single(request: Dict[str, Any]) -> Tuple[str, Optional[BehavioralFingerprint], Optional[str]]:
            async with semaphore:
                player_id = request["player_id"]
                try:
                    fingerprint = await self.analyze_player_behavior(
                        player_id=player_id,
                        match_history=request["match_history"],
                        video_paths=request.get("video_paths"),
                        force_refresh=request.get("force_refresh", False)
                    )
                    return player_id, fingerprint, None
                except Exception as e:
                    return player_id, None, str(e)
        
        # Process all requests concurrently
        tasks = [analyze_single(request) for request in analysis_requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions from gather
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                player_id = analysis_requests[i]["player_id"]
                final_results.append((player_id, None, str(result)))
            else:
                final_results.append(result)
        
        return final_results
    
    async def get_behavior_insights(
        self,
        player_id: str,
        insight_type: str = "summary"
    ) -> Dict[str, Any]:
        """
        Get behavioral insights for a player.
        
        Args:
            player_id: Player identifier
            insight_type: Type of insights to retrieve
            
        Returns:
            Dictionary with behavioral insights
        """
        fingerprint = await self._get_cached_analysis(player_id)
        if not fingerprint:
            raise ValueError(f"No analysis found for player {player_id}")
        
        if insight_type == "summary":
            return self._generate_summary_insights(fingerprint)
        elif insight_type == "recommendations":
            return self._generate_recommendations(fingerprint)
        elif insight_type == "comparison":
            return self._generate_comparison_insights(fingerprint)
        else:
            raise ValueError(f"Unknown insight type: {insight_type}")
    
    async def compare_players(
        self,
        player_ids: List[str],
        comparison_metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare behavioral profiles of multiple players.
        
        Args:
            player_ids: List of player identifiers
            comparison_metrics: Specific metrics to compare
            
        Returns:
            Dictionary with comparison results
        """
        # Get behavioral fingerprints for all players
        fingerprints = {}
        for player_id in player_ids:
            fingerprint = await self._get_cached_analysis(player_id)
            if fingerprint:
                fingerprints[player_id] = fingerprint
        
        if not fingerprints:
            raise ValueError("No analyses found for any specified players")
        
        # Generate comparison
        return self._generate_player_comparison(fingerprints, comparison_metrics)
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get service health status."""
        return {
            "service_name": self.service_name,
            "version": self.version,
            "status": "healthy",
            "metrics": self.analysis_metrics,
            "analyzer_initialized": self.behavioral_analyzer is not None,
            "cache_status": await self._get_cache_status(),
            "last_updated": datetime.now().isoformat()
        }
    
    async def _get_cached_analysis(self, player_id: str) -> Optional[BehavioralFingerprint]:
        """Get cached behavioral analysis."""
        try:
            cache_key = f"behavioral_analysis:{player_id}"
            cached_data = await self.cache_manager.get(cache_key)
            
            if cached_data:
                # Convert dict back to BehavioralFingerprint
                return BehavioralFingerprint(**cached_data)
            return None
            
        except Exception as e:
            logger.warning(f"Error retrieving cached analysis for {player_id}: {str(e)}")
            return None
    
    async def _cache_analysis_result(self, player_id: str, fingerprint: BehavioralFingerprint):
        """Cache behavioral analysis result."""
        try:
            cache_key = f"behavioral_analysis:{player_id}"
            await self.cache_manager.set(
                cache_key,
                fingerprint.__dict__,
                ttl=3600  # Cache for 1 hour
            )
        except Exception as e:
            logger.warning(f"Error caching analysis result for {player_id}: {str(e)}")
    
    async def _validate_analysis_inputs(self, player_id: str, match_history: List[Dict[str, Any]]):
        """Validate analysis inputs."""
        if not player_id:
            raise ValueError("Player ID is required")
        
        if not match_history:
            raise ValueError("Match history is required")
        
        if len(match_history) < 5:
            raise ValueError("Minimum 5 matches required for behavioral analysis")
        
        # Validate match data structure
        required_fields = ["kills", "deaths", "assists", "hero"]
        for i, match in enumerate(match_history):
            for field in required_fields:
                if field not in match:
                    raise ValueError(f"Missing field '{field}' in match {i}")
    
    async def _emit_analysis_event(
        self,
        event_type: str,
        player_id: str,
        processing_time: float,
        fingerprint: Optional[BehavioralFingerprint] = None,
        error: Optional[str] = None
    ):
        """Emit analysis event."""
        event_data = {
            "player_id": player_id,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        if fingerprint:
            event_data["confidence_score"] = fingerprint.confidence_score
            event_data["play_style"] = fingerprint.play_style.value
            event_data["matches_analyzed"] = fingerprint.matches_analyzed
        
        if error:
            event_data["error"] = error
        
        await self.event_bus.emit(f"behavioral_analysis_{event_type}", event_data)
    
    async def _update_analysis_metrics(self, processing_time: float, success: bool):
        """Update analysis metrics."""
        self.analysis_metrics["total_analyses"] += 1
        
        if success:
            self.analysis_metrics["successful_analyses"] += 1
        else:
            self.analysis_metrics["failed_analyses"] += 1
        
        # Update average processing time
        total_successful = self.analysis_metrics["successful_analyses"]
        if total_successful > 0:
            current_avg = self.analysis_metrics["avg_processing_time"]
            self.analysis_metrics["avg_processing_time"] = (
                (current_avg * (total_successful - 1) + processing_time) / total_successful
            )
        
        # Update cache hit rate
        cache_hits = self.analysis_metrics.get("cache_hits", 0)
        self.analysis_metrics["cache_hit_rate"] = cache_hits / self.analysis_metrics["total_analyses"]
    
    def _generate_summary_insights(self, fingerprint: BehavioralFingerprint) -> Dict[str, Any]:
        """Generate summary insights from behavioral fingerprint."""
        return {
            "player_id": fingerprint.player_id,
            "play_style": fingerprint.play_style.value,
            "risk_profile": fingerprint.risk_profile.value,
            "game_tempo": fingerprint.game_tempo.value,
            "key_strengths": fingerprint.strength_areas[:3],
            "improvement_areas": fingerprint.identified_flaws[:3],
            "confidence_score": fingerprint.confidence_score,
            "preferred_heroes": fingerprint.preferred_heroes,
            "behavioral_summary": {
                "map_awareness": fingerprint.map_awareness_score,
                "team_synergy": fingerprint.synergy_with_team,
                "mechanical_skill": fingerprint.mechanical_skill_score,
                "decision_making": fingerprint.decision_making_score
            }
        }
    
    def _generate_recommendations(self, fingerprint: BehavioralFingerprint) -> Dict[str, Any]:
        """Generate behavioral recommendations."""
        recommendations = []
        
        # Generate recommendations based on identified flaws
        for flaw in fingerprint.identified_flaws:
            if flaw == "overextending":
                recommendations.append({
                    "category": "positioning",
                    "priority": "high",
                    "suggestion": "Focus on safer positioning and map awareness",
                    "specific_actions": [
                        "Check minimap every 3-5 seconds",
                        "Maintain safe distance from enemies",
                        "Use wards for vision control"
                    ]
                })
            elif flaw == "low objective participation":
                recommendations.append({
                    "category": "strategy",
                    "priority": "medium",
                    "suggestion": "Increase participation in team objectives",
                    "specific_actions": [
                        "Rotate for turtle/lord fights",
                        "Coordinate with team before engaging",
                        "Prioritize objectives over kills"
                    ]
                })
        
        return {
            "player_id": fingerprint.player_id,
            "recommendations": recommendations,
            "priority_focus": recommendations[0]["category"] if recommendations else "general",
            "confidence": fingerprint.confidence_score
        }
    
    def _generate_comparison_insights(self, fingerprint: BehavioralFingerprint) -> Dict[str, Any]:
        """Generate comparison insights against meta patterns."""
        # This would compare against meta behavioral patterns
        return {
            "player_id": fingerprint.player_id,
            "meta_comparison": {
                "play_style_popularity": 0.3,  # Placeholder
                "risk_profile_effectiveness": 0.7,  # Placeholder
                "tempo_meta_alignment": 0.6  # Placeholder
            },
            "recommendations": "Consider adapting to more meta-aligned playstyles"
        }
    
    def _generate_player_comparison(
        self,
        fingerprints: Dict[str, BehavioralFingerprint],
        comparison_metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate comparison between multiple players."""
        if not comparison_metrics:
            comparison_metrics = [
                "map_awareness_score",
                "synergy_with_team",
                "mechanical_skill_score",
                "decision_making_score"
            ]
        
        comparison_data = {}
        
        for metric in comparison_metrics:
            comparison_data[metric] = {}
            for player_id, fingerprint in fingerprints.items():
                comparison_data[metric][player_id] = getattr(fingerprint, metric, 0.0)
        
        return {
            "players": list(fingerprints.keys()),
            "comparison_metrics": comparison_data,
            "summary": self._generate_comparison_summary(fingerprints)
        }
    
    def _generate_comparison_summary(self, fingerprints: Dict[str, BehavioralFingerprint]) -> Dict[str, Any]:
        """Generate summary of player comparisons."""
        play_styles = {}
        risk_profiles = {}
        
        for player_id, fingerprint in fingerprints.items():
            play_styles[player_id] = fingerprint.play_style.value
            risk_profiles[player_id] = fingerprint.risk_profile.value
        
        return {
            "play_styles": play_styles,
            "risk_profiles": risk_profiles,
            "most_aggressive": max(fingerprints.keys(), key=lambda p: 1 - fingerprints[p].behavioral_metrics.death_avoidance_score),
            "most_strategic": max(fingerprints.keys(), key=lambda p: fingerprints[p].behavioral_metrics.objective_participation)
        }
    
    async def _get_cache_status(self) -> Dict[str, Any]:
        """Get cache status information."""
        try:
            # This would depend on your cache implementation
            return {
                "status": "healthy",
                "hit_rate": self.analysis_metrics.get("cache_hit_rate", 0.0)
            }
        except Exception:
            return {"status": "unknown"}
    
    async def _register_event_handlers(self):
        """Register event handlers for monitoring."""
        pass  # Implement based on your event system needs
    
    async def _initialize_metrics(self):
        """Initialize metrics collection."""
        if self.metrics_collector:
            await self.metrics_collector.register_service(self.service_name, self.version)


# Factory function for service creation
async def create_behavioral_modeling_service(
    cache_manager: Optional[HybridCache] = None,
    event_bus: Optional[EventBus] = None,
    metrics_collector: Optional[MetricsCollector] = None,
    **kwargs
) -> BehavioralModelingService:
    """Create and initialize a behavioral modeling service."""
    service = BehavioralModelingService(
        cache_manager=cache_manager,
        event_bus=event_bus,
        metrics_collector=metrics_collector,
        **kwargs
    )
    
    await service.initialize()
    return service 