"""
Team Behavior Service for MLBB Coach AI
=======================================

This service provides team-level behavioral analysis by integrating with the
existing service-oriented architecture. It orchestrates team behavior analysis,
synergy matrix building, and generates comprehensive team insights.

Key Features:
- Team behavior analysis orchestration
- Player compatibility matrix generation
- Team synergy assessment
- Coordination pattern analysis
- Role overlap detection
- Collective feedback generation
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

from .base_service import BaseService
from ..analytics.team_behavior_analyzer import (
    TeamBehaviorAnalyzer, TeamBehaviorAnalysis, create_team_behavior_analyzer
)
from ..utils.synergy_matrix_builder import (
    SynergyMatrixBuilder, TeamSynergyMatrix, create_synergy_matrix_builder
)
from ..behavioral_modeling import BehavioralFingerprint
from ..cache.hybrid_cache import HybridCache
from ..events.event_bus import EventBus
from ..observability.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class TeamBehaviorService(BaseService):
    """
    Service orchestrator for team behavior analysis operations.
    
    Provides high-level interface for team behavioral analysis while maintaining
    compatibility with existing SOA patterns and performance requirements.
    """
    
    def __init__(
        self,
        cache_manager: Optional[HybridCache] = None,
        event_bus: Optional[EventBus] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize the team behavior service.
        
        Args:
            cache_manager: Cache manager for performance optimization
            event_bus: Event bus for monitoring and notifications
            metrics_collector: Metrics collector for performance monitoring
        """
        super().__init__(
            cache_manager=cache_manager,
            event_bus=event_bus,
            metrics_collector=metrics_collector
        )
        
        self.team_analyzer = None
        self.synergy_builder = None
        
        # Service configuration
        self.service_name = "team_behavior"
        self.version = "1.0.0"
        
        # Performance tracking
        self.analysis_metrics = {
            "total_team_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "avg_processing_time": 0.0,
            "cache_hit_rate": 0.0
        }
    
    async def initialize(self):
        """Initialize the team behavior service."""
        try:
            # Initialize team analyzer
            self.team_analyzer = await create_team_behavior_analyzer(
                self.cache_manager
            )
            
            # Initialize synergy matrix builder
            self.synergy_builder = create_synergy_matrix_builder()
            
            # Register event handlers
            await self._register_event_handlers()
            
            # Initialize metrics
            await self._initialize_metrics()
            
            logger.info(f"Team Behavior Service v{self.version} initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize team behavior service: {str(e)}")
            raise
    
    async def analyze_team_behavior(
        self,
        match_data: List[Dict[str, Any]],
        behavioral_profiles: Optional[List[BehavioralFingerprint]] = None,
        match_id: str = "unknown",
        team_id: str = "team_1",
        include_synergy_matrix: bool = True,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze team behavior patterns for a 5-player squad.
        
        Args:
            match_data: List of match data for all 5 players
            behavioral_profiles: Optional pre-computed behavioral profiles
            match_id: Match identifier
            team_id: Team identifier
            include_synergy_matrix: Whether to include synergy matrix analysis
            force_refresh: Force refresh cache
            
        Returns:
            Dictionary containing comprehensive team behavior analysis
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            await self._validate_team_inputs(match_data, team_id)
            
            # Check cache first
            if not force_refresh:
                cached_result = await self._get_cached_team_analysis(
                    match_id, team_id
                )
                if cached_result:
                    await self._update_cache_metrics(True)
                    return cached_result
            
            # Perform team behavior analysis
            team_analysis = await self.team_analyzer.analyze_team_behavior(
                match_data=match_data,
                behavioral_profiles=behavioral_profiles,
                match_id=match_id,
                team_id=team_id
            )
            
            # Build synergy matrix if requested
            synergy_matrix = None
            if include_synergy_matrix and behavioral_profiles:
                synergy_matrix = self.synergy_builder.build_synergy_matrix(
                    behavioral_profiles=behavioral_profiles,
                    team_id=team_id
                )
            
            # Generate comprehensive result
            result = await self._generate_team_analysis_result(
                team_analysis, synergy_matrix
            )
            
            # Cache the result
            await self._cache_team_analysis_result(
                match_id, team_id, result
            )
            
            # Update metrics
            processing_time = time.time() - start_time
            await self._update_analysis_metrics(processing_time, True)
            
            # Emit events
            await self._emit_team_analysis_event(
                "team_analysis_completed",
                team_id,
                match_id,
                processing_time,
                result
            )
            
            return result
            
        except Exception as e:
            # Update metrics
            processing_time = time.time() - start_time
            await self._update_analysis_metrics(processing_time, False)
            
            # Emit error event
            await self._emit_team_analysis_event(
                "team_analysis_failed",
                team_id,
                match_id,
                processing_time,
                error=str(e)
            )
            
            logger.error(f"Team behavior analysis failed: {str(e)}")
            raise
    
    async def build_compatibility_matrix(
        self,
        behavioral_profiles: List[BehavioralFingerprint],
        team_id: str = "team_1"
    ) -> TeamSynergyMatrix:
        """
        Build player compatibility matrix.
        
        Args:
            behavioral_profiles: List of behavioral profiles
            team_id: Team identifier
            
        Returns:
            TeamSynergyMatrix with compatibility analysis
        """
        try:
            if not self.synergy_builder:
                await self.initialize()
            
            synergy_matrix = self.synergy_builder.build_synergy_matrix(
                behavioral_profiles=behavioral_profiles,
                team_id=team_id
            )
            
            return synergy_matrix
            
        except Exception as e:
            logger.error(f"Failed to build compatibility matrix: {str(e)}")
            raise
    
    async def get_team_insights(
        self,
        match_id: str,
        team_id: str,
        insight_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Get team insights for a specific match.
        
        Args:
            match_id: Match identifier
            team_id: Team identifier
            insight_type: Type of insights to retrieve
            
        Returns:
            Dictionary containing team insights
        """
        try:
            # Check cache for existing analysis
            cached_analysis = await self._get_cached_team_analysis(
                match_id, team_id
            )
            
            if not cached_analysis:
                return {
                    "error": "No analysis found for this team and match",
                    "match_id": match_id,
                    "team_id": team_id
                }
            
            # Extract insights based on type
            if insight_type == "comprehensive":
                return self._extract_comprehensive_insights(cached_analysis)
            elif insight_type == "synergy":
                return self._extract_synergy_insights(cached_analysis)
            elif insight_type == "coordination":
                return self._extract_coordination_insights(cached_analysis)
            elif insight_type == "summary":
                return self._extract_summary_insights(cached_analysis)
            else:
                return cached_analysis
                
        except Exception as e:
            logger.error(f"Failed to get team insights: {str(e)}")
            raise
    
    async def compare_team_compositions(
        self,
        team_analyses: List[Dict[str, Any]],
        comparison_metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple team compositions.
        
        Args:
            team_analyses: List of team analysis results
            comparison_metrics: Optional metrics to compare
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            if len(team_analyses) < 2:
                raise ValueError("Need at least 2 team analyses for comparison")
            
            # Extract comparison metrics
            if not comparison_metrics:
                comparison_metrics = [
                    "overall_synergy_score",
                    "team_coordination_score",
                    "team_effectiveness_rating"
                ]
            
            comparison_results = {
                "teams_compared": len(team_analyses),
                "comparison_metrics": comparison_metrics,
                "team_rankings": [],
                "metric_analysis": {},
                "recommendations": []
            }
            
            # Analyze each metric
            for metric in comparison_metrics:
                metric_values = []
                for analysis in team_analyses:
                    team_behavior = analysis.get("team_behavior", {})
                    value = team_behavior.get(metric, 0)
                    metric_values.append({
                        "team_id": analysis.get("team_id", "unknown"),
                        "value": value
                    })
                
                # Sort by value
                metric_values.sort(key=lambda x: x["value"], reverse=True)
                comparison_results["metric_analysis"][metric] = metric_values
            
            # Generate overall rankings
            team_scores = {}
            for analysis in team_analyses:
                team_id = analysis.get("team_id", "unknown")
                team_behavior = analysis.get("team_behavior", {})
                
                # Calculate composite score
                composite_score = 0
                for metric in comparison_metrics:
                    composite_score += team_behavior.get(metric, 0)
                
                team_scores[team_id] = composite_score / len(comparison_metrics)
            
            # Sort teams by composite score
            sorted_teams = sorted(
                team_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            comparison_results["team_rankings"] = [
                {"team_id": team_id, "composite_score": score}
                for team_id, score in sorted_teams
            ]
            
            # Generate recommendations
            comparison_results["recommendations"] = self._generate_comparison_recommendations(
                team_analyses, comparison_results
            )
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Failed to compare team compositions: {str(e)}")
            raise
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get service health status."""
        try:
            cache_status = await self._get_cache_status()
            
            return {
                "service_name": self.service_name,
                "version": self.version,
                "status": "healthy",
                "initialized": self.team_analyzer is not None,
                "cache_status": cache_status,
                "metrics": self.analysis_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "service_name": self.service_name,
                "version": self.version,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # Private helper methods
    async def _validate_team_inputs(
        self, match_data: List[Dict[str, Any]], team_id: str
    ):
        """Validate team analysis inputs."""
        if not match_data:
            raise ValueError("Match data is required")
        
        if len(match_data) != 5:
            raise ValueError(f"Expected 5 players, got {len(match_data)}")
        
        if not team_id:
            raise ValueError("Team ID is required")
        
        # Validate each player's data
        for i, player_data in enumerate(match_data):
            if not isinstance(player_data, dict):
                raise ValueError(f"Player {i} data must be a dictionary")
            
            required_fields = ["kills", "deaths", "assists", "hero"]
            for field in required_fields:
                if field not in player_data:
                    raise ValueError(f"Player {i} missing required field: {field}")
    
    async def _get_cached_team_analysis(
        self, match_id: str, team_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached team analysis result."""
        if not self.cache_manager:
            return None
        
        cache_key = f"team_analysis:{match_id}:{team_id}"
        return await self.cache_manager.get(cache_key)
    
    async def _cache_team_analysis_result(
        self, match_id: str, team_id: str, result: Dict[str, Any]
    ):
        """Cache team analysis result."""
        if not self.cache_manager:
            return
        
        cache_key = f"team_analysis:{match_id}:{team_id}"
        await self.cache_manager.set(cache_key, result, ttl=3600)  # 1 hour
    
    async def _generate_team_analysis_result(
        self,
        team_analysis: TeamBehaviorAnalysis,
        synergy_matrix: Optional[TeamSynergyMatrix] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive team analysis result."""
        result = {
            "match_id": team_analysis.match_id,
            "team_id": team_analysis.team_id,
            "analysis_timestamp": team_analysis.analysis_timestamp.isoformat(),
            "team_behavior": team_analysis.to_dict(),
            "formatted_output": self._format_team_analysis_output(team_analysis)
        }
        
        if synergy_matrix:
            result["synergy_matrix"] = synergy_matrix.to_dict()
            result["formatted_output"]["synergy_analysis"] = self._format_synergy_output(synergy_matrix)
        
        return result
    
    def _format_team_analysis_output(
        self, team_analysis: TeamBehaviorAnalysis
    ) -> Dict[str, Any]:
        """Format team analysis for user-friendly output."""
        return {
            "team_overview": {
                "synergy_level": team_analysis.synergy_level.value.title(),
                "coordination_pattern": team_analysis.coordination_pattern.value.replace("_", " ").title(),
                "role_overlap_severity": team_analysis.role_overlap_severity.value.replace("_", " ").title(),
                "overall_effectiveness": f"{team_analysis.team_effectiveness_rating:.1%}"
            },
            "key_insights": [
                {
                    "category": insight.category,
                    "insight": insight.insight,
                    "severity": insight.severity.title(),
                    "recommendations": insight.suggestions[:2]  # Top 2 suggestions
                }
                for insight in team_analysis.team_insights[:5]  # Top 5 insights
            ],
            "performance_metrics": {
                "teamfight_spacing": f"{team_analysis.teamfight_spacing.spacing_score:.1%}",
                "objective_control": f"{team_analysis.objective_control.objective_timing_score:.1%}",
                "rotation_sync": f"{team_analysis.rotation_sync.rotation_efficiency:.1%}",
                "role_synergy": f"{team_analysis.role_analysis.synergy_score:.1%}"
            },
            "player_compatibility": [
                {
                    "players": f"{comp.player_a} & {comp.player_b}",
                    "compatibility": f"{comp.compatibility_score:.1%}",
                    "top_synergy": comp.synergy_factors[0] if comp.synergy_factors else "None",
                    "top_conflict": comp.conflict_factors[0] if comp.conflict_factors else "None"
                }
                for comp in team_analysis.compatibility_matrix[:3]  # Top 3 pairs
            ],
            "collective_feedback": team_analysis.collective_feedback
        }
    
    def _format_synergy_output(
        self, synergy_matrix: TeamSynergyMatrix
    ) -> Dict[str, Any]:
        """Format synergy matrix for user-friendly output."""
        return {
            "overall_synergy": f"{synergy_matrix.overall_team_synergy:.1%}",
            "strongest_pairs": [
                {
                    "players": f"{pair[0]} & {pair[1]}",
                    "synergy_score": f"{pair[2]:.1%}"
                }
                for pair in synergy_matrix.strongest_pairs
            ],
            "weakest_pairs": [
                {
                    "players": f"{pair[0]} & {pair[1]}",
                    "synergy_score": f"{pair[2]:.1%}"
                }
                for pair in synergy_matrix.weakest_pairs
            ],
            "synergy_distribution": synergy_matrix.synergy_distribution,
            "recommendations": synergy_matrix.team_recommendations
        }
    
    def _extract_comprehensive_insights(
        self, analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract comprehensive insights from analysis."""
        return {
            "team_overview": analysis["formatted_output"]["team_overview"],
            "key_insights": analysis["formatted_output"]["key_insights"],
            "performance_metrics": analysis["formatted_output"]["performance_metrics"],
            "player_compatibility": analysis["formatted_output"]["player_compatibility"],
            "collective_feedback": analysis["formatted_output"]["collective_feedback"]
        }
    
    def _extract_synergy_insights(
        self, analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract synergy-specific insights."""
        synergy_analysis = analysis.get("formatted_output", {}).get("synergy_analysis", {})
        return {
            "synergy_overview": synergy_analysis,
            "player_compatibility": analysis["formatted_output"]["player_compatibility"]
        }
    
    def _extract_coordination_insights(
        self, analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract coordination-specific insights."""
        return {
            "coordination_pattern": analysis["formatted_output"]["team_overview"]["coordination_pattern"],
            "performance_metrics": analysis["formatted_output"]["performance_metrics"],
            "coordination_insights": [
                insight for insight in analysis["formatted_output"]["key_insights"]
                if insight["category"] in ["Team Coordination", "Role Distribution"]
            ]
        }
    
    def _extract_summary_insights(
        self, analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract summary insights."""
        return {
            "team_overview": analysis["formatted_output"]["team_overview"],
            "top_insight": analysis["formatted_output"]["key_insights"][0] if analysis["formatted_output"]["key_insights"] else None,
            "collective_feedback": analysis["formatted_output"]["collective_feedback"]
        }
    
    def _generate_comparison_recommendations(
        self, team_analyses: List[Dict[str, Any]], comparison_results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on team comparison."""
        recommendations = []
        
        # Top team recommendations
        top_team = comparison_results["team_rankings"][0]
        recommendations.append(f"Team {top_team['team_id']} shows the strongest overall composition")
        
        # Metric-specific recommendations
        for metric, values in comparison_results["metric_analysis"].items():
            weakest_team = values[-1]
            if weakest_team["value"] < 0.5:
                recommendations.append(f"Team {weakest_team['team_id']} needs improvement in {metric.replace('_', ' ')}")
        
        # General recommendations
        recommendations.append("Focus on improving synergy and coordination for better team performance")
        
        return recommendations
    
    async def _update_analysis_metrics(
        self, processing_time: float, success: bool
    ):
        """Update analysis metrics."""
        self.analysis_metrics["total_team_analyses"] += 1
        
        if success:
            self.analysis_metrics["successful_analyses"] += 1
        else:
            self.analysis_metrics["failed_analyses"] += 1
        
        # Update average processing time
        total_analyses = self.analysis_metrics["total_team_analyses"]
        current_avg = self.analysis_metrics["avg_processing_time"]
        self.analysis_metrics["avg_processing_time"] = (
            (current_avg * (total_analyses - 1) + processing_time) / total_analyses
        )
    
    async def _update_cache_metrics(self, cache_hit: bool):
        """Update cache metrics."""
        # Simple cache hit rate calculation
        if cache_hit:
            self.analysis_metrics["cache_hit_rate"] = min(
                self.analysis_metrics["cache_hit_rate"] + 0.1, 1.0
            )
        else:
            self.analysis_metrics["cache_hit_rate"] = max(
                self.analysis_metrics["cache_hit_rate"] - 0.05, 0.0
            )
    
    async def _emit_team_analysis_event(
        self,
        event_type: str,
        team_id: str,
        match_id: str,
        processing_time: float,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Emit team analysis event."""
        if not self.event_bus:
            return
        
        event_data = {
            "service": self.service_name,
            "event_type": event_type,
            "team_id": team_id,
            "match_id": match_id,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        if result:
            event_data["result_summary"] = {
                "synergy_level": result.get("team_behavior", {}).get("synergy_level"),
                "coordination_pattern": result.get("team_behavior", {}).get("coordination_pattern"),
                "team_effectiveness": result.get("team_behavior", {}).get("team_effectiveness_rating")
            }
        
        if error:
            event_data["error"] = error
        
        await self.event_bus.emit(event_type, event_data)
    
    async def _get_cache_status(self) -> Dict[str, Any]:
        """Get cache status."""
        if not self.cache_manager:
            return {"status": "disabled"}
        
        try:
            # Test cache connectivity
            await self.cache_manager.set("health_check", "ok", ttl=60)
            test_result = await self.cache_manager.get("health_check")
            
            return {
                "status": "healthy" if test_result == "ok" else "unhealthy",
                "hit_rate": f"{self.analysis_metrics['cache_hit_rate']:.1%}"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _register_event_handlers(self):
        """Register event handlers."""
        if not self.event_bus:
            return
        
        # Register for relevant events
        await self.event_bus.subscribe("team_analysis_requested", self._handle_analysis_request)
    
    async def _handle_analysis_request(self, event_data: Dict[str, Any]):
        """Handle analysis request event."""
        logger.info(f"Received team analysis request: {event_data}")
    
    async def _initialize_metrics(self):
        """Initialize metrics collection."""
        if not self.metrics_collector:
            return
        
        # Register service metrics
        await self.metrics_collector.register_metric(
            "team_analyses_total",
            "Total number of team analyses performed"
        )
        
        await self.metrics_collector.register_metric(
            "team_analysis_duration",
            "Duration of team analysis operations"
        )


async def create_team_behavior_service(
    cache_manager: Optional[HybridCache] = None,
    event_bus: Optional[EventBus] = None,
    metrics_collector: Optional[MetricsCollector] = None
) -> TeamBehaviorService:
    """Create and initialize a team behavior service."""
    service = TeamBehaviorService(
        cache_manager=cache_manager,
        event_bus=event_bus,
        metrics_collector=metrics_collector
    )
    
    await service.initialize()
    return service 