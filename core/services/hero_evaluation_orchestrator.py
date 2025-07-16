"""
Hero Evaluation Orchestrator Service for MLBB Coach AI

This service coordinates the entire hero evaluation process:
- Hero role detection and mapping
- Role-specific evaluator selection
- Hero-specific override application
- Performance caching and optimization
- Event-driven feedback generation

Architecture:
- SOA microservice pattern
- Async/await for non-blocking operations
- Redis/Memory hybrid caching
- Event bus integration
- Comprehensive logging and metrics
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from ..role_evaluators import RoleEvaluatorFactory, RoleEvaluationResult
from ..cache.hybrid_cache import HybridCache
from ..events.event_bus import EventBus
from ..events.event_types import EvaluationEvent
from ..observability.metrics_collector import MetricsCollector
from ..base_evaluator import BaseEvaluator


logger = logging.getLogger(__name__)


@dataclass
class HeroEvaluationRequest:
    """Structured request for hero evaluation."""
    hero: str
    match_data: Dict[str, Any]
    player_ign: str
    match_duration: Optional[int] = None
    evaluation_mode: str = "comprehensive"  # comprehensive, quick, detailed
    cache_enabled: bool = True
    emit_events: bool = True


@dataclass
class HeroEvaluationResponse:
    """Comprehensive response from hero evaluation."""
    hero: str
    role: str
    overall_score: float
    performance_rating: str  # Poor, Average, Good, Excellent
    feedback: List[Tuple[str, str]]
    suggestions: List[str]
    role_specific_metrics: Dict[str, float]
    hero_specific_insights: Dict[str, Any]
    confidence: float
    evaluation_timestamp: str
    cache_hit: bool = False
    processing_time_ms: int = 0


class HeroEvaluationOrchestrator:
    """
    Central orchestrator for hero evaluation process.
    
    Responsibilities:
    1. Hero role detection and mapping
    2. Role-specific evaluator coordination
    3. Hero-specific override application
    4. Performance optimization and caching
    5. Event emission and metrics collection
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.cache = HybridCache(
            memory_ttl=300,  # 5 minutes
            redis_ttl=1800,  # 30 minutes
            max_memory_size=1000
        )
        self.event_bus = EventBus()
        self.metrics = MetricsCollector()
        
        # Load hero role mapping
        self._load_hero_mappings()
        
        # Performance optimization settings
        self.batch_size = self.config.get('batch_size', 10)
        self.max_concurrent = self.config.get('max_concurrent', 5)
        self.timeout_seconds = self.config.get('timeout_seconds', 30)
        
        logger.info("Hero Evaluation Orchestrator initialized")
    
    def _load_hero_mappings(self):
        """Load hero-to-role mappings from configuration."""
        try:
            # In production, this would load from the JSON file
            # For now, we'll use a subset for demonstration
            self.hero_mappings = {
                'tigreal': {'role': 'tank', 'sub_role': 'initiator'},
                'franco': {'role': 'tank', 'sub_role': 'initiator'},
                'lancelot': {'role': 'assassin', 'sub_role': 'mobility'},
                'hayabusa': {'role': 'assassin', 'sub_role': 'stealth'},
                'miya': {'role': 'marksman', 'sub_role': 'carry'},
                'layla': {'role': 'marksman', 'sub_role': 'carry'},
                'kagura': {'role': 'mage', 'sub_role': 'burst'},
                'chou': {'role': 'fighter', 'sub_role': 'assassin'},
                'estes': {'role': 'support', 'sub_role': 'healer'},
                'mathilda': {'role': 'support', 'sub_role': 'roamer'}
            }
            logger.info(f"Loaded {len(self.hero_mappings)} hero mappings")
        except Exception as e:
            logger.error(f"Failed to load hero mappings: {e}")
            self.hero_mappings = {}
    
    async def evaluate_hero(self, request: HeroEvaluationRequest) -> HeroEvaluationResponse:
        """
        Main entry point for hero evaluation.
        
        This method orchestrates the entire evaluation process:
        1. Input validation
        2. Cache checking
        3. Role detection
        4. Evaluator selection
        5. Evaluation execution
        6. Result processing
        7. Caching and event emission
        """
        start_time = datetime.now()
        
        try:
            # Input validation
            await self._validate_request(request)
            
            # Check cache first
            if request.cache_enabled:
                cached_result = await self._check_cache(request)
                if cached_result:
                    logger.debug(f"Cache hit for {request.hero}")
                    cached_result.cache_hit = True
                    return cached_result
            
            # Emit evaluation start event
            if request.emit_events:
                await self._emit_evaluation_start(request)
            
            # Detect hero role
            role_info = await self._detect_hero_role(request.hero)
            
            # Get role-specific evaluator
            evaluator = await self._get_evaluator(role_info['role'])
            
            # Execute evaluation
            evaluation_result = await self._execute_evaluation(
                evaluator, request, role_info)
            
            # Process and enhance results
            enhanced_result = await self._enhance_evaluation_result(
                evaluation_result, request, role_info)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            enhanced_result.processing_time_ms = int(processing_time)
            
            # Cache results
            if request.cache_enabled:
                await self._cache_result(request, enhanced_result)
            
            # Emit completion events
            if request.emit_events:
                await self._emit_evaluation_complete(request, enhanced_result)
            
            # Collect metrics
            await self._collect_metrics(request, enhanced_result)
            
            logger.info(f"Hero evaluation completed for {request.hero} "
                       f"in {processing_time:.2f}ms")
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Hero evaluation failed for {request.hero}: {e}")
            
            # Emit error event
            if request.emit_events:
                await self._emit_evaluation_error(request, str(e))
            
            # Return error response
            return self._create_error_response(request, str(e))
    
    async def evaluate_heroes_batch(self, 
                                    requests: List[HeroEvaluationRequest]) -> List[HeroEvaluationResponse]:
        """
        Batch evaluation for multiple heroes with performance optimization.
        """
        logger.info(f"Starting batch evaluation for {len(requests)} heroes")
        
        # Split into batches to avoid overwhelming the system
        batches = [requests[i:i + self.batch_size] 
                  for i in range(0, len(requests), self.batch_size)]
        
        all_results = []
        
        for batch in batches:
            # Process batch with concurrency limit
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def evaluate_with_semaphore(req):
                async with semaphore:
                    return await self.evaluate_hero(req)
            
            batch_tasks = [evaluate_with_semaphore(req) for req in batch]
            
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*batch_tasks),
                    timeout=self.timeout_seconds
                )
                all_results.extend(batch_results)
                
            except asyncio.TimeoutError:
                logger.error(f"Batch evaluation timeout after {self.timeout_seconds}s")
                # Add error responses for failed evaluations
                for req in batch:
                    error_response = self._create_error_response(req, "Timeout")
                    all_results.append(error_response)
        
        logger.info(f"Batch evaluation completed. {len(all_results)} results")
        return all_results
    
    async def _validate_request(self, request: HeroEvaluationRequest):
        """Validate evaluation request."""
        if not request.hero:
            raise ValueError("Hero name is required")
        
        if not request.match_data:
            raise ValueError("Match data is required")
        
        if not request.player_ign:
            raise ValueError("Player IGN is required")
        
        # Validate evaluation mode
        valid_modes = ['comprehensive', 'quick', 'detailed']
        if request.evaluation_mode not in valid_modes:
            raise ValueError(f"Invalid evaluation mode: {request.evaluation_mode}")
    
    async def _check_cache(self, request: HeroEvaluationRequest) -> Optional[HeroEvaluationResponse]:
        """Check cache for existing evaluation results."""
        cache_key = self._generate_cache_key(request)
        
        try:
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                return HeroEvaluationResponse(**cached_data)
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
        
        return None
    
    async def _detect_hero_role(self, hero_name: str) -> Dict[str, str]:
        """Detect hero role from hero mappings."""
        hero_key = hero_name.lower().replace(' ', '_').replace('-', '_')
        
        role_info = self.hero_mappings.get(hero_key)
        
        if not role_info:
            logger.warning(f"Unknown hero: {hero_name}, defaulting to fighter")
            return {'role': 'fighter', 'sub_role': 'versatile'}
        
        return role_info
    
    async def _get_evaluator(self, role: str) -> Any:
        """Get role-specific evaluator."""
        try:
            evaluator = RoleEvaluatorFactory.get_evaluator(role)
            logger.debug(f"Retrieved {role} evaluator")
            return evaluator
        except Exception as e:
            logger.error(f"Failed to get evaluator for role {role}: {e}")
            # Fallback to base evaluator
            return BaseEvaluator()
    
    async def _execute_evaluation(self, evaluator: Any, 
                                  request: HeroEvaluationRequest,
                                  role_info: Dict[str, str]) -> RoleEvaluationResult:
        """Execute the actual evaluation."""
        try:
            # Add role information to match data
            enhanced_data = request.match_data.copy()
            enhanced_data['hero'] = request.hero
            enhanced_data['role'] = role_info['role']
            enhanced_data['sub_role'] = role_info['sub_role']
            enhanced_data['player_ign'] = request.player_ign
            
            # Execute evaluation based on mode
            if request.evaluation_mode == 'quick':
                # Quick evaluation with reduced processing
                result = await evaluator._evaluate_role_specific(
                    enhanced_data, request.match_duration)
            else:
                # Full evaluation
                result = await evaluator.evaluate_async(
                    enhanced_data, request.match_duration)
            
            return result
            
        except Exception as e:
            logger.error(f"Evaluation execution failed: {e}")
            raise
    
    async def _enhance_evaluation_result(self, 
                                         result: RoleEvaluationResult,
                                         request: HeroEvaluationRequest,
                                         role_info: Dict[str, str]) -> HeroEvaluationResponse:
        """Enhance evaluation result with additional insights."""
        
        # Determine performance rating
        performance_rating = self._calculate_performance_rating(result.overall_score)
        
        # Generate hero-specific insights
        hero_insights = await self._generate_hero_insights(
            request.hero, result, request.match_data)
        
        return HeroEvaluationResponse(
            hero=request.hero,
            role=role_info['role'],
            overall_score=result.overall_score,
            performance_rating=performance_rating,
            feedback=result.feedback,
            suggestions=result.suggestions,
            role_specific_metrics=result.role_specific_metrics,
            hero_specific_insights=hero_insights,
            confidence=result.confidence,
            evaluation_timestamp=datetime.now().isoformat()
        )
    
    def _calculate_performance_rating(self, score: float) -> str:
        """Convert numerical score to performance rating."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Average"
        else:
            return "Poor"
    
    async def _generate_hero_insights(self, hero: str, 
                                      result: RoleEvaluationResult,
                                      match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hero-specific insights and recommendations."""
        insights = {}
        
        try:
            # Hero performance vs role average
            insights['role_comparison'] = {
                'above_average_metrics': [],
                'below_average_metrics': [],
                'standout_performance': []
            }
            
            # Identify standout metrics
            for metric, value in result.role_specific_metrics.items():
                if value > 0.8:
                    insights['role_comparison']['standout_performance'].append({
                        'metric': metric,
                        'value': value,
                        'description': f"Excellent {metric.replace('_', ' ')}"
                    })
            
            # Build optimization recommendations
            insights['optimization_focus'] = []
            
            # Find lowest performing metric for focus
            if result.role_specific_metrics:
                min_metric = min(result.role_specific_metrics.items(), key=lambda x: x[1])
                insights['optimization_focus'].append({
                    'priority': 'high',
                    'metric': min_metric[0],
                    'current_value': min_metric[1],
                    'target_value': min(1.0, min_metric[1] + 0.2),
                    'improvement_suggestions': [
                        f"Focus on improving {min_metric[0].replace('_', ' ')}",
                        "Review replay for missed opportunities"
                    ]
                })
            
            # Match context insights
            insights['match_context'] = {
                'duration_assessment': self._assess_match_duration(match_data),
                'team_performance_impact': self._assess_team_impact(match_data),
                'objective_control': self._assess_objective_control(match_data)
            }
            
        except Exception as e:
            logger.warning(f"Failed to generate hero insights: {e}")
            insights['error'] = "Insights generation failed"
        
        return insights
    
    def _assess_match_duration(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess performance relative to match duration."""
        duration = match_data.get('match_duration', 15)
        
        if duration < 10:
            return {
                'type': 'short_game',
                'description': 'Short game - early game performance crucial',
                'focus': 'early_impact'
            }
        elif duration > 20:
            return {
                'type': 'long_game',
                'description': 'Long game - late game scaling important',
                'focus': 'scaling_effectiveness'
            }
        else:
            return {
                'type': 'normal_game',
                'description': 'Normal game duration',
                'focus': 'balanced_performance'
            }
    
    def _assess_team_impact(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess impact on team performance."""
        damage_pct = match_data.get('damage_percentage', 0)
        teamfight_participation = match_data.get('teamfight_participation', 0)
        
        impact_score = (damage_pct + teamfight_participation) / 2
        
        if impact_score > 70:
            return {'level': 'high', 'description': 'High team impact'}
        elif impact_score > 40:
            return {'level': 'medium', 'description': 'Moderate team impact'}
        else:
            return {'level': 'low', 'description': 'Low team impact - focus on participation'}
    
    def _assess_objective_control(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess objective control contribution."""
        turret_damage = match_data.get('turret_damage', 0)
        lord_participation = match_data.get('lord_participation', 0)
        
        if turret_damage > 5000 or lord_participation > 80:
            return {'level': 'good', 'description': 'Good objective control'}
        else:
            return {'level': 'poor', 'description': 'Improve objective participation'}
    
    async def _cache_result(self, request: HeroEvaluationRequest, 
                            result: HeroEvaluationResponse):
        """Cache evaluation result."""
        cache_key = self._generate_cache_key(request)
        
        try:
            # Convert to dict for caching
            cache_data = asdict(result)
            await self.cache.set(cache_key, cache_data, ttl=1800)  # 30 minutes
            
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    def _generate_cache_key(self, request: HeroEvaluationRequest) -> str:
        """Generate cache key for evaluation request."""
        # Create a hash based on hero, match data hash, and evaluation mode
        import hashlib
        
        key_components = [
            request.hero,
            request.player_ign,
            request.evaluation_mode,
            str(hash(frozenset(request.match_data.items())))
        ]
        
        key_string = '|'.join(key_components)
        return f"hero_eval:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    async def _emit_evaluation_start(self, request: HeroEvaluationRequest):
        """Emit evaluation start event."""
        event = EvaluationEvent(
            type="hero_evaluation_start",
            hero=request.hero,
            player_ign=request.player_ign,
            evaluation_mode=request.evaluation_mode,
            timestamp=datetime.now().isoformat()
        )
        await self.event_bus.emit(event)
    
    async def _emit_evaluation_complete(self, request: HeroEvaluationRequest,
                                        result: HeroEvaluationResponse):
        """Emit evaluation complete event."""
        event = EvaluationEvent(
            type="hero_evaluation_complete",
            hero=request.hero,
            player_ign=request.player_ign,
            score=result.overall_score,
            performance_rating=result.performance_rating,
            processing_time_ms=result.processing_time_ms,
            timestamp=datetime.now().isoformat()
        )
        await self.event_bus.emit(event)
    
    async def _emit_evaluation_error(self, request: HeroEvaluationRequest, error: str):
        """Emit evaluation error event."""
        event = EvaluationEvent(
            type="hero_evaluation_error",
            hero=request.hero,
            player_ign=request.player_ign,
            error=error,
            timestamp=datetime.now().isoformat()
        )
        await self.event_bus.emit(event)
    
    async def _collect_metrics(self, request: HeroEvaluationRequest,
                               result: HeroEvaluationResponse):
        """Collect performance metrics."""
        try:
            await self.metrics.record_counter(
                'hero_evaluations_total',
                labels={'hero': request.hero, 'role': result.role}
            )
            
            await self.metrics.record_histogram(
                'hero_evaluation_duration_ms',
                result.processing_time_ms,
                labels={'hero': request.hero}
            )
            
            await self.metrics.record_gauge(
                'hero_evaluation_score',
                result.overall_score,
                labels={'hero': request.hero, 'rating': result.performance_rating}
            )
            
        except Exception as e:
            logger.warning(f"Failed to collect metrics: {e}")
    
    def _create_error_response(self, request: HeroEvaluationRequest, 
                               error: str) -> HeroEvaluationResponse:
        """Create error response for failed evaluations."""
        return HeroEvaluationResponse(
            hero=request.hero,
            role="unknown",
            overall_score=0.0,
            performance_rating="Error",
            feedback=[("error", f"Evaluation failed: {error}")],
            suggestions=["Please try again or contact support"],
            role_specific_metrics={},
            hero_specific_insights={'error': error},
            confidence=0.0,
            evaluation_timestamp=datetime.now().isoformat(),
            processing_time_ms=0
        )
    
    async def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        try:
            stats = {
                'cache_stats': await self.cache.get_stats(),
                'hero_mappings_count': len(self.hero_mappings),
                'available_roles': RoleEvaluatorFactory.get_available_roles(),
                'config': {
                    'batch_size': self.batch_size,
                    'max_concurrent': self.max_concurrent,
                    'timeout_seconds': self.timeout_seconds
                }
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'error': str(e)}
    
    async def warmup_cache(self, heroes: List[str]):
        """Warmup cache with common hero evaluations."""
        logger.info(f"Starting cache warmup for {len(heroes)} heroes")
        
        # Create dummy requests for cache warmup
        warmup_requests = []
        for hero in heroes:
            dummy_data = {
                'kills': 5, 'deaths': 3, 'assists': 8,
                'hero_damage': 50000, 'damage_taken': 30000,
                'gold_per_min': 600, 'match_duration': 15
            }
            
            request = HeroEvaluationRequest(
                hero=hero,
                match_data=dummy_data,
                player_ign='warmup_user',
                evaluation_mode='quick',
                emit_events=False
            )
            warmup_requests.append(request)
        
        # Execute warmup in batches
        await self.evaluate_heroes_batch(warmup_requests)
        logger.info("Cache warmup completed")
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self.cache.close()
            await self.event_bus.close()
            logger.info("Hero Evaluation Orchestrator cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


# Singleton instance for global access
_orchestrator_instance = None

def get_orchestrator(config: Dict[str, Any] = None) -> HeroEvaluationOrchestrator:
    """Get singleton orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = HeroEvaluationOrchestrator(config)
    return _orchestrator_instance 