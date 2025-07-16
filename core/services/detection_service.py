"""
Detection Service Implementation
Handles hero detection, trophy/medal detection, and other
visual recognition tasks
"""

import logging
from typing import Dict, Any, List

from .base_service import BaseService, ServiceResult

logger = logging.getLogger(__name__)


class DetectionService(BaseService):
    """Service for detection operations (hero, trophy, etc.)"""
    
    def __init__(self):
        super().__init__("DetectionService")
        self.hero_detector = None
        self.trophy_detector = None
        self.detection_cache = {}
    
    async def process(self, request: Dict[str, Any]) -> ServiceResult:
        """Process detection request"""
        try:
            detection_type = request.get("type", "hero")
            image_path = request.get("image_path")
            
            if not image_path:
                return ServiceResult(
                    success=False,
                    error="No image_path provided"
                )
            
            if detection_type == "hero":
                return await self._detect_hero(request)
            elif detection_type == "trophy":
                return await self._detect_trophy(request)
            elif detection_type == "combined":
                return await self._detect_combined(request)
            else:
                return ServiceResult(
                    success=False,
                    error=f"Unknown detection type: {detection_type}"
                )
            
        except Exception as e:
            logger.error(f"Detection processing failed: {str(e)}")
            return ServiceResult(
                success=False,
                error=str(e)
            )
    
    async def _detect_hero(self, request: Dict[str, Any]) -> ServiceResult:
        """Detect hero in screenshot"""
        # Initialize hero detector if needed
        if self.hero_detector is None:
            try:
                from ..advanced_hero_detector import AdvancedHeroDetector
                self.hero_detector = AdvancedHeroDetector()
            except ImportError:
                from core.advanced_hero_detector import AdvancedHeroDetector
                self.hero_detector = AdvancedHeroDetector()
        
        image_path = request.get("image_path")
        player_ign = request.get("player_ign", "")
        hero_override = request.get("hero_override")
        
        # Check cache
        cache_key = f"hero_{image_path}_{player_ign}"
        if cache_key in self.detection_cache:
            cached_result = self.detection_cache[cache_key]
            return ServiceResult(
                success=True,
                data=cached_result,
                metadata={"cache_hit": True}
            )
        
        # Perform detection
        hero, confidence, debug_info = (
            self.hero_detector.detect_hero_comprehensive(
                image_path, player_ign
            )
        )
        
        if hero_override:
            hero = hero_override
            confidence = 1.0
        
        result_data = {
            "hero": hero,
            "confidence": confidence,
            "debug_info": debug_info
        }
        
        # Cache result
        self.detection_cache[cache_key] = result_data
        self._cleanup_cache()
        
        return ServiceResult(
            success=True,
            data=result_data,
            metadata={
                "detection_method": debug_info.get("method", "unknown"),
                "cache_hit": False
            }
        )
    
    async def _detect_trophy(self, request: Dict[str, Any]) -> ServiceResult:
        """Detect trophy/medal in screenshot"""
        # Initialize trophy detector if needed
        if self.trophy_detector is None:
            try:
                from ..trophy_medal_detector_v2 import (
                    ImprovedTrophyMedalDetector
                )
                self.trophy_detector = ImprovedTrophyMedalDetector()
            except ImportError:
                from core.trophy_medal_detector_v2 import (
                    ImprovedTrophyMedalDetector
                )
                self.trophy_detector = ImprovedTrophyMedalDetector()
        
        image_path = request.get("image_path")
        player_row_y = request.get("player_row_y", 0)
        player_name_x = request.get("player_name_x")
        
        # Perform detection
        trophy_result = self.trophy_detector.detect_trophy_in_player_row(
            image_path=image_path,
            player_row_y=player_row_y,
            player_name_x=player_name_x,
            debug_mode=request.get("debug_mode", False)
        )
        
        return ServiceResult(
            success=True,
            data={
                "trophy_type": trophy_result.trophy_type.value,
                "confidence": trophy_result.confidence,
                "performance_label": trophy_result.performance_label.value,
                "detection_method": trophy_result.detection_method,
                "bounding_box": trophy_result.bounding_box,
                "color_analysis": trophy_result.color_analysis,
                "shape_analysis": trophy_result.shape_analysis
            },
            metadata=trophy_result.debug_info
        )
    
    async def _detect_combined(self, request: Dict[str, Any]) -> ServiceResult:
        """Perform both hero and trophy detection"""
        hero_result = await self._detect_hero(request)
        trophy_result = await self._detect_trophy(request)
        
        combined_data = {
            "hero_detection": (
                hero_result.data if hero_result.success else None
            ),
            "trophy_detection": (
                trophy_result.data if trophy_result.success else None
            )
        }
        
        combined_metadata = {
            "hero_metadata": hero_result.metadata,
            "trophy_metadata": trophy_result.metadata
        }
        
        return ServiceResult(
            success=hero_result.success and trophy_result.success,
            data=combined_data,
            metadata=combined_metadata,
            error=hero_result.error or trophy_result.error
        )
    
    def _cleanup_cache(self):
        """Clean up detection cache if it gets too large"""
        if len(self.detection_cache) > 100:
            # Remove oldest half of entries
            keys = list(self.detection_cache.keys())
            for key in keys[:50]:
                del self.detection_cache[key]
    
    async def batch_detect(
        self, requests: List[Dict[str, Any]]
    ) -> List[ServiceResult]:
        """Process multiple detection requests"""
        results = []
        
        for request in requests:
            result = await self.execute(request)
            results.append(result)
        
        return results 