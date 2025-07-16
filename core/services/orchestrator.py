"""
Analysis Orchestrator
Coordinates multiple services for parallel processing and optimal performance
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .base_service import ServiceResult
from .ocr_service import OCRService
from .detection_service import DetectionService
from .analysis_service import AnalysisService

logger = logging.getLogger(__name__)


@dataclass
class AnalysisRequest:
    """Request structure for analysis"""
    image_path: str
    ign: str
    hero_override: Optional[str] = None
    session_id: Optional[str] = None
    context: str = "scoreboard"
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Complete analysis result from orchestrator"""
    success: bool
    match_data: Dict[str, Any]
    confidence_scores: Dict[str, float]
    processing_time: float
    service_results: Dict[str, ServiceResult]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class AnalysisOrchestrator:
    """Orchestrates analysis across multiple services"""
    
    def __init__(self):
        self.ocr_service = OCRService()
        self.detection_service = DetectionService()
        self.analysis_service = AnalysisService()
        
        # Service dependencies
        self.service_dependencies = {
            "ocr": [],
            "hero_detection": [],
            "trophy_detection": [],
            "data_analysis": ["ocr", "hero_detection"]
        }
        
        # Processing metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_processing_time": 0.0,
            "service_timings": {}
        }
    
    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Main analysis method with parallel service execution"""
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        service_results = {}
        errors = []
        warnings = []
        
        try:
            # Execute services in parallel where possible
            async with asyncio.TaskGroup() as tg:
                # Services with no dependencies can run in parallel
                ocr_task = tg.create_task(
                    self._execute_ocr(request)
                )
                hero_task = tg.create_task(
                    self._execute_hero_detection(request)
                )
                trophy_task = tg.create_task(
                    self._execute_trophy_detection(request)
                )
            
            # Get results
            service_results["ocr"] = await ocr_task
            service_results["hero_detection"] = await hero_task
            service_results["trophy_detection"] = await trophy_task
            
            # Services with dependencies run after
            service_results["data_analysis"] = await self._execute_analysis(
                request, service_results
            )
            
            # Aggregate results
            match_data = self._aggregate_match_data(service_results)
            confidence_scores = self._calculate_confidence_scores(
                service_results
            )
            
            # Collect warnings and errors
            for result in service_results.values():
                if result.error:
                    errors.append(result.error)
                if result.metadata.get("warnings"):
                    warnings.extend(result.metadata["warnings"])
            
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, True)
            
            return AnalysisResult(
                success=len(errors) == 0,
                match_data=match_data,
                confidence_scores=confidence_scores,
                processing_time=processing_time,
                service_results=service_results,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Orchestration failed: {str(e)}")
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, False)
            
            return AnalysisResult(
                success=False,
                match_data={},
                confidence_scores={},
                processing_time=processing_time,
                service_results=service_results,
                errors=[str(e)],
                warnings=warnings
            )
    
    async def _execute_ocr(self, request: AnalysisRequest) -> ServiceResult:
        """Execute OCR service"""
        ocr_request = {
            "image_path": request.image_path,
            "preprocessing": "standard"
        }
        
        start = time.time()
        result = await self.ocr_service.execute(ocr_request)
        self.metrics["service_timings"]["ocr"] = time.time() - start
        
        return result
    
    async def _execute_hero_detection(
        self, request: AnalysisRequest
    ) -> ServiceResult:
        """Execute hero detection service"""
        detection_request = {
            "type": "hero",
            "image_path": request.image_path,
            "player_ign": request.ign,
            "hero_override": request.hero_override
        }
        
        start = time.time()
        result = await self.detection_service.execute(detection_request)
        self.metrics["service_timings"]["hero_detection"] = time.time() - start
        
        return result
    
    async def _execute_trophy_detection(
        self, request: AnalysisRequest
    ) -> ServiceResult:
        """Execute trophy detection service"""
        detection_request = {
            "type": "trophy",
            "image_path": request.image_path,
            "player_row_y": 0,  # Will be determined from OCR
            "player_name_x": None
        }
        
        start = time.time()
        result = await self.detection_service.execute(detection_request)
        elapsed = time.time() - start
        self.metrics["service_timings"]["trophy_detection"] = elapsed
        
        return result
    
    async def _execute_analysis(
        self, 
        request: AnalysisRequest,
        service_results: Dict[str, ServiceResult]
    ) -> ServiceResult:
        """Execute analysis service with dependencies"""
        # Extract data from dependent services
        ocr_success = service_results["ocr"].success
        hero_success = service_results["hero_detection"].success
        
        ocr_data = service_results["ocr"].data if ocr_success else {}
        hero_data = (
            service_results["hero_detection"].data if hero_success else {}
        )
        
        analysis_request = {
            "ocr_results": ocr_data.get("raw_results", []),
            "hero": hero_data.get("hero", "unknown"),
            "player_ign": request.ign,
            "context": request.context
        }
        
        start = time.time()
        result = await self.analysis_service.execute(analysis_request)
        elapsed = time.time() - start
        self.metrics["service_timings"]["data_analysis"] = elapsed
        
        return result
    
    def _aggregate_match_data(
        self, service_results: Dict[str, ServiceResult]
    ) -> Dict[str, Any]:
        """Aggregate data from all services"""
        match_data = {}
        
        # Get analysis data
        if service_results["data_analysis"].success:
            match_data.update(service_results["data_analysis"].data)
        
        # Add hero data
        if service_results["hero_detection"].success:
            hero_data = service_results["hero_detection"].data
            match_data["hero"] = hero_data.get("hero", "unknown")
            match_data["hero_confidence"] = hero_data.get("confidence", 0.0)
        
        # Add trophy data
        if service_results["trophy_detection"].success:
            trophy_data = service_results["trophy_detection"].data
            match_data["trophy_type"] = trophy_data.get("trophy_type")
            match_data["performance_label"] = trophy_data.get(
                "performance_label"
            )
        
        return match_data
    
    def _calculate_confidence_scores(
        self, service_results: Dict[str, ServiceResult]
    ) -> Dict[str, float]:
        """Calculate overall confidence scores"""
        scores = {}
        
        # OCR confidence
        if service_results["ocr"].success:
            ocr_data = service_results["ocr"].data
            scores["ocr_confidence"] = ocr_data.get("average_confidence", 0.0)
        
        # Hero detection confidence
        if service_results["hero_detection"].success:
            hero_data = service_results["hero_detection"].data
            scores["hero_confidence"] = hero_data.get("confidence", 0.0)
        
        # Trophy detection confidence
        if service_results["trophy_detection"].success:
            trophy_data = service_results["trophy_detection"].data
            scores["trophy_confidence"] = trophy_data.get("confidence", 0.0)
        
        # Overall confidence
        if scores:
            scores["overall_confidence"] = sum(scores.values()) / len(scores)
        else:
            scores["overall_confidence"] = 0.0
        
        return scores
    
    def _update_metrics(self, processing_time: float, success: bool):
        """Update orchestrator metrics"""
        if success:
            self.metrics["successful_requests"] += 1
        
        # Update rolling average
        total = self.metrics["total_requests"]
        current_avg = self.metrics["average_processing_time"]
        self.metrics["average_processing_time"] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    async def batch_analyze(
        self, requests: List[AnalysisRequest]
    ) -> List[AnalysisResult]:
        """Process multiple analysis requests"""
        # Sort by priority
        sorted_requests = sorted(
            requests, key=lambda r: r.priority, reverse=True
        )
        
        # Process in batches for optimal resource usage
        batch_size = 5
        results = []
        
        for i in range(0, len(sorted_requests), batch_size):
            batch = sorted_requests[i:i + batch_size]
            
            # Process batch in parallel
            batch_tasks = [
                self.analyze(request) for request in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics"""
        return {
            **self.metrics,
            "service_health": {
                "ocr": self.ocr_service.get_health_status(),
                "detection": self.detection_service.get_health_status(),
                "analysis": self.analysis_service.get_health_status()
            }
        } 