"""
Service-Oriented Architecture for MLBB Coach AI
This module contains service implementations for scalable analysis
"""

from .base_service import BaseService, ServiceResult
from .ocr_service import OCRService
from .detection_service import DetectionService
from .analysis_service import AnalysisService
from .orchestrator import AnalysisOrchestrator

__all__ = [
    'BaseService',
    'ServiceResult',
    'OCRService',
    'DetectionService',
    'AnalysisService',
    'AnalysisOrchestrator'
] 