"""
YOLOv8 Detection Service for MLBB Coach AI
==========================================

This service provides YOLOv8-based object detection capabilities for MLBB screenshots,
integrating seamlessly with the existing services architecture.

Features:
- Hero portrait detection
- UI element detection
- Stat box identification
- Confidence-based fallback integration
- Performance monitoring
- Integration with existing pipeline
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import torch

from .base_service import BaseService
from ..yolo_fallback import make_fallback_decision, FallbackDecision
from ..error_handler import error_handler

logger = logging.getLogger(__name__)


class YOLODetectionService(BaseService):
    """YOLOv8 detection service for MLBB object detection."""
    
    def __init__(self, 
                 model_path: str = "models/mlbb_yolo_best.pt",
                 confidence_threshold: float = 0.5,
                 enable_fallback: bool = True):
        """
        Initialize the YOLO detection service.
        
        Args:
            model_path: Path to trained YOLOv8 model
            confidence_threshold: Minimum confidence for detections
            enable_fallback: Whether to enable fallback logic
        """
        super().__init__(service_name="YOLODetectionService")
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.enable_fallback = enable_fallback
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # MLBB class definitions
        self.mlbb_classes = {
            0: "hero_portrait",
            1: "kda_box",
            2: "damage_box", 
            3: "gold_box",
            4: "minimap",
            5: "timer",
            6: "score_indicator",
            7: "mvp_badge",
            8: "medal_bronze",
            9: "medal_silver",
            10: "medal_gold",
            11: "team_indicator",
            12: "player_name",
            13: "hero_name",
            14: "kill_indicator",
            15: "death_indicator",
            16: "assist_indicator"
        }
        
        # Priority classes for different use cases
        self.hero_classes = ["hero_portrait", "hero_name"]
        self.stat_classes = ["kda_box", "damage_box", "gold_box"]
        self.ui_classes = ["minimap", "timer", "score_indicator"]
        self.badge_classes = ["mvp_badge", "medal_bronze", "medal_silver", "medal_gold"]
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'avg_inference_time': 0.0,
            'confidence_scores': [],
            'class_distribution': {cls: 0 for cls in self.mlbb_classes.values()}
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the YOLOv8 model."""
        try:
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                logger.info(f"✅ YOLOv8 model loaded from {self.model_path}")
                logger.info(f"🔧 Device: {self.device}")
                logger.info(f"📊 Confidence threshold: {self.confidence_threshold}")
            else:
                logger.warning(f"⚠️ Model file not found: {self.model_path}")
                logger.info("🔄 Using YOLOv8 nano as fallback")
                self.model = YOLO('yolov8n.pt')
        except Exception as e:
            logger.error(f"❌ Failed to load YOLOv8 model: {e}")
            error_handler.handle_error(e, "YOLOv8 model initialization")
            raise
    
    def detect_objects(self, 
                      image_path: str,
                      target_classes: Optional[List[str]] = None,
                      ocr_confidence: float = 1.0) -> Dict[str, Any]:
        """
        Detect objects in an image using YOLOv8.
        
        Args:
            image_path: Path to input image
            target_classes: Specific classes to detect (None for all)
            ocr_confidence: OCR confidence for fallback decision
            
        Returns:
            Detection results dictionary
        """
        start_time = time.time()
        
        try:
            # Check if YOLO should be used based on OCR confidence
            if self.enable_fallback:
                fallback_decision = make_fallback_decision(
                    ocr_confidence=ocr_confidence,
                    yolo_confidence=self.confidence_threshold
                )
                
                if not fallback_decision.should_use_yolo:
                    logger.info(f"🚫 YOLO fallback not triggered (OCR confidence: {ocr_confidence:.2f})")
                    return {
                        'used_yolo': False,
                        'reason': 'OCR confidence sufficient',
                        'detections': [],
                        'inference_time': 0.0
                    }
                
                logger.info(f"🎯 YOLO fallback triggered: {fallback_decision.reason.value}")
            
            # Run YOLO inference
            results = self.model(
                image_path,
                conf=self.confidence_threshold,
                iou=0.45,
                device=self.device
            )
            
            # Process results
            detections = []
            confidence_scores = []
            class_counts = {cls: 0 for cls in self.mlbb_classes.values()}
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = self.mlbb_classes.get(class_id, 'unknown')
                        confidence = float(box.conf[0])
                        
                        # Filter by target classes if specified
                        if target_classes and class_name not in target_classes:
                            continue
                        
                        # Get bounding box coordinates
                        xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        xywh = box.xywh[0].tolist()  # [x_center, y_center, width, height]
                        
                        detection = {
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox_xyxy': xyxy,
                            'bbox_xywh': xywh,
                            'area': xywh[2] * xywh[3],
                            'normalized_bbox': self._normalize_bbox(xyxy, image_path)
                        }
                        
                        detections.append(detection)
                        confidence_scores.append(confidence)
                        class_counts[class_name] += 1
            
            inference_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(detections, inference_time, confidence_scores)
            
            # Compile results
            results_dict = {
                'used_yolo': True,
                'image_path': image_path,
                'inference_time': inference_time,
                'detections': detections,
                'detection_count': len(detections),
                'confidence_scores': confidence_scores,
                'class_counts': class_counts,
                'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
                'max_confidence': max(confidence_scores) if confidence_scores else 0.0,
                'target_classes': target_classes
            }
            
            logger.info(f"🎯 YOLO detection completed: {len(detections)} objects in {inference_time:.3f}s")
            
            return results_dict
            
        except Exception as e:
            logger.error(f"❌ YOLO detection failed for {image_path}: {e}")
            error_handler.handle_error(e, f"YOLO detection: {image_path}")
            return {
                'used_yolo': False,
                'error': str(e),
                'detections': [],
                'inference_time': time.time() - start_time
            }
    
    def detect_hero_portraits(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect hero portraits specifically.
        FIXED: Enhanced error handling and fallback logic.
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of hero portrait detections with improved reliability
        """
        try:
            if not image_path or not os.path.exists(image_path):
                logger.warning(f"Invalid image path: {image_path}")
                return []
            
            results = self.detect_objects(image_path, target_classes=self.hero_classes)
            detections = results.get('detections', [])
            
            # Filter and validate detections
            valid_detections = []
            for detection in detections:
                if detection.get('confidence', 0) > 0.3:  # Minimum confidence threshold
                    valid_detections.append(detection)
            
            logger.debug(f"Hero portraits detected: {len(valid_detections)}/{len(detections)}")
            return valid_detections
            
        except Exception as e:
            logger.error(f"Hero portrait detection failed: {e}")
            return []
    
    def detect_ui_elements(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect UI elements specifically.
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of UI element detections
        """
        results = self.detect_objects(image_path, target_classes=self.ui_classes)
        return results.get('detections', [])
    
    def detect_stat_boxes(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect stat boxes specifically.
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of stat box detections
        """
        results = self.detect_objects(image_path, target_classes=self.stat_classes)
        return results.get('detections', [])
    
    def get_minimap_region(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Get minimap region for minimap tracker integration.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Minimap region data or None if not found
        """
        results = self.detect_objects(image_path, target_classes=["minimap"])
        
        minimap_detections = results.get('detections', [])
        if not minimap_detections:
            return None
        
        # Return the most confident minimap detection
        best_minimap = max(minimap_detections, key=lambda x: x['confidence'])
        
        return {
            'bbox': best_minimap['bbox_xyxy'],
            'confidence': best_minimap['confidence'],
            'normalized_bbox': best_minimap['normalized_bbox']
        }
    
    def enhance_ocr_regions(self, image_path: str, ocr_confidence: float = 0.6) -> Dict[str, Any]:
        """
        Enhance OCR by providing focused regions from YOLO detection.
        
        Args:
            image_path: Path to input image
            ocr_confidence: OCR confidence for fallback decision
            
        Returns:
            Enhanced regions for OCR processing
        """
        # Get stat box and text regions
        stat_detections = self.detect_stat_boxes(image_path)
        
        # Get text-related detections
        text_classes = ["player_name", "hero_name"]
        text_results = self.detect_objects(image_path, target_classes=text_classes)
        text_detections = text_results.get('detections', [])
        
        # Combine all text-relevant regions
        all_regions = stat_detections + text_detections
        
        # Sort by confidence and area (prefer larger, more confident regions)
        sorted_regions = sorted(all_regions, 
                              key=lambda x: (x['confidence'], x['area']), 
                              reverse=True)
        
        return {
            'enhanced_regions': sorted_regions,
            'region_count': len(sorted_regions),
            'confidence_boost': min(0.2, len(sorted_regions) * 0.05),  # Slight confidence boost
            'recommended_ocr_regions': [r['bbox_xyxy'] for r in sorted_regions[:5]]  # Top 5
        }
    
    def _normalize_bbox(self, bbox_xyxy: List[float], image_path: str) -> List[float]:
        """
        Normalize bounding box coordinates to 0-1 range.
        
        Args:
            bbox_xyxy: Bounding box in [x1, y1, x2, y2] format
            image_path: Path to image for getting dimensions
            
        Returns:
            Normalized bounding box
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return bbox_xyxy
            
            height, width = image.shape[:2]
            x1, y1, x2, y2 = bbox_xyxy
            
            return [
                x1 / width,
                y1 / height,
                x2 / width,
                y2 / height
            ]
        except Exception:
            return bbox_xyxy
    
    def _update_stats(self, detections: List[Dict], inference_time: float, confidence_scores: List[float]):
        """Update performance statistics."""
        self.detection_stats['total_detections'] += len(detections)
        
        # Update average inference time
        current_avg = self.detection_stats['avg_inference_time']
        self.detection_stats['avg_inference_time'] = (current_avg + inference_time) / 2
        
        # Update confidence scores
        self.detection_stats['confidence_scores'].extend(confidence_scores)
        
        # Update class distribution
        for detection in detections:
            class_name = detection['class_name']
            if class_name in self.detection_stats['class_distribution']:
                self.detection_stats['class_distribution'][class_name] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        confidence_scores = self.detection_stats['confidence_scores']
        
        return {
            'total_detections': self.detection_stats['total_detections'],
            'avg_inference_time': self.detection_stats['avg_inference_time'],
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
            'confidence_std': np.std(confidence_scores) if confidence_scores else 0.0,
            'class_distribution': self.detection_stats['class_distribution'],
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold
        }
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for BaseService compliance.
        
        Args:
            request: Dictionary containing 'image_path' and optional parameters
            
        Returns:
            ServiceResult-compatible detection results
        """
        from .base_service import ServiceResult
        
        try:
            image_path = request.get('image_path')
            target_classes = request.get('target_classes')
            ocr_confidence = request.get('ocr_confidence', 1.0)
            
            if not image_path:
                return ServiceResult(
                    success=False,
                    error="image_path required in request",
                    service_name="YOLODetectionService"
                )
            
            # Call main detection method
            results = self.detect_objects(
                image_path=image_path,
                target_classes=target_classes,
                ocr_confidence=ocr_confidence
            )
            
            return ServiceResult(
                success=True,
                data=results,
                service_name="YOLODetectionService",
                processing_time=results.get('inference_time', 0.0)
            )
            
        except Exception as e:
            return ServiceResult(
                success=False,
                error=str(e),
                service_name="YOLODetectionService"
            )
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of the detection service."""
        try:
            # Quick inference test
            test_result = self.model("https://ultralytics.com/images/bus.jpg", conf=0.5)
            
            return {
                'status': 'healthy',
                'model_loaded': self.model is not None,
                'device': self.device,
                'model_path': self.model_path,
                'test_inference': len(test_result) > 0,
                'total_detections': self.detection_stats['total_detections']
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'model_loaded': self.model is not None
            }


# Global service instance
yolo_detection_service = YOLODetectionService()


def get_yolo_detection_service() -> YOLODetectionService:
    """Get the global YOLOv8 detection service instance."""
    return yolo_detection_service 