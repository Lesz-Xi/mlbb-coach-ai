"""
OCR Service Implementation
Handles all OCR-related operations with async support
"""

import logging
import sys
import os
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_service import BaseService, ServiceResult

# Import OCR functionality
try:
    from ..data_collector import get_ocr_reader
except ImportError:
    from core.data_collector import get_ocr_reader

logger = logging.getLogger(__name__)


class OCRService(BaseService):
    """Service for OCR operations"""
    
    def __init__(self):
        super().__init__("OCRService")
        self.reader = None
        self.preprocessing_cache = {}
    
    async def process(self, request: Dict[str, Any]) -> ServiceResult:
        """Process OCR request"""
        try:
            image_path = request.get("image_path")
            region = request.get("region")  # Optional ROI
            preprocessing = request.get("preprocessing", "standard")
            
            if not image_path:
                return ServiceResult(
                    success=False,
                    error="No image_path provided"
                )
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return ServiceResult(
                    success=False,
                    error=f"Failed to load image: {image_path}"
                )
            
            # Apply region if specified
            if region:
                x, y, w, h = region
                image = image[y:y+h, x:x+w]
            
            # Preprocess image
            processed_image = await self._preprocess_image(
                image, preprocessing
            )
            
            # Perform OCR
            ocr_results = await self._perform_ocr(processed_image)
            
            # Extract text and structure
            extracted_data = self._structure_ocr_results(ocr_results)
            
            return ServiceResult(
                success=True,
                data=extracted_data,
                metadata={
                    "total_text_regions": len(ocr_results),
                    "preprocessing": preprocessing,
                    "image_shape": image.shape
                }
            )
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            return ServiceResult(
                success=False,
                error=str(e)
            )
    
    async def _preprocess_image(
        self, image: np.ndarray, method: str
    ) -> np.ndarray:
        """Preprocess image for better OCR results"""
        
        # Check cache
        cache_key = f"{id(image)}_{method}"
        if cache_key in self.preprocessing_cache:
            return self.preprocessing_cache[cache_key]
        
        if method == "standard":
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply denoising
            denoised = cv2.fastNlDenoising(gray)
            
            # Apply adaptive thresholding
            processed = cv2.adaptiveThreshold(
                denoised, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        
        elif method == "aggressive":
            # More aggressive preprocessing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Morphological operations
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(
                enhanced, cv2.MORPH_CLOSE, kernel
            )
        
        else:
            processed = image
        
        # Cache result
        self.preprocessing_cache[cache_key] = processed
        
        # Limit cache size
        if len(self.preprocessing_cache) > 100:
            # Remove oldest entries
            keys = list(self.preprocessing_cache.keys())
            for key in keys[:50]:
                del self.preprocessing_cache[key]
        
        return processed
    
    async def _perform_ocr(
        self, image: np.ndarray
    ) -> List[Tuple[List, str, float]]:
        """Perform OCR on preprocessed image"""
        
        # Initialize reader if needed
        if self.reader is None:
            self.reader = get_ocr_reader()
        
        # Perform OCR
        results = self.reader.readtext(image, detail=1)
        
        return results
    
    def _structure_ocr_results(
        self, ocr_results: List[Tuple[List, str, float]]
    ) -> Dict[str, Any]:
        """Structure OCR results into organized data"""
        
        structured_data = {
            "raw_results": ocr_results,
            "text_regions": [],
            "full_text": "",
            "confidence_scores": []
        }
        
        text_parts = []
        
        for bbox, text, confidence in ocr_results:
            # Calculate bounding box
            points = np.array(bbox)
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            
            region = {
                "text": text,
                "confidence": confidence,
                "bbox": {
                    "x": int(x_min),
                    "y": int(y_min),
                    "width": int(x_max - x_min),
                    "height": int(y_max - y_min)
                },
                "center": {
                    "x": int((x_min + x_max) / 2),
                    "y": int((y_min + y_max) / 2)
                }
            }
            
            structured_data["text_regions"].append(region)
            structured_data["confidence_scores"].append(confidence)
            text_parts.append(text)
        
        structured_data["full_text"] = " ".join(text_parts)
        structured_data["average_confidence"] = (
            sum(structured_data["confidence_scores"]) / 
            len(structured_data["confidence_scores"])
            if structured_data["confidence_scores"] else 0.0
        )
        
        return structured_data
    
    async def batch_process(
        self, requests: List[Dict[str, Any]]
    ) -> List[ServiceResult]:
        """Process multiple OCR requests in batch"""
        results = []
        
        for request in requests:
            result = await self.execute(request)
            results.append(result)
        
        return results 