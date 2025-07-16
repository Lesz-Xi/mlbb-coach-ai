"""
Optimized Image Preprocessing System

High-performance image preprocessing pipeline for MLBB Coach AI with:
- Intelligent preprocessing caching
- Quality-based adaptive processing
- Memory-efficient operations
- Parallel preprocessing stages
"""

import cv2
import numpy as np
import logging
import hashlib
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class ImageQualityMetrics:
    """Container for image quality assessment results."""
    resolution_score: float
    sharpness_score: float
    contrast_score: float
    brightness_score: float
    noise_score: float
    overall_score: float
    processing_recommendation: str


@dataclass
class PreprocessingResult:
    """Container for preprocessing results."""
    processed_image: np.ndarray
    quality_metrics: ImageQualityMetrics
    processing_time: float
    cache_hit: bool
    processing_method: str


class OptimizedImageProcessor:
    """
    High-performance image preprocessing system with intelligent optimizations.
    
    Features:
    - Smart preprocessing cache with SHA-256 hashing
    - Quality-based adaptive processing pipelines
    - Memory-efficient image operations
    - Parallel processing capabilities
    - Performance monitoring
    """
    
    def __init__(self, cache_size: int = 100, enable_cache: bool = True):
        self.cache_size = cache_size
        self.enable_cache = enable_cache
        self.preprocessing_cache: Dict[str, PreprocessingResult] = {}
        self.quality_cache: Dict[str, ImageQualityMetrics] = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        
        # Thread safety
        self._cache_lock = threading.Lock()
        
        # Performance tracking
        self.processing_times = []
        self.quality_assessment_times = []
        
        # Optimized processing pipelines
        self.processing_pipelines = {
            'minimal': self._minimal_preprocessing,
            'moderate': self._moderate_preprocessing,
            'aggressive': self._aggressive_preprocessing
        }
        
        logger.info(f"🚀 Optimized Image Processor initialized (cache: {enable_cache}, size: {cache_size})")
    
    def process_image(
        self, 
        image_path: str, 
        force_quality_assessment: bool = False,
        debug_mode: bool = False
    ) -> PreprocessingResult:
        """
        Main entry point for optimized image preprocessing.
        
        Args:
            image_path: Path to the image file
            force_quality_assessment: Force quality assessment even if cached
            debug_mode: Enable debug output and timing
            
        Returns:
            PreprocessingResult with processed image and metadata
        """
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key(image_path, force_quality_assessment)
        
        # Check cache first
        if self.enable_cache and not force_quality_assessment:
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.cache_stats["hits"] += 1
                if debug_mode:
                    logger.debug(f"⚡ Cache hit for {Path(image_path).name}")
                return cached_result
        
        self.cache_stats["misses"] += 1
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        
        # Quick quality assessment
        quality_start = time.time()
        quality_metrics = self._assess_image_quality_optimized(image)
        quality_time = time.time() - quality_start
        self.quality_assessment_times.append(quality_time)
        
        # Select optimal processing pipeline
        processing_method = quality_metrics.processing_recommendation
        pipeline_func = self.processing_pipelines[processing_method]
        
        # Process image
        processed_image = pipeline_func(image)
        
        total_time = time.time() - start_time
        self.processing_times.append(total_time)
        
        # Create result
        result = PreprocessingResult(
            processed_image=processed_image,
            quality_metrics=quality_metrics,
            processing_time=total_time,
            cache_hit=False,
            processing_method=processing_method
        )
        
        # Cache result
        if self.enable_cache:
            self._store_in_cache(cache_key, result)
        
        if debug_mode:
            logger.debug(f"✅ Processed {Path(image_path).name} using {processing_method} in {total_time:.3f}s")
        
        return result
    
    def _assess_image_quality_optimized(self, image: np.ndarray) -> ImageQualityMetrics:
        """Optimized image quality assessment with reduced computational overhead."""
        
        height, width = image.shape[:2]
        
        # Convert to grayscale once for all operations
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Use smaller sample for quality assessment to speed up processing
        sample_factor = max(1, min(width, height) // 400)  # Sample every N pixels
        if sample_factor > 1:
            gray_sample = gray[::sample_factor, ::sample_factor]
        else:
            gray_sample = gray
        
        # 1. Resolution score (quick calculation)
        resolution_score = min(1.0, (height * width) / (1920 * 1080))
        
        # 2. Sharpness using optimized Laplacian
        laplacian = cv2.Laplacian(gray_sample, cv2.CV_64F)
        sharpness_score = min(1.0, float(np.var(laplacian)) / 500.0)
        
        # 3. Contrast using standard deviation (on sample)
        contrast_score = min(1.0, float(np.std(gray_sample)) / 80.0)
        
        # 4. Brightness assessment (on sample)
        mean_brightness = float(np.mean(gray_sample))
        if 100 <= mean_brightness <= 160:
            brightness_score = 1.0
        else:
            brightness_score = max(0.0, 1.0 - abs(mean_brightness - 130) / 130)
        
        # 5. Simplified noise estimation
        blur_kernel = (5, 5)
        blurred = cv2.GaussianBlur(gray_sample, blur_kernel, 0)
        noise_estimate = float(np.mean(np.abs(gray_sample.astype(np.float32) - blurred.astype(np.float32))))
        noise_score = max(0.0, 1.0 - noise_estimate / 20.0)
        
        # Calculate weighted score
        weights = [0.15, 0.25, 0.20, 0.20, 0.20]
        scores = [resolution_score, sharpness_score, contrast_score, brightness_score, noise_score]
        overall_score = sum(w * s for w, s in zip(weights, scores))
        
        # Determine processing recommendation
        if overall_score > 0.7:
            processing_recommendation = 'minimal'
        elif overall_score > 0.5:
            processing_recommendation = 'moderate'
        else:
            processing_recommendation = 'aggressive'
        
        return ImageQualityMetrics(
            resolution_score=resolution_score,
            sharpness_score=sharpness_score,
            contrast_score=contrast_score,
            brightness_score=brightness_score,
            noise_score=noise_score,
            overall_score=overall_score,
            processing_recommendation=processing_recommendation
        )
    
    def _minimal_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Minimal preprocessing for high-quality images."""
        # Just convert to grayscale
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()
    
    def _moderate_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Moderate preprocessing for medium-quality images."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Light denoising with optimized parameters
        denoised = cv2.fastNlMeansDenoising(gray, h=8, templateWindowSize=7, searchWindowSize=21)
        return denoised
    
    def _aggressive_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Aggressive preprocessing for low-quality images."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Optimized morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return cleaned
    
    def _generate_cache_key(self, image_path: str, force_quality: bool) -> str:
        """Generate cache key from image path and file modification time."""
        try:
            path_obj = Path(image_path)
            # Include file size and modification time for cache invalidation
            stat = path_obj.stat()
            cache_data = f"{image_path}_{stat.st_size}_{stat.st_mtime}_{force_quality}"
            return hashlib.sha256(cache_data.encode()).hexdigest()[:16]
        except:
            # Fallback to simple path hash if stat fails
            return hashlib.sha256(image_path.encode()).hexdigest()[:16]
    
    def _get_from_cache(self, cache_key: str) -> Optional[PreprocessingResult]:
        """Thread-safe cache retrieval."""
        with self._cache_lock:
            result = self.preprocessing_cache.get(cache_key)
            if result:
                # Update cache hit flag
                return PreprocessingResult(
                    processed_image=result.processed_image.copy(),
                    quality_metrics=result.quality_metrics,
                    processing_time=result.processing_time,
                    cache_hit=True,
                    processing_method=result.processing_method
                )
        return None
    
    def _store_in_cache(self, cache_key: str, result: PreprocessingResult):
        """Thread-safe cache storage with size management."""
        with self._cache_lock:
            # Manage cache size
            if len(self.preprocessing_cache) >= self.cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.preprocessing_cache))
                del self.preprocessing_cache[oldest_key]
            
            # Store copy to prevent external modifications
            cached_result = PreprocessingResult(
                processed_image=result.processed_image.copy(),
                quality_metrics=result.quality_metrics,
                processing_time=result.processing_time,
                cache_hit=False,
                processing_method=result.processing_method
            )
            self.preprocessing_cache[cache_key] = cached_result
    
    def batch_process_images(
        self, 
        image_paths: list, 
        max_workers: int = 4
    ) -> Dict[str, PreprocessingResult]:
        """Process multiple images in parallel."""
        
        results = {}
        
        def process_single(path):
            try:
                return path, self.process_image(path)
            except Exception as e:
                logger.error(f"Failed to process {path}: {str(e)}")
                return path, None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(process_single, path): path for path in image_paths}
            
            for future in future_to_path:
                path, result = future.result()
                if result:
                    results[path] = result
        
        logger.info(f"📊 Batch processed {len(results)}/{len(image_paths)} images successfully")
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        
        cache_hit_rate = 0.0
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total_requests > 0:
            cache_hit_rate = self.cache_stats["hits"] / total_requests
        
        stats = {
            "cache_stats": {
                "hit_rate": cache_hit_rate,
                "total_hits": self.cache_stats["hits"],
                "total_misses": self.cache_stats["misses"],
                "cache_size": len(self.preprocessing_cache),
                "cache_capacity": self.cache_size
            },
            "processing_performance": {},
            "quality_assessment_performance": {}
        }
        
        if self.processing_times:
            stats["processing_performance"] = {
                "avg_time": np.mean(self.processing_times),
                "min_time": np.min(self.processing_times),
                "max_time": np.max(self.processing_times),
                "total_operations": len(self.processing_times)
            }
        
        if self.quality_assessment_times:
            stats["quality_assessment_performance"] = {
                "avg_time": np.mean(self.quality_assessment_times),
                "min_time": np.min(self.quality_assessment_times),
                "max_time": np.max(self.quality_assessment_times)
            }
        
        return stats
    
    def clear_cache(self):
        """Clear the preprocessing cache."""
        with self._cache_lock:
            self.preprocessing_cache.clear()
            self.quality_cache.clear()
            self.cache_stats = {"hits": 0, "misses": 0}
        logger.info("🗑️ Preprocessing cache cleared")
    
    def optimize_for_performance(self):
        """Apply performance optimizations based on usage patterns."""
        
        # Analyze processing method usage
        method_usage = {}
        for result in self.preprocessing_cache.values():
            method = result.processing_method
            method_usage[method] = method_usage.get(method, 0) + 1
        
        if method_usage:
            most_used = max(method_usage, key=method_usage.get)
            logger.info(f"📈 Most used processing method: {most_used} ({method_usage[most_used]} times)")
            
            # Adjust cache size based on usage
            if len(self.preprocessing_cache) == self.cache_size:
                logger.info("💡 Consider increasing cache size for better hit rates")


# Global optimized processor instance
optimized_processor = OptimizedImageProcessor()


# Integration function for easy replacement
def preprocess_image_optimized(image_path: str, debug_mode: bool = False) -> np.ndarray:
    """
    Drop-in replacement for existing preprocessing functions.
    
    Args:
        image_path: Path to the image file
        debug_mode: Enable debug output
        
    Returns:
        Preprocessed image as numpy array
    """
    result = optimized_processor.process_image(image_path, debug_mode=debug_mode)
    return result.processed_image


# Performance monitoring decorator
def monitor_preprocessing_performance(func):
    """Decorator to monitor preprocessing performance."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            processing_time = time.time() - start_time
            logger.debug(f"⏱️ {func.__name__} completed in {processing_time:.3f}s")
            return result
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ {func.__name__} failed after {processing_time:.3f}s: {str(e)}")
            raise
    return wrapper 