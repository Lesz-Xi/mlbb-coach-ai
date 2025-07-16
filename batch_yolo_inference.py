#!/usr/bin/env python3
"""
Batch YOLOv8 Inference for MLBB Screenshots
==========================================

This script processes multiple MLBB screenshots using trained YOLOv8 models
for hero portrait detection, stat box identification, and UI element recognition.

Features:
- Batch processing of screenshot directories
- Confidence filtering and NMS
- JSON export of detections
- Integration with existing MLBB pipeline
- Performance monitoring

Usage:
    python batch_yolo_inference.py --model models/best.pt --input screenshots/ --output results/
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLBBYOLOInference:
    """Batch inference engine for MLBB screenshots using YOLOv8."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to trained YOLOv8 model
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # MLBB class mapping (must match training)
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
        
        # Performance tracking
        self.inference_times = []
        self.detection_counts = []
        
        self._load_model()
    
    def _load_model(self):
        """Load the YOLOv8 model."""
        logger.info(f"🔄 Loading YOLOv8 model from {self.model_path}")
        
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"✅ Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def process_single_image(self, image_path: str, save_annotated: bool = False) -> Dict[str, Any]:
        """
        Process a single image and return detections.
        
        Args:
            image_path: Path to input image
            save_annotated: Whether to save annotated image
            
        Returns:
            Dictionary containing detection results
        """
        start_time = time.time()
        
        try:
            # Run inference
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
                        
                        # Get bounding box coordinates
                        xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        xywh = box.xywh[0].tolist()  # [x_center, y_center, width, height]
                        
                        detection = {
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox_xyxy': xyxy,
                            'bbox_xywh': xywh,
                            'area': xywh[2] * xywh[3]
                        }
                        
                        detections.append(detection)
                        confidence_scores.append(confidence)
                        class_counts[class_name] += 1
                
                # Save annotated image if requested
                if save_annotated:
                    annotated_path = str(Path(image_path).parent / f"annotated_{Path(image_path).name}")
                    annotated_img = result.plot()
                    cv2.imwrite(annotated_path, annotated_img)
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.detection_counts.append(len(detections))
            
            # Compile results
            results_dict = {
                'image_path': image_path,
                'inference_time': inference_time,
                'detections': detections,
                'detection_count': len(detections),
                'confidence_scores': confidence_scores,
                'class_counts': class_counts,
                'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
                'max_confidence': max(confidence_scores) if confidence_scores else 0.0,
                'min_confidence': min(confidence_scores) if confidence_scores else 0.0
            }
            
            logger.info(f"🎯 Processed {Path(image_path).name}: "
                       f"{len(detections)} detections in {inference_time:.3f}s")
            
            return results_dict
            
        except Exception as e:
            logger.error(f"❌ Error processing {image_path}: {e}")
            return {
                'image_path': image_path,
                'error': str(e),
                'detections': [],
                'detection_count': 0
            }
    
    def process_directory(self, 
                         input_dir: str, 
                         output_dir: str,
                         save_annotated: bool = False,
                         file_extensions: List[str] = ['.jpg', '.jpeg', '.png']) -> Dict[str, Any]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results
            save_annotated: Whether to save annotated images
            file_extensions: Valid image file extensions
            
        Returns:
            Batch processing results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"📂 Processing directory: {input_dir}")
        logger.info(f"📁 Output directory: {output_dir}")
        
        # Find all image files
        image_files = []
        for ext in file_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        logger.info(f"📊 Found {len(image_files)} images to process")
        
        # Process each image
        all_results = []
        successful_processed = 0
        failed_processed = 0
        
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"🔄 Processing {i}/{len(image_files)}: {image_file.name}")
            
            result = self.process_single_image(str(image_file), save_annotated)
            all_results.append(result)
            
            if 'error' in result:
                failed_processed += 1
            else:
                successful_processed += 1
        
        # Compile batch statistics
        batch_stats = {
            'total_images': len(image_files),
            'successful_processed': successful_processed,
            'failed_processed': failed_processed,
            'total_detections': sum(r.get('detection_count', 0) for r in all_results),
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0.0,
            'total_processing_time': sum(self.inference_times),
            'avg_detections_per_image': np.mean(self.detection_counts) if self.detection_counts else 0.0
        }
        
        # Class distribution across all images
        class_distribution = {cls: 0 for cls in self.mlbb_classes.values()}
        for result in all_results:
            for cls, count in result.get('class_counts', {}).items():
                class_distribution[cls] += count
        
        # Final results
        final_results = {
            'batch_stats': batch_stats,
            'class_distribution': class_distribution,
            'individual_results': all_results,
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold,
            'processing_timestamp': time.time()
        }
        
        # Save results to JSON
        results_file = output_path / 'batch_inference_results.json'
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"📊 Batch processing completed:")
        logger.info(f"  - Total images: {batch_stats['total_images']}")
        logger.info(f"  - Successful: {batch_stats['successful_processed']}")
        logger.info(f"  - Failed: {batch_stats['failed_processed']}")
        logger.info(f"  - Total detections: {batch_stats['total_detections']}")
        logger.info(f"  - Avg inference time: {batch_stats['avg_inference_time']:.3f}s")
        logger.info(f"  - Results saved to: {results_file}")
        
        return final_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.inference_times:
            return {"message": "No inference data available"}
        
        return {
            'total_inferences': len(self.inference_times),
            'avg_inference_time': np.mean(self.inference_times),
            'min_inference_time': min(self.inference_times),
            'max_inference_time': max(self.inference_times),
            'total_processing_time': sum(self.inference_times),
            'avg_detections_per_image': np.mean(self.detection_counts),
            'throughput_images_per_second': len(self.inference_times) / sum(self.inference_times)
        }
    
    def filter_detections_by_class(self, 
                                  results: Dict[str, Any], 
                                  target_classes: List[str]) -> Dict[str, Any]:
        """
        Filter detections to only include specific classes.
        
        Args:
            results: Results from process_single_image or process_directory
            target_classes: List of class names to keep
            
        Returns:
            Filtered results
        """
        if 'individual_results' in results:
            # Batch results
            for result in results['individual_results']:
                if 'detections' in result:
                    result['detections'] = [
                        det for det in result['detections'] 
                        if det['class_name'] in target_classes
                    ]
        else:
            # Single image results
            if 'detections' in results:
                results['detections'] = [
                    det for det in results['detections'] 
                    if det['class_name'] in target_classes
                ]
        
        return results
    
    def export_for_mlbb_pipeline(self, 
                                results: Dict[str, Any], 
                                output_path: str) -> str:
        """
        Export results in format compatible with MLBB pipeline.
        
        Args:
            results: Inference results
            output_path: Path to save exported data
            
        Returns:
            Path to exported file
        """
        # Convert to MLBB pipeline format
        mlbb_format = {
            'detection_method': 'yolo_v8',
            'confidence_threshold': self.confidence_threshold,
            'timestamp': time.time(),
            'detections': {}
        }
        
        # Process individual results
        if 'individual_results' in results:
            for result in results['individual_results']:
                if 'detections' in result:
                    image_name = Path(result['image_path']).name
                    mlbb_format['detections'][image_name] = {
                        'hero_portraits': [],
                        'ui_elements': [],
                        'stat_boxes': [],
                        'confidence_scores': []
                    }
                    
                    for detection in result['detections']:
                        class_name = detection['class_name']
                        
                        # Categorize detections
                        if class_name == 'hero_portrait':
                            mlbb_format['detections'][image_name]['hero_portraits'].append(detection)
                        elif class_name in ['kda_box', 'damage_box', 'gold_box']:
                            mlbb_format['detections'][image_name]['stat_boxes'].append(detection)
                        else:
                            mlbb_format['detections'][image_name]['ui_elements'].append(detection)
                        
                        mlbb_format['detections'][image_name]['confidence_scores'].append(
                            detection['confidence']
                        )
        
        # Save exported data
        export_file = Path(output_path) / 'mlbb_pipeline_export.json'
        with open(export_file, 'w') as f:
            json.dump(mlbb_format, f, indent=2)
        
        logger.info(f"📦 MLBB pipeline export saved to: {export_file}")
        return str(export_file)


def main():
    """Main batch inference script."""
    parser = argparse.ArgumentParser(description='Batch YOLOv8 inference for MLBB screenshots')
    parser.add_argument('--model', required=True, help='Path to trained YOLOv8 model')
    parser.add_argument('--input', required=True, help='Input directory containing images')
    parser.add_argument('--output', required=True, help='Output directory for results')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--annotate', action='store_true', help='Save annotated images')
    parser.add_argument('--export-mlbb', action='store_true', help='Export for MLBB pipeline')
    parser.add_argument('--filter-classes', nargs='+', help='Filter specific classes')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference_engine = MLBBYOLOInference(args.model, args.confidence)
    
    # Process directory
    results = inference_engine.process_directory(
        args.input,
        args.output,
        save_annotated=args.annotate
    )
    
    # Filter classes if specified
    if args.filter_classes:
        results = inference_engine.filter_detections_by_class(results, args.filter_classes)
        logger.info(f"🔍 Filtered results for classes: {args.filter_classes}")
    
    # Export for MLBB pipeline if requested
    if args.export_mlbb:
        export_path = inference_engine.export_for_mlbb_pipeline(results, args.output)
        logger.info(f"📦 MLBB pipeline export completed: {export_path}")
    
    # Print performance stats
    stats = inference_engine.get_performance_stats()
    logger.info(f"📈 Performance Statistics: {stats}")
    
    logger.info("🎉 Batch inference completed successfully!")


if __name__ == "__main__":
    main() 