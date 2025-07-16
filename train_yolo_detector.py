#!/usr/bin/env python3
"""
YOLOv8 Training Pipeline for MLBB Coach AI
==========================================

This script provides a complete training pipeline for YOLOv8 object detection
to identify hero portraits, stat boxes, and UI elements in MLBB screenshots.

Features:
- Dataset preparation and validation
- Custom class configuration
- Advanced training parameters
- Model evaluation and metrics
- Export for production use
- Integration with existing pipeline

Usage:
    python train_yolo_detector.py --dataset ./dataset --epochs 100 --batch 16
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
import yaml
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLBBYOLOTrainer:
    """YOLOv8 trainer specifically designed for MLBB object detection."""
    
    def __init__(self, dataset_path: str, model_size: str = "yolov8n.pt"):
        """
        Initialize the YOLO trainer.
        
        Args:
            dataset_path: Path to the dataset directory
            model_size: YOLOv8 model size (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
        """
        self.dataset_path = Path(dataset_path)
        self.model_size = model_size
        self.model = None
        self.results = None
        
        # MLBB-specific class configuration
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
        
        logger.info(f"🎯 MLBB YOLO Trainer initialized with {len(self.mlbb_classes)} classes")
        
    def prepare_dataset(self) -> Dict[str, Any]:
        """
        Prepare and validate the dataset structure.
        
        Returns:
            Dataset configuration dictionary
        """
        logger.info("📂 Preparing dataset...")
        
        # Check dataset structure
        required_dirs = ['train', 'val', 'test']
        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            if not dir_path.exists():
                logger.error(f"❌ Missing directory: {dir_path}")
                raise FileNotFoundError(f"Dataset directory {dir_path} not found")
        
        # Count files in each split
        dataset_stats = {}
        for split in required_dirs:
            split_path = self.dataset_path / split
            image_files = list(split_path.glob('*.jpg')) + list(split_path.glob('*.png'))
            label_files = list(split_path.glob('*.txt'))
            
            dataset_stats[split] = {
                'images': len(image_files),
                'labels': len(label_files),
                'path': str(split_path)
            }
            
            logger.info(f"📊 {split.upper()}: {len(image_files)} images, {len(label_files)} labels")
        
        # Create dataset configuration
        dataset_config = {
            'path': str(self.dataset_path),
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'names': self.mlbb_classes
        }
        
        # Save dataset configuration
        config_path = self.dataset_path / 'dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"✅ Dataset configuration saved to {config_path}")
        return dataset_stats
        
    def validate_annotations(self, sample_size: int = 10) -> Dict[str, Any]:
        """
        Validate annotation format and quality.
        
        Args:
            sample_size: Number of samples to validate
            
        Returns:
            Validation results
        """
        logger.info(f"🔍 Validating annotations (sample size: {sample_size})...")
        
        validation_results = {
            'valid_annotations': 0,
            'invalid_annotations': 0,
            'class_distribution': {cls: 0 for cls in self.mlbb_classes.values()},
            'issues': []
        }
        
        # Sample validation files
        train_path = self.dataset_path / 'train'
        label_files = list(train_path.glob('*.txt'))[:sample_size]
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        validation_results['issues'].append(
                            f"{label_file.name}:{line_num} - Invalid format"
                        )
                        validation_results['invalid_annotations'] += 1
                        continue
                    
                    class_id = int(parts[0])
                    if class_id not in self.mlbb_classes:
                        validation_results['issues'].append(
                            f"{label_file.name}:{line_num} - Unknown class {class_id}"
                        )
                        validation_results['invalid_annotations'] += 1
                        continue
                    
                    # Check bounding box format
                    try:
                        bbox = [float(x) for x in parts[1:]]
                        if not all(0 <= x <= 1 for x in bbox):
                            validation_results['issues'].append(
                                f"{label_file.name}:{line_num} - Invalid bbox coordinates"
                            )
                            validation_results['invalid_annotations'] += 1
                            continue
                    except ValueError:
                        validation_results['issues'].append(
                            f"{label_file.name}:{line_num} - Invalid bbox format"
                        )
                        validation_results['invalid_annotations'] += 1
                        continue
                    
                    # Valid annotation
                    validation_results['valid_annotations'] += 1
                    class_name = self.mlbb_classes[class_id]
                    validation_results['class_distribution'][class_name] += 1
                    
            except Exception as e:
                validation_results['issues'].append(f"{label_file.name} - {str(e)}")
                validation_results['invalid_annotations'] += 1
        
        # Log results
        logger.info(f"✅ Valid annotations: {validation_results['valid_annotations']}")
        logger.info(f"❌ Invalid annotations: {validation_results['invalid_annotations']}")
        
        if validation_results['issues']:
            logger.warning(f"⚠️ Found {len(validation_results['issues'])} issues:")
            for issue in validation_results['issues'][:5]:  # Show first 5
                logger.warning(f"  - {issue}")
        
        return validation_results
    
    def train_model(self, 
                   epochs: int = 100,
                   batch_size: int = 16,
                   img_size: int = 640,
                   device: str = 'auto',
                   patience: int = 50,
                   save_period: int = 10) -> Dict[str, Any]:
        """
        Train the YOLOv8 model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Input image size
            device: Device to use ('auto', 'cpu', 'cuda')
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            
        Returns:
            Training results
        """
        logger.info("🚀 Starting YOLOv8 training...")
        
        # Initialize model
        self.model = YOLO(self.model_size)
        
        # Training configuration
        training_config = {
            'data': str(self.dataset_path / 'dataset.yaml'),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'device': device,
            'patience': patience,
            'save_period': save_period,
            'project': 'mlbb_yolo_training',
            'name': f'mlbb_detector_{int(time.time())}',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        }
        
        logger.info(f"📋 Training configuration: {training_config}")
        
        # Start training
        start_time = time.time()
        self.results = self.model.train(**training_config)
        training_time = time.time() - start_time
        
        logger.info(f"🎉 Training completed in {training_time:.2f} seconds")
        
        # Get training results
        results_dict = {
            'training_time': training_time,
            'epochs_completed': epochs,
            'best_model_path': str(self.results.save_dir / 'weights' / 'best.pt'),
            'last_model_path': str(self.results.save_dir / 'weights' / 'last.pt'),
            'results_dir': str(self.results.save_dir)
        }
        
        return results_dict
    
    def evaluate_model(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            model_path: Path to model weights (uses best if None)
            
        Returns:
            Evaluation metrics
        """
        logger.info("📊 Evaluating model...")
        
        if model_path:
            model = YOLO(model_path)
        elif self.results:
            model = YOLO(self.results.save_dir / 'weights' / 'best.pt')
        else:
            raise ValueError("No model available for evaluation")
        
        # Validate on test set
        validation_results = model.val(
            data=str(self.dataset_path / 'dataset.yaml'),
            split='test',
            save_json=True,
            save_hybrid=True
        )
        
        # Extract key metrics
        metrics = {
            'map_50': validation_results.box.map50,
            'map_50_95': validation_results.box.map,
            'precision': validation_results.box.p.mean(),
            'recall': validation_results.box.r.mean(),
            'f1': validation_results.box.f1.mean(),
            'class_metrics': {}
        }
        
        # Per-class metrics
        for i, class_name in self.mlbb_classes.items():
            if i < len(validation_results.box.ap):
                metrics['class_metrics'][class_name] = {
                    'ap_50': validation_results.box.ap50[i],
                    'ap_50_95': validation_results.box.ap[i],
                    'precision': validation_results.box.p[i],
                    'recall': validation_results.box.r[i],
                    'f1': validation_results.box.f1[i]
                }
        
        logger.info(f"📈 Model Performance:")
        logger.info(f"  - mAP@0.5: {metrics['map_50']:.3f}")
        logger.info(f"  - mAP@0.5:0.95: {metrics['map_50_95']:.3f}")
        logger.info(f"  - Precision: {metrics['precision']:.3f}")
        logger.info(f"  - Recall: {metrics['recall']:.3f}")
        logger.info(f"  - F1: {metrics['f1']:.3f}")
        
        return metrics
    
    def export_model(self, 
                    model_path: str, 
                    export_format: str = 'onnx',
                    output_dir: str = 'models') -> str:
        """
        Export model for production use.
        
        Args:
            model_path: Path to trained model
            export_format: Export format ('onnx', 'torchscript', 'tensorrt')
            output_dir: Output directory for exported model
            
        Returns:
            Path to exported model
        """
        logger.info(f"📦 Exporting model to {export_format}...")
        
        model = YOLO(model_path)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export model
        export_path = model.export(
            format=export_format,
            half=False,
            int8=False,
            dynamic=True,
            simplify=True,
            opset=11 if export_format == 'onnx' else None
        )
        
        logger.info(f"✅ Model exported to: {export_path}")
        return export_path
    
    def test_inference(self, model_path: str, test_image: str) -> Dict[str, Any]:
        """
        Test model inference on a sample image.
        
        Args:
            model_path: Path to model weights
            test_image: Path to test image
            
        Returns:
            Inference results
        """
        logger.info(f"🔍 Testing inference on {test_image}...")
        
        model = YOLO(model_path)
        
        # Run inference
        results = model(test_image, conf=0.5, iou=0.45)
        
        # Process results
        inference_results = {
            'detections': [],
            'confidence_scores': [],
            'class_counts': {cls: 0 for cls in self.mlbb_classes.values()}
        }
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.mlbb_classes.get(class_id, 'unknown')
                    confidence = float(box.conf[0])
                    bbox = box.xywh[0].tolist()
                    
                    inference_results['detections'].append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    
                    inference_results['confidence_scores'].append(confidence)
                    inference_results['class_counts'][class_name] += 1
        
        # Log results
        logger.info(f"🎯 Detected {len(inference_results['detections'])} objects")
        for class_name, count in inference_results['class_counts'].items():
            if count > 0:
                logger.info(f"  - {class_name}: {count}")
        
        return inference_results


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train YOLOv8 for MLBB object detection')
    parser.add_argument('--dataset', required=True, help='Path to dataset directory')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLOv8 model size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size')
    parser.add_argument('--device', default='auto', help='Device to use')
    parser.add_argument('--validate', action='store_true', help='Validate dataset only')
    parser.add_argument('--export', help='Export model to specified format')
    parser.add_argument('--test-image', help='Test image for inference')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = MLBBYOLOTrainer(args.dataset, args.model)
    
    # Prepare dataset
    dataset_stats = trainer.prepare_dataset()
    
    # Validate annotations
    validation_results = trainer.validate_annotations()
    
    if args.validate:
        logger.info("✅ Dataset validation completed")
        return
    
    # Train model
    training_results = trainer.train_model(
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device
    )
    
    # Evaluate model
    evaluation_results = trainer.evaluate_model()
    
    # Export model if requested
    if args.export:
        export_path = trainer.export_model(
            training_results['best_model_path'],
            args.export
        )
        logger.info(f"📦 Model exported to: {export_path}")
    
    # Test inference if requested
    if args.test_image:
        inference_results = trainer.test_inference(
            training_results['best_model_path'],
            args.test_image
        )
        logger.info("🔍 Inference test completed")
    
    logger.info("🎉 Training pipeline completed successfully!")


if __name__ == "__main__":
    main() 