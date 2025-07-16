# YOLOv8 Integration Guide for MLBB Coach AI

## 🎯 Overview

This guide provides complete instructions for integrating YOLOv8 object detection into your MLBB Coach AI system. YOLOv8 will enhance your current OCR-based system with precise detection of:

- **Hero Portraits** - Identify heroes in post-match screens
- **Stat Boxes** - KDA, damage, gold, and other statistics
- **UI Elements** - Timers, minimap, score indicators, badges
- **Minimap Regions** - Enhanced minimap tracking

## 🚀 Installation

### 1. Install YOLOv8 Dependencies

```bash
# Install required packages
pip install ultralytics==8.0.196
pip install supervision==0.18.0
pip install roboflow==1.1.9

# Verify installation
python -c "from ultralytics import YOLO; print('YOLOv8 installed successfully')"
```

### 2. Download Pre-trained Model (Optional)

```bash
# Download YOLOv8 nano model (fastest)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Or download larger models for better accuracy
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"  # Small
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"  # Medium
```

### 3. Set Up Directory Structure

```bash
# Create required directories
mkdir -p models/
mkdir -p dataset/train/
mkdir -p dataset/val/
mkdir -p dataset/test/
mkdir -p results/
```

## 📊 Dataset Preparation

### 1. Annotation Format

Your dataset should follow YOLO format with these classes:

```yaml
# Class mapping (dataset.yaml)
names:
  0: hero_portrait
  1: kda_box
  2: damage_box
  3: gold_box
  4: minimap
  5: timer
  6: score_indicator
  7: mvp_badge
  8: medal_bronze
  9: medal_silver
  10: medal_gold
  11: team_indicator
  12: player_name
  13: hero_name
  14: kill_indicator
  15: death_indicator
  16: assist_indicator
```

### 2. Annotation Tools

**Recommended tools:**

- [Roboflow](https://roboflow.com/) - Web-based annotation
- [LabelImg](https://github.com/tzutalin/labelImg) - Desktop annotation
- [CVAT](https://github.com/openvinotoolkit/cvat) - Advanced annotation

### 3. Dataset Structure

```
dataset/
├── train/
│   ├── image1.jpg
│   ├── image1.txt
│   ├── image2.jpg
│   └── image2.txt
├── val/
│   ├── image3.jpg
│   ├── image3.txt
│   └── ...
├── test/
│   ├── image4.jpg
│   ├── image4.txt
│   └── ...
└── dataset.yaml
```

## 🏋️ Training

### 1. Validate Your Dataset

```bash
# Validate dataset format
python train_yolo_detector.py --dataset ./dataset --validate
```

### 2. Train the Model

```bash
# Basic training
python train_yolo_detector.py \
    --dataset ./dataset \
    --epochs 100 \
    --batch 16 \
    --model yolov8n.pt

# Advanced training with custom parameters
python train_yolo_detector.py \
    --dataset ./dataset \
    --epochs 200 \
    --batch 32 \
    --model yolov8s.pt \
    --img-size 640 \
    --device cuda
```

### 3. Monitor Training

Training results will be saved to `mlbb_yolo_training/mlbb_detector_[timestamp]/`:

- `weights/best.pt` - Best model weights
- `weights/last.pt` - Last checkpoint
- `results.png` - Training curves
- `confusion_matrix.png` - Validation results

### 4. Export Model

```bash
# Export to ONNX for production
python train_yolo_detector.py \
    --dataset ./dataset \
    --export onnx \
    --model models/best.pt
```

## 🔧 Integration with Existing System

### 1. Update Configuration

Copy your trained model to the models directory:

```bash
# Copy best model
cp mlbb_yolo_training/mlbb_detector_*/weights/best.pt models/mlbb_yolo_best.pt
```

### 2. Test YOLOv8 Detection Service

```python
# Test the detection service
from core.services.yolo_detection_service import get_yolo_detection_service

service = get_yolo_detection_service()
results = service.detect_objects("path/to/screenshot.png")
print(f"Detected {len(results['detections'])} objects")
```

### 3. Integration with OCR Fallback

The system automatically uses YOLOv8 when OCR confidence is below 0.7:

```python
from core.yolo_fallback import should_use_yolo, make_fallback_decision

# Simple fallback check
if should_use_yolo(ocr_confidence=0.6):
    # Use YOLOv8 detection
    yolo_results = yolo_service.detect_objects(image_path)

# Advanced fallback decision
decision = make_fallback_decision(
    ocr_confidence=0.6,
    yolo_confidence=0.8,
    quality_metrics={"has_glare": True}
)
```

### 4. Enhanced Minimap Tracking

```python
from core.enhanced_minimap_tracker import create_enhanced_minimap_tracker

# Create enhanced tracker with YOLOv8
tracker = create_enhanced_minimap_tracker("YourIGN", use_yolo=True)

# Track movement with enhanced detection
movement_events = tracker.track_movement_from_frame(timestamped_frame)
```

## 🎮 Usage Examples

### 1. Batch Processing Screenshots

```bash
# Process multiple screenshots
python batch_yolo_inference.py \
    --model models/mlbb_yolo_best.pt \
    --input screenshots/ \
    --output results/ \
    --confidence 0.5 \
    --annotate \
    --export-mlbb
```

### 2. Real-time Detection

```python
from core.services.yolo_detection_service import get_yolo_detection_service

service = get_yolo_detection_service()

# Detect hero portraits
heroes = service.detect_hero_portraits("screenshot.png")
for hero in heroes:
    print(f"Hero: {hero['class_name']}, Confidence: {hero['confidence']:.2f}")

# Detect UI elements
ui_elements = service.detect_ui_elements("screenshot.png")
print(f"Found {len(ui_elements)} UI elements")

# Get minimap region
minimap = service.get_minimap_region("screenshot.png")
if minimap:
    print(f"Minimap found with confidence: {minimap['confidence']:.2f}")
```

### 3. Integration with Existing Analysis

```python
# In your existing analysis pipeline
from core.ultimate_parsing_system import ultimate_parsing_system
from core.services.yolo_detection_service import get_yolo_detection_service

def enhanced_analysis(image_path, ign):
    # Use existing ultimate parsing
    ocr_result = ultimate_parsing_system.analyze_screenshot_ultimate(
        image_path, ign
    )

    # Enhance with YOLOv8 if needed
    yolo_service = get_yolo_detection_service()

    if ocr_result.confidence < 0.7:
        yolo_result = yolo_service.detect_objects(
            image_path,
            ocr_confidence=ocr_result.confidence
        )

        # Combine results
        enhanced_result = combine_ocr_yolo_results(ocr_result, yolo_result)
        return enhanced_result

    return ocr_result
```

## 📈 Performance Optimization

### 1. Model Selection

- **YOLOv8n**: Fastest inference (~10ms), good for real-time
- **YOLOv8s**: Balanced speed/accuracy (~20ms)
- **YOLOv8m**: Better accuracy (~40ms)
- **YOLOv8l/x**: Highest accuracy (~80ms+)

### 2. Batch Processing

```python
# Process multiple images efficiently
results = []
for image_batch in batch_images(image_paths, batch_size=8):
    batch_results = yolo_service.detect_objects_batch(image_batch)
    results.extend(batch_results)
```

### 3. GPU Acceleration

```python
# Ensure GPU usage
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name()}")

# YOLOv8 will automatically use GPU if available
```

## 🔍 Best Practices

### 1. Confidence Thresholds

```python
# Adjust confidence thresholds by use case
hero_detection_conf = 0.7  # High confidence for hero identification
ui_detection_conf = 0.5    # Medium confidence for UI elements
minimap_detection_conf = 0.6  # Balanced for minimap tracking
```

### 2. Class Balancing

Ensure your training dataset has balanced representation:

- Hero portraits: 30-40% of annotations
- Stat boxes: 25-35% of annotations
- UI elements: 20-30% of annotations
- Badges/medals: 10-15% of annotations

### 3. Data Augmentation

```python
# Training with augmentation (automatic in YOLOv8)
training_config = {
    'hsv_h': 0.015,      # Hue augmentation
    'hsv_s': 0.7,        # Saturation augmentation
    'hsv_v': 0.4,        # Value augmentation
    'degrees': 0.0,      # Rotation (usually 0 for UI elements)
    'translate': 0.1,    # Translation
    'scale': 0.5,        # Scale variation
    'fliplr': 0.5,       # Horizontal flip
    'mosaic': 1.0,       # Mosaic augmentation
}
```

## 🐛 Troubleshooting

### 1. Common Issues

**Model not found:**

```bash
# Check model path
ls -la models/
# Re-download if needed
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**CUDA out of memory:**

```bash
# Reduce batch size
python train_yolo_detector.py --batch 8  # Instead of 16
```

**Poor detection accuracy:**

```bash
# Check annotation quality
python train_yolo_detector.py --dataset ./dataset --validate
# Increase training epochs
python train_yolo_detector.py --epochs 200
```

### 2. Performance Monitoring

```python
# Monitor detection performance
from core.services.yolo_detection_service import get_yolo_detection_service

service = get_yolo_detection_service()
health = service.health_check()
print(f"Service status: {health['status']}")

stats = service.get_performance_stats()
print(f"Average inference time: {stats['avg_inference_time']:.3f}s")
```

## 📝 Advanced Configuration

### 1. Custom Classes

To add new classes, update the class mapping in:

- `train_yolo_detector.py` (MLBBYOLOTrainer.mlbb_classes)
- `core/services/yolo_detection_service.py` (YOLODetectionService.mlbb_classes)
- `batch_yolo_inference.py` (MLBBYOLOInference.mlbb_classes)

### 2. Model Ensemble

```python
# Use multiple models for better accuracy
def ensemble_detection(image_path):
    model1 = YOLO('models/yolov8n_best.pt')
    model2 = YOLO('models/yolov8s_best.pt')

    results1 = model1(image_path)
    results2 = model2(image_path)

    # Combine results with weighted voting
    return combine_ensemble_results(results1, results2)
```

### 3. Real-time Optimization

```python
# Optimize for real-time processing
service = YOLODetectionService(
    model_path="models/yolov8n_best.pt",  # Use nano model
    confidence_threshold=0.6,              # Slightly higher threshold
    enable_fallback=True                   # Keep fallback logic
)
```

## 🎉 Success Metrics

After successful integration, you should see:

1. **Detection Accuracy**: >85% mAP@0.5 on test set
2. **Inference Speed**: <50ms per image on GPU
3. **Fallback Rate**: 20-30% of frames use YOLOv8 (depends on OCR quality)
4. **System Integration**: Seamless operation with existing pipeline

## 📞 Support

For issues or questions:

1. Check the troubleshooting section above
2. Review YOLOv8 documentation: https://docs.ultralytics.com
3. Examine log files in `temp/diagnostics/`
4. Test individual components with provided scripts

## 🔄 Updates and Maintenance

### 1. Model Updates

```bash
# Retrain with new data
python train_yolo_detector.py --dataset ./updated_dataset --epochs 50

# Update model in production
cp new_model.pt models/mlbb_yolo_best.pt
```

### 2. Performance Monitoring

```python
# Regular performance checks
from core.yolo_fallback import get_fallback_stats

stats = get_fallback_stats()
print(f"YOLO fallback rate: {stats['yolo_fallback_rate']:.1%}")
```

This completes your YOLOv8 integration! Your MLBB Coach AI system now has enhanced object detection capabilities with intelligent fallback logic.
