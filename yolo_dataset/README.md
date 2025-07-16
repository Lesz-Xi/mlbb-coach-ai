# MLBB YOLO Dataset Annotation Setup

## 🎯 Overview

This directory contains the complete setup for annotating MLBB (Mobile Legends: Bang Bang) post-match screenshots for YOLO object detection training. The system is designed to detect 51 different classes of game elements including hero icons, performance badges, statistics, and UI elements.

## 📁 Directory Structure

```
yolo_dataset/
├── images/                    # Training images
│   ├── train/                # Training set (70%)
│   ├── val/                  # Validation set (15%)
│   └── test/                 # Test set (15%)
├── labels/                   # YOLO format annotations
│   ├── train/                # Training labels
│   ├── val/                  # Validation labels
│   └── test/                 # Test labels
├── configs/                  # Configuration files
│   ├── classes.yaml          # Class definitions
│   ├── dataset.yaml          # YOLO dataset config
│   └── label_studio_config.xml # Label Studio interface
├── annotations/              # Annotation workspace
│   ├── exports/              # Exported annotations
│   ├── backups/              # Annotation backups
│   └── guidelines.md         # Detailed annotation guidelines
├── setup_annotation.py       # Automated setup script
├── start_annotation.sh       # Label Studio startup script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Initial Setup

```bash
# Navigate to the annotation directory
cd skillshift-ai/yolo_dataset

# Install dependencies (if not already done)
pip install -r requirements.txt

# Run setup script (creates directories and configs)
python setup_annotation.py
```

### 2. Prepare Images

Place your MLBB screenshots in the appropriate directories:

```bash
# Training images (recommended: 500-1000 per category)
images/train/screenshot_001.jpg
images/train/screenshot_002.jpg
...

# Validation images (recommended: 100-200 per category)
images/val/screenshot_val_001.jpg
...

# Test images (reserved for final evaluation)
images/test/screenshot_test_001.jpg
...
```

### 3. Start Annotation

```bash
# Start Label Studio annotation server
./start_annotation.sh

# Or manually:
label-studio start --port 8080
```

Then:

1. Open http://localhost:8080 in your browser
2. Create a new account or login
3. Create a new project
4. Import the configuration: `configs/label_studio_config.xml`
5. Add your images and start annotating!

## 🏷️ Class Definitions

The system detects **51 classes** organized into categories:

### Hero Detection (10 classes)

- `hero_icon_ally_1-5`: Ally team hero portraits
- `hero_icon_enemy_1-5`: Enemy team hero portraits

### Performance Indicators (5 classes)

- `mvp_badge`, `mvp_loss_badge`: MVP recognition
- `medal_gold`, `medal_silver`, `medal_bronze`: Performance medals

### Match Information (6 classes)

- `match_type_classic/ranked/brawl/custom`: Game mode indicators
- `victory_text`, `defeat_text`: Match outcome

### Player Statistics (9 classes)

- `kda_display`: Kill/Death/Assist values
- `gold_amount`, `damage_dealt/taken`: Core metrics
- `healing_done`, `turret_damage`: Additional stats
- `gpm_display`, `participation_rate`, `match_duration`: Advanced metrics

### UI Elements (5 classes)

- `player_name`, `player_level`: Player identification
- `team_indicator_ally/enemy`: Team sections
- `scoreboard_container`: Main interface area

### Items & Equipment (8 classes)

- `item_icon_1-6`: Equipment slots
- `battle_spell_1-2`: Battle spell icons

### Special Elements (8 classes)

- `role_indicator`, `position_rank`: Player role and ranking
- `savage/maniac/legendary_indicator`: Achievement badges
- `ui_complete`, `text_readable`, `icons_clear`: Quality markers

## 📝 Annotation Guidelines

### Priority Order

1. **HIGH**: Hero icons, MVP badges, KDA displays, player names
2. **MEDIUM**: Performance medals, match results, damage statistics
3. **LOW**: Items, achievements, quality indicators

### Quality Standards

- **Bounding Boxes**: Tight, complete coverage of objects
- **Team Order**: Ally/Enemy 1-5 from top to bottom
- **Text Elements**: Include entire readable area
- **Clarity**: Only annotate clearly visible elements

### What to Annotate ✅

- Clear, unobstructed objects
- Complete UI elements
- Readable text at normal resolution
- Recognizable hero portraits

### What to Skip ❌

- Blurry or pixelated elements
- Partially cut-off objects
- Overlapping UI elements
- Corrupted/loading images

## 🔧 Configuration Files

### `classes.yaml`

Comprehensive class hierarchy with metadata and annotation priorities.

### `dataset.yaml`

YOLO-compatible dataset configuration with paths and class mappings.

### `label_studio_config.xml`

Label Studio interface configuration with:

- Color-coded class labels
- Team-based organization (green=ally, red=enemy)
- Comprehensive annotation instructions
- Quality control guidelines

## 📊 Data Requirements

### Minimum Dataset Size

- **High Priority Classes**: 200+ annotations each
- **Medium Priority**: 100+ annotations each
- **Low Priority**: 50+ annotations each

### Recommended Distribution

- **Training**: 70% of annotated data
- **Validation**: 15% of annotated data
- **Testing**: 15% of annotated data

### Image Quality

- **Resolution**: 1080p or higher preferred
- **Format**: JPG, PNG supported
- **Clarity**: Sharp, unblurred screenshots
- **Completeness**: Full post-match screen visible

## 🛠️ Export and Training Preparation

### Label Studio Export

1. Go to your project in Label Studio
2. Export → YOLO format
3. Download annotations
4. Extract to `labels/` directories

### Validation

```bash
# Verify dataset structure
ls images/train/ | wc -l  # Count training images
ls labels/train/ | wc -l  # Should match image count

# Check class distribution
python -c "
import glob
labels = glob.glob('labels/**/*.txt', recursive=True)
print(f'Total annotations: {len(labels)}')
"
```

### Training Integration

The `configs/dataset.yaml` file is ready for YOLO training:

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Train on MLBB dataset
model.train(data='configs/dataset.yaml', epochs=100)
```

## 🚨 Troubleshooting

### Common Issues

**Label Studio won't start**

```bash
# Check if port 8080 is available
lsof -i :8080

# Try different port
label-studio start --port 8081
```

**Images not loading**

- Ensure `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true`
- Check file permissions
- Verify image paths

**Export format issues**

- Verify class names match `dataset.yaml`
- Check bounding box coordinates (0-1 normalized)
- Ensure proper train/val/test split

### Quality Control

**Annotation Validation**

```bash
# Check for missing labels
find images/ -name "*.jpg" | while read img; do
    label="${img//images/labels}"
    label="${label//.jpg/.txt}"
    [ ! -f "$label" ] && echo "Missing: $label"
done
```

**Class Distribution**

```bash
# Count annotations per class
cat labels/**/*.txt | cut -d' ' -f1 | sort | uniq -c | sort -nr
```

## 📈 Next Steps

1. **Phase 1**: Complete annotation setup ✅
2. **Phase 2**: Collect and annotate 1000+ screenshots
3. **Phase 3**: Train initial YOLO model
4. **Phase 4**: Evaluate and iterate on model performance
5. **Phase 5**: Deploy trained model to MLBB Coach AI pipeline

## 🤝 Contributing

When adding new screenshots:

1. Use descriptive filenames (e.g., `mlbb_ranked_victory_001.jpg`)
2. Maintain consistent quality standards
3. Follow annotation guidelines strictly
4. Update documentation for new edge cases

## 📚 Resources

- [Label Studio Documentation](https://labelstud.io/guide/)
- [YOLO Training Guide](https://docs.ultralytics.com/modes/train/)
- [MLBB Coach AI Project](../README.md)
- [Annotation Guidelines](annotations/guidelines.md)

---

**Status**: Phase 1 Complete ✅  
**Next**: Begin screenshot collection and annotation  
**Target**: 1000+ high-quality annotations for production model
