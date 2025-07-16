#!/usr/bin/env python3
"""
MLBB YOLO Dataset Annotation Setup Script

This script automates the setup of Label Studio for MLBB screenshot annotation.
It creates the project, configures the labeling interface, and prepares the
dataset structure.
"""

import os
import sys
import subprocess
from pathlib import Path


class MLBBAnnotationSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.images_dir = self.project_root / "images"
        self.labels_dir = self.project_root / "labels"
        self.configs_dir = self.project_root / "configs"
        self.annotations_dir = self.project_root / "annotations"
        
    def check_dependencies(self):
        """Check if required dependencies are installed."""
        print("🔍 Checking dependencies...")
        
        try:
            __import__('label_studio')
            print("✅ Label Studio found")
        except ImportError:
            print("❌ Label Studio not found. Installing...")
            self.install_label_studio()
    
    def install_label_studio(self):
        """Install Label Studio and required dependencies."""
        print("📦 Installing Label Studio...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "label-studio==1.9.2",
                "label-studio-tools",
                "Pillow>=9.0.0",
                "PyYAML>=6.0"
            ])
            print("✅ Label Studio installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Label Studio: {e}")
            sys.exit(1)
    
    def create_project_structure(self):
        """Create the necessary directory structure."""
        print("📁 Creating project structure...")
        
        directories = [
            self.images_dir / "train",
            self.images_dir / "val", 
            self.images_dir / "test",
            self.labels_dir / "train",
            self.labels_dir / "val",
            self.labels_dir / "test",
            self.annotations_dir / "exports",
            self.annotations_dir / "backups"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ {directory}")
    
    def create_dataset_config(self):
        """Create YOLO dataset configuration file."""
        print("⚙️  Creating dataset configuration...")
        
        config = {
            "path": str(self.project_root.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 51,  # Number of classes
            "names": [
                # Hero Detection - Ally Team
                "hero_icon_ally_1", "hero_icon_ally_2", "hero_icon_ally_3", 
                "hero_icon_ally_4", "hero_icon_ally_5",
                # Hero Detection - Enemy Team  
                "hero_icon_enemy_1", "hero_icon_enemy_2", "hero_icon_enemy_3",
                "hero_icon_enemy_4", "hero_icon_enemy_5",
                # Performance Indicators
                "mvp_badge", "mvp_loss_badge", "medal_gold", "medal_silver", "medal_bronze",
                # Match Information
                "match_type_classic", "match_type_ranked", "match_type_brawl", "match_type_custom",
                "victory_text", "defeat_text",
                # Player Statistics
                "kda_display", "gold_amount", "damage_dealt", "damage_taken", 
                "healing_done", "turret_damage", "gpm_display", "participation_rate", "match_duration",
                # UI Elements
                "player_name", "player_level", "team_indicator_ally", "team_indicator_enemy", 
                "scoreboard_container",
                # Items and Equipment
                "item_icon_1", "item_icon_2", "item_icon_3", "item_icon_4", "item_icon_5", "item_icon_6",
                # Spells and Effects
                "battle_spell_1", "battle_spell_2",
                # Role and Achievements
                "role_indicator", "position_rank", "savage_indicator", "maniac_indicator", "legendary_indicator",
                # Quality Indicators
                "ui_complete", "text_readable", "icons_clear"
            ]
        }
        
        config_path = self.configs_dir / "dataset.yaml"
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Dataset config created: {config_path}")
    
    def create_annotation_guidelines(self):
        """Create detailed annotation guidelines."""
        print("📝 Creating annotation guidelines...")
        
        guidelines = """
# MLBB YOLO Annotation Guidelines

## Overview
This guide provides comprehensive instructions for annotating MLBB post-match screenshots for YOLO object detection training.

## Annotation Priority (High to Low)

### 1. HIGH PRIORITY - Core Detection Targets
- **Hero Icons**: Both ally and enemy team hero portraits
- **MVP Badges**: MVP crowns and performance badges
- **KDA Display**: Kill/Death/Assist statistics
- **Player Names**: IGN text for player identification

### 2. MEDIUM PRIORITY - Match Information
- **Performance Medals**: Gold, Silver, Bronze medals
- **Match Results**: Victory/Defeat text
- **Match Types**: Classic, Ranked, Brawl indicators
- **Damage Statistics**: Damage dealt/taken values

### 3. LOW PRIORITY - Additional Elements
- **Items**: Equipment slot icons
- **Achievements**: Savage, Maniac, Legendary indicators
- **Quality Markers**: UI completeness, text readability

## Bounding Box Guidelines

### General Rules
1. **Tight Boxes**: Draw boxes as close to object boundaries as possible
2. **Complete Coverage**: Ensure all parts of the object are included
3. **No Overlap**: Avoid overlapping bounding boxes when possible
4. **Readable Text**: For text elements, include entire readable area

### Hero Icon Annotation
- **Positioning**: Use ally_1-5 and enemy_1-5 from top to bottom
- **Size**: Include entire circular/square hero portrait
- **Team Identification**: Green = Ally, Red = Enemy
- **Clarity**: Only annotate clearly visible hero faces

### Text Element Annotation
- **Player Names**: Include entire IGN text area
- **KDA Values**: Capture complete "X/Y/Z" format
- **Statistics**: Include both number and unit (e.g., "67,551")
- **Labels**: Ensure text is readable at annotation resolution

### Performance Indicator Annotation
- **MVP Badges**: Include crown/star and any surrounding glow
- **Medals**: Capture entire medal circle/shape
- **Achievement Icons**: Include icon and any accompanying text

## Quality Control

### What to Annotate
✅ Clear, unobstructed objects
✅ Complete UI elements
✅ Readable text (normal resolution)
✅ Recognizable hero portraits

### What to Skip
❌ Blurry or pixelated elements
❌ Partially cut-off objects
❌ Overlapping UI elements
❌ Corrupted/loading images

## Export Guidelines

### Label Studio Export
1. Export in YOLO format
2. Verify class mappings match dataset.yaml
3. Check bounding box coordinates
4. Validate annotation completeness

### File Organization
- Images: `images/train|val|test/`
- Labels: `labels/train|val|test/`
- Maintain consistent naming convention

## Troubleshooting

### Common Issues
- **Overlapping Heroes**: Use position-based ordering (ally_1, ally_2, etc.)
- **Multiple MVP Badges**: Choose most prominent/clear badge
- **Unclear Text**: Skip if not clearly readable
- **Partial Screenshots**: Only annotate complete, visible elements

### Quality Metrics
- Aim for 95%+ annotation accuracy
- Minimum 200 annotations per high-priority class
- Balanced distribution across match types and scenarios
"""
        
        guidelines_path = self.annotations_dir / "guidelines.md"
        with open(guidelines_path, 'w') as f:
            f.write(guidelines)
        
        print(f"✅ Guidelines created: {guidelines_path}")
    
    def create_startup_script(self):
        """Create script to start Label Studio with proper configuration."""
        print("🚀 Creating startup script...")
        
        startup_script = f"""#!/bin/bash
# MLBB YOLO Annotation Startup Script

echo "🎯 Starting MLBB YOLO Annotation Environment..."

# Set environment variables
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={self.images_dir.absolute()}

# Create data directory if it doesn't exist
mkdir -p ~/.local/share/label-studio

echo "📁 Project Directory: {self.project_root.absolute()}"
echo "🖼️  Images Directory: {self.images_dir.absolute()}"
echo "🏷️  Config File: {self.configs_dir.absolute()}/label_studio_config.xml"

# Start Label Studio
echo "🚀 Starting Label Studio..."
echo "   💻 Open http://localhost:8080 in your browser"
echo "   📧 Create account or login"
echo "   🎯 Import config from: {self.configs_dir.absolute()}/label_studio_config.xml"

label-studio start --port 8080 \\
    --data-dir ~/.local/share/label-studio \\
    --log-level INFO

echo "👋 Label Studio session ended"
"""
        
        startup_path = self.project_root / "start_annotation.sh"
        with open(startup_path, 'w') as f:
            f.write(startup_script)
        
        # Make script executable
        os.chmod(startup_path, 0o755)
        
        print(f"✅ Startup script created: {startup_path}")
    
    def run_setup(self):
        """Execute the complete setup process."""
        print("🎯 MLBB YOLO Annotation Setup")
        print("=" * 50)
        
        try:
            self.check_dependencies()
            self.create_project_structure()
            self.create_dataset_config()
            self.create_annotation_guidelines()
            self.create_startup_script()
            
            print("\n" + "=" * 50)
            print("✅ Setup completed successfully!")
            print("\n📋 Next Steps:")
            print("1. Place MLBB screenshots in: images/train/, images/val/, images/test/")
            print("2. Run: ./start_annotation.sh")
            print("3. Open http://localhost:8080 in browser")
            print("4. Create Label Studio project")
            print("5. Import config: configs/label_studio_config.xml")
            print("6. Start annotating!")
            print("\n📖 Read guidelines: annotations/guidelines.md")
            
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    setup = MLBBAnnotationSetup()
    setup.run_setup() 