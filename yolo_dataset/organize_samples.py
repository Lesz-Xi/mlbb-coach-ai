#!/usr/bin/env python3
"""
Organize MLBB Screenshots for YOLO Dataset

This script automatically copies and organizes existing MLBB screenshots
into the proper train/val/test directory structure for annotation.
"""

import shutil
from pathlib import Path


def organize_screenshots():
    """Organize existing screenshots into YOLO dataset structure."""
    
    # Define paths
    project_root = Path(__file__).parent.parent
    yolo_root = Path(__file__).parent
    
    # Source directories
    sources = [
        {
            'path': project_root / "Screenshot-Test-Excellent",
            'prefix': "excellent",
            'split': "train"  # High quality for training
        },
        {
            'path': project_root / "Screenshot-Test-Good", 
            'prefix': "good",
            'split': "train"  # Good quality for training
        },
        {
            'path': project_root / "Screenshot-Test-Average",
            'prefix': "average", 
            'split': "val"  # Average for validation
        },
        {
            'path': project_root / "Screenshot-Test-Poor",
            'prefix': "poor",
            'split': "test"  # Poor quality for testing robustness
        },
        {
            'path': project_root / "dashboard-ui" / "uploads",
            'prefix': "upload",
            'split': "train"  # Additional training data
        }
    ]
    
    print("🎯 Organizing MLBB Screenshots for YOLO Dataset")
    print("=" * 50)
    
    copied_files = []
    
    for source in sources:
        source_path = source['path']
        
        if not source_path.exists():
            print(f"⚠️ Skipping {source_path} (doesn't exist)")
            continue
            
        # Get image files
        image_files = []
        for ext in ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
            image_files.extend(source_path.glob(ext))
        
        # Filter out .DS_Store and other non-image files
        image_files = [f for f in image_files if f.name != '.DS_Store']
        
        if not image_files:
            print(f"⚠️ No images found in {source_path}")
            continue
            
        print(f"\n📁 Processing {source['prefix']}: {len(image_files)} images")
        
        # Determine target directory
        target_dir = yolo_root / "images" / source['split']
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy and rename files
        for i, image_file in enumerate(image_files, 1):
            # Create descriptive filename
            new_name = f"mlbb_{source['prefix']}_{i:02d}.jpg"
            target_path = target_dir / new_name
            
            try:
                # Copy and convert to JPG format
                shutil.copy2(image_file, target_path)
                copied_files.append({
                    'source': str(image_file),
                    'target': str(target_path),
                    'split': source['split'],
                    'quality': source['prefix']
                })
                print(f"  ✅ {image_file.name} → {new_name}")
                
            except Exception as e:
                print(f"  ❌ Failed to copy {image_file.name}: {e}")
    
    # Summary
    print("\n📊 Organization Summary:")
    train_count = sum(1 for f in copied_files if f['split'] == 'train')
    val_count = sum(1 for f in copied_files if f['split'] == 'val') 
    test_count = sum(1 for f in copied_files if f['split'] == 'test')
    
    print(f"  📚 Training Set: {train_count} images")
    print(f"  📊 Validation Set: {val_count} images")
    print(f"  🧪 Test Set: {test_count} images")
    print(f"  📈 Total: {len(copied_files)} images")
    
    # Quality distribution
    quality_dist = {}
    for file_info in copied_files:
        quality = file_info['quality']
        quality_dist[quality] = quality_dist.get(quality, 0) + 1
    
    print("\n🎨 Quality Distribution:")
    for quality, count in quality_dist.items():
        print(f"  {quality.capitalize()}: {count} images")
    
    # Next steps
    print("\n🚀 Next Steps:")
    print("1. Run: ./start_annotation.sh")
    print("2. Open: http://localhost:8080")
    print("3. Create new Label Studio project")
    print("4. Import config: configs/label_studio_config.xml")
    print("5. Start annotating! 🎯")
    
    print("\n💡 Annotation Tips:")
    print("- Start with 'excellent' and 'good' quality images")
    print("- Focus on HIGH priority classes first:")
    print("  • Hero icons (ally_1-5, enemy_1-5)")
    print("  • MVP badges and medals")
    print("  • KDA displays")
    print("  • Player names")
    print("- Use validation set to check annotation quality")
    
    return copied_files


def validate_organization():
    """Validate the organized dataset structure."""
    print("\n🔍 Validating dataset organization...")
    
    yolo_root = Path(__file__).parent
    
    for split in ['train', 'val', 'test']:
        images_dir = yolo_root / "images" / split
        
        if images_dir.exists():
            jpg_count = len(list(images_dir.glob("*.jpg")))
            png_count = len(list(images_dir.glob("*.png")))
            image_count = jpg_count + png_count
            print(f"  📁 {split}: {image_count} images ready for annotation")
        else:
            print(f"  ⚠️ {split}: Directory not found")
    
    print("\n✅ Dataset ready for annotation!")


if __name__ == "__main__":
    copied_files = organize_screenshots()
    validate_organization()
    
    print("\n" + "=" * 50)
    print("🎉 Screenshot organization complete!")
    print("Ready to start annotation workflow! 🏷️") 