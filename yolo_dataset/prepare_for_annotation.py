#!/usr/bin/env python3
"""
Prepare MLBB Screenshots for YOLO Annotation

This script organizes collected screenshots from the strategic collection
structure into the proper YOLO train/val/test structure for annotation.
"""

import shutil
from pathlib import Path


def organize_for_annotation():
    """Organize collected screenshots into YOLO annotation structure."""
    
    collection_dir = Path("collection")
    images_dir = Path("images")
    
    print("🎯 Preparing Screenshots for YOLO Annotation")
    print("=" * 50)
    
    # Clear existing images
    for split in ['train', 'val', 'test']:
        split_dir = images_dir / split
        for img in split_dir.glob("*"):
            if img.is_file():
                img.unlink()
    
    collected_images = []
    
    # Collect all screenshots from strategic structure
    for match_type in ['1_RANKED_MATCHES', '2_CLASSIC_MATCHES', '3_OTHER_MODES']:
        match_dir = collection_dir / match_type
        
        if not match_dir.exists():
            continue
            
        for performance in ['excellent_performance', 'good_performance', 
                          'average_performance', 'poor_performance']:
            perf_dir = match_dir / performance
            
            if not perf_dir.exists():
                continue
            
            # Collect from both damage_view and kda_view
            for view_type in ['damage_view', 'kda_view']:
                view_dir = perf_dir / view_type
                
                if not view_dir.exists():
                    continue
                
                # Find all images
                for ext in ['*.jpg', '*.png', '*.jpeg']:
                    for img_path in view_dir.glob(ext):
                        collected_images.append({
                            'path': img_path,
                            'match_type': match_type,
                            'performance': performance,
                            'view_type': view_type
                        })
    
    if not collected_images:
        print("❌ No screenshots found in collection directories!")
        print("\n💡 First collect screenshots using the strategic structure:")
        print("   1. collection/1_RANKED_MATCHES/good_performance/damage_view/")
        print("   2. collection/1_RANKED_MATCHES/good_performance/kda_view/")
        print("   3. etc...")
        return
    
    # Distribute to train/val/test (70/15/15 split)
    total_images = len(collected_images)
    train_count = int(total_images * 0.70)
    val_count = int(total_images * 0.15)
    
    # Sort for consistent distribution
    collected_images.sort(key=lambda x: str(x['path']))
    
    train_images = collected_images[:train_count]
    val_images = collected_images[train_count:train_count + val_count]
    test_images = collected_images[train_count + val_count:]
    
    # Copy images to annotation structure
    datasets = {
        'train': train_images,
        'val': val_images, 
        'test': test_images
    }
    
    for split, image_list in datasets.items():
        split_dir = images_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 {split.upper()} SET: {len(image_list)} images")
        
        for i, img_info in enumerate(image_list, 1):
            source_path = img_info['path']
            
            # Create descriptive filename
            match_clean = img_info['match_type'].replace('_MATCHES', '').lower()
            perf_clean = img_info['performance'].replace('_performance', '')
            view_clean = img_info['view_type'].replace('_view', '')
            
            new_name = f"mlbb_{match_clean}_{perf_clean}_{view_clean}_{i:03d}.jpg"
            target_path = split_dir / new_name
            
            try:
                shutil.copy2(source_path, target_path)
                print(f"  ✅ {source_path.name} → {new_name}")
            except Exception as e:
                print(f"  ❌ Failed to copy {source_path.name}: {e}")
    
    print("\n📊 Summary:")
    print(f"  📚 Training: {len(train_images)} images")
    print(f"  📊 Validation: {len(val_images)} images")
    print(f"  🧪 Testing: {len(test_images)} images")
    print(f"  📈 Total: {total_images} images")
    
    print("\n🚀 Next Steps:")
    print("1. Run: ./start_annotation.sh")
    print("2. Open: http://localhost:8080")
    print("3. Import config: configs/label_studio_config.xml")
    print("4. Start annotating! 🎯")


if __name__ == "__main__":
    organize_for_annotation() 