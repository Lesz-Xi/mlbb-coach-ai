#!/usr/bin/env python3
"""
MLBB YOLO Dataset Validation Script

This script validates the quality and completeness of the annotated dataset.
It checks for missing files, validates bounding box formats, and provides
statistics about class distribution.
"""

import glob
from pathlib import Path
from collections import Counter
import yaml


class DatasetValidator:
    def __init__(self, dataset_root):
        self.dataset_root = Path(dataset_root)
        self.images_dir = self.dataset_root / "images"
        self.labels_dir = self.dataset_root / "labels"
        self.config_file = self.dataset_root / "configs" / "dataset.yaml"
        
        # Load class names from config
        self.class_names = self._load_class_names()
        
    def _load_class_names(self):
        """Load class names from dataset configuration."""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('names', [])
        except Exception as e:
            print(f"⚠️ Could not load class names: {e}")
            return []
    
    def validate_file_structure(self):
        """Validate that all images have corresponding label files."""
        print("🔍 Validating file structure...")
        
        results = {
            'train': {'missing_labels': [], 'orphaned_labels': []},
            'val': {'missing_labels': [], 'orphaned_labels': []},
            'test': {'missing_labels': [], 'orphaned_labels': []}
        }
        
        for split in ['train', 'val', 'test']:
            # Check for missing label files
            image_files = glob.glob(str(self.images_dir / split / "*.jpg"))
            image_files.extend(glob.glob(str(self.images_dir / split / "*.png")))
            
            for image_path in image_files:
                image_name = Path(image_path).stem
                label_path = self.labels_dir / split / f"{image_name}.txt"
                
                if not label_path.exists():
                    results[split]['missing_labels'].append(image_name)
            
            # Check for orphaned label files
            label_files = glob.glob(str(self.labels_dir / split / "*.txt"))
            
            for label_path in label_files:
                label_name = Path(label_path).stem
                image_path1 = self.images_dir / split / f"{label_name}.jpg"
                image_path2 = self.images_dir / split / f"{label_name}.png"
                
                if not image_path1.exists() and not image_path2.exists():
                    results[split]['orphaned_labels'].append(label_name)
        
        # Report results
        total_issues = 0
        for split, issues in results.items():
            missing = len(issues['missing_labels'])
            orphaned = len(issues['orphaned_labels'])
            total_issues += missing + orphaned
            
            if missing > 0:
                print(f"❌ {split}: {missing} images missing labels")
                if missing <= 5:  # Show first 5 examples
                    for example in issues['missing_labels'][:5]:
                        print(f"   - {example}")
            
            if orphaned > 0:
                print(f"⚠️ {split}: {orphaned} orphaned labels")
        
        if total_issues == 0:
            print("✅ File structure validation passed!")
        
        return results
    
    def validate_label_format(self):
        """Validate YOLO label file format."""
        print("\n🔍 Validating label format...")
        
        format_errors = []
        bbox_errors = []
        class_errors = []
        
        for split in ['train', 'val', 'test']:
            label_files = glob.glob(str(self.labels_dir / split / "*.txt"))
            
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                        
                        parts = line.split()
                        
                        # Check format: class_id x_center y_center width height
                        if len(parts) != 5:
                            format_errors.append(
                                f"{label_file}:{line_num} - "
                                f"Expected 5 values, got {len(parts)}"
                            )
                            continue
                        
                        try:
                            class_id = int(parts[0])
                            bbox = [float(x) for x in parts[1:]]
                        except ValueError:
                            format_errors.append(
                                f"{label_file}:{line_num} - "
                                f"Invalid number format"
                            )
                            continue
                        
                        # Validate class ID
                        if class_id < 0 or class_id >= len(self.class_names):
                            class_errors.append(
                                f"{label_file}:{line_num} - "
                                f"Invalid class ID: {class_id}"
                            )
                        
                        # Validate bounding box coordinates (0-1 range)
                        for i, coord in enumerate(bbox):
                            if coord < 0 or coord > 1:
                                bbox_errors.append(
                                    f"{label_file}:{line_num} - "
                                    f"Coordinate {i+1} out of range: {coord}"
                                )
                
                except Exception as e:
                    format_errors.append(f"{label_file} - Could not read: {e}")
        
        # Report validation results
        total_errors = len(format_errors) + len(bbox_errors) + len(class_errors)
        
        if format_errors:
            print(f"❌ {len(format_errors)} format errors found")
            for error in format_errors[:3]:  # Show first 3
                print(f"   - {error}")
        
        if bbox_errors:
            print(f"❌ {len(bbox_errors)} bounding box errors found")
            for error in bbox_errors[:3]:  # Show first 3
                print(f"   - {error}")
        
        if class_errors:
            print(f"❌ {len(class_errors)} class ID errors found")
            for error in class_errors[:3]:  # Show first 3
                print(f"   - {error}")
        
        if total_errors == 0:
            print("✅ Label format validation passed!")
        
        return {
            'format_errors': format_errors,
            'bbox_errors': bbox_errors,
            'class_errors': class_errors
        }
    
    def analyze_class_distribution(self):
        """Analyze and report class distribution across splits."""
        print("\n📊 Analyzing class distribution...")
        
        distribution = {
            'train': Counter(),
            'val': Counter(),
            'test': Counter(),
            'total': Counter()
        }
        
        for split in ['train', 'val', 'test']:
            label_files = glob.glob(str(self.labels_dir / split / "*.txt"))
            
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        line = line.strip()
                        if line:
                            class_id = int(line.split()[0])
                            distribution[split][class_id] += 1
                            distribution['total'][class_id] += 1
                
                except (ValueError, IndexError):
                    continue  # Skip malformed lines
        
        # Report distribution
        print(f"\n📈 Class Distribution Summary:")
        print(f"{'Class':<20} {'Train':<8} {'Val':<6} {'Test':<6} {'Total':<8}")
        print("-" * 50)
        
        for class_id in sorted(distribution['total'].keys()):
            if class_id < len(self.class_names):
                class_name = self.class_names[class_id][:18]  # Truncate long names
                train_count = distribution['train'][class_id]
                val_count = distribution['val'][class_id]
                test_count = distribution['test'][class_id]
                total_count = distribution['total'][class_id]
                
                print(f"{class_name:<20} {train_count:<8} {val_count:<6} "
                      f"{test_count:<6} {total_count:<8}")
        
        # Identify underrepresented classes
        print(f"\n⚠️ Classes with < 50 annotations:")
        for class_id, count in distribution['total'].items():
            if count < 50 and class_id < len(self.class_names):
                print(f"   - {self.class_names[class_id]}: {count} annotations")
        
        return distribution
    
    def generate_report(self):
        """Generate comprehensive validation report."""
        print("🎯 MLBB YOLO Dataset Validation Report")
        print("=" * 50)
        
        # File structure validation
        file_validation = self.validate_file_structure()
        
        # Label format validation
        format_validation = self.validate_label_format()
        
        # Class distribution analysis
        distribution = self.analyze_class_distribution()
        
        # Summary statistics
        total_images = 0
        total_annotations = 0
        
        for split in ['train', 'val', 'test']:
            images = len(glob.glob(str(self.images_dir / split / "*.jpg")))
            images += len(glob.glob(str(self.images_dir / split / "*.png")))
            annotations = sum(distribution[split].values())
            
            total_images += images
            total_annotations += annotations
            
            print(f"\n📁 {split.upper()} Set:")
            print(f"   Images: {images}")
            print(f"   Annotations: {annotations}")
            if images > 0:
                print(f"   Avg annotations/image: {annotations/images:.1f}")
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Total Images: {total_images}")
        print(f"   Total Annotations: {total_annotations}")
        print(f"   Unique Classes: {len(distribution['total'])}")
        print(f"   Class Coverage: "
              f"{len(distribution['total'])}/{len(self.class_names)} "
              f"({100*len(distribution['total'])/len(self.class_names):.1f}%)")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if total_annotations < 1000:
            print("   - Consider collecting more annotations "
                  "(target: 1000+ total)")
        
        if len(distribution['total']) < len(self.class_names) * 0.7:
            print("   - Many classes are missing annotations")
        
        underrepresented = sum(1 for count in distribution['total'].values() 
                             if count < 50)
        if underrepresented > 0:
            print(f"   - {underrepresented} classes need more annotations")
        
        total_errors = (len(format_validation['format_errors']) + 
                       len(format_validation['bbox_errors']) + 
                       len(format_validation['class_errors']))
        
        if total_errors == 0:
            print("   - Dataset quality looks good! ✅")
        else:
            print(f"   - Fix {total_errors} annotation errors before training")
        
        print("\n" + "=" * 50)
        print("Validation complete! 🎉")


if __name__ == "__main__":
    validator = DatasetValidator(".")
    validator.generate_report() 