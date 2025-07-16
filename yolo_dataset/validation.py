#!/usr/bin/env python3
"""
MLBB YOLO Dataset Annotation Validation Script

This script validates annotation quality according to the annotation guidelines,
checking for proper dimensions, aspect ratios, class consistency, and other
quality standards.

Usage:
    python validation.py --labels_dir labels/train --images_dir images/train
    python validation.py --annotation_file labels/train/image_001.txt --image_file images/train/image_001.jpg
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Container for validation results"""
    passed: bool
    warnings: List[str]
    errors: List[str]
    metrics: Dict[str, float]


class AnnotationValidator:
    """Validates MLBB YOLO annotations according to quality guidelines"""
    
    def __init__(self):
        """Initialize validator with quality standards"""
        self.validation_rules = {
            # Hero Icon Detection Standards
            'hero_icons': {
                'classes': ['hero_icon_ally_1', 'hero_icon_ally_2', 'hero_icon_ally_3', 
                          'hero_icon_ally_4', 'hero_icon_ally_5',
                          'hero_icon_enemy_1', 'hero_icon_enemy_2', 'hero_icon_enemy_3',
                          'hero_icon_enemy_4', 'hero_icon_enemy_5'],
                'min_width': 50,
                'max_width': 150,
                'preferred_width': (70, 90),
                'aspect_ratio': (0.9, 1.1),  # Nearly square
                'min_area': 2500  # 50x50
            },
            
            # Medal Detection Standards
            'medals': {
                'classes': ['mvp_badge', 'mvp_loss_badge', 'medal_gold', 
                          'medal_silver', 'medal_bronze'],
                'min_width': 30,
                'max_width': 100,
                'preferred_width': (50, 70),
                'aspect_ratio': (0.8, 1.2),
                'min_area': 900  # 30x30
            },
            
            # Text Elements Standards
            'text_elements': {
                'classes': ['player_name', 'kda_display', 'victory_text', 'defeat_text',
                          'match_type_classic', 'match_type_ranked', 'match_type_brawl', 
                          'match_type_custom'],
                'min_width': 100,
                'min_height': 20,
                'max_width': 400,
                'max_height': 100,
                'aspect_ratio': (2.0, 20.0)  # Wide text
            },
            
            # Statistics Standards
            'statistics': {
                'classes': ['gold_amount', 'damage_dealt', 'damage_taken', 'healing_done',
                          'gpm_display', 'participation_rate', 'turret_damage', 'match_duration'],
                'min_width': 60,
                'max_width': 200,
                'min_height': 15,
                'max_height': 50,
                'aspect_ratio': (1.5, 10.0)
            },
            
            # UI Elements Standards
            'ui_elements': {
                'classes': ['team_indicator_ally', 'team_indicator_enemy', 
                          'scoreboard_container', 'ui_complete', 'text_readable', 'icons_clear'],
                'min_width': 50,
                'max_width': 800,
                'min_height': 30,
                'max_height': 600,
                'aspect_ratio': (0.5, 15.0)
            },
            
            # Items & Equipment Standards
            'items': {
                'classes': ['item_icon_1', 'item_icon_2', 'item_icon_3', 
                          'item_icon_4', 'item_icon_5', 'item_icon_6',
                          'battle_spell_1', 'battle_spell_2'],
                'min_width': 25,
                'max_width': 80,
                'preferred_width': (35, 55),
                'aspect_ratio': (0.8, 1.2),
                'min_area': 625  # 25x25
            },
            
            # Achievement Standards
            'achievements': {
                'classes': ['savage_indicator', 'maniac_indicator', 'legendary_indicator',
                          'role_indicator', 'position_rank'],
                'min_width': 30,
                'max_width': 150,
                'min_height': 20,
                'max_height': 100,
                'aspect_ratio': (0.5, 3.0)
            }
        }
        
        # Class mapping for validation
        self.class_to_category = {}
        for category, rules in self.validation_rules.items():
            for class_name in rules['classes']:
                self.class_to_category[class_name] = category
    
    def validate_annotation_file(self, annotation_file: str, image_file: str) -> ValidationResult:
        """Validate a single annotation file against its corresponding image"""
        result = ValidationResult(
            passed=True,
            warnings=[],
            errors=[],
            metrics={}
        )
        
        try:
            # Load image to get dimensions
            if not os.path.exists(image_file):
                result.errors.append(f"Image file not found: {image_file}")
                result.passed = False
                return result
                
            image = cv2.imread(image_file)
            if image is None:
                result.errors.append(f"Cannot load image: {image_file}")
                result.passed = False
                return result
                
            img_height, img_width = image.shape[:2]
            result.metrics['image_width'] = img_width
            result.metrics['image_height'] = img_height
            
            # Load annotations
            if not os.path.exists(annotation_file):
                result.warnings.append(f"No annotation file found: {annotation_file}")
                return result
                
            annotations = self._load_yolo_annotations(annotation_file)
            result.metrics['annotation_count'] = len(annotations)
            
            if not annotations:
                result.warnings.append("No annotations found in file")
                return result
            
            # Validate each annotation
            category_counts = {}
            for i, annotation in enumerate(annotations):
                ann_result = self._validate_single_annotation(
                    annotation, img_width, img_height, i
                )
                
                # Merge results
                result.warnings.extend(ann_result.warnings)
                result.errors.extend(ann_result.errors)
                
                if not ann_result.passed:
                    result.passed = False
                
                # Count categories
                class_id = annotation['class_id']
                if 0 <= class_id < len(self._get_class_names()):
                    class_name = self._get_class_names()[class_id]
                    category = self.class_to_category.get(class_name, 'unknown')
                    category_counts[category] = category_counts.get(category, 0) + 1
            
            result.metrics['category_counts'] = category_counts
            
            # Validate overall annotation consistency
            self._validate_annotation_consistency(annotations, result)
            
        except Exception as e:
            result.errors.append(f"Validation error: {str(e)}")
            result.passed = False
        
        return result
    
    def _validate_single_annotation(self, annotation: Dict, img_width: int, 
                                  img_height: int, index: int) -> ValidationResult:
        """Validate a single bounding box annotation"""
        result = ValidationResult(passed=True, warnings=[], errors=[], metrics={})
        
        class_id = annotation['class_id']
        x_center, y_center, width, height = annotation['bbox']
        
        # Convert normalized coordinates to pixel coordinates
        pixel_width = width * img_width
        pixel_height = height * img_height
        pixel_x = x_center * img_width
        pixel_y = y_center * img_height
        
        # Get class name and category
        class_names = self._get_class_names()
        if class_id >= len(class_names):
            result.errors.append(f"Annotation {index}: Invalid class ID {class_id}")
            result.passed = False
            return result
            
        class_name = class_names[class_id]
        category = self.class_to_category.get(class_name, 'unknown')
        
        if category == 'unknown':
            result.warnings.append(f"Annotation {index}: Unknown class '{class_name}'")
        
        # Get validation rules for this category
        rules = self.validation_rules.get(category, {})
        
        # Validate dimensions
        if 'min_width' in rules and pixel_width < rules['min_width']:
            result.errors.append(
                f"Annotation {index} ({class_name}): Width {pixel_width:.1f} < minimum {rules['min_width']}"
            )
            result.passed = False
            
        if 'max_width' in rules and pixel_width > rules['max_width']:
            result.errors.append(
                f"Annotation {index} ({class_name}): Width {pixel_width:.1f} > maximum {rules['max_width']}"
            )
            result.passed = False
            
        if 'min_height' in rules and pixel_height < rules['min_height']:
            result.errors.append(
                f"Annotation {index} ({class_name}): Height {pixel_height:.1f} < minimum {rules['min_height']}"
            )
            result.passed = False
            
        if 'max_height' in rules and pixel_height > rules['max_height']:
            result.errors.append(
                f"Annotation {index} ({class_name}): Height {pixel_height:.1f} > maximum {rules['max_height']}"
            )
            result.passed = False
        
        # Validate aspect ratio
        if 'aspect_ratio' in rules and pixel_height > 0:
            aspect_ratio = pixel_width / pixel_height
            min_ratio, max_ratio = rules['aspect_ratio']
            
            if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                result.warnings.append(
                    f"Annotation {index} ({class_name}): Aspect ratio {aspect_ratio:.2f} "
                    f"outside range [{min_ratio:.1f}, {max_ratio:.1f}]"
                )
        
        # Validate area
        if 'min_area' in rules:
            area = pixel_width * pixel_height
            if area < rules['min_area']:
                result.errors.append(
                    f"Annotation {index} ({class_name}): Area {area:.0f} < minimum {rules['min_area']}"
                )
                result.passed = False
        
        # Validate preferred dimensions
        if 'preferred_width' in rules:
            pref_min, pref_max = rules['preferred_width']
            if not (pref_min <= pixel_width <= pref_max):
                result.warnings.append(
                    f"Annotation {index} ({class_name}): Width {pixel_width:.1f} "
                    f"outside preferred range [{pref_min}, {pref_max}]"
                )
        
        # Validate bounds (should be within image)
        left = pixel_x - pixel_width / 2
        right = pixel_x + pixel_width / 2
        top = pixel_y - pixel_height / 2
        bottom = pixel_y + pixel_height / 2
        
        if left < 0 or right > img_width or top < 0 or bottom > img_height:
            result.errors.append(
                f"Annotation {index} ({class_name}): Bounding box extends outside image bounds"
            )
            result.passed = False
        
        # Store metrics
        result.metrics.update({
            f'ann_{index}_width': pixel_width,
            f'ann_{index}_height': pixel_height,
            f'ann_{index}_aspect_ratio': aspect_ratio if pixel_height > 0 else 0,
            f'ann_{index}_area': pixel_width * pixel_height
        })
        
        return result
    
    def _validate_annotation_consistency(self, annotations: List[Dict], result: ValidationResult):
        """Validate consistency across all annotations in the file"""
        
        # Check for hero team completeness
        ally_heroes = [ann for ann in annotations 
                      if self._get_class_names()[ann['class_id']].startswith('hero_icon_ally_')]
        enemy_heroes = [ann for ann in annotations 
                       if self._get_class_names()[ann['class_id']].startswith('hero_icon_enemy_')]
        
        if ally_heroes and len(ally_heroes) < 5:
            result.warnings.append(f"Incomplete ally team: only {len(ally_heroes)}/5 heroes annotated")
            
        if enemy_heroes and len(enemy_heroes) < 5:
            result.warnings.append(f"Incomplete enemy team: only {len(enemy_heroes)}/5 heroes annotated")
        
        # Check for duplicate positions
        self._check_duplicate_positions(ally_heroes, 'ally', result)
        self._check_duplicate_positions(enemy_heroes, 'enemy', result)
        
        # Check for overlapping bounding boxes
        self._check_overlapping_boxes(annotations, result)
    
    def _check_duplicate_positions(self, hero_annotations: List[Dict], team: str, result: ValidationResult):
        """Check for duplicate hero position numbers"""
        positions = []
        class_names = self._get_class_names()
        
        for ann in hero_annotations:
            class_name = class_names[ann['class_id']]
            if class_name.endswith('_1'): positions.append(1)
            elif class_name.endswith('_2'): positions.append(2)
            elif class_name.endswith('_3'): positions.append(3)
            elif class_name.endswith('_4'): positions.append(4)
            elif class_name.endswith('_5'): positions.append(5)
        
        duplicates = [pos for pos in set(positions) if positions.count(pos) > 1]
        if duplicates:
            result.errors.append(f"Duplicate {team} hero positions: {duplicates}")
            result.passed = False
    
    def _check_overlapping_boxes(self, annotations: List[Dict], result: ValidationResult):
        """Check for significantly overlapping bounding boxes"""
        for i, ann1 in enumerate(annotations):
            for j, ann2 in enumerate(annotations[i+1:], i+1):
                overlap = self._calculate_box_overlap(ann1['bbox'], ann2['bbox'])
                if overlap > 0.3:  # More than 30% overlap
                    class_names = self._get_class_names()
                    class1 = class_names[ann1['class_id']]
                    class2 = class_names[ann2['class_id']]
                    result.warnings.append(
                        f"High overlap ({overlap:.1%}) between {class1} and {class2}"
                    )
    
    def _calculate_box_overlap(self, bbox1: Tuple[float, float, float, float], 
                              bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate overlap ratio between two normalized bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to corner coordinates
        left1, top1 = x1 - w1/2, y1 - h1/2
        right1, bottom1 = x1 + w1/2, y1 + h1/2
        left2, top2 = x2 - w2/2, y2 - h2/2
        right2, bottom2 = x2 + w2/2, y2 + h2/2
        
        # Calculate intersection
        left = max(left1, left2)
        top = max(top1, top2)
        right = min(right1, right2)
        bottom = min(bottom1, bottom2)
        
        if left >= right or top >= bottom:
            return 0.0
        
        intersection = (right - left) * (bottom - top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _load_yolo_annotations(self, annotation_file: str) -> List[Dict]:
        """Load YOLO format annotations from file"""
        annotations = []
        
        with open(annotation_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) != 5:
                        raise ValueError(f"Expected 5 values, got {len(parts)}")
                    
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Validate normalized coordinates
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                           0 < width <= 1 and 0 < height <= 1):
                        raise ValueError("Coordinates must be normalized between 0 and 1")
                    
                    annotations.append({
                        'class_id': class_id,
                        'bbox': (x_center, y_center, width, height),
                        'line_num': line_num
                    })
                    
                except ValueError as e:
                    print(f"Warning: Invalid annotation at line {line_num}: {e}")
                    continue
        
        return annotations
    
    def _get_class_names(self) -> List[str]:
        """Get ordered list of all class names"""
        all_classes = []
        for rules in self.validation_rules.values():
            all_classes.extend(rules['classes'])
        return sorted(list(set(all_classes)))
    
    def validate_dataset(self, labels_dir: str, images_dir: str) -> Dict[str, ValidationResult]:
        """Validate entire dataset directory"""
        results = {}
        
        labels_path = Path(labels_dir)
        images_path = Path(images_dir)
        
        if not labels_path.exists():
            print(f"Error: Labels directory not found: {labels_dir}")
            return results
        
        if not images_path.exists():
            print(f"Error: Images directory not found: {images_dir}")
            return results
        
        # Find all annotation files
        annotation_files = list(labels_path.glob("*.txt"))
        
        print(f"Validating {len(annotation_files)} annotation files...")
        
        for ann_file in annotation_files:
            # Find corresponding image file
            image_name = ann_file.stem
            image_file = None
            
            # Try common image extensions
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                potential_image = images_path / f"{image_name}{ext}"
                if potential_image.exists():
                    image_file = str(potential_image)
                    break
            
            if not image_file:
                print(f"Warning: No image found for {ann_file.name}")
                continue
            
            # Validate annotation
            result = self.validate_annotation_file(str(ann_file), image_file)
            results[ann_file.name] = result
            
            # Print progress
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{status} {ann_file.name}")
            
            if result.errors:
                for error in result.errors:
                    print(f"  ERROR: {error}")
            
            if result.warnings:
                for warning in result.warnings[:3]:  # Limit warnings displayed
                    print(f"  WARNING: {warning}")
                if len(result.warnings) > 3:
                    print(f"  ... and {len(result.warnings) - 3} more warnings")
        
        return results
    
    def generate_validation_report(self, results: Dict[str, ValidationResult], 
                                 output_file: Optional[str] = None) -> Dict:
        """Generate comprehensive validation report"""
        report = {
            'summary': {
                'total_files': len(results),
                'passed': sum(1 for r in results.values() if r.passed),
                'failed': sum(1 for r in results.values() if not r.passed),
                'total_warnings': sum(len(r.warnings) for r in results.values()),
                'total_errors': sum(len(r.errors) for r in results.values())
            },
            'category_statistics': {},
            'common_issues': {},
            'recommendations': []
        }
        
        # Aggregate category statistics
        all_categories = {}
        for result in results.values():
            category_counts = result.metrics.get('category_counts', {})
            for category, count in category_counts.items():
                if category not in all_categories:
                    all_categories[category] = []
                all_categories[category].append(count)
        
        for category, counts in all_categories.items():
            report['category_statistics'][category] = {
                'total_annotations': sum(counts),
                'avg_per_image': np.mean(counts),
                'min_per_image': min(counts),
                'max_per_image': max(counts)
            }
        
        # Identify common issues
        all_errors = []
        all_warnings = []
        for result in results.values():
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
        
        # Count error patterns
        error_patterns = {}
        for error in all_errors:
            # Extract error type (everything before the first colon)
            error_type = error.split(':')[0] if ':' in error else error
            error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
        
        report['common_issues'] = {
            'top_errors': sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:5],
            'error_examples': all_errors[:10],
            'warning_examples': all_warnings[:10]
        }
        
        # Generate recommendations
        recommendations = []
        
        if report['summary']['failed'] > report['summary']['passed'] * 0.1:
            recommendations.append("High failure rate detected. Review annotation guidelines and retrain annotators.")
        
        if 'hero_icons' in all_categories:
            hero_stats = report['category_statistics']['hero_icons']
            if hero_stats['avg_per_image'] < 8:
                recommendations.append("Low hero icon detection rate. Ensure all visible heroes are annotated.")
        
        if report['summary']['total_warnings'] > report['summary']['total_files'] * 3:
            recommendations.append("High warning count. Consider adjusting validation thresholds or improving annotation quality.")
        
        report['recommendations'] = recommendations
        
        # Save report if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Validation report saved to {output_file}")
        
        return report


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Validate MLBB YOLO annotations")
    parser.add_argument('--labels_dir', help='Directory containing label files')
    parser.add_argument('--images_dir', help='Directory containing image files')
    parser.add_argument('--annotation_file', help='Single annotation file to validate')
    parser.add_argument('--image_file', help='Corresponding image file')
    parser.add_argument('--output_report', help='Output file for validation report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    validator = AnnotationValidator()
    
    if args.annotation_file and args.image_file:
        # Validate single file
        print(f"Validating single annotation: {args.annotation_file}")
        result = validator.validate_annotation_file(args.annotation_file, args.image_file)
        
        print(f"\nResult: {'PASS' if result.passed else 'FAIL'}")
        print(f"Warnings: {len(result.warnings)}")
        print(f"Errors: {len(result.errors)}")
        
        if args.verbose:
            for warning in result.warnings:
                print(f"WARNING: {warning}")
            for error in result.errors:
                print(f"ERROR: {error}")
        
    elif args.labels_dir and args.images_dir:
        # Validate dataset
        print(f"Validating dataset: {args.labels_dir}")
        results = validator.validate_dataset(args.labels_dir, args.images_dir)
        
        # Generate report
        report = validator.generate_validation_report(results, args.output_report)
        
        print(f"\n📊 Validation Summary:")
        print(f"Total files: {report['summary']['total_files']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Pass rate: {report['summary']['passed']/report['summary']['total_files']*100:.1f}%")
        print(f"Total warnings: {report['summary']['total_warnings']}")
        print(f"Total errors: {report['summary']['total_errors']}")
        
        if report['common_issues']['top_errors']:
            print(f"\n🚨 Top Error Types:")
            for error_type, count in report['common_issues']['top_errors']:
                print(f"  {error_type}: {count} occurrences")
        
        if report['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main() 