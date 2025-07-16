#!/usr/bin/env python3
"""
Mobile Legends Hero Data Cleaner

This script processes and cleans hero data from the Mobile Legends scraped
JSON files. It performs data validation, normalization, and enrichment with
role information.

Author: Data Integrity Team
Created: 2025-01-15
"""

import json
import re
from typing import Dict, List, Optional
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HeroDataCleaner:
    """
    A comprehensive data cleaner for Mobile Legends hero data.
    
    This class handles loading, cleaning, validating, and enriching hero data
    from multiple JSON sources including scraped hero data and role mappings.
    """
    
    def __init__(self, input_file: str, role_mapping_file: str, 
                 output_file: str):
        """
        Initialize the cleaner with file paths.
        
        Args:
            input_file: Path to the scraped hero data JSON file
            role_mapping_file: Path to the hero role mapping JSON file
            output_file: Path where cleaned data will be saved
        """
        self.input_file = input_file
        self.role_mapping_file = role_mapping_file
        self.output_file = output_file
        
        # Statistics tracking
        self.stats = {
            'total_heroes': 0,
            'duplicates_removed': 0,
            'invalid_entries': 0,
            'missing_fields': 0,
            'invalid_urls': 0,
            'suspicious_entries': 0,
            'role_mappings_added': 0,
            'cleaned_names': []
        }
        
        # Suspicious entry patterns to flag
        self.suspicious_patterns = [
            r'new\s+hero',
            r'coming\s+soon',
            r'placeholder',
            r'test',
            r'ad\s*',
            r'advertisement',
            r'banner',
            r'promo'
        ]
        
        # Valid image extensions
        self.valid_image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
        
    def load_json_file(self, file_path: str) -> Optional[Dict]:
        """
        Load and parse a JSON file with error handling.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Parsed JSON data or None if loading fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def normalize_hero_name(self, name: str) -> str:
        """
        Normalize hero name with proper casing and whitespace handling.
        
        Args:
            name: Raw hero name
            
        Returns:
            Normalized hero name
        """
        if not name:
            return ""
        
        # Strip whitespace and normalize
        name = name.strip()
        
        # Handle special cases for names with apostrophes, periods, etc.
        # Examples: "Chang'e", "X.Borg", "Luo Yi"
        special_cases = {
            "chang'e": "Chang'e",
            "x.borg": "X.Borg",
            "luo yi": "Luo Yi",
            "yu zhong": "Yu Zhong",
            "yi sun-shin": "Yi Sun-shin",
            "popol and kupa": "Popol and Kupa"
        }
        
        # Check for special cases first
        name_lower = name.lower()
        if name_lower in special_cases:
            return special_cases[name_lower]
        
        # Standard title case for regular names
        return name.title()
    
    def create_slug(self, name: str) -> str:
        """
        Create a URL-friendly slug from hero name.
        
        Args:
            name: Hero name
            
        Returns:
            URL-friendly slug
        """
        # Convert to lowercase and replace spaces/special chars with hyphens
        slug = re.sub(r'[^a-zA-Z0-9\s-]', '', name.lower())
        slug = re.sub(r'[-\s]+', '-', slug)
        return slug.strip('-')
    
    def validate_image_url(self, url: str) -> bool:
        """
        Validate image URL format and extension.
        
        Args:
            url: Image URL to validate
            
        Returns:
            True if URL is valid, False otherwise
        """
        if not url:
            return False
        
        # Check HTTPS protocol
        if not url.startswith('https://'):
            return False
        
        # Check valid image extension
        url_lower = url.lower()
        extensions = self.valid_image_extensions
        if not any(url_lower.endswith(ext) for ext in extensions):
            return False
        
        # Check for reasonable URL structure
        if len(url) < 10 or len(url) > 500:
            return False
        
        return True
    
    def validate_page_url(self, url: str) -> bool:
        """
        Validate page URL format.
        
        Args:
            url: Page URL to validate
            
        Returns:
            True if URL is valid, False otherwise
        """
        if not url:
            return False
        
        # Check HTTPS protocol
        if not url.startswith('https://'):
            return False
        
        # Check for reasonable URL structure
        if len(url) < 10 or len(url) > 500:
            return False
        
        return True
    
    def is_suspicious_entry(self, hero_data: Dict) -> bool:
        """
        Check if a hero entry looks suspicious (ads, placeholders, etc.).
        
        Args:
            hero_data: Hero data dictionary
            
        Returns:
            True if entry looks suspicious, False otherwise
        """
        name = hero_data.get('hero_name', '').lower()
        
        # Check against suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return True
        
        # Check for obviously invalid names
        if len(name) < 2 or len(name) > 50:
            return True
        
        # Check for numeric-only names
        if name.isdigit():
            return True
        
        return False
    
    def clean_hero_entry(self, hero_data: Dict) -> Optional[Dict]:
        """
        Clean and validate a single hero entry.
        
        Args:
            hero_data: Raw hero data dictionary
            
        Returns:
            Cleaned hero data dictionary or None if invalid
        """
        # Check for suspicious entries
        if self.is_suspicious_entry(hero_data):
            self.stats['suspicious_entries'] += 1
            return None
        
        # Extract and validate required fields
        hero_name = hero_data.get('hero_name', '').strip()
        image_url = hero_data.get('image_url', '').strip()
        page_url = hero_data.get('correct_hero_url', '').strip()
        
        # Validate hero name
        if not hero_name:
            self.stats['missing_fields'] += 1
            return None
        
        # Normalize hero name
        normalized_name = self.normalize_hero_name(hero_name)
        
        # Validate image URL
        if not self.validate_image_url(image_url):
            self.stats['invalid_urls'] += 1
            msg = f"Invalid image URL for hero {normalized_name}: {image_url}"
            logger.warning(msg)
            # Don't skip the entry, just log the warning
        
        # Validate page URL
        if not self.validate_page_url(page_url):
            msg = f"Invalid page URL for hero {normalized_name}: {page_url}"
            logger.warning(msg)
            # Don't skip the entry, just log the warning
        
        # Create cleaned entry
        cleaned_entry = {
            'name': normalized_name,
            'slug': self.create_slug(normalized_name),
            'image_url': image_url,
            'page_url': page_url,
            'extraction_method': hero_data.get('extraction_method', 'unknown')
        }
        
        # Track cleaned names for statistics
        if normalized_name != hero_name:
            self.stats['cleaned_names'].append({
                'original': hero_name,
                'cleaned': normalized_name
            })
        
        return cleaned_entry
    
    def merge_role_data(self, hero_data: List[Dict], 
                        role_mapping: Dict) -> List[Dict]:
        """
        Merge role information from role mapping into hero data.
        
        Args:
            hero_data: List of cleaned hero data
            role_mapping: Role mapping data
            
        Returns:
            Hero data enriched with role information
        """
        role_heroes = role_mapping.get('heroes', {})
        
        for hero in hero_data:
            hero_name = hero['name']
            
            # Try to find role mapping (exact match first)
            if hero_name in role_heroes:
                role_info = role_heroes[hero_name]
                hero.update({
                    'role': role_info.get('role', 'unknown'),
                    'sub_role': role_info.get('sub_role', 'unknown'),
                    'specialty': role_info.get('sub_role', 'unknown'),
                    'priority': role_info.get('priority', 'unknown')
                })
                self.stats['role_mappings_added'] += 1
            else:
                # Set default values for unmapped heroes
                hero.update({
                    'role': 'unknown',
                    'sub_role': 'unknown',
                    'specialty': 'unknown',
                    'priority': 'unknown'
                })
                msg = f"No role mapping found for hero: {hero_name}"
                logger.warning(msg)
        
        return hero_data
    
    def remove_duplicates(self, hero_data: List[Dict]) -> List[Dict]:
        """
        Remove duplicate heroes based on name.
        
        Args:
            hero_data: List of hero data
            
        Returns:
            List with duplicates removed
        """
        seen_names = set()
        unique_heroes = []
        
        for hero in hero_data:
            name = hero['name']
            if name not in seen_names:
                seen_names.add(name)
                unique_heroes.append(hero)
            else:
                self.stats['duplicates_removed'] += 1
                logger.info(f"Removed duplicate hero: {name}")
        
        return unique_heroes
    
    def group_by_role(self, hero_data: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group heroes by their primary role.
        
        Args:
            hero_data: List of hero data
            
        Returns:
            Dictionary with roles as keys and hero lists as values
        """
        grouped = {}
        
        for hero in hero_data:
            role = hero.get('role', 'unknown')
            if role not in grouped:
                grouped[role] = []
            grouped[role].append(hero)
        
        return grouped
    
    def save_cleaned_data(self, cleaned_data: Dict) -> bool:
        """
        Save cleaned data to output file.
        
        Args:
            cleaned_data: Cleaned and processed data
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully saved cleaned data to {self.output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving cleaned data: {e}")
            return False
    
    def generate_summary_report(self) -> Dict:
        """
        Generate a summary report of the cleaning process.
        
        Returns:
            Dictionary containing summary statistics
        """
        report = {
            'processing_date': datetime.now().isoformat(),
            'input_file': self.input_file,
            'output_file': self.output_file,
            'statistics': self.stats,
            'validation_summary': {
                'total_processed': self.stats['total_heroes'],
                'successfully_cleaned': (
                    self.stats['total_heroes'] - 
                    self.stats['invalid_entries']
                ),
                'duplicates_removed': self.stats['duplicates_removed'],
                'suspicious_entries_filtered': self.stats['suspicious_entries'],
                'role_mappings_added': self.stats['role_mappings_added']
            }
        }
        
        return report
    
    def clean_data(self) -> bool:
        """
        Main method to clean hero data.
        
        Returns:
            True if cleaning successful, False otherwise
        """
        logger.info("Starting hero data cleaning process...")
        
        # Load input data
        hero_data = self.load_json_file(self.input_file)
        if not hero_data:
            logger.error("Failed to load hero data")
            return False
        
        # Load role mapping data
        role_mapping = self.load_json_file(self.role_mapping_file)
        if not role_mapping:
            logger.error("Failed to load role mapping data")
            return False
        
        # Track total heroes
        self.stats['total_heroes'] = len(hero_data)
        logger.info(f"Processing {self.stats['total_heroes']} heroes...")
        
        # Clean individual entries
        cleaned_entries = []
        for hero in hero_data:
            cleaned_entry = self.clean_hero_entry(hero)
            if cleaned_entry:
                cleaned_entries.append(cleaned_entry)
            else:
                self.stats['invalid_entries'] += 1
        
        # Remove duplicates
        cleaned_entries = self.remove_duplicates(cleaned_entries)
        
        # Merge role data
        cleaned_entries = self.merge_role_data(cleaned_entries, role_mapping)
        
        # Group by role for bonus feature
        grouped_by_role = self.group_by_role(cleaned_entries)
        
        # Prepare final output
        final_output = {
            'metadata': {
                'version': '1.0.0',
                'last_updated': datetime.now().isoformat(),
                'total_heroes': len(cleaned_entries),
                'source': 'Mobile Legends Official Website',
                'processed_by': 'Hero Data Cleaner v1.0'
            },
            'heroes': cleaned_entries,
            'grouped_by_role': grouped_by_role,
            'summary_report': self.generate_summary_report()
        }
        
        # Save cleaned data
        if self.save_cleaned_data(final_output):
            logger.info("Hero data cleaning completed successfully!")
            self.print_summary_report()
            return True
        else:
            logger.error("Failed to save cleaned data")
            return False
    
    def print_summary_report(self):
        """Print a formatted summary report to console."""
        print("\n" + "="*60)
        print("HERO DATA CLEANING SUMMARY REPORT")
        print("="*60)
        print(f"📊 Total Heroes Processed: {self.stats['total_heroes']}")
        processed = self.stats['total_heroes'] - self.stats['invalid_entries']
        print(f"✅ Successfully Cleaned: {processed}")
        print(f"🗑️  Duplicates Removed: {self.stats['duplicates_removed']}")
        filtered = self.stats['suspicious_entries']
        print(f"⚠️  Suspicious Entries Filtered: {filtered}")
        print(f"❌ Invalid Entries: {self.stats['invalid_entries']}")
        print(f"🔗 Role Mappings Added: {self.stats['role_mappings_added']}")
        print(f"🌐 Invalid URLs Found: {self.stats['invalid_urls']}")
        print(f"📝 Names Cleaned: {len(self.stats['cleaned_names'])}")
        
        if self.stats['cleaned_names']:
            print("\n📋 Examples of cleaned names:")
            for example in self.stats['cleaned_names'][:5]:  # Show first 5
                orig = example['original']
                cleaned = example['cleaned']
                print(f"   '{orig}' → '{cleaned}'")
            if len(self.stats['cleaned_names']) > 5:
                remaining = len(self.stats['cleaned_names']) - 5
                print(f"   ... and {remaining} more")
        
        print("\n✨ Output saved to:", self.output_file)
        print("="*60)


def main():
    """Main entry point for the script."""
    # File paths
    input_file = 'mlbb-heroes-corrected.json'
    role_mapping_file = 'hero_role_mapping.json'
    output_file = 'heroes_cleaned.json'
    
    # Create cleaner instance
    cleaner = HeroDataCleaner(input_file, role_mapping_file, output_file)
    
    # Run cleaning process
    success = cleaner.clean_data()
    
    if success:
        print("\n🎉 Hero data cleaning completed successfully!")
        exit(0)
    else:
        print("\n❌ Hero data cleaning failed!")
        exit(1)


if __name__ == "__main__":
    main() 