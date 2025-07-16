#!/usr/bin/env python3
"""
Test script for the Enhanced Data Collector

This script demonstrates how to use the EnhancedDataCollector class properly.
"""

import sys
import os
import logging

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

try:
    from core.enhanced_data_collector import EnhancedDataCollector
    print("✅ Enhanced Data Collector imported successfully!")
    
    # Create an instance
    collector = EnhancedDataCollector()
    print("✅ Enhanced Data Collector instance created successfully!")
    
    # Test the class methods
    print("\n📋 Available methods:")
    methods = [method for method in dir(collector) 
               if not method.startswith('_')]
    for method in methods:
        print(f"  - {method}")
    
    print("\n🎯 Enhanced Data Collector is ready to use!")
    print("Usage example:")
    print("  result = collector.analyze_screenshot_with_session(")
    print("      image_path='path/to/screenshot.png',")
    print("      ign='YourIGN',")
    print("      session_id='optional_session_id'")
    print("  )")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("This suggests there are missing dependencies or import issues.")
except Exception as e:
    print(f"❌ Error: {e}") 