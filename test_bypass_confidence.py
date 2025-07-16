#!/usr/bin/env python3
"""
Test to bypass the elite confidence scorer and see if the real extracted data flows through.
"""

import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_bypass_confidence_scorer():
    """Test the system bypassing the elite confidence scorer."""
    from core.ultimate_parsing_system import UltimateParsingSystem
    
    print("🔍 TESTING SYSTEM BYPASSING ELITE CONFIDENCE SCORER")
    print("=" * 60)
    
    # Temporarily modify ultimate parsing system to skip elite confidence scoring
    parsing_system = UltimateParsingSystem()
    
    test_image = 'dashboard-ui/uploads/5c19c434-ac9d-40d4-82c1-8779ca5cb7ba.png'
    ign = 'Lesz XVII'
    
    print(f"📸 Analyzing: {test_image}")
    print(f"👤 IGN: {ign}")
    print()
    
    # Focus on data extraction and completion stages
    print("📊 Stage 3: Basic Data Extraction")
    from core.enhanced_data_collector import EnhancedDataCollector
    data_collector = EnhancedDataCollector()
    
    basic_result = data_collector.analyze_screenshot_with_session(
        image_path=test_image,
        ign=ign
    )
    
    print("   Basic Result Keys:", list(basic_result.keys()) if basic_result else "None")
    if basic_result and 'data' in basic_result:
        extracted_data = basic_result['data']
        print(f"   🎯 EXTRACTED DATA:")
        print(json.dumps(extracted_data, indent=4))
        
        # Test if the data is good
        key_fields = ['kills', 'deaths', 'assists', 'hero_damage', 'hero']
        found_fields = [field for field in key_fields if field in extracted_data and extracted_data[field] not in [None, 0, "", "unknown"]]
        print(f"\n   ✅ GOOD FIELDS FOUND: {found_fields}")
        print(f"   📊 DATA QUALITY: {len(found_fields)}/{len(key_fields)} fields ({len(found_fields)/len(key_fields)*100:.1f}%)")
        
        # Show the medal detection
        if 'medal_type' in extracted_data:
            print(f"   🏆 MEDAL: {extracted_data['medal_type']} (confidence: {extracted_data.get('trophy_confidence', 0):.1f})")
    
    print(f"\n🧠 Stage 5: Intelligent Data Completion")
    from core.intelligent_data_completer import IntelligentDataCompleter
    data_completer = IntelligentDataCompleter()
    
    if basic_result and 'data' in basic_result:
        completion_result = data_completer.complete_data(
            raw_data=basic_result['data'],
            ocr_results=basic_result.get('debug_info', {}).get('ocr_results', []),
            image_path=test_image,
            context={'screenshot_type': 'scoreboard'}
        )
        
        print(f"   Completion: {completion_result.completeness_score:.1f}% complete, {completion_result.confidence:.1f}% confidence")
        
        if completion_result.fields:
            final_data = {}
            for field_name, field in completion_result.fields.items():
                final_data[field_name] = field.value
                
            print(f"\n   🎯 FINAL COMPLETED DATA:")
            print(json.dumps(final_data, indent=4))
            
            # Check if the real extracted data is preserved
            if final_data.get('kills', 0) == 3 and final_data.get('deaths', 0) == 1 and final_data.get('assists', 0) == 8:
                print(f"\n   🎉 SUCCESS! Real KDA data preserved: {final_data.get('kills')}/{final_data.get('deaths')}/{final_data.get('assists')}")
            else:
                print(f"\n   ❌ FAILURE! Real KDA data lost or corrupted")

if __name__ == "__main__":
    test_bypass_confidence_scorer() 