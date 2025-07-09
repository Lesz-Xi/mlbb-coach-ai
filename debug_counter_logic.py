#!/usr/bin/env python3
"""Debug script to understand counter logic."""

from core.meta_analyzer import MetaAnalyzer

def main():
    analyzer = MetaAnalyzer()
    
    # Test enemy heroes (using only heroes that exist in the data)
    enemy_heroes = ["Angela", "Melissa"]
    
    print("🔍 Debug Counter Logic")
    print("=" * 30)
    
    print(f"\nEnemy heroes: {enemy_heroes}")
    
    # Check what counters each enemy hero
    for enemy in enemy_heroes:
        enemy_data = analyzer.get_hero_meta(enemy)
        if enemy_data:
            print(f"\n{enemy}:")
            print(f"  Counter heroes: {enemy_data.counter_heroes}")
            print(f"  Win rate: {enemy_data.win_rate}%")
            print(f"  Tier: {enemy_data.tier}")
    
    # Test counter recommendations
    print(f"\n🎯 Counter Recommendations:")
    recommendations = analyzer.get_counter_recommendations(enemy_heroes)
    
    if not recommendations:
        print("No recommendations found!")
        print("Debug: Let's check the counter logic manually...")
        
        # Manual check
        for hero in analyzer.meta_data.data:
            counter_count = 0
            for enemy in enemy_heroes:
                enemy_data = analyzer.get_hero_meta(enemy)
                if enemy_data and hero.hero.lower() in [c.lower() for c in enemy_data.counter_heroes]:
                    counter_count += 1
            if counter_count > 0:
                print(f"  {hero.hero} counters {counter_count} enemies")
        
    else:
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"{i}. {rec.hero} - {rec.confidence:.2f}")
            print(f"   {rec.reasoning}")
    
    # Test individual hero lookups
    print(f"\n🔍 Hero Lookup Test:")
    test_heroes = ["Franco", "Chou", "Jawhead"]
    for hero in test_heroes:
        hero_data = analyzer.get_hero_meta(hero)
        if hero_data:
            print(f"{hero}: Found - WR: {hero_data.win_rate}%, Tier: {hero_data.tier}")
        else:
            print(f"{hero}: Not found")
    
    # Check if Franco counters Angela
    print(f"\n🎯 Specific Counter Check:")
    angela_data = analyzer.get_hero_meta("Angela")
    if angela_data:
        print(f"Angela counter_heroes: {angela_data.counter_heroes}")
        print(f"Does Franco counter Angela? {'Franco' in angela_data.counter_heroes}")

if __name__ == "__main__":
    main()