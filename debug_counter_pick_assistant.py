#!/usr/bin/env python3
"""Debug script for counter-pick assistant."""

from core.counter_pick_assistant import CounterPickAssistant, DraftState
from core.meta_analyzer import MetaAnalyzer

def main():
    # Initialize systems
    meta_analyzer = MetaAnalyzer()
    counter_assistant = CounterPickAssistant(meta_analyzer)
    
    # Simulate draft state
    draft_state = DraftState(
        enemy_picks=["Angela", "Melissa"],
        ally_picks=["Franco"],
        enemy_bans=["Beatrix"],
        ally_bans=["Tigreal"],
        current_phase="pick2"
    )
    
    print("🔍 Debug Counter-Pick Assistant")
    print("=" * 35)
    
    # Debug the get_counter_pick_suggestions method
    print("\n1. Testing get_counter_recommendations from meta_analyzer:")
    counter_recs = meta_analyzer.get_counter_recommendations(draft_state.enemy_picks)
    print(f"Found {len(counter_recs)} counter recommendations")
    for rec in counter_recs:
        print(f"  - {rec.hero}: {rec.confidence:.2f}")
    
    print("\n2. Testing unavailable heroes filtering:")
    unavailable_heroes = set(draft_state.enemy_picks + draft_state.ally_picks + 
                           draft_state.enemy_bans + draft_state.ally_bans)
    print(f"Unavailable heroes: {unavailable_heroes}")
    
    available_recs = [rec for rec in counter_recs if rec.hero not in unavailable_heroes]
    print(f"Available recommendations: {len(available_recs)}")
    for rec in available_recs:
        print(f"  - {rec.hero}: {rec.confidence:.2f}")
    
    print("\n3. Testing counter-pick suggestions:")
    suggestions = counter_assistant.get_counter_pick_suggestions(draft_state)
    print(f"Final suggestions: {len(suggestions)}")
    for suggestion in suggestions:
        print(f"  - {suggestion.hero}: {suggestion.priority} priority")

if __name__ == "__main__":
    main()