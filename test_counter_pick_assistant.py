#!/usr/bin/env python3
"""Demo script for Post-Match Analysis and Retrospective Counter-Pick Learning."""

from core.post_match_analyzer import PostMatchAnalyzer, MatchResult
from core.meta_analyzer import MetaAnalyzer


def main():
    print("🎯 SkillShift Post-Match Analysis Demo")
    print("=" * 45)
    
    # Initialize systems
    meta_analyzer = MetaAnalyzer()
    post_match_analyzer = PostMatchAnalyzer(meta_analyzer)
    
    # Simulate completed match data
    match_result = MatchResult(
        player_hero="Franco",
        ally_team=["Franco", "Layla", "Cyclops", "Ruby"],
        enemy_team=["Angela", "Melissa", "Tigreal", "Grock"],
        match_outcome="defeat",
        match_duration=18,
        player_performance={
            "kda": 1.5,
            "damage_dealt": 45000,
            "damage_taken": 65000,
            "kill_participation": 0.6
        }
    )
    
    print("\n📋 Match Summary:")
    print(f"Outcome: {match_result.match_outcome.upper()}")
    print(f"Duration: {match_result.match_duration} minutes")
    print(f"Your Hero: {match_result.player_hero}")
    print(f"Your Team: {', '.join(match_result.ally_team)}")
    print(f"Enemy Team: {', '.join(match_result.enemy_team)}")
    
    # Perform comprehensive post-match analysis
    print("\n🔍 Post-Match Analysis:")
    analysis = post_match_analyzer.analyze_match(match_result)
    
    # Show enemy threat assessment
    print("\n⚠️ Enemy Threat Assessment:")
    threats = analysis["threat_assessment"]
    for i, threat in enumerate(threats[:3], 1):
        print(f"\n{i}. {threat.hero} - Threat Level: {threat.threat_level:.2f}")
        print(f"   Tier: {threat.meta_data.tier}")
        print(f"   Reasons: {', '.join(threat.threat_reasons)}")
        if threat.suggested_counters:
            print(f"   Suggested Counters: {', '.join(threat.suggested_counters)}")
    
    # Show retrospective counter-pick suggestions
    print("\n🎯 Retrospective Counter-Pick Suggestions:")
    print("(What you could have picked instead for better matchup)")
    
    counter_suggestions = analysis["counter_suggestions"]
    for i, suggestion in enumerate(counter_suggestions[:5], 1):
        print(f"\n{i}. {suggestion.hero} [{suggestion.meta_data.tier} TIER]")
        print(f"   Confidence: {suggestion.confidence:.2f}")
        print(f"   Win Rate: {suggestion.meta_data.win_rate:.1f}%")
        print(f"   Why: {suggestion.reasoning}")
    
    # Show team composition analysis
    print("\n📊 Team Composition Analysis:")
    comp_analysis = analysis["team_composition_analysis"]
    print(f"Balance Score: {comp_analysis['balance_score']:.2f}")
    print(f"Your Team Roles: {comp_analysis['ally_roles']}")
    print(f"Enemy Team Roles: {comp_analysis['enemy_roles']}")
    
    if comp_analysis['strengths']:
        print(f"Strengths: {', '.join(comp_analysis['strengths'])}")
    if comp_analysis['weaknesses']:
        print(f"Weaknesses: {', '.join(comp_analysis['weaknesses'])}")
    
    # Show meta awareness analysis
    print("\n🧠 Meta Awareness Analysis:")
    meta_analysis = analysis["meta_awareness"]
    print(f"Your Team Meta Score: {meta_analysis['ally_meta_score']:.1f}")
    print(f"Enemy Team Meta Score: {meta_analysis['enemy_meta_score']:.1f}")
    print(f"Meta Advantage: {meta_analysis['meta_advantage']:+.1f}")
    print(f"Meta Awareness Rating: {meta_analysis['meta_awareness_rating'].upper()}")
    
    # Show actionable learning points
    print("\n💡 Learning Points for Future Matches:")
    learning_points = analysis["learning_points"]
    
    for i, insight in enumerate(learning_points, 1):
        print(f"\n{i}. {insight.title} [{insight.priority.upper()} PRIORITY]")
        print(f"   Issue: {insight.description}")
        print(f"   Action: {insight.actionable_advice}")
        if insight.relevant_heroes:
            print(f"   Heroes: {', '.join(insight.relevant_heroes)}")
    
    # Show additional coaching insights
    print("\n🎓 Coaching Insights:")
    
    # Analyze what went wrong
    if match_result.match_outcome == "defeat":
        high_threat_enemies = [t for t in threats if t.threat_level > 0.7]
        if high_threat_enemies:
            print(f"⚠️ High-threat enemies that dominated: {', '.join([t.hero for t in high_threat_enemies])}")
        
        if meta_analysis['meta_advantage'] < -20:
            print("📉 Your team was significantly out-meta'd")
        
        if comp_analysis['balance_score'] < 0.5:
            print("⚖️ Team composition was poorly balanced")
    
    # Success factors
    elif match_result.match_outcome == "victory":
        print("🎉 Victory! Here's what worked:")
        if meta_analysis['meta_advantage'] > 10:
            print("✅ Strong meta advantage")
        if comp_analysis['balance_score'] > 0.8:
            print("✅ Well-balanced team composition")
    
    # Future improvement suggestions
    print("\n🚀 For Your Next Match:")
    if counter_suggestions:
        top_suggestion = counter_suggestions[0]
        print(f"Consider learning: {top_suggestion.hero} (Tier {top_suggestion.meta_data.tier})")
    
    if threats:
        most_threatening = threats[0]
        print(f"Ban priority against similar teams: {most_threatening.hero}")
    
    print("\n📈 Overall Assessment:")
    threat_level = analysis["match_summary"]["enemy_threat_level"]
    if threat_level > 0.7:
        print("🔥 You faced a very strong enemy composition")
    elif threat_level > 0.5:
        print("⚖️ Enemy team had moderate threat level")
    else:
        print("😌 Enemy team was relatively weak")
    
    if match_result.match_outcome == "defeat" and len(counter_suggestions) > 0:
        print(f"💡 Key takeaway: {counter_suggestions[0].hero} could have significantly improved your team's chances")


if __name__ == "__main__":
    main()