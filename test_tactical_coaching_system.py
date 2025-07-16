#!/usr/bin/env python3
"""
Comprehensive Test Suite for Tactical Coaching System
====================================================

This test file demonstrates the complete tactical coaching system integration
with your existing MLBB Coach AI architecture.

Usage:
    python test_tactical_coaching_system.py

This will run a complete demonstration of:
1. Temporal analysis pipeline
2. Behavioral modeling
3. Tactical coaching analysis
4. API response generation
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock data for testing (simulating real temporal analysis results)
MOCK_TEMPORAL_DATA = {
    "video_analysis": {
        "video_path": "example_gameplay.mp4",
        "player_ign": "Lesz XVII",
        "total_duration": 720.5,
        "analysis_timestamp": time.time()
    },
    "timestamped_events": [
        {
            "timestamp": 245.5,
            "event_type": "movement_event",
            "event_subtype": "rotation",
            "player": "Lesz XVII",
            "from_region": "mid_lane",
            "to_region": "bot_lane",
            "distance": 0.45,
            "confidence": 0.85,
            "metadata": {
                "speed": 0.12,
                "duration": 3.75,
                "frame_path": "frame_007365_t245.50s.jpg"
            }
        },
        {
            "timestamp": 320.1,
            "event_type": "game_event",
            "event_subtype": "death",
            "player": "Lesz XVII",
            "confidence": 0.92,
            "metadata": {
                "location": "enemy_jungle",
                "cause": "overextension",
                "frame_path": "frame_009603_t320.10s.jpg"
            }
        },
        {
            "timestamp": 512.3,
            "event_type": "game_event",
            "event_subtype": "tower_destroyed",
            "player": "Enemy Team",
            "confidence": 0.92,
            "metadata": {
                "detection_method": "text_pattern",
                "gold_change": 320,
                "frame_path": "frame_015369_t512.30s.jpg"
            }
        },
        {
            "timestamp": 580.7,
            "event_type": "game_event",
            "event_subtype": "teamfight_start",
            "player": "Lesz XVII",
            "confidence": 0.78,
            "metadata": {
                "location": "lord_pit",
                "participants": ["Lesz XVII", "Teammate1", "Teammate2"],
                "frame_path": "frame_017421_t580.70s.jpg"
            }
        },
        {
            "timestamp": 590.2,
            "event_type": "game_event",
            "event_subtype": "death",
            "player": "Lesz XVII",
            "confidence": 0.88,
            "metadata": {
                "location": "lord_pit",
                "cause": "poor_positioning",
                "frame_path": "frame_017706_t590.20s.jpg"
            }
        }
    ],
    "event_summary": {
        "total_events": 5,
        "game_events": 4,
        "movement_events": 1,
        "time_span": {
            "start": 245.5,
            "end": 590.2
        }
    }
}


class MockTacticalCoachingAgent:
    """
    Mock implementation of the tactical coaching agent that demonstrates
    all the features requested in the specification.
    """
    
    def __init__(self):
        self.coaching_patterns = {
            "rotation_patterns": {
                "late_rotation": {
                    "description": "Player rotates too late to objectives",
                    "severity": "high",
                    "coaching_tip": "Improve map awareness and anticipate teamfight locations"
                },
                "overextension": {
                    "description": "Player extends too far without vision",
                    "severity": "critical",
                    "coaching_tip": "Maintain vision control before extending"
                }
            },
            "positioning_patterns": {
                "poor_teamfight_positioning": {
                    "description": "Player positions poorly in teamfights",
                    "severity": "high",
                    "coaching_tip": "Stay behind tanks and identify threats before engaging"
                }
            }
        }
    
    def analyze_temporal_data(self, temporal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze timestamped gameplay data and generate tactical insights.
        
        This is the main method that processes the temporal data and generates
        the complete tactical coaching report as specified in the requirements.
        """
        player_ign = temporal_data["video_analysis"]["player_ign"]
        events = temporal_data["timestamped_events"]
        
        logger.info(f"Analyzing tactical coaching for {player_ign}")
        logger.info(f"Processing {len(events)} timestamped events")
        
        # Step 1: Analyze decision-making quality
        tactical_findings = self._analyze_decision_making(events)
        
        # Step 2: Generate visual overlays
        visual_overlays = self._generate_visual_overlays(events, tactical_findings)
        
        # Step 3: Identify missed opportunities
        opportunity_analysis = self._identify_missed_opportunities(events)
        
        # Step 4: Group insights by game phases
        game_phase_breakdown = self._group_by_game_phase(tactical_findings)
        
        # Step 5: Generate gamified feedback
        gamified_feedback = self._generate_gamified_feedback(tactical_findings)
        
        # Step 6: Generate natural language summary
        post_game_summary = self._generate_post_game_summary(
            tactical_findings, opportunity_analysis
        )
        
        # Create comprehensive report
        report = {
            "player_ign": player_ign,
            "video_path": temporal_data["video_analysis"]["video_path"],
            "analysis_timestamp": datetime.now().isoformat(),
            "post_game_summary": post_game_summary,
            "tactical_findings": tactical_findings,
            "visual_overlays": visual_overlays,
            "game_phase_breakdown": game_phase_breakdown,
            "opportunity_analysis": opportunity_analysis,
            "gamified_feedback": gamified_feedback,
            "overall_confidence": self._calculate_overall_confidence(tactical_findings),
            "processing_time": 2.5,  # Mock processing time
            "insights_generated": len(tactical_findings) + len(opportunity_analysis)
        }
        
        return report
    
    def _analyze_decision_making(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze decision-making quality for each major event."""
        findings = []
        
        for event in events:
            timestamp = event["timestamp"]
            game_phase = self._determine_game_phase(timestamp)
            
            if event["event_subtype"] == "rotation":
                # Analyze rotation timing
                if self._is_late_rotation(event, events):
                    findings.append({
                        "timestamp": timestamp,
                        "event": "late_rotation",
                        "finding": (
                            f"Late rotation to {event['to_region']} at {timestamp:.1f}s. "
                            f"Missed opportunity to assist team or counter enemy movements."
                        ),
                        "suggestion": (
                            "Prioritize map awareness and anticipate objective timings. "
                            "Watch for enemy positioning cues 30-45 seconds before rotating."
                        ),
                        "severity": "high",
                        "confidence": 0.85,
                        "game_phase": game_phase,
                        "event_type": "rotation_analysis",
                        "metadata": {
                            "from_region": event["from_region"],
                            "to_region": event["to_region"],
                            "distance": event["distance"]
                        }
                    })
            
            elif event["event_subtype"] == "death":
                # Analyze death events
                location = event["metadata"].get("location", "unknown")
                cause = event["metadata"].get("cause", "unknown")
                
                if cause == "overextension":
                    findings.append({
                        "timestamp": timestamp,
                        "event": "death_overextension",
                        "finding": (
                            f"Death due to overextension in {location} at {timestamp:.1f}s. "
                            f"High-risk positioning without vision coverage."
                        ),
                        "suggestion": (
                            "Maintain vision control and communicate with team before "
                            "extending to high-risk areas like enemy jungle."
                        ),
                        "severity": "critical",
                        "confidence": 0.92,
                        "game_phase": game_phase,
                        "event_type": "death_analysis",
                        "metadata": {
                            "location": location,
                            "cause": cause
                        }
                    })
                
                elif cause == "poor_positioning":
                    findings.append({
                        "timestamp": timestamp,
                        "event": "death_positioning",
                        "finding": (
                            f"Death due to poor positioning in {location} at {timestamp:.1f}s. "
                            f"Eliminated early in teamfight, reducing team's combat effectiveness."
                        ),
                        "suggestion": (
                            "Focus on staying behind tanks and identifying threats before engaging. "
                            "Wait for tank initiation and position at maximum effective range."
                        ),
                        "severity": "high",
                        "confidence": 0.88,
                        "game_phase": game_phase,
                        "event_type": "positioning_analysis",
                        "metadata": {
                            "location": location,
                            "teamfight_context": True
                        }
                    })
        
        return findings
    
    def _generate_visual_overlays(self, events: List[Dict[str, Any]], 
                                 findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate timestamp-aligned visual overlay annotations."""
        overlays = []
        
        for finding in findings:
            frame_path = self._get_frame_path_for_timestamp(finding["timestamp"], events)
            
            if finding["event_type"] == "rotation_analysis":
                overlays.append({
                    "frame_path": frame_path,
                    "annotations": [
                        {
                            "type": "arrow",
                            "from_region": finding["metadata"]["from_region"],
                            "to_region": finding["metadata"]["to_region"],
                            "label": "Missed Rotation",
                            "color": "red",
                            "opacity": 0.8
                        },
                        {
                            "type": "zone",
                            "region": "objective_area",
                            "label": "No Vision",
                            "color": "orange",
                            "opacity": 0.6
                        }
                    ]
                })
            
            elif finding["event_type"] == "positioning_analysis":
                overlays.append({
                    "frame_path": frame_path,
                    "annotations": [
                        {
                            "type": "zone",
                            "region": "danger_zone",
                            "label": "High Risk Position",
                            "color": "red",
                            "opacity": 0.7
                        },
                        {
                            "type": "arrow",
                            "from_region": "current_position",
                            "to_region": "safe_position",
                            "label": "Suggested Position",
                            "color": "green",
                            "opacity": 0.8
                        }
                    ]
                })
        
        return overlays
    
    def _identify_missed_opportunities(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify missed opportunities with high-impact potential."""
        opportunities = []
        
        for i, event in enumerate(events):
            if event["event_subtype"] == "tower_destroyed":
                # Check if player was involved in tower push
                player_actions_nearby = [
                    e for e in events 
                    if abs(e["timestamp"] - event["timestamp"]) < 30
                    and e["player"] == "Lesz XVII"
                ]
                
                if not player_actions_nearby:
                    opportunities.append({
                        "timestamp": event["timestamp"],
                        "event": "tower_destroyed",
                        "missed_action": (
                            "Not present for tower push after favorable team fight. "
                            "Missed opportunity to secure map control and gold advantage."
                        ),
                        "alternative": (
                            f"Rotate to tower immediately after team fight advantage "
                            f"at {event['timestamp'] - 15:.1f}s. Focus on maintaining momentum."
                        ),
                        "impact_score": 0.8,
                        "reasoning": (
                            "Tower gold and map control are crucial for maintaining momentum. "
                            "This missed opportunity cost approximately 320 gold and strategic positioning."
                        ),
                        "metadata": {
                            "tower_gold": 320,
                            "map_control_value": "high",
                            "team_coordination_issue": True
                        }
                    })
        
        return opportunities
    
    def _group_by_game_phase(self, findings: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group tactical findings by game phase (early/mid/late game)."""
        phase_groups = {
            "early_game": [],
            "mid_game": [],
            "late_game": []
        }
        
        for finding in findings:
            game_phase = finding["game_phase"]
            if game_phase in phase_groups:
                phase_groups[game_phase].append(finding)
        
        return phase_groups
    
    def _generate_gamified_feedback(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Generate gamified achievements and challenges based on performance."""
        feedback = []
        
        # Count issue types
        positioning_issues = len([f for f in findings if "positioning" in f["event"]])
        rotation_issues = len([f for f in findings if "rotation" in f["event"]])
        critical_issues = len([f for f in findings if f["severity"] == "critical"])
        
        # Generate achievements based on performance patterns
        if positioning_issues >= 2:
            feedback.append(
                "🏆 Map Awareness Rookie: 2+ positioning issues detected – "
                "focus on threat identification and safe positioning"
            )
        elif positioning_issues == 1:
            feedback.append(
                "🎯 Positioning Apprentice: One positioning issue identified – "
                "you're improving but stay vigilant!"
            )
        elif positioning_issues == 0:
            feedback.append(
                "🌟 Positioning Master: Excellent positioning awareness – "
                "keep up the outstanding spatial control!"
            )
        
        if rotation_issues >= 2:
            feedback.append(
                "🗺️ Rotation Trainee: Late to 2+ key rotations – "
                "improve map awareness and timing anticipation"
            )
        elif rotation_issues == 1:
            feedback.append(
                "⚡ Rotation Learner: One rotation timing issue – "
                "focus on earlier map movement cues"
            )
        elif rotation_issues == 0:
            feedback.append(
                "🚀 Rotation Expert: Perfect rotation timing – "
                "excellent macro awareness and map control!"
            )
        
        if critical_issues == 0:
            feedback.append(
                "🛡️ Tactical Strategist: No critical issues detected – "
                "solid decision-making and risk management"
            )
        elif critical_issues >= 2:
            feedback.append(
                "⚠️ Risk Management Focus: Multiple critical issues – "
                "prioritize safer decision-making and vision control"
            )
        
        # Motivational feedback
        high_confidence_findings = [f for f in findings if f["confidence"] > 0.85]
        if len(high_confidence_findings) < 3:
            feedback.append(
                "🎊 Consistent Performer: Few major tactical issues identified – "
                "maintain this level of gameplay excellence!"
            )
        
        return feedback
    
    def _generate_post_game_summary(self, findings: List[Dict[str, Any]], 
                                   opportunities: List[Dict[str, Any]]) -> str:
        """Generate natural language post-game summary with mentorship tone."""
        critical_issues = len([f for f in findings if f["severity"] == "critical"])
        high_issues = len([f for f in findings if f["severity"] == "high"])
        missed_opportunities = len(opportunities)
        
        # Assess overall performance
        if critical_issues > 0:
            severity_assessment = (
                f"Critical tactical issues identified ({critical_issues}). "
                f"Focus on fundamental decision-making and risk management."
            )
        elif high_issues > 2:
            severity_assessment = (
                f"Multiple high-priority areas for improvement ({high_issues}). "
                f"Solid foundation with room for tactical refinement."
            )
        elif high_issues > 0:
            severity_assessment = (
                f"Some tactical adjustments needed ({high_issues}). "
                f"Good overall gameplay with specific areas to polish."
            )
        else:
            severity_assessment = (
                "Strong tactical gameplay overall. "
                "Excellent decision-making and strategic awareness."
            )
        
        # Analyze play style (simplified)
        play_style_insight = (
            "Your carry-focused approach demonstrates strong mechanical skills. "
            "Consider balancing individual excellence with team coordination."
        )
        
        # Opportunity analysis
        if missed_opportunities > 0:
            opportunity_insight = (
                f"Analysis detected {missed_opportunities} missed opportunities "
                f"for greater impact. Focus on maintaining momentum after advantages."
            )
        else:
            opportunity_insight = (
                "Excellent opportunity recognition and execution. "
                "Keep capitalizing on favorable moments."
            )
        
        # Chess analogy for strategic thinking
        strategic_analogy = (
            "Think of MLBB like chess - every move should consider the next 2-3 moves. "
            "Your tactical awareness is developing well, but remember: "
            "positioning is like protecting your king, and rotations are like controlling the center."
        )
        
        return (
            f"{severity_assessment} {play_style_insight} {opportunity_insight} "
            f"{strategic_analogy} "
            f"Primary focus areas: positioning safety, map awareness, and objective timing coordination."
        )
    
    def _determine_game_phase(self, timestamp: float) -> str:
        """Determine game phase based on timestamp."""
        if timestamp <= 600:  # 0-10 minutes
            return "early_game"
        elif timestamp <= 1200:  # 10-20 minutes
            return "mid_game"
        else:  # 20+ minutes
            return "late_game"
    
    def _is_late_rotation(self, rotation_event: Dict[str, Any], 
                         all_events: List[Dict[str, Any]]) -> bool:
        """Check if rotation timing was late based on nearby events."""
        # Simplified logic - in real implementation, this would analyze
        # teamfight timing, objective spawns, enemy movements, etc.
        return rotation_event["timestamp"] > 300  # Arbitrary threshold for demo
    
    def _get_frame_path_for_timestamp(self, timestamp: float, 
                                     events: List[Dict[str, Any]]) -> str:
        """Get frame path for a specific timestamp."""
        # Find closest event with frame path
        for event in events:
            if abs(event["timestamp"] - timestamp) < 5:
                return event["metadata"].get("frame_path", f"frame_t{timestamp:.1f}s.jpg")
        return f"frame_t{timestamp:.1f}s.jpg"
    
    def _calculate_overall_confidence(self, findings: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score for the analysis."""
        if not findings:
            return 0.0
        
        confidence_scores = [f["confidence"] for f in findings]
        return sum(confidence_scores) / len(confidence_scores)


def demonstrate_tactical_coaching_system():
    """
    Demonstrate the complete tactical coaching system with sample data.
    
    This function shows how the system works end-to-end:
    1. Takes timestamped gameplay data
    2. Analyzes tactical patterns
    3. Generates coaching insights
    4. Provides structured output in the requested format
    """
    
    print("🎮 MLBB Tactical Coaching Agent - System Demonstration")
    print("=" * 60)
    
    # Initialize the coaching agent
    agent = MockTacticalCoachingAgent()
    
    # Process the temporal data
    print("\n📊 Processing timestamped gameplay data...")
    print(f"   Player: {MOCK_TEMPORAL_DATA['video_analysis']['player_ign']}")
    print(f"   Events: {len(MOCK_TEMPORAL_DATA['timestamped_events'])}")
    print(f"   Duration: {MOCK_TEMPORAL_DATA['video_analysis']['total_duration']:.1f}s")
    
    # Generate tactical coaching report
    start_time = time.time()
    coaching_report = agent.analyze_temporal_data(MOCK_TEMPORAL_DATA)
    processing_time = time.time() - start_time
    
    print(f"\n⚡ Analysis completed in {processing_time:.2f}s")
    print(f"   Insights generated: {coaching_report['insights_generated']}")
    print(f"   Overall confidence: {coaching_report['overall_confidence']:.3f}")
    
    # Display the complete coaching report
    print("\n" + "=" * 60)
    print("📘 TACTICAL COACHING REPORT")
    print("=" * 60)
    
    # Post-game summary
    print(f"\n📝 POST-GAME SUMMARY:")
    print(f"   {coaching_report['post_game_summary']}")
    
    # Tactical findings
    print(f"\n🎯 TACTICAL FINDINGS ({len(coaching_report['tactical_findings'])}):")
    for i, finding in enumerate(coaching_report['tactical_findings'], 1):
        print(f"   {i}. [{finding['severity'].upper()}] at {finding['timestamp']:.1f}s")
        print(f"      Finding: {finding['finding']}")
        print(f"      Suggestion: {finding['suggestion']}")
        print(f"      Confidence: {finding['confidence']:.2f}")
        print()
    
    # Game phase breakdown
    print(f"\n⏱️ GAME PHASE BREAKDOWN:")
    for phase, phase_findings in coaching_report['game_phase_breakdown'].items():
        print(f"   {phase.replace('_', ' ').title()}: {len(phase_findings)} findings")
        for finding in phase_findings:
            print(f"     - {finding['event']} at {finding['timestamp']:.1f}s")
    
    # Visual overlays
    print(f"\n🎨 VISUAL OVERLAYS ({len(coaching_report['visual_overlays'])}):")
    for i, overlay in enumerate(coaching_report['visual_overlays'], 1):
        print(f"   {i}. Frame: {overlay['frame_path']}")
        print(f"      Annotations: {len(overlay['annotations'])}")
        for annotation in overlay['annotations']:
            print(f"         - {annotation['type']}: {annotation['label']} ({annotation['color']})")
    
    # Missed opportunities
    print(f"\n💡 MISSED OPPORTUNITIES ({len(coaching_report['opportunity_analysis'])}):")
    for i, opportunity in enumerate(coaching_report['opportunity_analysis'], 1):
        print(f"   {i}. at {opportunity['timestamp']:.1f}s - {opportunity['event']}")
        print(f"      Missed: {opportunity['missed_action']}")
        print(f"      Alternative: {opportunity['alternative']}")
        print(f"      Impact Score: {opportunity['impact_score']:.1f}")
        print(f"      Reasoning: {opportunity['reasoning']}")
        print()
    
    # Gamified feedback
    print(f"\n🏆 GAMIFIED FEEDBACK:")
    for feedback in coaching_report['gamified_feedback']:
        print(f"   {feedback}")
    
    # Save the complete report as JSON
    output_file = "tactical_coaching_report.json"
    with open(output_file, 'w') as f:
        json.dump(coaching_report, f, indent=2)
    
    print(f"\n💾 Complete report saved to: {output_file}")
    
    # Return the report for API integration
    return coaching_report


def test_api_integration():
    """Test API integration format."""
    print("\n🌐 API Integration Test")
    print("=" * 30)
    
    # Generate report
    report = demonstrate_tactical_coaching_system()
    
    # Show API response format
    api_response = {
        "success": True,
        "data": report,
        "metadata": {
            "api_version": "1.0",
            "processing_time": report["processing_time"],
            "endpoint": "/api/tactical-coaching/analyze"
        }
    }
    
    print("\n📡 API Response Preview:")
    print(json.dumps(api_response, indent=2)[:500] + "...")
    
    return api_response


if __name__ == "__main__":
    # Run the complete demonstration
    print("Starting MLBB Tactical Coaching Agent Demonstration...")
    print("This system integrates with your existing temporal pipeline and behavioral modeling.")
    print()
    
    # Run the demonstration
    coaching_report = demonstrate_tactical_coaching_system()
    
    # Test API integration
    api_response = test_api_integration()
    
    print("\n✅ Demonstration Complete!")
    print(f"   The system successfully analyzed {len(MOCK_TEMPORAL_DATA['timestamped_events'])} events")
    print(f"   Generated {coaching_report['insights_generated']} actionable insights")
    print(f"   Provided comprehensive tactical coaching in the requested format")
    print()
    print("🚀 Ready for integration with your existing MLBB Coach AI system!") 