(.venv) lesz@MacBook-Pro skillshift-ai % python3 test_tactical_coaching_system.py
Starting MLBB Tactical Coaching Agent Demonstration...
This system integrates with your existing temporal pipeline and behavioral modeling.

🎮 MLBB Tactical Coaching Agent - System Demonstration
============================================================

📊 Processing timestamped gameplay data...
   Player: Lesz XVII
   Events: 5
   Duration: 720.5s
INFO:__main__:Analyzing tactical coaching for Lesz XVII
INFO:__main__:Processing 5 timestamped events

⚡ Analysis completed in 0.00s
   Insights generated: 3
   Overall confidence: 0.900

============================================================
📘 TACTICAL COACHING REPORT
============================================================

📝 POST-GAME SUMMARY:
   Critical tactical issues identified (1). Focus on fundamental decision-making and risk management. Your carry-focused approach demonstrates strong mechanical skills. Consider balancing individual excellence with team coordination. Analysis detected 1 missed opportunities for greater impact. Focus on maintaining momentum after advantages. Think of MLBB like chess - every move should consider the next 2-3 moves. Your tactical awareness is developing well, but remember: positioning is like protecting your king, and rotations are like controlling the center. Primary focus areas: positioning safety, map awareness, and objective timing coordination.

🎯 TACTICAL FINDINGS (2):
   1. [CRITICAL] at 320.1s
      Finding: Death due to overextension in enemy_jungle at 320.1s. High-risk positioning without vision coverage.
      Suggestion: Maintain vision control and communicate with team before extending to high-risk areas like enemy jungle.
      Confidence: 0.92

   2. [HIGH] at 590.2s
      Finding: Death due to poor positioning in lord_pit at 590.2s. Eliminated early in teamfight, reducing team's combat effectiveness.
      Suggestion: Focus on staying behind tanks and identifying threats before engaging. Wait for tank initiation and position at maximum effective range.
      Confidence: 0.88


⏱️ GAME PHASE BREAKDOWN:
   Early Game: 2 findings
     - death_overextension at 320.1s
     - death_positioning at 590.2s
   Mid Game: 0 findings
   Late Game: 0 findings

🎨 VISUAL OVERLAYS (1):
   1. Frame: frame_017706_t590.20s.jpg
      Annotations: 2
         - zone: High Risk Position (red)
         - arrow: Suggested Position (green)

💡 MISSED OPPORTUNITIES (1):
   1. at 512.3s - tower_destroyed
      Missed: Not present for tower push after favorable team fight. Missed opportunity to secure map control and gold advantage.
      Alternative: Rotate to tower immediately after team fight advantage at 497.3s. Focus on maintaining momentum.
      Impact Score: 0.8
      Reasoning: Tower gold and map control are crucial for maintaining momentum. This missed opportunity cost approximately 320 gold and strategic positioning.


🏆 GAMIFIED FEEDBACK:
   🎯 Positioning Apprentice: One positioning issue identified – you're improving but stay vigilant!
   🚀 Rotation Expert: Perfect rotation timing – excellent macro awareness and map control!
   🎊 Consistent Performer: Few major tactical issues identified – maintain this level of gameplay excellence!

💾 Complete report saved to: tactical_coaching_report.json

🌐 API Integration Test
==============================
🎮 MLBB Tactical Coaching Agent - System Demonstration
============================================================

📊 Processing timestamped gameplay data...
   Player: Lesz XVII
   Events: 5
   Duration: 720.5s
INFO:__main__:Analyzing tactical coaching for Lesz XVII
INFO:__main__:Processing 5 timestamped events

⚡ Analysis completed in 0.00s
   Insights generated: 3
   Overall confidence: 0.900

============================================================
📘 TACTICAL COACHING REPORT
============================================================

📝 POST-GAME SUMMARY:
   Critical tactical issues identified (1). Focus on fundamental decision-making and risk management. Your carry-focused approach demonstrates strong mechanical skills. Consider balancing individual excellence with team coordination. Analysis detected 1 missed opportunities for greater impact. Focus on maintaining momentum after advantages. Think of MLBB like chess - every move should consider the next 2-3 moves. Your tactical awareness is developing well, but remember: positioning is like protecting your king, and rotations are like controlling the center. Primary focus areas: positioning safety, map awareness, and objective timing coordination.

🎯 TACTICAL FINDINGS (2):
   1. [CRITICAL] at 320.1s
      Finding: Death due to overextension in enemy_jungle at 320.1s. High-risk positioning without vision coverage.
      Suggestion: Maintain vision control and communicate with team before extending to high-risk areas like enemy jungle.
      Confidence: 0.92

   2. [HIGH] at 590.2s
      Finding: Death due to poor positioning in lord_pit at 590.2s. Eliminated early in teamfight, reducing team's combat effectiveness.
      Suggestion: Focus on staying behind tanks and identifying threats before engaging. Wait for tank initiation and position at maximum effective range.
      Confidence: 0.88


⏱️ GAME PHASE BREAKDOWN:
   Early Game: 2 findings
     - death_overextension at 320.1s
     - death_positioning at 590.2s
   Mid Game: 0 findings
   Late Game: 0 findings

🎨 VISUAL OVERLAYS (1):
   1. Frame: frame_017706_t590.20s.jpg
      Annotations: 2
         - zone: High Risk Position (red)
         - arrow: Suggested Position (green)

💡 MISSED OPPORTUNITIES (1):
   1. at 512.3s - tower_destroyed
      Missed: Not present for tower push after favorable team fight. Missed opportunity to secure map control and gold advantage.
      Alternative: Rotate to tower immediately after team fight advantage at 497.3s. Focus on maintaining momentum.
      Impact Score: 0.8
      Reasoning: Tower gold and map control are crucial for maintaining momentum. This missed opportunity cost approximately 320 gold and strategic positioning.


🏆 GAMIFIED FEEDBACK:
   🎯 Positioning Apprentice: One positioning issue identified – you're improving but stay vigilant!
   🚀 Rotation Expert: Perfect rotation timing – excellent macro awareness and map control!
   🎊 Consistent Performer: Few major tactical issues identified – maintain this level of gameplay excellence!

💾 Complete report saved to: tactical_coaching_report.json

📡 API Response Preview:
{
  "success": true,
  "data": {
    "player_ign": "Lesz XVII",
    "video_path": "example_gameplay.mp4",
    "analysis_timestamp": "2025-07-13T15:19:11.857636",
    "post_game_summary": "Critical tactical issues identified (1). Focus on fundamental decision-making and risk management. Your carry-focused approach demonstrates strong mechanical skills. Consider balancing individual excellence with team coordination. Analysis detected 1 missed opportunities for greater impact. Focus on maintaining...

✅ Demonstration Complete!
   The system successfully analyzed 5 events
   Generated 3 actionable insights
   Provided comprehensive tactical coaching in the requested format

🚀 Ready for integration with your existing MLBB Coach AI system!
(.venv) lesz@MacBook-Pro skillshift-ai % 