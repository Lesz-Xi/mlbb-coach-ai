# Franco-specific evaluation logic
from typing import List, Dict, Any


def evaluate(data: Dict[str, Any], minutes: int = None) -> List[str]:
    """
    Evaluate Franco's performance with dynamic thresholds.
    
    Args:
        data: Match data dictionary
        minutes: Match duration in minutes
        
    Returns:
        List of feedback messages
    """
    fb = []
    
    # Default match duration if not provided
    if minutes is None:
        minutes = data.get('match_duration', 15)
    
    # --- Constants (will be loaded from YAML in full implementation) ---
    KDA_LOW, KDA_HIGH = 2.5, 4.0
    GPM_MIN = 450  # Base GPM for tank
    DMG_TAKEN_PER_MIN = 4000  # Franco should absorb damage
    PARTICIPATION_MIN = 50  # Tank teamfight participation
    HOOKS_MIN = 4  # Franco-specific hook requirement
    ENGAGES_MIN = 3  # Team engagement requirement
    VISION_MIN = 5  # Vision control requirement
    
    # --- KDA Evaluation ---
    kills = data.get("kills", 0)
    deaths = max(1, data.get("deaths", 1))
    assists = data.get("assists", 0)
    kda = (kills + assists) / deaths
    
    if kda < KDA_LOW:
        fb.append(("warning",
                   f"KDA {kda:.1f} (< {KDA_LOW}). Ensure your deaths result "
                   f"in won teamfights or key objectives."))
    elif kda > KDA_HIGH:
        fb.append(("success",
                   f"Solid KDA {kda:.1f} for a tank! You're clearly setting "
                   f"up successful plays."))
    else:
        fb.append(("info",
                   f"Decent KDA ({kda:.1f}). Keep initiating and peeling "
                   f"for your team."))
    
    # --- GPM Evaluation (scaled by match duration) ---
    gpm = data.get("gold_per_min", 0)
    
    # Scale expectations based on game phase
    if minutes < 10:
        gpm_threshold = GPM_MIN * 0.8
    elif minutes > 20:
        gpm_threshold = GPM_MIN * 1.1
    else:
        gpm_threshold = GPM_MIN
        
    if gpm < gpm_threshold:
        fb.append(("warning",
                   f"GPM {gpm} (< {gpm_threshold:.0f}). Get assists and "
                   f"help clear waves for defensive items."))
    
    # --- Teamfight Participation ---
    tfp = data.get("teamfight_participation", 0)
    if tfp < PARTICIPATION_MIN:
        fb.append(("warning",
                   f"Low fight presence ({tfp}% < {PARTICIPATION_MIN}%). "
                   f"Respond faster to team pings."))
    
    # --- Damage Taken (scaled by minutes) ---
    damage_taken = data.get("damage_taken", 0)
    dmg_taken_needed = DMG_TAKEN_PER_MIN * minutes
    
    if damage_taken < dmg_taken_needed:
        fb.append(("warning",
                   f"Damage taken {damage_taken:,} (< {dmg_taken_needed:,}). "
                   f"Your frontline presence is lacking."))
    
    # --- Positioning Check ---
    if data.get("positioning_rating", "").lower() == "low":
        fb.append(("warning",
                   "Positioning flagged low. Stay just ahead of your "
                   "squishies—not in Narnia."))
    
    # --- Franco-Specific Checks ---
    
    # Death check (Franco can die but not excessively)
    if deaths > 7:
        fb.append(("critical",
                   f"{deaths} deaths is too high. Franco's value is in "
                   f"disruption, not dying for no reason."))
    
    # Hook accuracy
    hooks_landed = data.get("hooks_landed", 0)
    if hooks_landed < HOOKS_MIN:
        fb.append(("warning",
                   f"Only {hooks_landed} hooks landed (< {HOOKS_MIN}). "
                   f"Focus on prediction and baiting enemy movement."))
    
    # Team engages
    team_engages = data.get("team_engages", 0)
    if team_engages < ENGAGES_MIN:
        fb.append(("warning",
                   f"Only {team_engages} team engages (< {ENGAGES_MIN}). "
                   f"Create openings and apply pressure."))
    
    # Vision score
    vision_score = data.get("vision_score", 0)
    if vision_score < VISION_MIN:
        fb.append(("warning",
                   f"Vision score {vision_score} (< {VISION_MIN}). Place "
                   f"yourself near choke points for control."))
    
    # Return formatted feedback
    # For now, return severity + message combined
    return [f"{severity}: {msg}" if severity != "info" else msg 
            for severity, msg in fb] 