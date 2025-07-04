# Miya-specific evaluation logic
from typing import List, Dict, Any


def evaluate(data: Dict[str, Any], minutes: int = None) -> List[str]:
    """
    Evaluate Miya's performance with dynamic thresholds.
    
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
    KDA_LOW, KDA_HIGH = 4.0, 6.0
    GPM_MIN = 700  # Base GPM for marksman
    DMG_PER_MIN = 5000  # Miya-specific damage expectation
    TURRET_DMG_PER_MIN = 800  # Marksman objective damage
    PARTICIPATION_MIN = 50  # Marksman teamfight participation
    
    # --- KDA Evaluation ---
    kills = data.get("kills", 0)
    deaths = max(1, data.get("deaths", 1))
    assists = data.get("assists", 0)
    kda = (kills + assists) / deaths
    
    if kda < KDA_LOW:
        fb.append(("warning",
                   f"KDA {kda:.1f} (< {KDA_LOW}). Stay behind tanks and "
                   f"focus on safe positioning."))
    elif kda > KDA_HIGH:
        fb.append(("success",
                   f"Outstanding KDA {kda:.1f}! Your positioning and target "
                   f"selection are clearly a strength."))
    else:
        fb.append(("info",
                   f"Solid KDA ({kda:.1f}). Keep maximizing damage while "
                   f"staying alive."))
    
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
                   f"GPM {gpm} (< {gpm_threshold:.0f}). Prioritize "
                   f"last-hitting and rotate to jungle camps."))
    
    # --- Teamfight Participation ---
    tfp = data.get("teamfight_participation", 0)
    if tfp < PARTICIPATION_MIN:
        fb.append(("warning",
                   f"Low fight presence ({tfp}% < {PARTICIPATION_MIN}%). "
                   f"Respond faster to team pings."))
    
    # --- Hero Damage (scaled by minutes) ---
    damage = data.get("hero_damage", 0)
    dmg_needed = DMG_PER_MIN * minutes
    
    if damage < dmg_needed:
        fb.append(("warning",
                   f"Damage {damage:,} (< {dmg_needed:,}). Focus on hitting "
                   f"key targets in fights."))
    
    # --- Turret Damage (marksman-specific) ---
    turret_damage = data.get("turret_damage", 0)
    turret_needed = TURRET_DMG_PER_MIN * minutes
    
    if turret_damage < turret_needed:
        fb.append(("warning",
                   f"Turret damage {turret_damage:,} (< {turret_needed:,}). "
                   f"Marksmen are key to taking objectives."))
    
    # --- Positioning Check ---
    if data.get("positioning_rating", "").lower() == "low":
        fb.append(("warning",
                   "Positioning flagged low. Stay in the backline during "
                   "fights."))
    
    # --- Miya-Specific Checks ---
    
    # Death check (Miya should have very low deaths)
    if deaths > 5:
        fb.append(("critical",
                   f"{deaths} deaths is too high. Use Miya's ult to "
                   f"reposition or escape dangerous situations."))
    
    # Ultimate usage
    if data.get("ult_usage", "").lower() == "low":
        fb.append(("warning",
                   "Low ultimate usage. Use Miya's ult to reposition, "
                   "chase, or escape."))
    
    # Return formatted feedback
    # For now, return severity + message combined
    return [f"{severity}: {msg}" if severity != "info" else msg 
            for severity, msg in fb]