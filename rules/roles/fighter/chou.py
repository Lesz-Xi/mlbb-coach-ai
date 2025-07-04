# Chou-specific evaluation logic
from typing import List, Dict, Any


def evaluate(data: Dict[str, Any], minutes: int = None) -> List[str]:
    """
    Evaluate Chou's performance with dynamic thresholds.
    
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
    KDA_LOW, KDA_HIGH = 3.0, 5.0
    GPM_MIN = 650  # Base GPM for fighter
    DMG_PER_MIN = 3750  # Chou-specific damage expectation
    PARTICIPATION_MIN = 60  # Chou needs high participation
    
    # --- KDA Evaluation ---
    kills = data.get("kills", 0)
    deaths = max(1, data.get("deaths", 1))
    assists = data.get("assists", 0)
    kda = (kills + assists) / deaths
    
    if kda < KDA_LOW:
        fb.append(("warning",
                   f"KDA {kda:.1f} (< {KDA_LOW}). Pick safer angles and "
                   f"exit with 2nd skill."))
    elif kda > KDA_HIGH:
        fb.append(("success",
                   f"Great KDA {kda:.1f}! Keep pressuring their back-line."))
    else:
        fb.append(("info",
                   f"Decent KDA ({kda:.1f}). You're on track—keep "
                   f"snowballing."))
    
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
                   f"GPM {gpm} (< {gpm_threshold:.0f}). Clear side waves "
                   f"between ganks."))
    
    # --- Teamfight Participation ---
    tfp = data.get("teamfight_participation", 0)
    if tfp < PARTICIPATION_MIN:
        fb.append(("warning",
                   f"Low fight presence ({tfp}% < {PARTICIPATION_MIN}%). "
                   f"Collapse faster on ally engages."))
    
    # --- Hero Damage (scaled by minutes) ---
    damage = data.get("hero_damage", 0)
    dmg_needed = DMG_PER_MIN * minutes
    
    if damage < dmg_needed:
        fb.append(("warning",
                   f"Damage {damage:,} (< {dmg_needed:,.0f}). Prioritise "
                   f"isolated kills."))
    
    # --- Positioning Check ---
    if data.get("positioning_rating", "").lower() == "low":
        fb.append(("warning",
                   "Positioning flagged low. Hold Retribution for wall "
                   "dashes."))
    
    # --- Chou-Specific Checks ---
    
    # Death check (Chou should die less than average fighter)
    if deaths > 6:
        fb.append(("critical",
                   f"{deaths} deaths is too high. Chou has great mobility—"
                   f"use Shunpo to disengage after combos."))
    
    # Damage ratio check
    damage_taken = data.get("damage_taken", 1)
    if damage_taken > 0 and damage > 0:
        damage_ratio = damage / damage_taken
        if damage_ratio < 1.2:
            fb.append(("warning",
                       f"Damage ratio {damage_ratio:.1f}. You're taking more "
                       f"damage than dealing—use immunity frames better."))
    
    # Ultimate usage
    if data.get("ult_usage", "").lower() == "low":
        fb.append(("warning",
                   "Low ultimate usage. Way of Dragon is your playmaking "
                   "tool—look for isolated carries to kick."))
    
    # Return formatted feedback
    # For now, return severity + message combined
    return [f"{severity}: {msg}" if severity != "info" else msg 
            for severity, msg in fb]