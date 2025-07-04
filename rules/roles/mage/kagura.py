# Kagura-specific evaluation logic
from typing import List, Dict, Any


def evaluate(data: Dict[str, Any], minutes: int = None) -> List[str]:
    """
    Evaluate Kagura's performance with dynamic thresholds.
    
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
    GPM_MIN = 600  # Base GPM for mage
    DMG_PER_MIN = 6000  # Kagura-specific damage expectation (high burst)
    PARTICIPATION_MIN = 60  # Mage teamfight participation
    
    # --- KDA Evaluation ---
    kills = data.get("kills", 0)
    deaths = max(1, data.get("deaths", 1))
    assists = data.get("assists", 0)
    kda = (kills + assists) / deaths
    
    if kda < KDA_LOW:
        fb.append(("warning",
                   f"KDA {kda:.1f} (< {KDA_LOW}). Use your umbrella teleport "
                   f"to escape after combos."))
    elif kda > KDA_HIGH:
        fb.append(("success",
                   f"Great KDA {kda:.1f}! You're making a huge impact in "
                   f"fights with those combos."))
    else:
        fb.append(("info",
                   f"Solid KDA ({kda:.1f}). Keep landing those powerful "
                   f"umbrella combos."))
    
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
                   f"GPM {gpm} (< {gpm_threshold:.0f}). Clear minion waves "
                   f"and jungle camps between fights."))
    
    # --- Teamfight Participation ---
    tfp = data.get("teamfight_participation", 0)
    if tfp < PARTICIPATION_MIN:
        fb.append(("warning",
                   f"Low fight presence ({tfp}% < {PARTICIPATION_MIN}%). "
                   f"Your burst damage is crucial in clashes."))
    
    # --- Hero Damage (scaled by minutes) ---
    damage = data.get("hero_damage", 0)
    dmg_needed = DMG_PER_MIN * minutes
    
    if damage < dmg_needed:
        fb.append(("warning",
                   f"Damage {damage:,} (< {dmg_needed:,}). Poke more with "
                   f"umbrella and look for combo opportunities."))
    
    # --- Positioning Check ---
    if data.get("positioning_rating", "").lower() == "low":
        fb.append(("warning",
                   "Positioning flagged low. Stay behind your tank and use "
                   "umbrella to zone from safe distance."))
    
    # --- Kagura-Specific Checks ---
    
    # Death check (Kagura is mobile but squishy)
    if deaths > 6:
        fb.append(("critical",
                   f"{deaths} deaths is too high. Use umbrella teleport to "
                   f"reposition safely after casting spells."))
    
    # Damage ratio check (Kagura should deal much more than she takes)
    damage_taken = data.get("damage_taken", 1)
    if damage_taken > 0 and damage > 0:
        damage_ratio = damage / damage_taken
        if damage_ratio < 3.0:
            fb.append(("warning",
                       f"Damage ratio {damage_ratio:.1f}. Use umbrella "
                       f"mobility to stay safe while dealing damage."))
    
    # Ultimate usage
    if data.get("ult_usage", "").lower() == "low":
        fb.append(("warning",
                   "Low ultimate usage. Look for opportunities to use ult "
                   "with umbrella to catch enemies."))
    
    # Return formatted feedback
    # For now, return severity + message combined
    return [f"{severity}: {msg}" if severity != "info" else msg 
            for severity, msg in fb]
