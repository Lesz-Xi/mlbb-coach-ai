# Lancelot-specific evaluation logic
from typing import List, Dict, Any


def evaluate(data: Dict[str, Any], minutes: int = None) -> List[str]:
    """
    Evaluate Lancelot's performance with dynamic thresholds.
    
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
    GPM_MIN = 750  # Base GPM for assassin
    DMG_PER_MIN = 4500  # Lancelot-specific damage expectation
    PARTICIPATION_MIN = 50  # Assassin teamfight participation
    
    # --- KDA Evaluation ---
    kills = data.get("kills", 0)
    deaths = max(1, data.get("deaths", 1))
    assists = data.get("assists", 0)
    kda = (kills + assists) / deaths
    
    if kda < KDA_LOW:
        fb.append(("warning",
                   f"KDA {kda:.1f} (< {KDA_LOW}). Pick off key targets and "
                   f"escape safely with your mobility."))
    elif kda > KDA_HIGH:
        fb.append(("success",
                   f"Excellent KDA {kda:.1f}! Your ability to secure kills "
                   f"while staying alive is top-tier."))
    else:
        fb.append(("info",
                   f"Good KDA ({kda:.1f}). Keep snowballing your lead and "
                   f"target priority enemies."))
    
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
                   f"GPM {gpm} (< {gpm_threshold:.0f}). A fed assassin is "
                   f"scary—farm efficiently between ganks."))
    
    # --- Teamfight Participation ---
    tfp = data.get("teamfight_participation", 0)
    if tfp < PARTICIPATION_MIN:
        fb.append(("warning",
                   f"Low fight presence ({tfp}% < {PARTICIPATION_MIN}%). "
                   f"Reposition or respond faster to pings."))
    
    # --- Hero Damage (scaled by minutes) ---
    damage = data.get("hero_damage", 0)
    dmg_needed = DMG_PER_MIN * minutes
    
    if damage < dmg_needed:
        fb.append(("warning",
                   f"Damage {damage:,} (< {dmg_needed:,}). Focus on "
                   f"isolating and eliminating key targets."))
    
    # --- Positioning Check ---
    if data.get("positioning_rating", "").lower() == "low":
        fb.append(("warning",
                   "Positioning flagged low. You're vulnerable to burst "
                   "damage—watch your positioning."))
    
    # --- Lancelot-Specific Checks ---
    
    # Death check (Assassins should have low deaths)
    if deaths > 4:
        fb.append(("critical",
                   f"{deaths} deaths is too high. Use Lancelot's mobility "
                   f"to engage and disengage safely."))
    
    # Damage ratio check
    damage_taken = data.get("damage_taken", 1)
    if damage_taken > 0 and damage > 0:
        damage_ratio = damage / damage_taken
        if damage_ratio < 2.0:
            fb.append(("warning",
                       f"Damage ratio {damage_ratio:.1f}. Use hit-and-run "
                       f"tactics to minimize damage taken."))
    
    # Ultimate usage
    if data.get("ult_usage", "").lower() == "low":
        fb.append(("warning",
                   "Low ultimate usage. Lancelot's ult provides immunity—"
                   "use it to engage or escape."))
    
    # Return formatted feedback
    # For now, return severity + message combined
    return [f"{severity}: {msg}" if severity != "info" else msg 
            for severity, msg in fb] 