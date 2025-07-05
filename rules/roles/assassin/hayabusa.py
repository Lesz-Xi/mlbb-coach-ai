# Hayabusa-specific evaluation logic
from typing import List, Dict, Any


def evaluate(data: Dict[str, Any], minutes: int = None) -> List[str]:
    """
    Evaluate Hayabusa's performance with dynamic thresholds.
    
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
    KDA_LOW, KDA_HIGH = 3.5, 5.5
    GPM_MIN = 700  # Base GPM for assassin (slightly lower than Lancelot)
    DMG_PER_MIN = 4200  # Hayabusa-specific damage expectation
    PARTICIPATION_MIN = 45  # Hayabusa teamfight participation
    
    # --- KDA Evaluation ---
    kills = data.get("kills", 0)
    actual_deaths = data.get("deaths", 0)
    deaths = max(1, actual_deaths)  # For calculation only
    assists = data.get("assists", 0)
    kda = (kills + assists) / deaths
    
    # Display raw KDA for clarity
    raw_kda = f"{kills}/{actual_deaths}/{assists}"
    
    if kda < KDA_LOW:
        fb.append(("warning",
                   f"KDA {raw_kda} (calculated: {kda:.1f} < {KDA_LOW}). Use shadow techniques to "
                   f"eliminate targets and escape with your ultimate."))
    elif kda > KDA_HIGH:
        if actual_deaths == 0:
            fb.append(("success",
                       f"Perfect KDA {raw_kda}! Flawless performance - your shadow mastery is "
                       f"creating fear in the enemy team."))
        else:
            fb.append(("success",
                       f"Outstanding KDA {raw_kda} (calculated: {kda:.1f})! Your shadow mastery is "
                       f"creating fear in the enemy team."))
    else:
        fb.append(("info",
                   f"Solid KDA {raw_kda} (calculated: {kda:.1f}). Keep using your mobility to "
                   f"outplay enemies and secure eliminations."))
    
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
                   f"GPM {gpm} (< {gpm_threshold:.0f}). Farm jungle efficiently "
                   f"and look for gank opportunities to snowball."))
    
    # --- Teamfight Participation ---
    tfp = data.get("teamfight_participation", 0)
    if tfp < PARTICIPATION_MIN:
        fb.append(("warning",
                   f"Low teamfight presence ({tfp}% < {PARTICIPATION_MIN}%). "
                   f"Use your ultimate to engage or clean up fights."))
    
    # --- Hero Damage (scaled by minutes) ---
    damage = data.get("hero_damage", 0)
    dmg_needed = DMG_PER_MIN * minutes
    
    if damage < dmg_needed:
        fb.append(("warning",
                   f"Damage {damage:,} (< {dmg_needed:,}). Focus on bursting "
                   f"down squishy targets with your shadow combo."))
    
    # --- Positioning Check ---
    if data.get("positioning_rating", "").lower() == "low":
        fb.append(("warning",
                   "Positioning flagged low. Stay in shadows and wait for "
                   "the right moment to strike."))
    
    # --- Hayabusa-Specific Checks ---
    
    # Death check (Assassins should have low deaths)
    if actual_deaths > 4:
        fb.append(("critical",
                   f"{actual_deaths} deaths is too high. Use Hayabusa's ultimate "
                   f"for both engagement and escape—master the timing."))
    
    # Damage ratio check
    damage_taken = data.get("damage_taken", 1)
    if damage_taken > 0 and damage > 0:
        damage_ratio = damage / damage_taken
        if damage_ratio < 1.8:
            fb.append(("warning",
                       f"Damage ratio {damage_ratio:.1f}. Use hit-and-run "
                       f"tactics with your shadow clones to minimize damage."))
    
    # Ultimate usage (crucial for Hayabusa)
    if data.get("ult_usage", "").lower() == "low":
        fb.append(("warning",
                   "Low ultimate usage. Hayabusa's ult is your main engage "
                   "and escape tool—use it wisely for picks."))
    
    # Jungle efficiency (Hayabusa is jungle-dependent)
    if kills + assists < 8 and minutes > 12:
        fb.append(("warning",
                   "Low kill participation for mid-game. Hayabusa needs "
                   "early kills to snowball—be more aggressive."))
    
    # Shadow combo effectiveness
    if damage > 0 and kills > 0:
        damage_per_kill = damage / (kills + 1)
        if damage_per_kill > 8000:
            fb.append(("info",
                       "High damage per kill suggests good burst combos. "
                       "Keep perfecting your shadow techniques."))
    
    # Survival tip for Hayabusa
    if actual_deaths > 2 and minutes < 15:
        fb.append(("warning",
                   "Dying too much in early game. Use your shadows to "
                   "scout and avoid unnecessary risks."))
    
    # Return formatted feedback as tuples
    return fb