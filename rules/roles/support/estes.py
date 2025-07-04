from typing import List, Dict, Any


def evaluate(data: Dict[str, Any], minutes: int = None) -> List[str]:
    """
    Evaluate Estes's performance with dynamic thresholds.
    
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
    KDA_LOW, KDA_HIGH = 3.5, 5.0
    GPM_MIN = 500  # Base GPM for support
    PARTICIPATION_MIN = 60  # Support teamfight participation (critical)
    
    # --- KDA Evaluation ---
    kills = data.get("kills", 0)
    deaths = max(1, data.get("deaths", 1))
    assists = data.get("assists", 0)
    kda = (kills + assists) / deaths
    
    if kda < KDA_LOW:
        fb.append(("warning",
                   f"KDA {kda:.1f} (< {KDA_LOW}). Focus on maximizing "
                   f"assists by healing teammates at crucial moments."))
    elif kda > KDA_HIGH:
        fb.append(("success",
                   f"Fantastic KDA {kda:.1f} as support! Your high assist "
                   f"count shows excellent timing."))
    else:
        fb.append(("info",
                   f"Good KDA ({kda:.1f}). Keep supporting your team in "
                   f"key moments."))
    
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
                   f"GPM {gpm} (< {gpm_threshold:.0f}). Ensure you're "
                   f"getting assist gold from ganks and teamfights."))
    
    # --- Teamfight Participation ---
    tfp = data.get("teamfight_participation", 0)
    if tfp < PARTICIPATION_MIN:
        fb.append(("critical",
                   f"Low fight presence ({tfp}% < {PARTICIPATION_MIN}%). "
                   f"You need to be with your team for every engagement."))
    
    # --- Positioning Check ---
    if data.get("positioning_rating", "").lower() == "low":
        fb.append(("warning",
                   "Positioning flagged low. Stay safe while keeping "
                   "teammates in heal range."))
    
    # --- Estes-Specific Checks ---
    
    # Death check (Support should have low deaths)
    if deaths > 5:
        fb.append(("critical",
                   f"{deaths} deaths is too high. Estes needs to stay "
                   f"alive to provide sustained healing."))
    
    # Assist ratio check (Supports should have high assist participation)
    total_kills_assists = kills + assists
    if total_kills_assists < 15:
        fb.append(("warning",
                   f"Low kill participation ({total_kills_assists}). "
                   f"Be present for more team fights and ganks."))
    
    # Ultimate usage
    if data.get("ult_usage", "").lower() == "low":
        fb.append(("warning",
                   "Low ultimate usage. Estes's ult can turn teamfights—"
                   "use it proactively."))
    
    # Return formatted feedback
    # For now, return severity + message combined
    return [f"{severity}: {msg}" if severity != "info" else msg 
            for severity, msg in fb] 