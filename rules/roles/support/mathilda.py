"""
Coaching rules for Mathilda - The Swift Support
"""


def evaluate(match_data, minutes=None):
    """
    Evaluates Mathilda's performance and provides coaching feedback.
    
    Mathilda is a support/tank hybrid who excels at:
    - Team mobility and engagement
    - Crowd control and protection
    - Roaming and map control
    - Enabling team fights with her ultimate
    """
    feedback = []
    
    # Extract key stats
    kills = match_data.get('kills', 0)
    deaths = match_data.get('deaths', 1)
    assists = match_data.get('assists', 0)
    hero_damage = match_data.get('hero_damage', 0)
    turret_damage = match_data.get('turret_damage', 0)
    damage_taken = match_data.get('damage_taken', 0)
    gold = match_data.get('gold', 0)
    teamfight_participation = match_data.get('teamfight_participation', 0)
    
    duration = minutes or match_data.get('match_duration', 10)
    kda = (kills + assists) / max(deaths, 1)
    gpm = gold / max(duration, 1) if duration > 0 else 0
    
    # Support Role Analysis - Assists are crucial
    if assists >= 20:
        feedback.append((
            "info", 
            f"Exceptional team support ({assists} assists)! You're enabling your team perfectly."
        ))
    elif assists >= 15:
        feedback.append((
            "info", 
            f"Great assist count ({assists}). You're fulfilling Mathilda's support role well."
        ))
    elif assists >= 10:
        feedback.append((
            "info", 
            f"Good team participation ({assists} assists). Keep roaming and helping teammates."
        ))
    else:
        feedback.append((
            "warning", 
            f"Low assists ({assists}). As Mathilda, you should be in every team fight providing support."
        ))
    
    # Death Analysis for Support
    if deaths > 7:
        feedback.append((
            "critical", 
            f"Too many deaths ({deaths}) for a support. Focus on positioning behind your team and using your mobility wisely."
        ))
    elif deaths > 4:
        feedback.append((
            "warning", 
            f"High death count ({deaths}). Use Mathilda's dash and speed boost to escape dangerous situations."
        ))
    elif deaths <= 2:
        feedback.append((
            "info", 
            f"Excellent survivability ({deaths} deaths). You're mastering Mathilda's mobility and positioning!"
        ))
    
    # Damage Analysis (Mathilda can deal decent damage)
    if hero_damage >= 30000:
        feedback.append((
            "info", 
            f"Strong damage output ({hero_damage:,}) for a support! You're maximizing Mathilda's damage potential."
        ))
    elif hero_damage >= 20000:
        feedback.append((
            "info", 
            f"Good damage contribution ({hero_damage:,}). Keep using your skills aggressively in fights."
        ))
    elif hero_damage >= 12000:
        feedback.append((
            "warning", 
            f"Moderate damage ({hero_damage:,}). Try to be more aggressive with your skill combinations."
        ))
    else:
        feedback.append((
            "warning", 
            f"Low damage output ({hero_damage:,}). Mathilda can deal significant damage - work on your positioning and combos."
        ))
    
    # Damage Taken Analysis (Should tank some damage but not too much)
    if damage_taken >= 40000:
        feedback.append((
            "warning", 
            f"Very high damage taken ({damage_taken:,}). You might be overextending - let your tanks initiate first."
        ))
    elif damage_taken >= 25000:
        feedback.append((
            "info", 
            f"Good frontline presence ({damage_taken:,} damage taken). You're absorbing some damage for squishier teammates."
        ))
    elif damage_taken < 15000:
        feedback.append((
            "warning", 
            f"Low damage taken ({damage_taken:,}). You might be playing too passively - get closer to support your team."
        ))
    
    # KDA Analysis for Support
    if kda >= 3.0:
        feedback.append((
            "info", 
            f"Outstanding KDA ({kda:.1f}) for a support! Perfect balance of aggression and safety."
        ))
    elif kda >= 2.0:
        feedback.append((
            "info", 
            f"Excellent KDA ({kda:.1f}). Great support performance."
        ))
    elif kda >= 1.5:
        feedback.append((
            "info", 
            f"Good KDA ({kda:.1f}) for a support role."
        ))
    else:
        feedback.append((
            "warning", 
            f"Low KDA ({kda:.1f}). Focus on staying alive to provide continuous support."
        ))
    
    # Teamfight Participation
    if teamfight_participation >= 80:
        feedback.append((
            "info", 
            f"Excellent teamfight participation ({teamfight_participation}%)! You're always there when needed."
        ))
    elif teamfight_participation >= 60:
        feedback.append((
            "info", 
            f"Good teamfight presence ({teamfight_participation}%)."
        ))
    elif teamfight_participation < 50:
        feedback.append((
            "warning", 
            f"Low teamfight participation ({teamfight_participation}%). Mathilda should be in most team engagements."
        ))
    
    # Gold Efficiency for Support
    if gpm >= 350:
        feedback.append((
            "info", 
            f"Good gold income ({gpm:.0f} GPM) for a support."
        ))
    elif gpm < 250:
        feedback.append((
            "warning", 
            f"Low GPM ({gpm:.0f}). Consider roaming more efficiently and securing assists."
        ))
    
    # Kills Analysis (Support shouldn't take too many kills)
    if kills >= 8:
        feedback.append((
            "warning", 
            f"High kill count ({kills}). As a support, try to let your carries secure more kills."
        ))
    elif kills >= 4:
        feedback.append((
            "info", 
            f"Decent kill participation ({kills}). Good balance for a support."
        ))
    
    # Mathilda-specific advice
    feedback.append((
        "info", 
        "Mathilda Tips: Use your S1 to scout and escape, combo S2+S1 for engage/disengage, "
        "and time your ultimate perfectly to turn team fights around!"
    ))
    
    # Specific Mathilda gameplay advice
    if assists < 10:
        feedback.append((
            "critical", 
            "Focus on roaming more! Mathilda excels at rotating between lanes and helping teammates. "
            "Use your mobility to be everywhere your team needs support."
        ))
    
    # Performance-based advice
    if kda >= 2.5 and assists >= 15:
        feedback.append((
            "info", 
            "Perfect Mathilda performance! You're providing excellent support while staying alive."
        ))
    elif deaths > 5 and assists < 12:
        feedback.append((
            "critical", 
            "Work on your positioning and map awareness. Stay behind your frontline and use your "
            "mobility skills to help teammates rather than engaging directly."
        ))
    
    return feedback