"""
Coaching rules for Fredrinn - The Noble Warrior Tank
"""


def evaluate(match_data, minutes=None):
    """
    Evaluates Fredrinn's performance and provides coaching feedback.
    
    Fredrinn is a tank/fighter hybrid who excels at:
    - Crowd control and team engagement
    - Damage dealing while maintaining tankiness
    - Area denial and battlefield control
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
    
    duration = minutes or match_data.get('match_duration', 10)
    kda = (kills + assists) / max(deaths, 1)
    gpm = gold / max(duration, 1) if duration > 0 else 0
    
    # Tank Performance Analysis
    if deaths > 8:
        feedback.append((
            "critical", 
            f"Too many deaths ({deaths}) for a tank. Focus on positioning and timing your engages."
        ))
    elif deaths > 5:
        feedback.append((
            "warning", 
            f"High death count ({deaths}). As Fredrinn, use your shield and mobility to stay alive longer."
        ))
    elif deaths <= 3:
        feedback.append((
            "info", 
            f"Excellent survivability ({deaths} deaths). You're mastering Fredrinn's defensive capabilities!"
        ))
    
    # Damage Analysis (Fredrinn should deal respectable damage for a tank)
    if hero_damage >= 40000:
        feedback.append((
            "info", 
            f"Outstanding damage output ({hero_damage:,}) for a tank! You're maximizing Fredrinn's hybrid potential."
        ))
    elif hero_damage >= 25000:
        feedback.append((
            "info", 
            f"Good damage dealing ({hero_damage:,}). Keep using your skills aggressively in team fights."
        ))
    elif hero_damage >= 15000:
        feedback.append((
            "warning", 
            f"Moderate damage ({hero_damage:,}). Try to be more aggressive with your skill combinations."
        ))
    else:
        feedback.append((
            "critical", 
            f"Low damage output ({hero_damage:,}). Fredrinn should deal significant damage - work on your combos."
        ))
    
    # Damage Taken Analysis
    if damage_taken >= 50000:
        feedback.append((
            "info", 
            f"Excellent frontline presence ({damage_taken:,} damage taken). You're absorbing damage for your team!"
        ))
    elif damage_taken >= 30000:
        feedback.append((
            "info", 
            f"Good tanking ({damage_taken:,} damage taken). Keep positioning yourself between enemies and carries."
        ))
    else:
        feedback.append((
            "warning", 
            f"Low damage taken ({damage_taken:,}). As a tank, you should absorb more damage for your team."
        ))
    
    # KDA for tank standards
    if kda >= 2.5:
        feedback.append((
            "info", 
            f"Excellent KDA ({kda:.1f}) for a tank! You're getting kills while staying alive."
        ))
    elif kda >= 1.5:
        feedback.append((
            "info", 
            f"Good KDA ({kda:.1f}). Solid performance for a frontline fighter."
        ))
    else:
        feedback.append((
            "warning", 
            f"Low KDA ({kda:.1f}). Focus on timing your engages and disengages better."
        ))
    
    # Assists (very important for tanks)
    if assists >= 15:
        feedback.append((
            "info", 
            f"Outstanding assist count ({assists})! You're enabling your team effectively."
        ))
    elif assists >= 10:
        feedback.append((
            "info", 
            f"Good team participation ({assists} assists)."
        ))
    elif assists < 6:
        feedback.append((
            "warning", 
            f"Low assists ({assists}). As Fredrinn, you should be in every team fight."
        ))
    
    # Gold efficiency for tank
    if gpm >= 400:
        feedback.append((
            "info", 
            f"Good gold income ({gpm:.0f} GPM) for a tank."
        ))
    elif gpm < 250:
        feedback.append((
            "warning", 
            f"Low GPM ({gpm:.0f}). Consider roaming more efficiently and assisting in kills."
        ))
    
    # Fredrinn-specific advice
    feedback.append((
        "info", 
        "Fredrinn Tips: Use your passive shield effectively, combo your skills for maximum damage, "
        "and don't forget to use your ultimate for team fight initiation or escape!"
    ))
    
    # Performance-based advice
    if kda >= 2.0 and damage_taken >= 30000:
        feedback.append((
            "info", 
            "Excellent Fredrinn performance! You're balancing damage dealing with tanking perfectly."
        ))
    elif deaths > 6:
        feedback.append((
            "critical", 
            "Work on your positioning. Use Fredrinn's mobility and shield to engage safely and escape when needed."
        ))
    
    return feedback