"""
Generic coaching rules for unknown heroes.
Provides general MLBB advice when hero-specific coaching isn't available.
"""


def evaluate(match_data, minutes=None):
    """
    Provides generic coaching feedback for unknown heroes.
    
    Args:
        match_data: Dictionary containing match statistics
        minutes: Match duration in minutes
        
    Returns:
        List of (severity, message) tuples
    """
    feedback = []
    
    # Extract key stats
    kills = match_data.get('kills', 0)
    deaths = match_data.get('deaths', 1)
    assists = match_data.get('assists', 0)
    gold = match_data.get('gold', 0)
    hero_damage = match_data.get('hero_damage', 0)
    turret_damage = match_data.get('turret_damage', 0)
    
    duration = minutes or match_data.get('match_duration', 10)
    kda = (kills + assists) / max(deaths, 1)
    gpm = gold / max(duration, 1) if duration > 0 else 0
    
    # Hero identification issue
    feedback.append((
        "warning", 
        "Hero identification failed. Providing general coaching advice."
    ))
    
    # KDA Analysis
    if kda >= 3.0:
        feedback.append((
            "info", 
            f"Excellent KDA ratio of {kda:.1f}! You're making smart plays and staying alive."
        ))
    elif kda >= 2.0:
        feedback.append((
            "info", 
            f"Good KDA ratio of {kda:.1f}. Room for improvement but solid performance."
        ))
    elif kda >= 1.0:
        feedback.append((
            "warning", 
            f"Average KDA of {kda:.1f}. Focus on playing safer and securing more kills/assists."
        ))
    else:
        feedback.append((
            "critical", 
            f"Low KDA of {kda:.1f}. You're dying too much - prioritize positioning and map awareness."
        ))
    
    # Death analysis
    if deaths > 8:
        feedback.append((
            "critical", 
            f"Too many deaths ({deaths}). Focus on map awareness and positioning."
        ))
    elif deaths > 5:
        feedback.append((
            "warning", 
            f"High death count ({deaths}). Be more cautious in team fights."
        ))
    elif deaths <= 2:
        feedback.append((
            "info", 
            f"Great death management ({deaths} deaths). Keep up the safe play!"
        ))
    
    # Gold efficiency
    if gpm >= 600:
        feedback.append((
            "info", 
            f"Excellent gold efficiency ({gpm:.0f} GPM). You're farming effectively."
        ))
    elif gpm >= 400:
        feedback.append((
            "info", 
            f"Good gold income ({gpm:.0f} GPM). Consider improving last-hitting."
        ))
    else:
        feedback.append((
            "warning", 
            f"Low gold income ({gpm:.0f} GPM). Focus more on farming and objectives."
        ))
    
    # Damage analysis
    if hero_damage > 0:
        if hero_damage >= 50000:
            feedback.append((
                "info", 
                f"High damage output ({hero_damage:,}). You're contributing well to team fights."
            ))
        elif hero_damage >= 20000:
            feedback.append((
                "info", 
                f"Decent damage output ({hero_damage:,}). Consider being more aggressive."
            ))
        else:
            feedback.append((
                "warning", 
                f"Low damage output ({hero_damage:,}). Focus on dealing more damage in fights."
            ))
    
    # Objective damage
    if turret_damage > 0:
        if turret_damage >= 10000:
            feedback.append((
                "info", 
                f"Good objective focus ({turret_damage:,} turret damage)."
            ))
        else:
            feedback.append((
                "warning", 
                "Low turret damage. Help your team push objectives."
            ))
    
    # General advice based on performance
    if kda >= 2.5 and gpm >= 500:
        feedback.append((
            "info", 
            "Strong overall performance! Keep focusing on objectives and team coordination."
        ))
    elif deaths > 6 or gpm < 300:
        feedback.append((
            "critical", 
            "Focus on the fundamentals: safer positioning, better farming, and map awareness."
        ))
    
    # Always provide constructive advice
    feedback.append((
        "info", 
        "For better analysis, please ensure hero names are clearly visible in screenshots, "
        "or use the manual hero override feature."
    ))
    
    return feedback