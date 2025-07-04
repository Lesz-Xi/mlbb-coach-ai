def validate_data(data):
    # Defines the fields that are absolutely required for the program to run.
    required_fields = [
        "hero", "kills", "deaths", "assists", "gold_per_min",
        "damage_taken", "positioning_rating", "ult_usage",
        "hero_damage", "turret_damage", "teamfight_participation"
    ]

    # Loop through each required field.
    for field in required_fields:
        # Check if the field is missing from the provided data.
        if field not in data:
            # If a field is missing, return False immediately.
            return False

    # If all fields are present, return True.
    return True