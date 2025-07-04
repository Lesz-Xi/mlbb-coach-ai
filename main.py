import json
from coach import generate_feedback

with open("data/sample_match.json", "r") as f:
    matches = json.load(f)

for i, match_data in enumerate(matches):
    feedback = generate_feedback(match_data)
    hero = match_data.get("hero", "Unknown")

    print(f"🧠 Match {i + 1} Coaching Report (Hero: {hero.title()})")
    for line in feedback:
        print(f"- {line}")
    print()  # Add a blank line for readability