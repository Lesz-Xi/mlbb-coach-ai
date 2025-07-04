import json
from coach import generate_feedback
from core.data_collector import DataCollector
from core.mental_coach import MentalCoach


def main():
    """
    Main function to run the coaching analysis.
    """
    # Instantiate the data collector
    collector = DataCollector()

    # Load player history for the Mental Coach
    try:
        with open("data/player_history.json", "r") as f:
            player_data = json.load(f)
        history = player_data.get("match_history", [])
        goal = player_data.get("player_defined_goal", "general_improvement")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Could not load player history: {e}")
        history, goal = [], "general_improvement"

    # Instantiate the mental coach with the player's data
    mental_coach = MentalCoach(player_history=history, player_goal=goal)

    # Use the collector to load and validate data from the JSON file
    try:
        matches = collector.from_json_upload("data/sample_match.json")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    # Process each match and print the feedback
    for i, match_data in enumerate(matches):
        # 1. Get statistical feedback
        stats_feedback = generate_feedback(match_data, include_severity=True)
        hero = match_data.get("hero", "Unknown")

        print(f"🧠 Match {i + 1} Coaching Report (Hero: {hero.title()})")
        for severity, line in stats_feedback:
            print(f"- {severity.upper()}: {line}")
        
        print("-" * 20)  # Separator

        # 2. Get mental boost feedback
        mental_boost = mental_coach.get_mental_boost(match_data)
        print(f"💡 Mental Boost: {mental_boost}")
        print()  # Add a blank line for readability

    # After processing all matches, generate a progress journal summary.
    print("\n" + "="*40)
    print("📖 Player Progress Journal")
    print("="*40)
    journal_entry = mental_coach.get_progress_journal()
    print(journal_entry)
    print("="*40)


if __name__ == "__main__":
    main()