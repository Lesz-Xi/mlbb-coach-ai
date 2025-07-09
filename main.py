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

    # --- New Screenshot-based Workflow ---
    try:
        ign = "Lesz XVII"  # This can now be dynamically set or detected
        hero_override = "hayabusa"  # Manually specify the hero here
        kda_image_path = "Screenshot-Test/IMG_1523.PNG"
        
        print(f"🔬 Analyzing screenshot for player: {ign} (Hero: {hero_override})")
        
        # Add known IGNs for better validation
        known_igns = ["Lesz XVII", "Player1", "Enemy1", "Enemy2", "Enemy3", "Enemy4"]
        
        result = collector.from_screenshot(
            ign, kda_image_path, hero_override=hero_override, known_igns=known_igns
        )
        
        if result and result.get("data"):
            match_data = result["data"]
            stats_feedback = generate_feedback(match_data, include_severity=True)
            hero = match_data.get("hero", "Unknown")

            print(f"\n🧠 Match Coaching Report (Hero: {hero.title()})")
            for severity, line in stats_feedback:
                print(f"- {severity.upper()}: {line}")
            
            print("-" * 20)

            mental_boost = mental_coach.get_mental_boost(match_data)
            print(f"💡 Mental Boost: {mental_boost}")
            print()

            if result.get("warnings"):
                print("\n⚠️  Parsing Warnings:")
                for warning in result["warnings"]:
                    print(f"- {warning}")
        else:
            print("\n❌ Analysis failed. No data was extracted.")
            if result.get("warnings"):
                print("\n⚠️  Parsing Warnings:")
                for warning in result["warnings"]:
                    print(f"- {warning}")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    # --- Old JSON-based workflow (commented out for now) ---
    # try:
    #     matches = collector.from_json_upload("data/sample_match.json")
    # except (FileNotFoundError, ValueError) as e:
    #     print(f"Error: {e}")
    #     return
    # # Process each match and print the feedback
    # for i, match_data in enumerate(matches):
    #     # 1. Get statistical feedback
    #     stats_feedback = generate_feedback(match_data, include_severity=True)
    #     hero = match_data.get("hero", "Unknown")
    #     print(f"🧠 Match {i + 1} Coaching Report (Hero: {hero.title()})")
    #     for severity, line in stats_feedback:
    #         print(f"- {severity.upper()}: {line}")
    #     print("-" * 20)  # Separator
    #     # 2. Get mental boost feedback
    #     mental_boost = mental_coach.get_mental_boost(match_data)
    #     print(f"💡 Mental Boost: {mental_boost}")
    #     print()  # Add a blank line for readability

    # After processing all matches, generate a progress journal summary.
    print("\n" + "="*40)
    print("📖 Player Progress Journal")
    print("="*40)
    journal_entry = mental_coach.get_progress_journal()
    print(journal_entry)
    print("="*40)


if __name__ == "__main__":
    main()