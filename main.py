from coach import generate_feedback
from core.data_collector import DataCollector


def main():
    """
    Main function to run the coaching analysis.
    """
    # Instantiate the data collector
    collector = DataCollector()

    # Use the collector to load and validate data from the JSON file
    try:
        matches = collector.from_json_upload("data/sample_match.json")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    # Process each match and print the feedback
    for i, match_data in enumerate(matches):
        feedback = generate_feedback(match_data, include_severity=True)
        hero = match_data.get("hero", "Unknown")

        print(f"🧠 Match {i + 1} Coaching Report (Hero: {hero.title()})")
        # Print feedback with severity icons from config
        for severity, line in feedback:
            # This is a placeholder for where you'd get icons
            # For now, we just show the severity level
            print(f"- {severity.upper()}: {line}")
        print()  # Add a blank line for readability


if __name__ == "__main__":
    main()