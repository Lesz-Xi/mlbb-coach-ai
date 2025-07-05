import json
import os
from coach import generate_feedback
from core.data_collector import DataCollector
from core.mental_coach import MentalCoach
from pprint import pprint


def test_screenshot(file_path: str):
    """
    Runs a full end-to-end test for a single screenshot.
    """
    print("="*50)
    print(f"🚀 TESTING SCREENSHOT: {file_path}")
    print("="*50)

    # 1. --- OCR and Parsing ---
    print("\n[1/3] 🔍 Parsing screenshot with DataCollector...")
    collector = DataCollector()
    parsed_result = collector.from_screenshot(file_path)
    
    print("\n--- Raw Parser Output ---")
    pprint(parsed_result)
    print("-------------------------\n")

    match_data = parsed_result.get("data")
    if not match_data:
        print("❌ ERROR: Could not parse valid match data. Aborting test for this file.")
        return

    # 2. --- Statistical Feedback ---
    print("[2/3] 🧠 Generating statistical feedback...")
    stats_feedback = generate_feedback(match_data, include_severity=True)
    
    print("\n--- Coaching Report ---")
    for severity, line in stats_feedback:
        print(f"- {severity.upper()}: {line}")
    print("-----------------------\n")

    # 3. --- Mental Coaching ---
    print("[3/3] 💡 Generating mental coaching feedback...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        history_path = os.path.join(script_dir, "data", "player_history.json")
        with open(history_path, "r") as f:
            player_data = json.load(f)
        history = player_data.get("match_history", [])
        goal = player_data.get("player_defined_goal", "general_improvement")
    except (FileNotFoundError, json.JSONDecodeError):
        history, goal = [], "general_improvement"

    mental_coach = MentalCoach(player_history=history, player_goal=goal)
    mental_boost = mental_coach.get_mental_boost(match_data)
    
    print("\n--- Mental Boost ---")
    print(mental_boost)
    print("--------------------\n")


if __name__ == "__main__":
    # This script is in 'mlbb-coach-ai', and screenshots are in a sibling directory.
    # To make paths robust, construct absolute paths from this script's location.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    screenshot_paths = [
        os.path.join(project_root, "Screenshot-Test", "IMG_1523.PNG"),
        os.path.join(project_root, "Screenshot-Test", "IMG_1524.PNG"),
    ]

    for path in screenshot_paths:
        test_screenshot(path)

    print("✅ All tests complete.") 