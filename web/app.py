import json
import os
import shutil
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, HTTPException, UploadFile, File

# Local application imports
# Assumes the app is run from the 'mlbb-coach-ai' directory.
from coach import generate_feedback
from core.data_collector import DataCollector
from core.mental_coach import MentalCoach
from core.schemas import AnyMatch


app = FastAPI(
    title="MLBB Coach AI",
    description="An AI-powered coaching tool for Mobile Legends: Bang Bang.",
    version="0.1.0",
)


@app.get("/")
def read_root():
    """
    Root endpoint that returns a welcome message.
    """
    return {"message": "Welcome to the MLBB Coach AI API!"}


@app.post("/analyze-screenshot/")
async def analyze_screenshot(file: UploadFile = File(...)):
    """
    Analyzes a match from a screenshot and returns coaching feedback.
    """
    # Use a temporary file to save the upload
    try:
        # The `NamedTemporaryFile` creates a file that is automatically
        # deleted when the 'with' block is exited.
        with NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
    finally:
        file.file.close()

    try:
        # --- OCR and Parsing ---
        data_collector = DataCollector()
        match_data_dict = data_collector.from_screenshot(temp_file_path)

        if not match_data_dict:
            raise HTTPException(
                status_code=400,
                detail="Could not parse valid match data from the screenshot."
            )

        # --- Feedback Generation ---
        # Use the same logic as the /analyze endpoint
        statistical_feedback = generate_feedback(match_data_dict)

        # For mental coach, we need to load history
        history_path = os.path.join("data", "player_history.json")
        with open(history_path, 'r') as f:
            history = json.load(f)

        mental_coach = MentalCoach(
            history=history,
            goal="improve_early_game"  # Goal could be another parameter later
        )
        mental_feedback = mental_coach.get_mental_boost()
        
        return {
            "statistical_feedback": statistical_feedback,
            "mental_feedback": mental_feedback,
            "parsed_data": match_data_dict
        }

    except Exception as e:
        # Broad exception to catch issues during OCR or feedback generation
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # --- Cleanup ---
        # Ensure the temporary file is deleted
        os.unlink(temp_file_path)


@app.post("/analyze/")
def analyze_match(match_data: AnyMatch):
    """
    Analyzes a single match and returns coaching feedback.
    """
    try:
        # Convert Pydantic model back to dict for processing
        match_dict = match_data.model_dump()

        # 1. Get statistical feedback
        stats_feedback = generate_feedback(match_dict, include_severity=True)

        # 2. Get mental boost feedback
        history_path = os.path.join("data", "player_history.json")
        try:
            with open(history_path, "r") as f:
                player_data = json.load(f)
            history = player_data.get("match_history", [])
            goal = player_data.get(
                "player_defined_goal", "general_improvement"
            )
        except (FileNotFoundError, json.JSONDecodeError):
            history, goal = [], "general_improvement"

        mental_coach = MentalCoach(player_history=history, player_goal=goal)
        mental_boost = mental_coach.get_mental_boost(match_dict)

        return {
            "hero": match_data.hero,
            "statistical_feedback": stats_feedback,
            "mental_boost": mental_boost,
        }
    except Exception as e:
        # Catch-all for any other unexpected errors during processing
        raise HTTPException(status_code=500, detail=str(e))