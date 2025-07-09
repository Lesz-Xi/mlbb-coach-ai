import json
import os
import shutil
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# Local application imports
# Assumes the app is run from the 'skillshift-ai' directory.
from coach import generate_feedback
from core.data_collector import DataCollector
from core.enhanced_data_collector import enhanced_data_collector
from core.mental_coach import MentalCoach
from core.schemas import AnyMatch
from core.video_reader import VideoReader
from core.session_manager import session_manager


app = FastAPI(
    title="MLBB Coach AI",
    description="An AI-powered coaching tool for Mobile Legends: Bang Bang.",
    version="0.1.0",
)

# Add CORS middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001",
        "http://localhost:5173",  # Vite default
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001", 
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    """
    Root endpoint that returns a welcome message.
    """
    return {"message": "Welcome to the MLBB Coach AI API!"}


@app.post("/analyze-screenshot/")
async def analyze_screenshot(
    file: UploadFile = File(...),
    ign: str = "Lesz XVII"
):
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
        # The collector returns a dict with keys:
        # 'data', 'confidence', and 'warnings'
        parsed_result = data_collector.from_screenshot(
            ign=ign,
            image_path=temp_file_path
        )

        # Extract the actual match data for feedback generation
        match_data_dict = parsed_result.get("data", {})

        if not match_data_dict:
            raise HTTPException(
                status_code=400,
                detail=f"Could not parse valid match data. "
                       f"Warnings: {parsed_result.get('warnings', [])}"
            )

        # --- Feedback Generation ---
        statistical_feedback = generate_feedback(
            match_data_dict,
            include_severity=True
        )

        # For mental coach, we need to load history
        history_path = os.path.join("data", "player_history.json")
        try:
            with open(history_path, 'r') as f:
                player_data = json.load(f)
                history = player_data.get("match_history", [])
                goal = player_data.get(
                    "player_defined_goal",
                    "general_improvement"
                )
        except (FileNotFoundError, json.JSONDecodeError):
            history, goal = [], "general_improvement"


        mental_coach = MentalCoach(
            player_history=history,
            player_goal=goal
        )
        mental_feedback = mental_coach.get_mental_boost(match_data_dict)
        
        # Return all the information: feedback, parsed data, and debug info
        return {
            "statistical_feedback": statistical_feedback,
            "mental_feedback": mental_feedback,
            "parsed_data": match_data_dict,
            "confidence_scores": parsed_result.get("confidence", {}),
            "parsing_warnings": parsed_result.get("warnings", [])
        }

    except Exception as e:
        # Broad exception to catch issues during OCR or feedback generation
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # --- Cleanup ---
        # Ensure the temporary file is deleted
        os.unlink(temp_file_path)


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    ign: str = "Lesz XVII"
):
    """
    Frontend-compatible endpoint that analyzes a screenshot and returns coaching feedback.
    Returns response in format expected by React frontend.
    """
    # Use a temporary file to save the upload
    try:
        with NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
    finally:
        file.file.close()

    try:
        # --- OCR and Parsing ---
        data_collector = DataCollector()
        parsed_result = data_collector.from_screenshot(
            ign=ign,
            image_path=temp_file_path
        )
        match_data_dict = parsed_result.get("data", {})

        if not match_data_dict:
            raise HTTPException(
                status_code=400,
                detail=f"Could not parse valid match data. Warnings: {parsed_result.get('warnings', [])}"
            )

        # --- Feedback Generation ---
        statistical_feedback = generate_feedback(
            match_data_dict,
            include_severity=True
        )

        # Load player history for mental coach
        history_path = os.path.join("data", "player_history.json")
        try:
            with open(history_path, 'r') as f:
                player_data = json.load(f)
                history = player_data.get("match_history", [])
                goal = player_data.get(
                    "player_defined_goal",
                    "general_improvement"
                )
        except (FileNotFoundError, json.JSONDecodeError):
            history, goal = [], "general_improvement"

        mental_coach = MentalCoach(player_history=history, player_goal=goal)
        mental_boost = mental_coach.get_mental_boost(match_data_dict)
        
        # Format response for frontend compatibility
        # Transform statistical_feedback to match frontend expectations
        feedback_items = []
        for item in statistical_feedback:
            if isinstance(item, tuple) and len(item) == 2:
                # New format: (severity, message)
                severity, message = item
                feedback_items.append({
                    "type": severity,
                    "message": message,
                    "category": "Performance"
                })
            elif isinstance(item, dict):
                # Dict format with severity/feedback keys
                feedback_items.append({
                    "type": item.get("severity", "info"),
                    "message": item.get("feedback", ""),
                    "category": item.get("category", "General")
                })
            else:
                # Fallback for string messages
                feedback_items.append({
                    "type": "info",
                    "message": str(item),
                    "category": "General"
                })
        
        # Determine overall rating based on feedback severity
        critical_count = len([f for f in feedback_items if f["type"] == "critical"])
        warning_count = len([f for f in feedback_items if f["type"] == "warning"])
        
        if critical_count > 2:
            overall_rating = "Poor"
        elif critical_count > 0 or warning_count > 3:
            overall_rating = "Average"
        elif warning_count > 0:
            overall_rating = "Good"
        else:
            overall_rating = "Excellent"

        return {
            "feedback": feedback_items,
            "mental_boost": mental_boost,
            "overall_rating": overall_rating
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # --- Cleanup ---
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


@app.post("/analyze-video/")
async def analyze_video(
    file: UploadFile = File(...),
    ign: str = "Lesz XVII",
    hero_override: str = ""
):
    """
    Analyzes a gameplay video and extracts match statistics from score screens.
    
    Args:
        file: Uploaded video file
        ign: Player's in-game name to look for (default: "Lesz XVII")
        hero_override: Manually specified hero name (optional)
    """
    filename = str(file.filename)
    if not filename.lower().endswith(
        ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    ):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported video format. Please upload MP4, AVI, MOV, MKV, FLV, or WMV files."
        )
    
    # Use a temporary file to save the upload
    try:
        with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
    finally:
        file.file.close()
    
    try:
        # Initialize video reader
        video_reader = VideoReader()
        
        # Get known IGNs for validation
        known_igns = [ign, "Player1", "Enemy1", "Enemy2", "Enemy3", "Enemy4"]
        
        # Analyze video
        result = video_reader.analyze_video(
            video_path=temp_file_path,
            ign=ign,
            known_igns=known_igns,
            hero_override=hero_override
        )
        
        if not result.success:
            raise HTTPException(
                status_code=400,
                detail=f"Video analysis failed. Warnings: {result.warnings}"
            )
        
        # Generate coaching feedback
        match_data = result.match_data
        statistical_feedback = generate_feedback(
            match_data,
            include_severity=True
        )
        
        # Load player history for mental coach
        history_path = os.path.join("data", "player_history.json")
        try:
            with open(history_path, 'r') as f:
                player_data = json.load(f)
                history = player_data.get("match_history", [])
                goal = player_data.get(
                    "player_defined_goal",
                    "general_improvement"
                )
        except (FileNotFoundError, json.JSONDecodeError):
            history, goal = [], "general_improvement"
        
        mental_coach = MentalCoach(player_history=history, player_goal=goal)
        mental_feedback = mental_coach.get_mental_boost(match_data)
        
        # Get video info
        video_info = video_reader.get_video_info(temp_file_path)
        
        return {
            "statistical_feedback": statistical_feedback,
            "mental_feedback": mental_feedback,
            "match_data": match_data,
            "video_analysis": {
                "success": result.success,
                "confidence_score": result.confidence_score,
                "frame_count": result.frame_count,
                "processed_frames": result.processed_frames,
                "processing_time": result.processing_time,
                "warnings": result.warnings
            },
            "video_info": video_info
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis error: {str(e)}")
    finally:
        # Cleanup temporary file
        try:
            os.unlink(temp_file_path)
        except Exception:
            pass


@app.post("/api/analyze-enhanced/")
async def analyze_enhanced(
    file: UploadFile = File(...),
    ign: str = "Lesz XVII",
    session_id: str = None,
    hero_override: str = ""
):
    """
    Enhanced screenshot analysis with session management and improved hero detection.
    Supports multi-screenshot processing for better accuracy.
    """
    # Use a temporary file to save the upload
    try:
        with NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
    finally:
        file.file.close()

    try:
        # Use enhanced data collector with session management
        result = enhanced_data_collector.analyze_screenshot_with_session(
            image_path=temp_file_path,
            ign=ign,
            session_id=session_id,
            hero_override=hero_override if hero_override else None
        )
        
        match_data = result.get("data", {})
        
        if not match_data or not any(v for k, v in match_data.items() if k != "_session_info"):
            raise HTTPException(
                status_code=400,
                detail=f"Could not parse valid match data. Warnings: {result.get('warnings', [])}"
            )

        # Generate coaching feedback
        statistical_feedback = generate_feedback(match_data, include_severity=True)

        # Load player history for mental coach
        history_path = os.path.join("data", "player_history.json")
        try:
            with open(history_path, 'r') as f:
                player_data = json.load(f)
                history = player_data.get("match_history", [])
                goal = player_data.get("player_defined_goal", "general_improvement")
        except (FileNotFoundError, json.JSONDecodeError):
            history, goal = [], "general_improvement"

        mental_coach = MentalCoach(player_history=history, player_goal=goal)
        mental_boost = mental_coach.get_mental_boost(match_data)
        
        # Format response for frontend compatibility
        feedback_items = []
        for item in statistical_feedback:
            if isinstance(item, tuple) and len(item) == 2:
                severity, message = item
                feedback_items.append({
                    "type": severity,
                    "message": message,
                    "category": "Performance"
                })
            elif isinstance(item, dict):
                feedback_items.append({
                    "type": item.get("severity", "info"),
                    "message": item.get("feedback", ""),
                    "category": item.get("category", "General")
                })
            else:
                feedback_items.append({
                    "type": "info",
                    "message": str(item),
                    "category": "General"
                })
        
        # Determine overall rating
        critical_count = len([f for f in feedback_items if f["type"] == "critical"])
        warning_count = len([f for f in feedback_items if f["type"] == "warning"])
        
        if critical_count > 2:
            overall_rating = "Poor"
        elif critical_count > 0 or warning_count > 3:
            overall_rating = "Average"
        elif warning_count > 0:
            overall_rating = "Good"
        else:
            overall_rating = "Excellent"

        return {
            "feedback": feedback_items,
            "mental_boost": mental_boost,
            "overall_rating": overall_rating,
            "session_info": {
                "session_id": result.get("session_id"),
                "screenshot_type": result.get("screenshot_type"),
                "type_confidence": result.get("type_confidence", 0),
                "session_complete": result.get("session_complete", False),
                "screenshot_count": result.get("debug_info", {}).get("screenshot_count", 1)
            },
            "debug_info": result.get("debug_info", {}),
            "warnings": result.get("warnings", [])
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced analysis error: {str(e)}")
    finally:
        os.unlink(temp_file_path)


@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """Get information about an analysis session."""
    session_info = enhanced_data_collector.get_session_info(session_id)
    if not session_info:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_info


@app.get("/api/sessions/")
async def list_active_sessions():
    """List all active analysis sessions."""
    active_sessions = session_manager.list_active_sessions()
    return {"active_sessions": active_sessions, "count": len(active_sessions)}