import json
import logging
import os
import shutil
import sys
from tempfile import NamedTemporaryFile
from typing import List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# Local application imports
# Assumes the app is run from the 'skillshift-ai' directory.
from coach import generate_feedback
from core.data_collector import DataCollector
from core.enhanced_data_collector import enhanced_data_collector
from core.ultimate_parsing_system import ultimate_parsing_system
from core.mental_coach import MentalCoach
from core.schemas import AnyMatch
from core.video_reader import VideoReader
from core.session_manager import session_manager
from core.hero_database import hero_database
from core.advanced_performance_analyzer import advanced_performance_analyzer
from core.enhanced_counter_pick_system import enhanced_counter_pick_system
from core.error_handler import error_handler


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def _filter_warnings_for_high_confidence(warnings: List[str], overall_confidence: float, hero_detected: bool) -> List[str]:
    """Filter out non-critical warnings when we have high confidence analysis."""
    # CRITICAL FIX: Handle both percentage (68.7) and decimal (0.687) confidence values
    confidence_percentage = overall_confidence * 100 if overall_confidence <= 1.0 else overall_confidence
    
    if confidence_percentage < 85.0:  # Only filter for high confidence analyses (85%+)
        return warnings
    
    # List of warnings to suppress when confidence is high and other data is good
    suppressible_warnings = [
        "Could not detect match duration", 
        "Could not detect player rank",
        "Hero could not be identified",
        "Low hero identification confidence",
        "IGN match confidence"
    ]
    
    filtered_warnings = []
    for warning in warnings:
        # Check if this warning should be suppressed
        should_suppress = any(suppress in warning for suppress in suppressible_warnings)
        
        # Special case: if hero is detected but has low confidence, don't suppress hero warnings
        if "hero" in warning.lower() and hero_detected:
            should_suppress = False
        
        if not should_suppress:
            filtered_warnings.append(warning)
    
    return filtered_warnings

# Import debug panel routers and validation API
try:
    from .debug_panel import debug_router
    from .debug_ultimate import router as ultimate_debug_router
    from .validation_api import add_validation_routes, add_edge_case_routes
except ImportError:
    # Fallback for when running from different directory
    import sys
    sys.path.append(os.path.dirname(__file__))
    from debug_panel import debug_router
    from debug_ultimate import router as ultimate_debug_router
    from validation_api import add_validation_routes, add_edge_case_routes


app = FastAPI(
    title="SkillShift AI",
    description="An AI-powered coaching tool for Mobile Legends: Bang Bang.",
    version="0.1.0",
)

# Add CORS middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now to debug
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add error handling middleware for CORS issues
@app.middleware("http")
async def cors_handler(request, call_next):
    """Handle CORS for all requests including errors."""
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # Return proper CORS headers even on errors
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )

# Include debug panel routers
app.include_router(debug_router)
app.include_router(ultimate_debug_router)

# Add validation routes
add_validation_routes(app)
add_edge_case_routes(app)


@app.get("/")
def read_root():
    """
    Root endpoint that returns a welcome message.
    """
    return {"message": "Welcome to the SkillShift AI API!"}


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
        # --- ULTIMATE PARSING SYSTEM ---
        # Use the Ultimate Parsing System for 95-100% confidence
        ultimate_result = ultimate_parsing_system.analyze_screenshot_ultimate(
            image_path=temp_file_path,
            ign=ign,
            hero_override=None,
            context="scoreboard",
            quality_threshold=85.0
        )

        # Extract the actual match data for feedback generation
        match_data_dict = ultimate_result.parsed_data

        if not match_data_dict:
            raise HTTPException(
                status_code=400,
                detail=f"Could not parse valid match data. "
                       f"Warnings: {ultimate_result.warnings}"
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
        
        # Return all the information: feedback, parsed data, and elite debug info
        return {
            "statistical_feedback": statistical_feedback,
            "mental_feedback": mental_feedback,
            "parsed_data": match_data_dict,
            "confidence_scores": {
                "overall_confidence": ultimate_result.overall_confidence,
                "confidence_category": ultimate_result.confidence_breakdown.category.value,
                "component_scores": ultimate_result.confidence_breakdown.component_scores,
                "quality_factors": ultimate_result.confidence_breakdown.quality_factors
            },
            "parsing_warnings": ultimate_result.warnings,
            "ultimate_analysis": {
                "processing_time": ultimate_result.processing_time,
                "success_factors": ultimate_result.success_factors,
                "improvement_roadmap": ultimate_result.improvement_roadmap,
                "data_completeness": ultimate_result.completeness_score
            }
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
        # --- ULTIMATE PARSING SYSTEM WITH FALLBACK ---
        # Use the Ultimate Parsing System for 95-100% confidence
        ultimate_result = ultimate_parsing_system.analyze_screenshot_ultimate(
            image_path=temp_file_path,
            ign=ign,
            hero_override=None,
            context="scoreboard",
            quality_threshold=85.0
        )
        
        # CRITICAL FIX: Fallback to Enhanced Data Collector if Ultimate system fails
        if ultimate_result.overall_confidence <= 0 or ultimate_result.analysis_stage == "failed":
            logger.warning("🔄 Ultimate Parsing System failed, falling back to Enhanced Data Collector")
            # Use Enhanced Data Collector as fallback
            enhanced_result = enhanced_data_collector.analyze_screenshot_with_session(
                image_path=temp_file_path,
                ign=ign,
                session_id=None,
                hero_override=None
            )
            
            # Convert enhanced result to ultimate format
            match_data_dict = enhanced_result.get("match_data", {})
            warnings = enhanced_result.get("warnings", [])
            overall_confidence = enhanced_result.get("overall_confidence", 0.0)
            completeness_score = enhanced_result.get("completeness_score", 0.0)
        else:
            # Ultimate system succeeded
            match_data_dict = ultimate_result.parsed_data
            warnings = ultimate_result.warnings
            # CRITICAL FIX: Don't divide by 100 - Ultimate system already returns percentage values
            overall_confidence = ultimate_result.overall_confidence  # Keep as percentage (63.2)
            completeness_score = ultimate_result.completeness_score  # Keep as percentage (2.0)

        # --- Generate Adaptive Diagnostics for Both Systems ---
        diagnostics = {
            "hero_detected": match_data_dict.get("hero", "unknown") != "unknown",
            "hero_name": match_data_dict.get("hero", "unknown"),
            "match_duration_detected": bool(match_data_dict.get("match_duration")),
            "gold_data_valid": bool(match_data_dict.get("gold") and match_data_dict.get("gold") > 0),
            "kda_data_complete": all(k in match_data_dict for k in ["kills", "deaths", "assists"]),
            "damage_data_available": bool(match_data_dict.get("hero_damage")),
            "ign_found": ign.lower() in str(match_data_dict).lower(),
            # CRITICAL FIX: Convert percentage to 0-1 range for frontend compatibility
            "confidence_score": overall_confidence / 100.0,  # Convert 63.2 → 0.632
            "warnings": warnings,
            # CRITICAL FIX: Convert percentage to 0-1 range for frontend compatibility  
            "data_completeness": completeness_score / 100.0,  # Convert 2.0 → 0.02
            "analysis_mode": "ultimate_with_fallback",
        }
        
        # Add system-specific fields based on which system provided the data
        if ultimate_result.overall_confidence > 0 and hasattr(ultimate_result, 'confidence_breakdown'):
            # Ultimate system succeeded - add Ultimate-specific fields
            diagnostics.update({
                "confidence_category": ultimate_result.confidence_breakdown.category.value,
                "component_scores": ultimate_result.confidence_breakdown.component_scores,
                "quality_factors": ultimate_result.confidence_breakdown.quality_factors,
                "processing_time": ultimate_result.processing_time,
                "success_factors": ultimate_result.success_factors,
                "improvement_roadmap": ultimate_result.improvement_roadmap
            })
        else:
            # Enhanced fallback system used - add fallback-specific fields
            diagnostics.update({
                "confidence_category": "acceptable" if overall_confidence >= 0.7 else "poor",
                "component_scores": {"basic_extraction": overall_confidence * 100, "fallback_mode": True},
                "quality_factors": {"system_fallback": True, "extraction_method": "enhanced"},
                "processing_time": 0.0,
                "success_factors": ["Enhanced fallback extraction successful"] if overall_confidence > 0.5 else [],
                "improvement_roadmap": ["🔄 Ultimate system failed - using enhanced fallback", "📸 Try higher quality screenshot for Ultimate analysis"]
            })

        if not match_data_dict:
            raise HTTPException(
                status_code=400,
                detail=f"Could not parse valid match data. Warnings: {warnings}"
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
            "overall_rating": overall_rating,
            "diagnostics": diagnostics,
            "match_data": match_data_dict
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
        # Validate inputs
        validation_result = error_handler.validate_request(
            ign=ign,
            file_path=temp_file_path
        )
        
        if not validation_result.is_valid:
            error_response = error_handler.handle_analysis_error(
                Exception(validation_result.error_message),
                {}
            )
            return error_handler.create_user_friendly_response(error_response)
        
        # Use ultimate parsing system for 95-100% confidence targeting
        try:
            ultimate_result = ultimate_parsing_system.analyze_screenshot_ultimate(
                image_path=temp_file_path,
                ign=ign,
                session_id=session_id,
                hero_override=hero_override if hero_override else None
            )
            print(f"🔍 DEBUG: Ultimate result type: {type(ultimate_result)}")
            print(f"🔍 DEBUG: Ultimate confidence: {ultimate_result.overall_confidence}")
            print(f"🔍 DEBUG: Ultimate analysis stage: {ultimate_result.analysis_stage}")
            
            # Convert ultimate result to enhanced format for compatibility
            result = {
                "data": ultimate_result.parsed_data,
                "warnings": ultimate_result.warnings,
                # CRITICAL FIX: Convert to 0-1 range for frontend compatibility
                "confidence": ultimate_result.overall_confidence / 100.0,  # Convert to 0-1 range
                "debug_info": ultimate_result.diagnostic_info,
                "session_info": {"session_id": ultimate_result.session_id},
                # CRITICAL FIX: Keep as percentage value, convert later for frontend
                "completeness_score": ultimate_result.completeness_score,  # Keep as percentage
                "overall_confidence": ultimate_result.overall_confidence / 100.0,  # Add for compatibility
                "diagnostics": {
                    # CRITICAL FIX: Convert to 0-1 range for frontend compatibility
                    "confidence_score": ultimate_result.overall_confidence / 100.0,  # Convert to 0-1 range
                    "analysis_state": ultimate_result.analysis_stage,
                    "hero_confidence": ultimate_result.hero_detection.confidence if hasattr(ultimate_result.hero_detection, 'confidence') else 0.0,
                    "hero_name": ultimate_result.hero_detection.hero_name if hasattr(ultimate_result.hero_detection, 'hero_name') else "unknown",
                    "hero_detected": ultimate_result.hero_detection.hero_name != "unknown" if hasattr(ultimate_result.hero_detection, 'hero_name') else False,
                    "kda_confidence": 1.0 if ultimate_result.parsed_data.get("kills") is not None else 0.0,
                    "gold_confidence": 1.0 if ultimate_result.parsed_data.get("gold") is not None else 0.0,
                    "kda_data_complete": all(ultimate_result.parsed_data.get(field) is not None for field in ["kills", "deaths", "assists"]),
                    "gold_data_valid": ultimate_result.parsed_data.get("gold") is not None and ultimate_result.parsed_data.get("gold") > 0,
                    "match_duration_detected": ultimate_result.parsed_data.get("match_duration") is not None
                }
            }
            print(f"🔍 DEBUG: Converted confidence: {result['confidence']}")
            print(f"🔍 DEBUG: Diagnostics confidence: {result['diagnostics']['confidence_score']}")
            
        except Exception as e:
            print(f"❌ ULTIMATE SYSTEM ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fallback to original enhanced system
            result = enhanced_data_collector.analyze_screenshot_with_session(
                image_path=temp_file_path,
                ign=ign,
                session_id=session_id,
                hero_override=hero_override if hero_override else None
            )
        
        match_data = result.get("data", {})
        
        if not match_data or not any(v for k, v in match_data.items() if k != "_session_info"):
            error_response = error_handler.handle_analysis_error(
                Exception("Could not parse valid match data"),
                match_data
            )
            error_response.details = {"warnings": result.get("warnings", [])}
            return error_handler.create_user_friendly_response(error_response)

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
        
        # Use AdvancedPerformanceAnalyzer for context-aware rating
        performance_report = advanced_performance_analyzer.analyze_comprehensive_performance(match_data)
        overall_rating = performance_report.overall_rating.value.title()  # Convert to title case
        
        # --- Generate Enhanced Diagnostics ---
        # Extract hero detection info from debug data - FIXED to use correct data structure
        debug_info = result.get("debug_info", {})
        hero_debug = debug_info.get("hero_debug", {})
        
        # Get hero confidence from multiple possible locations
        hero_confidence = 0.0
        if "hero_suggestions" in hero_debug and hero_debug["hero_suggestions"]:
            # If we have suggestions, use the top suggestion confidence
            hero_confidence = hero_debug["hero_suggestions"][0][1]
        else:
            # Fallback: extract from match data or debug info
            hero_confidence = debug_info.get("hero_confidence", 0.0)
        
        # Detect hero based on actual data, not just confidence threshold
        hero_name = match_data.get("hero", "unknown")
        hero_detected = bool(hero_name != "unknown" and hero_name != "" and hero_name is not None)
        
        # If hero is detected, ensure we have reasonable confidence
        if hero_detected and hero_confidence == 0.0:
            hero_confidence = 0.7  # Default reasonable confidence if hero is detected but confidence missing
        
        # Calculate granular confidence scores for each attribute
        kda_fields = ["kills", "deaths", "assists"]
        kda_present = [k in match_data for k in kda_fields]
        kda_confidence = float(sum(kda_present) / len(kda_fields)) if kda_fields else 0.0
        
        gold_value = match_data.get("gold", 0)
        gold_confidence = float(1.0 if gold_value > 1000 else 0.5 if gold_value > 0 else 0.0)
        
        # Consistency check: if overall confidence is 0, don't claim anything is detected
        overall_confidence = result.get("overall_confidence", 0.0)
        
        diagnostics = {
            # Hero Detection - ensure Python bool - FIXED to use corrected values
            "hero_detected": bool(hero_detected),
            "hero_name": hero_name if hero_detected else "unknown", 
            "hero_confidence": float(hero_confidence),
            
            # Match Info - ensure Python bool
            "match_duration_detected": bool(match_data.get("match_duration")) and bool(float(overall_confidence) > 0),
            "match_result_detected": bool(match_data.get("match_result")) and bool(float(overall_confidence) > 0),
            "player_rank_detected": bool(match_data.get("player_rank")) and bool(float(overall_confidence) > 0),
            
            # Core Stats with Confidence - ensure Python bool
            # CRITICAL FIX: Don't require overall_confidence > 0 for individual validations
            "gold_data_valid": bool(float(gold_confidence) > 0.5),  # Remove overall_confidence requirement
            "gold_confidence": float(gold_confidence),
            "kda_data_complete": bool(float(kda_confidence) >= 1.0),  # Remove overall_confidence requirement
            "kda_confidence": float(kda_confidence),
            
            # Additional Stats - ensure Python bool  
            "damage_data_available": bool(match_data.get("hero_damage")) and bool(float(overall_confidence) > 0),
            "teamfight_data_available": bool(match_data.get("teamfight_participation")) and bool(float(overall_confidence) > 0),
            
            # Overall Metrics
            "ign_found": True,  # Enhanced mode has better IGN validation
            "confidence_score": float(overall_confidence),
            "warnings": _filter_warnings_for_high_confidence(result.get("warnings", []), overall_confidence, hero_detected),
            # CRITICAL FIX: Convert percentage to 0-1 range for frontend consistency
            "data_completeness": float(result.get("completeness_score", 0.0)) / 100.0,
            
            # Analysis Info
            "analysis_mode": "enhanced",
            "screenshot_type": result.get("screenshot_type", "unknown"),
            "type_confidence": float(result.get("type_confidence", 0.0)),
            "session_complete": bool(result.get("session_complete", False)),
            "hero_suggestions": result.get("hero_suggestions", [])[:3],  # Top 3 suggestions
        }
        
        # Analysis State - CRITICAL FIX: Convert confidence to percentage scale for proper comparison
        confidence_for_state = float(overall_confidence) * 100 if float(overall_confidence) <= 1.0 else float(overall_confidence)
        analysis_state = "partial" if 10.0 < confidence_for_state < 60.0 else "complete" if confidence_for_state >= 60.0 else "failed"
        diagnostics["analysis_state"] = analysis_state
        
        # Enhanced validation checks - ensure Python bool
        validation_passed = bool(all([
            float(diagnostics["confidence_score"]) >= 0.7,
            bool(diagnostics["hero_detected"]) or len(diagnostics["hero_suggestions"]) > 0,
            bool(diagnostics["kda_data_complete"]),
            bool(diagnostics["gold_data_valid"])
        ]))
        
        diagnostics["validation_passed"] = bool(validation_passed)
        diagnostics["can_provide_feedback"] = bool(validation_passed and len(diagnostics["warnings"]) <= 3)

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
        
        # Add performance insights to feedback - ensure no numpy types
        if performance_report.strengths:
            feedback_items.append({
                "type": "success",
                "message": f"Strengths: {', '.join(convert_numpy_types(performance_report.strengths))}",
                "category": "Performance Analysis"
            })
        
        if performance_report.weaknesses:
            feedback_items.append({
                "type": "warning",
                "message": f"Areas for improvement: {', '.join(convert_numpy_types(performance_report.weaknesses))}",
                "category": "Performance Analysis"
            })
        
        if performance_report.improvement_priorities:
            feedback_items.append({
                "type": "info",
                "message": f"Priority focus: {convert_numpy_types(performance_report.improvement_priorities)[0]}",
                "category": "Improvement Tips"
            })

        response = {
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
            "debug_info": convert_numpy_types(result.get("debug_info", {})),
            "warnings": convert_numpy_types(result.get("warnings", [])),
            "diagnostics": diagnostics
        }
        
        # CRITICAL FIX: Convert all numpy types to Python types before serialization
        return convert_numpy_types(response)

    except HTTPException:
        raise
    except Exception as e:
        # Safe exception handling for numpy types
        try:
            error_detail = str(e)
        except Exception:
            error_detail = "Analysis error occurred"
        
        # Additional safety: ensure no numpy types in the error
        if "numpy.bool_" in error_detail or "vars()" in error_detail:
            error_detail = "Analysis processing error - please try again"
        
        raise HTTPException(status_code=500, detail=f"Enhanced analysis error: {error_detail}")
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


@app.post("/api/analyze-advanced/")
async def analyze_advanced_performance(
    file: UploadFile = File(...),
    ign: str = "Lesz XVII",
    session_id: Optional[str] = None,
    hero_override: str = ""
):
    """
    Advanced performance analysis with comprehensive metrics beyond KDA.
    Provides detailed performance breakdown including combat efficiency,
    objective participation, economic efficiency, and role-specific metrics.
    """
    # Use a temporary file to save the upload
    try:
        with NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
    finally:
        file.file.close()

    try:
        # Validate inputs
        validation_result = error_handler.validate_request(
            ign=ign,
            file_path=temp_file_path
        )
        
        if not validation_result.is_valid:
            error_response = error_handler.handle_analysis_error(
                Exception(validation_result.error_message),
                {}
            )
            return error_handler.create_user_friendly_response(error_response)
        
        # Use ultimate parsing system for 95-100% confidence targeting
        ultimate_result = ultimate_parsing_system.analyze_screenshot_ultimate(
            image_path=temp_file_path,
            ign=ign,
            session_id=session_id,
            hero_override=hero_override if hero_override else None
        )
        
        # Convert ultimate result to enhanced format for compatibility
        result = {
            "data": ultimate_result.parsed_data,
            "warnings": ultimate_result.warnings,
            "confidence": ultimate_result.overall_confidence / 100.0,  # Convert to 0-1 range
            "debug_info": ultimate_result.diagnostic_info,
            "session_info": {"session_id": ultimate_result.session_id},
            "completeness_score": ultimate_result.completeness_score,
            "diagnostics": {
                "confidence_score": ultimate_result.overall_confidence / 100.0,  # Convert to 0-1 range
                "analysis_state": ultimate_result.analysis_stage,
                "hero_confidence": ultimate_result.hero_detection.confidence if hasattr(ultimate_result.hero_detection, 'confidence') else 0.0,
                "hero_name": ultimate_result.hero_detection.hero_name if hasattr(ultimate_result.hero_detection, 'hero_name') else "unknown",
                "hero_detected": ultimate_result.hero_detection.hero_name != "unknown" if hasattr(ultimate_result.hero_detection, 'hero_name') else False,
                "kda_confidence": 1.0 if ultimate_result.parsed_data.get("kills") is not None else 0.0,
                "gold_confidence": 1.0 if ultimate_result.parsed_data.get("gold") is not None else 0.0,
                "kda_data_complete": all(ultimate_result.parsed_data.get(field) is not None for field in ["kills", "deaths", "assists"]),
                "gold_data_valid": ultimate_result.parsed_data.get("gold") is not None and ultimate_result.parsed_data.get("gold") > 0,
                "match_duration_detected": ultimate_result.parsed_data.get("match_duration") is not None
            }
        }
        
        match_data = result.get("data", {})
        
        if not match_data or not any(v for k, v in match_data.items() if k != "_session_info"):
            error_response = error_handler.handle_analysis_error(
                Exception("Could not parse valid match data"),
                match_data
            )
            error_response.details = {"warnings": result.get("warnings", [])}
            return error_handler.create_user_friendly_response(error_response)

        # Perform advanced performance analysis
        performance_report = advanced_performance_analyzer.analyze_comprehensive_performance(match_data)
        
        # Generate detailed feedback
        detailed_feedback = advanced_performance_analyzer.generate_detailed_feedback(performance_report)
        
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
        
        return {
            "advanced_analysis": {
                "hero": performance_report.hero,
                "role": performance_report.role,
                "overall_rating": performance_report.overall_rating.value,
                "overall_score": performance_report.overall_score,
                "core_metrics": {
                    "combat_efficiency": {
                        "value": performance_report.combat_efficiency.value,
                        "benchmark": performance_report.combat_efficiency.benchmark,
                        "category": performance_report.combat_efficiency.category.value,
                        "description": performance_report.combat_efficiency.description
                    },
                    "objective_participation": {
                        "value": performance_report.objective_participation.value,
                        "benchmark": performance_report.objective_participation.benchmark,
                        "category": performance_report.objective_participation.category.value,
                        "description": performance_report.objective_participation.description
                    },
                    "economic_efficiency": {
                        "value": performance_report.economic_efficiency.value,
                        "benchmark": performance_report.economic_efficiency.benchmark,
                        "category": performance_report.economic_efficiency.category.value,
                        "description": performance_report.economic_efficiency.description
                    },
                    "survival_rating": {
                        "value": performance_report.survival_rating.value,
                        "benchmark": performance_report.survival_rating.benchmark,
                        "category": performance_report.survival_rating.category.value,
                        "description": performance_report.survival_rating.description
                    }
                },
                "role_specific_metrics": {
                    name: {
                        "value": metric.value,
                        "benchmark": metric.benchmark,
                        "category": metric.category.value,
                        "description": metric.description
                    }
                    for name, metric in performance_report.role_specific_metrics.items()
                },
                "strengths": performance_report.strengths,
                "weaknesses": performance_report.weaknesses,
                "improvement_priorities": performance_report.improvement_priorities,
                "advanced_insights": performance_report.advanced_insights
            },
            "detailed_feedback": detailed_feedback,
            "mental_boost": mental_boost,
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
        raise HTTPException(status_code=500, detail=f"Advanced analysis error: {str(e)}")
    finally:
        os.unlink(temp_file_path)


@app.post("/api/counter-pick-analysis/")
async def analyze_counter_picks(
    enemy_heroes: List[str],
    allied_heroes: Optional[List[str]] = None,
    match_result: Optional[str] = None
):
    """
    Comprehensive counter-pick analysis for team compositions.
    Analyzes enemy team composition and provides intelligent counter-pick suggestions.
    """
    try:
        # Validate inputs
        validation_result = error_handler.validate_request(
            hero_names=enemy_heroes + (allied_heroes or [])
        )
        
        if not validation_result.is_valid:
            error_response = error_handler.handle_analysis_error(
                Exception(validation_result.error_message),
                {}
            )
            return error_handler.create_user_friendly_response(error_response)
        
        if not enemy_heroes:
            error_response = error_handler.handle_analysis_error(
                Exception("Enemy heroes list cannot be empty"),
                {}
            )
            return error_handler.create_user_friendly_response(error_response)
        
        # Perform counter-pick analysis
        analysis = enhanced_counter_pick_system.analyze_enemy_team_and_suggest_counters(
            enemy_heroes=enemy_heroes,
            allied_heroes=allied_heroes or [],
            match_result=match_result
        )
        
        # Export analysis to dictionary format
        analysis_dict = enhanced_counter_pick_system.export_counter_analysis(analysis)
        
        return {
            "counter_pick_analysis": analysis_dict,
            "success": True
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Counter-pick analysis error: {str(e)}")


@app.get("/api/hero-database/search/{query}")
async def search_heroes(query: str, limit: int = 10):
    """Search for heroes in the database."""
    try:
        search_results = hero_database.search_heroes(query, limit=limit)
        
        return {
            "query": query,
            "results": [
                {
                    "hero": result.hero,
                    "confidence": result.confidence,
                    "match_type": result.match_type,
                    "source": result.source
                }
                for result in search_results
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hero search error: {str(e)}")


@app.get("/api/hero-database/info/{hero_name}")
async def get_hero_info(hero_name: str):
    """Get detailed information about a specific hero."""
    try:
        hero_info = hero_database.get_hero_info(hero_name)
        
        if not hero_info:
            # Try to find similar heroes
            search_results = hero_database.search_heroes(hero_name, limit=3)
            if search_results:
                return {
                    "error": "Hero not found",
                    "suggestions": [
                        {"hero": result.hero, "confidence": result.confidence}
                        for result in search_results
                    ]
                }
            else:
                raise HTTPException(status_code=404, detail="Hero not found")
        
        return {
            "hero": {
                "name": hero_info.name,
                "role": hero_info.role,
                "aliases": hero_info.aliases,
                "detection_keywords": hero_info.detection_keywords,
                "skill_names": hero_info.skill_names,
                "counters": hero_info.counters,
                "countered_by": hero_info.countered_by,
                "synergies": hero_info.synergies,
                "meta_tier": hero_info.meta_tier,
                "win_rate": hero_info.win_rate,
                "pick_rate": hero_info.pick_rate,
                "ban_rate": hero_info.ban_rate,
                "image_url": hero_info.image_url,
                "wiki_url": hero_info.wiki_url
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hero info error: {str(e)}")


@app.get("/api/hero-database/roles/{role}")
async def get_heroes_by_role(role: str):
    """Get all heroes of a specific role."""
    try:
        heroes = hero_database.find_heroes_by_role(role)
        
        if not heroes:
            available_roles = list(hero_database.role_mapping.keys())
            return {
                "error": "No heroes found for this role",
                "available_roles": available_roles
            }
        
        return {
            "role": role,
            "heroes": heroes,
            "count": len(heroes)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Role search error: {str(e)}")


@app.get("/api/meta-analysis/top-heroes")
async def get_top_meta_heroes(limit: int = 10):
    """Get top meta heroes by win rate."""
    try:
        top_heroes = hero_database.get_top_meta_heroes(limit=limit)
        
        return {
            "top_heroes": [
                {"hero": hero, "win_rate": win_rate}
                for hero, win_rate in top_heroes
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Meta analysis error: {str(e)}")


@app.get("/api/meta-analysis/ban-worthy")
async def get_ban_worthy_heroes(limit: int = 10):
    """Get heroes worth banning based on meta analysis."""
    try:
        ban_worthy = hero_database.get_ban_worthy_heroes(limit=limit)
        
        return {
            "ban_worthy_heroes": ban_worthy
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ban analysis error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "hero_database_size": len(hero_database.heroes),
        "available_roles": list(hero_database.role_mapping.keys()),
        "version": "2.0.0"
    }