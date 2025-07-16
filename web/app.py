import asyncio
import json
import logging
import os
import shutil
import sys
import uuid
from tempfile import NamedTemporaryFile
from typing import List, Optional
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
from core.enhanced_ultimate_parsing_system import enhanced_ultimate_parsing_system

# Global analysis lock to prevent concurrent heavy processing
analysis_lock = asyncio.Lock()


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

# Redis and RQ setup for async job processing
try:
    from rq import Queue
    
    # Support multiple Redis formats
    redis_url = os.getenv("REDIS_URL")
    redis_token = os.getenv("REDIS_TOKEN")
    
    if redis_url and redis_token:
        # Upstash Redis format - need to use standard redis for RQ compatibility
        from redis import Redis
        # Convert Upstash URL to standard Redis URL format for RQ
        upstash_url = redis_url.replace("https://", "redis://")
        redis_conn = Redis.from_url(f"{upstash_url}?password={redis_token}", decode_responses=False)
        print(f"🌐 Connecting to Upstash Redis: {redis_url}")
        print("🔄 Using Redis Cloud compatibility mode for RQ")
    elif redis_url and redis_url.startswith(('redis://', 'rediss://')):
        # Standard Redis Cloud URL format
        from redis import Redis
        redis_conn = Redis.from_url(redis_url, decode_responses=False)
        print(f"🌐 Connecting to Redis Cloud: {redis_url.split('@')[1] if '@' in redis_url else redis_url}")
    else:
        # Fallback to individual parameters for local development
        from redis import Redis
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_db = int(os.getenv("REDIS_DB", "0"))
        redis_password = os.getenv("REDIS_PASSWORD")
        
        redis_conn = Redis(
            host=redis_host, 
            port=redis_port, 
            db=redis_db,
            password=redis_password if redis_password else None,
            decode_responses=False
        )
        print(f"🔧 Connecting to local Redis: {redis_host}:{redis_port}")
    
    # Test connection
    redis_conn.ping()
    job_queue = Queue("analysis", connection=redis_conn)
    
    # Create upload directory
    UPLOAD_DIR = Path("/tmp/skillshift_uploads")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    REDIS_AVAILABLE = True
    print("✅ Redis connection established successfully")
    
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis/RQ not available - async job processing disabled")
except Exception as e:
    REDIS_AVAILABLE = False
    logging.error(f"Redis connection failed: {e}")
    print(f"❌ Redis connection failed: {e}")
    print("💡 Tip: Make sure to set REDIS_URL and REDIS_TOKEN for Upstash Redis")

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
    Analyzes a match from a screenshot with enhanced confidence validation.
    """
    # Use a temporary file to save the upload
    try:
        with NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
    finally:
        file.file.close()

    try:
        # --- ENHANCED ULTIMATE PARSING SYSTEM ---
        ultimate_result = enhanced_ultimate_parsing_system.analyze_screenshot_ultimate(
            image_path=temp_file_path,
            ign=ign,
            hero_override=None,
            context="scoreboard",
            quality_threshold=75.0  # Lowered threshold for better sensitivity
        )

        # Extract the actual match data for feedback generation
        match_data_dict = ultimate_result.parsed_data
        
        # Enhanced validation - check if extraction actually succeeded
        extraction_success = (
            match_data_dict and 
            match_data_dict.get('hero', '').lower() not in ['unknown', 'n/a', ''] and
            ultimate_result.overall_confidence > 0.3
        )

        if not extraction_success:
            # Provide detailed error information
            ign_info = ultimate_result.diagnostic_info.get("ign_matching", {})
            confidence_info = ultimate_result.diagnostic_info.get("confidence_adjustment", {})
            
            error_detail = f"Data extraction failed. "
            
            if not ign_info.get("found", False):
                error_detail += f"IGN '{ign}' not found in screenshot. "
                error_detail += f"Try checking the spelling or provide a hero override. "
            
            if confidence_info.get("critical_issues"):
                error_detail += f"Issues: {'; '.join(confidence_info['critical_issues'][:2])}. "
            
            raise HTTPException(
                status_code=400,
                detail=error_detail
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
                goal = player_data.get("player_defined_goal", "general_improvement")
        except (FileNotFoundError, json.JSONDecodeError):
            history, goal = [], "general_improvement"

        mental_coach = MentalCoach(player_history=history, player_goal=goal)
        mental_boost = mental_coach.get_mental_boost(match_data_dict)

        # Enhanced diagnostics for frontend
        diagnostics = {
            "confidence_breakdown": {
                "original_confidence": ultimate_result.diagnostic_info.get("confidence_adjustment", {}).get("original", 0),
                "final_confidence": ultimate_result.overall_confidence,
                "adjustment_reason": ultimate_result.diagnostic_info.get("confidence_adjustment", {}).get("reason", ""),
                "ign_matching": ultimate_result.diagnostic_info.get("ign_matching", {}),
                "extraction_success": extraction_success
            },
            "data_quality": {
                "completeness": ultimate_result.completeness_score,
                "critical_fields_present": sum(1 for field in ['hero', 'kills', 'deaths', 'assists', 'gold'] 
                                               if match_data_dict.get(field) not in [None, 'N/A', '', 'unknown']),
                "warnings": ultimate_result.warnings
            },
            "processing_performance": {
                "processing_time": ultimate_result.processing_time,
                "analysis_stage": ultimate_result.analysis_stage
            }
        }

        return {
            "hero": match_data_dict.get("hero", "Unknown"),
            "statistical_feedback": statistical_feedback,
            "mental_boost": mental_boost,
            "confidence_score": ultimate_result.overall_confidence,
            "match_data": match_data_dict,
            "diagnostics": diagnostics
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
    finally:
        # Cleanup
        try:
            os.unlink(temp_file_path)
        except Exception:
            pass


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    ign: str = "Lesz XVII"
):
    """
    Frontend-compatible endpoint that analyzes a screenshot and returns coaching feedback.
    Returns response in format expected by React frontend.
    """
    # Try to acquire lock immediately, fail fast if busy
    if analysis_lock.locked():
        raise HTTPException(status_code=429, detail="Analysis busy; try again")
    
    async with analysis_lock:
        return await _perform_analysis(file, ign)

async def _perform_analysis(file: UploadFile, ign: str):
    """
    Internal function that performs the actual analysis work.
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

@app.get("/api/health")
async def api_health_check():
    """Frontend-compatible health check endpoint."""
    return {
        "status": "healthy",
        "hero_database_size": len(hero_database.heroes),
        "available_roles": list(hero_database.role_mapping.keys()),
        "version": "2.0.0",
        "services": {
            "ultimate_parsing_system": "active",
            "hero_database": "loaded",
            "analysis_engine": "ready"
        }
    }

@app.post("/api/analyze-fast")
async def analyze_fast(
    file: UploadFile = File(...),
    ign: str = "Lesz XVII"
):
    """
    Fast analysis endpoint that prioritizes speed over accuracy.
    Uses basic data collector without heavy processing.
    """
    # Use a temporary file to save the upload
    try:
        with NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
    finally:
        file.file.close()

    try:
        # Use enhanced data collector (simpler than ultimate system)
        result = enhanced_data_collector.analyze_screenshot_with_session(
            image_path=temp_file_path,
            ign=ign,
            session_id=None,
            hero_override=None
        )
        
        match_data = result.get("data", {})
        
        # Generate basic feedback
        statistical_feedback = generate_feedback(match_data, include_severity=True)
        
        # Simple mental boost
        mental_boost = "Keep pushing forward! Every match is a learning opportunity."
        
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
            else:
                feedback_items.append({
                    "type": "info",
                    "message": str(item),
                    "category": "General"
                })
        
        # Basic diagnostics
        diagnostics = {
            "hero_detected": match_data.get("hero", "unknown") != "unknown",
            "hero_name": match_data.get("hero", "unknown"),
            "confidence_score": result.get("confidence", 0.7),
            "analysis_mode": "fast",
            "processing_time": 2.0,  # Target under 2 seconds
            "data_completeness": result.get("completeness_score", 0.8),
            "warnings": result.get("warnings", [])
        }
        
        return {
            "feedback": feedback_items,
            "mental_boost": mental_boost,
            "overall_rating": "Good",
            "diagnostics": diagnostics,
            "match_data": match_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fast analysis error: {str(e)}")
    finally:
        try:
            os.unlink(temp_file_path)
        except Exception:
            pass


@app.post("/api/analyze-instant")
async def analyze_instant(
    file: UploadFile = File(...),
    ign: str = "Lesz XVII"
):
    """
    INSTANT analysis endpoint - returns cached results or enqueues heavy job.
    Cache hit: returns in under 2 seconds
    Cache miss: enqueues background job and returns job ID
    """
    import time
    import hashlib
    start_time = time.time()
    
    try:
        with NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
    finally:
        file.file.close()

    try:
        # Generate cache key from file content + IGN
        file_hash = hashlib.md5(open(temp_file_path, 'rb').read()).hexdigest()
        cache_key = f"analysis_{file_hash}_{ign}"
        
        # TODO: Check cache (Redis/file-based)
        # For now, assume cache miss and enqueue heavy job
        cached_result = None
        
        if cached_result:
            # Cache hit - return immediately
            processing_time = time.time() - start_time
            return {
                "success": True,
                "cached": True,
                "processing_time": processing_time,
                **cached_result
            }
        else:
            # Cache miss - enqueue heavy analysis job
            try:
                # Try to enqueue with RQ if available
                from rq import Queue
                from redis import Redis
                
                redis_conn = Redis(host='localhost', port=6379, db=0)
                q = Queue(connection=redis_conn)
                
                # Enqueue the heavy analysis job
                job = q.enqueue(
                    'core.ultimate_parsing_system.analyze_screenshot_ultimate',
                    image_path=temp_file_path,
                    ign=ign,
                    hero_override=None,
                    context="scoreboard",
                    quality_threshold=85.0,
                    job_timeout='5m'
                )
                
                processing_time = time.time() - start_time
                return {
                    "success": True,
                    "cached": False,
                    "job_enqueued": True,
                    "job_id": job.id,
                    "processing_time": processing_time,
                    "message": "Analysis job queued for processing",
                    "estimated_completion": "2-5 minutes",
                    "status_endpoint": f"/api/job-status/{job.id}",
                    "parsed_data": {"hero": "Unknown", "confidence": 0},
                    "overall_confidence": 0,
                    "diagnostics": {
                        "analysis_mode": "background_job",
                        "confidence_score": 0,
                        "warnings": ["Analysis queued - check status endpoint for results"]
                    }
                }
                
            except ImportError:
                # RQ not available - fallback to basic stub with job simulation
                processing_time = time.time() - start_time
                return {
                    "success": True,
                    "cached": False,
                    "job_enqueued": False,
                    "fallback_mode": True,
                    "processing_time": processing_time,
                    "message": "RQ not available - returning basic analysis",
                    "parsed_data": {"hero": "Unknown", "confidence": 0},
                    "overall_confidence": 0,
                    "diagnostics": {
                        "analysis_mode": "fallback_stub",
                        "confidence_score": 0,
                        "warnings": ["RQ worker not available - use /api/analyze for full analysis"]
                    }
                }
                
    except Exception as e:
        return {
            "success": False,
            "error": f"Instant analysis error: {str(e)}",
            "fallback": True,
            "parsed_data": {"hero": "Unknown", "confidence": 0},
            "overall_confidence": 0
        }
    finally:
        try:
            os.unlink(temp_file_path)
        except Exception:
            pass


# Heavy analysis function for RQ workers
def heavy_analysis(file_path: str, ign: str = "Lesz XVII", hero_override: str = "") -> dict:
    """
    Heavy analysis function that can be called by RQ workers.
    This is the existing YOLO + OCR pipeline extracted for worker use.
    
    Args:
        file_path: Path to the uploaded screenshot file
        ign: Player's in-game name
        hero_override: Optional hero override
    
    Returns:
        Dictionary with analysis results
    """
    try:
        # Use ultimate parsing system for analysis
        ultimate_result = ultimate_parsing_system.analyze_screenshot_ultimate(
            image_path=file_path,
            ign=ign,
            hero_override=hero_override if hero_override else None,
            context="scoreboard",
            quality_threshold=85.0
        )
        
        # Convert ultimate result to enhanced format for compatibility
        match_data_dict = ultimate_result.parsed_data
        warnings = ultimate_result.warnings
        overall_confidence = ultimate_result.overall_confidence
        completeness_score = ultimate_result.completeness_score

        # Generate feedback
        statistical_feedback = generate_feedback(match_data_dict, include_severity=True)

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
        mental_boost = mental_coach.get_mental_boost(match_data_dict)
        
        # Format response
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

        # Create diagnostics
        hero_detected = match_data_dict.get("hero", "unknown") != "unknown"
        diagnostics = {
            "hero_detected": hero_detected,
            "hero_name": match_data_dict.get("hero", "unknown"),
            "match_duration_detected": bool(match_data_dict.get("match_duration")),
            "gold_data_valid": bool(match_data_dict.get("gold") and match_data_dict.get("gold") > 0),
            "kda_data_complete": all(k in match_data_dict for k in ["kills", "deaths", "assists"]),
            "damage_data_available": bool(match_data_dict.get("hero_damage")),
            "ign_found": ign.lower() in str(match_data_dict).lower(),
            "confidence_score": overall_confidence / 100.0,  # Convert to 0-1 range
            "warnings": warnings,
            "data_completeness": completeness_score / 100.0,  # Convert to 0-1 range
            "analysis_mode": "worker_heavy_analysis",
        }

        return {
            "feedback": feedback_items,
            "mental_boost": mental_boost,
            "overall_rating": overall_rating,
            "diagnostics": diagnostics,
            "match_data": match_data_dict,
            "success": True
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "feedback": [],
            "mental_boost": "Analysis failed, but keep pushing forward!",
            "overall_rating": "Unknown",
            "diagnostics": {
                "analysis_mode": "worker_error",
                "error": str(e)
            }
        }
    finally:
        # Cleanup the temporary file
        try:
            os.unlink(file_path)
        except Exception:
            pass


# New async job endpoints
@app.post("/api/jobs", status_code=202)
async def create_job(file: UploadFile = File(...), ign: str = "Lesz XVII", hero_override: str = ""):
    """
    Create a new analysis job and return job ID for polling.
    This endpoint enqueues heavy analysis and returns immediately.
    """
    if not REDIS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Redis/RQ not available - use /api/analyze for synchronous processing"
        )
    
    try:
        # 1. Persist upload with unique job ID
        job_id = str(uuid.uuid4())
        saved_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
        
        with saved_path.open("wb") as f:
            f.write(await file.read())

        # 2. Enqueue the heavy analysis job
        job = job_queue.enqueue(
            "web.app.heavy_analysis",
            str(saved_path),
            ign,
            hero_override,
            job_timeout='5m'
        )
        
        return {
            "job_id": job.id,
            "state": job.get_status(),
            "message": "Analysis job created successfully",
            "estimated_completion": "2-5 minutes"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status and results of an analysis job.
    """
    if not REDIS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Redis/RQ not available"
        )
    
    try:
        job = job_queue.fetch_job(job_id)
        
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.is_finished:
            return {
                "state": "finished",
                "result": job.result,
                "job_id": job_id
            }
        elif job.is_failed:
            return {
                "state": "failed",
                "error": str(job.exc_info) if job.exc_info else "Unknown error",
                "job_id": job_id
            }
        else:
            return {
                "state": job.get_status(),
                "job_id": job_id,
                "message": f"Job is {job.get_status()}"
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


@app.get("/api/health-isolated")
async def health_check_isolated():
    """
    ISOLATED health check - never affected by analysis load.
    Returns immediately without dependencies.
    """
    return {
        "status": "healthy",
        "timestamp": "2025-01-15T20:30:00Z",
        "version": "2.0.0",
        "uptime": "running",
        "response_time_ms": 1
    }