"""
Tactical Coaching API for MLBB Coach AI
======================================

This module provides REST API endpoints for tactical coaching services,
integrating with the existing web application architecture.
"""

import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path
import json

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import existing system components
from ..core.services.tactical_coaching_service import (
    create_tactical_coaching_service
)
from ..core.temporal_pipeline import create_temporal_pipeline
from ..core.behavioral_modeling import BehavioralAnalyzer
from ..core.cache.hybrid_cache import HybridCache

logger = logging.getLogger(__name__)

# Request/Response models
class TacticalCoachingRequest(BaseModel):
    """Request model for tactical coaching analysis."""
    player_ign: str = Field(..., description="Player's in-game name")
    video_path: Optional[str] = Field(None, description="Path to video file")
    coaching_focus: list[str] = Field(default=[], description="Specific coaching areas to focus on")
    include_visual_overlays: bool = Field(default=True, description="Include visual annotations")
    include_gamified_feedback: bool = Field(default=True, description="Include gamified achievements")


class TacticalCoachingResponse(BaseModel):
    """Response model for tactical coaching analysis."""
    success: bool
    player_ign: str
    post_game_summary: str
    tactical_findings: list[Dict[str, Any]]
    visual_overlays: list[Dict[str, Any]]
    game_phase_breakdown: Dict[str, list[Dict[str, Any]]]
    opportunity_analysis: list[Dict[str, Any]]
    gamified_feedback: list[str]
    overall_confidence: float
    processing_time: float
    insights_generated: int
    error_message: Optional[str] = None


# Initialize services
cache_manager = HybridCache()
tactical_coaching_service = create_tactical_coaching_service(cache_manager)
behavioral_analyzer = BehavioralAnalyzer(cache_manager)

# Create API router
router = APIRouter(prefix="/api/tactical-coaching", tags=["tactical-coaching"])


@router.post("/analyze", response_model=TacticalCoachingResponse)
async def analyze_tactical_coaching(
    request: TacticalCoachingRequest
) -> TacticalCoachingResponse:
    """
    Perform tactical coaching analysis on timestamped gameplay data.
    
    This endpoint processes video gameplay and returns comprehensive tactical insights
    including natural language coaching, visual overlays, and strategic recommendations.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting tactical coaching analysis for {request.player_ign}")
        
        # Step 1: Run temporal analysis pipeline
        temporal_pipeline = create_temporal_pipeline(request.player_ign)
        
        if not request.video_path or not Path(request.video_path).exists():
            raise HTTPException(
                status_code=400,
                detail="Valid video path is required for analysis"
            )
        
        # Run complete temporal analysis
        temporal_result = temporal_pipeline.analyze_video(
            request.video_path,
            output_dir="temp/tactical_analysis"
        )
        
        logger.info(f"Temporal analysis completed: {len(temporal_result.game_events)} events detected")
        
        # Step 2: Generate behavioral profile
        behavioral_profile = await behavioral_analyzer.analyze_player_behavior(
            player_id=request.player_ign,
            match_history=[],  # Would use actual match history if available
            video_paths=[request.video_path]
        )
        
        logger.info(f"Behavioral profile generated: {behavioral_profile.play_style.value}")
        
        # Step 3: Perform tactical coaching analysis
        coaching_request = {
            "temporal_analysis": temporal_result,
            "behavioral_profile": behavioral_profile,
            "coaching_focus": request.coaching_focus
        }
        
        coaching_result = await tactical_coaching_service.process(coaching_request)
        
        if not coaching_result.success:
            raise HTTPException(
                status_code=500,
                detail=f"Tactical coaching analysis failed: {coaching_result.error}"
            )
        
        # Step 4: Format response
        coaching_data = coaching_result.data
        
        response = TacticalCoachingResponse(
            success=True,
            player_ign=request.player_ign,
            post_game_summary=coaching_data.get("post_game_summary", ""),
            tactical_findings=coaching_data.get("tactical_findings", []),
            visual_overlays=coaching_data.get("visual_overlays", []) if request.include_visual_overlays else [],
            game_phase_breakdown=coaching_data.get("game_phase_breakdown", {}),
            opportunity_analysis=coaching_data.get("opportunity_analysis", []),
            gamified_feedback=coaching_data.get("gamified_feedback", []) if request.include_gamified_feedback else [],
            overall_confidence=coaching_data.get("overall_confidence", 0.0),
            processing_time=time.time() - start_time,
            insights_generated=coaching_data.get("insights_generated", 0)
        )
        
        logger.info(f"Tactical coaching completed in {response.processing_time:.2f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in tactical coaching analysis: {str(e)}")
        return TacticalCoachingResponse(
            success=False,
            player_ign=request.player_ign,
            post_game_summary="",
            tactical_findings=[],
            visual_overlays=[],
            game_phase_breakdown={},
            opportunity_analysis=[],
            gamified_feedback=[],
            overall_confidence=0.0,
            processing_time=time.time() - start_time,
            insights_generated=0,
            error_message=str(e)
        )


@router.post("/analyze-from-json")
async def analyze_from_json_data(
    player_ign: str = Form(...),
    temporal_data: UploadFile = File(...),
    coaching_focus: Optional[str] = Form(default="[]")
) -> JSONResponse:
    """
    Analyze tactical coaching from pre-processed JSON temporal data.
    
    This endpoint accepts timestamped JSON data directly, bypassing video processing.
    Useful for analyzing previously processed gameplay data.
    """
    try:
        # Parse temporal data
        temporal_content = await temporal_data.read()
        temporal_json = json.loads(temporal_content)
        
        # Parse coaching focus
        focus_areas = json.loads(coaching_focus) if coaching_focus else []
        
        # Create coaching request
        coaching_request = {
            "temporal_analysis": temporal_json,
            "behavioral_profile": None,  # Will be generated from events
            "coaching_focus": focus_areas
        }
        
        # Process tactical coaching
        coaching_result = await tactical_coaching_service.process(coaching_request)
        
        if not coaching_result.success:
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {coaching_result.error}"
            )
        
        # Return formatted response
        return JSONResponse(content={
            "success": True,
            "data": coaching_result.data,
            "metadata": coaching_result.metadata
        })
        
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON format in temporal data"
        )
    except Exception as e:
        logger.error(f"Error in JSON analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis error: {str(e)}"
        )


@router.get("/sample-response")
async def get_sample_response() -> JSONResponse:
    """
    Get a sample tactical coaching response for testing/documentation.
    
    This endpoint returns a mock response showing the expected format
    of tactical coaching analysis results.
    """
    sample_response = {
        "success": True,
        "player_ign": "Lesz XVII",
        "post_game_summary": (
            "Strong tactical gameplay overall. Your carry focused approach shows "
            "developing decision-making patterns. Good opportunity recognition and "
            "execution. Focus areas: positioning, map awareness, and objective timing."
        ),
        "tactical_findings": [
            {
                "timestamp": 245.5,
                "event": "late_rotation",
                "finding": "Late rotation to teamfight area. Arrived 8.2s after engagement started.",
                "suggestion": "Improve map awareness and anticipate teamfight locations. Watch for enemy positioning cues.",
                "severity": "high",
                "confidence": 0.80,
                "game_phase": "mid_game",
                "event_type": "rotation_analysis",
                "metadata": {"delay": 8.2}
            },
            {
                "timestamp": 512.3,
                "event": "death_positioning",
                "finding": "Poor positioning led to elimination at 512.3s. Pattern indicates positioning issues.",
                "suggestion": "Focus on staying behind tanks and identifying threats before engaging.",
                "severity": "medium",
                "confidence": 0.75,
                "game_phase": "late_game",
                "event_type": "positioning_analysis",
                "metadata": {"positioning_score": 0.45}
            }
        ],
        "visual_overlays": [
            {
                "frame_path": "frame_007365_t245.50s.jpg",
                "annotations": [
                    {
                        "type": "arrow",
                        "from_region": "current_position",
                        "to_region": "objective_location",
                        "label": "Missed Rotation",
                        "color": "red"
                    },
                    {
                        "type": "zone",
                        "region": "objective_area",
                        "label": "No Vision",
                        "color": "orange"
                    }
                ]
            }
        ],
        "game_phase_breakdown": {
            "early_game": [],
            "mid_game": [
                {
                    "timestamp": 245.5,
                    "event": "late_rotation",
                    "finding": "Late rotation to teamfight area. Arrived 8.2s after engagement started.",
                    "suggestion": "Improve map awareness and anticipate teamfight locations.",
                    "severity": "high",
                    "confidence": 0.80,
                    "game_phase": "mid_game"
                }
            ],
            "late_game": [
                {
                    "timestamp": 512.3,
                    "event": "death_positioning",
                    "finding": "Poor positioning led to elimination at 512.3s.",
                    "suggestion": "Focus on staying behind tanks and identifying threats.",
                    "severity": "medium",
                    "confidence": 0.75,
                    "game_phase": "late_game"
                }
            ]
        },
        "opportunity_analysis": [
            {
                "timestamp": 512.3,
                "event": "tower_destroyed",
                "missed_action": "Not present for tower push after favorable team fight",
                "alternative": "Rotate to tower immediately after team fight advantage at 497.3s",
                "impact_score": 0.8,
                "reasoning": "Tower gold and map control are crucial for maintaining momentum",
                "metadata": {"tower_gold": 320, "map_control_value": "high"}
            }
        ],
        "gamified_feedback": [
            "🗺️ Rotation Trainee: Late to 2+ key rotations – improve map awareness and timing",
            "🎯 Positioning Pro: Excellent positioning awareness – keep it up!",
            "🌟 Consistent Performer: Few major tactical issues identified – solid gameplay!"
        ],
        "overall_confidence": 0.775,
        "processing_time": 15.2,
        "insights_generated": 3
    }
    
    return JSONResponse(content=sample_response)


@router.get("/coaching-patterns")
async def get_coaching_patterns() -> JSONResponse:
    """
    Get available tactical coaching patterns and templates.
    
    This endpoint returns information about what types of tactical patterns
    the system can detect and analyze.
    """
    patterns = {
        "rotation_patterns": {
            "late_rotation": {
                "description": "Player rotates too late to objectives",
                "severity": "high",
                "coaching_focus": "map_awareness"
            },
            "overextension": {
                "description": "Player extends too far without vision",
                "severity": "critical",
                "coaching_focus": "vision_control"
            }
        },
        "farming_patterns": {
            "inefficient_farming": {
                "description": "Player farms inefficiently during key moments",
                "severity": "medium",
                "coaching_focus": "macro_play"
            }
        },
        "teamfight_patterns": {
            "poor_positioning": {
                "description": "Player positions poorly in teamfights",
                "severity": "high",
                "coaching_focus": "positioning"
            }
        },
        "available_focus_areas": [
            "positioning",
            "map_awareness",
            "vision_control",
            "macro_play",
            "teamfight_execution",
            "objective_prioritization"
        ]
    }
    
    return JSONResponse(content=patterns)


# Health check endpoint
@router.get("/health")
async def health_check() -> JSONResponse:
    """Check the health of the tactical coaching service."""
    try:
        # Test basic service functionality
        test_start = time.time()
        
        # Check cache connectivity
        cache_healthy = await cache_manager.health_check() if hasattr(cache_manager, 'health_check') else True
        
        test_time = time.time() - test_start
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "tactical_coaching",
            "cache_status": "healthy" if cache_healthy else "degraded",
            "response_time": test_time,
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "tactical_coaching",
                "error": str(e),
                "timestamp": time.time()
            }
        ) 