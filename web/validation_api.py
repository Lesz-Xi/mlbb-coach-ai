"""
Validation API Endpoints for Real-User Validation Framework

Provides endpoints for:
- User feedback submission (/report-feedback)
- Ground truth annotation (/annotate)
- Validation statistics and dashboard (/validation-stats, /validation-dashboard)
- Edge case testing management (/edge-cases)
- Batch validation processing (/batch-validate)
"""

import os
import json
import shutil
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import validation framework
try:
    from core.validation_manager import validation_manager
    from core.validation_schemas import (
        UserFeedback, GroundTruthData, ValidationStatsRequest,
        FeedbackSubmissionRequest, AnnotationRequest, DeviceType, LocaleType
    )
except ImportError:
    # Fallback: create minimal validation system
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Validation system not available - creating minimal fallback")
    
    class MockValidationManager:
        def submit_user_feedback(self, *args, **kwargs):
            return True
        def get_validation_stats(self, *args, **kwargs):
            return {"success": False, "message": "Validation system not available"}
        def get_dashboard_data(self):
            return {"message": "Validation system not available"}
        def create_validation_entry(self, *args, **kwargs):
            return None
            
    validation_manager = MockValidationManager()
    
    # Mock classes
    class UserFeedback:
        def __init__(self, **kwargs):
            pass
    class GroundTruthData:
        def __init__(self, **kwargs):
            pass
    class ValidationStatsRequest:
        def __init__(self, **kwargs):
            pass
    class DeviceType:
        pass
    class LocaleType:
        pass

logger = logging.getLogger(__name__)


# ============ REQUEST/RESPONSE MODELS ============

class FeedbackRequest(BaseModel):
    """Simplified feedback request for web interface."""
    entry_id: str
    is_correct: bool
    incorrect_fields: List[str] = []
    corrections: Dict[str, Any] = {}
    user_rating: int  # 1-5 scale
    comments: str = ""
    user_id: Optional[str] = None


class AnnotationFormRequest(BaseModel):
    """Manual annotation form data."""
    entry_id: str
    player_ign: str
    hero_played: str
    kills: int
    deaths: int
    assists: int
    hero_damage: int
    turret_damage: int
    damage_taken: int
    teamfight_participation: int
    gold_per_min: int
    match_duration_minutes: int
    match_result: str  # "Victory" or "Defeat"
    annotator_notes: str = ""


class DeviceInfoRequest(BaseModel):
    """Device information for context."""
    device_type: Optional[str] = None
    device_model: Optional[str] = None
    game_locale: Optional[str] = None
    user_agent: Optional[str] = None


class ValidationDashboardResponse(BaseModel):
    """Dashboard data response."""
    success: bool
    dashboard_data: Dict[str, Any]
    generated_at: str


# ============ VALIDATION API ENDPOINTS ============

def add_validation_routes(app: FastAPI):
    """Add validation routes to the FastAPI app."""
    
    @app.post("/api/report-feedback/")
    async def report_feedback(
        feedback_data: FeedbackRequest,
        background_tasks: BackgroundTasks
    ):
        """
        Submit user feedback on analysis accuracy.
        
        This is the main endpoint for collecting real-user validation data.
        """
        try:
            # Create UserFeedback object
            user_feedback = UserFeedback(
                validation_entry_id=feedback_data.entry_id,
                is_analysis_correct=feedback_data.is_correct,
                incorrect_fields=feedback_data.incorrect_fields,
                user_corrections=feedback_data.corrections,
                confidence_in_feedback=0.8,  # Default confidence
                ease_of_use_rating=feedback_data.user_rating,
                analysis_speed_rating=feedback_data.user_rating,  # Use same rating
                user_comments=feedback_data.comments,
                user_id=feedback_data.user_id,
                submission_timestamp=datetime.now()
            )
            
            # Submit feedback (async in background)
            background_tasks.add_task(
                validation_manager.submit_user_feedback,
                feedback_data.entry_id,
                user_feedback
            )
            
            logger.info(f"Received feedback for entry {feedback_data.entry_id}")
            
            return {
                "success": True,
                "message": "Feedback submitted successfully",
                "feedback_id": user_feedback.feedback_id
            }
            
        except Exception as e:
            logger.error(f"Failed to submit feedback: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.post("/api/annotate/")
    async def annotate_entry(
        annotation_data: AnnotationFormRequest,
        background_tasks: BackgroundTasks
    ):
        """
        Submit manual annotation for ground truth data.
        
        Used by annotators to provide correct values for validation.
        """
        try:
            # Create GroundTruthData object
            ground_truth = GroundTruthData(
                player_ign=annotation_data.player_ign,
                hero_played=annotation_data.hero_played,
                kills=annotation_data.kills,
                deaths=annotation_data.deaths,
                assists=annotation_data.assists,
                hero_damage=annotation_data.hero_damage,
                turret_damage=annotation_data.turret_damage,
                damage_taken=annotation_data.damage_taken,
                teamfight_participation=annotation_data.teamfight_participation,
                gold_per_min=annotation_data.gold_per_min,
                match_duration_minutes=annotation_data.match_duration_minutes,
                match_result=annotation_data.match_result,
                annotator_id="web_annotator",
                annotation_confidence=0.95,  # High confidence for manual annotation
                annotation_timestamp=datetime.now(),
                annotation_notes=annotation_data.annotator_notes
            )
            
            # Submit annotation (async in background)
            background_tasks.add_task(
                validation_manager.annotate_entry,
                annotation_data.entry_id,
                ground_truth,
                annotation_data.annotator_notes
            )
            
            logger.info(f"Received annotation for entry {annotation_data.entry_id}")
            
            return {
                "success": True,
                "message": "Annotation submitted successfully",
                "entry_id": annotation_data.entry_id
            }
            
        except Exception as e:
            logger.error(f"Failed to submit annotation: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.get("/api/validation-stats/")
    async def get_validation_stats(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        device_filter: Optional[str] = None,
        locale_filter: Optional[str] = None
    ):
        """
        Get validation statistics and performance metrics.
        
        Supports filtering by date range, device type, and locale.
        """
        try:
            # Parse filters
            start_dt = datetime.fromisoformat(start_date) if start_date else None
            end_dt = datetime.fromisoformat(end_date) if end_date else None
            
            device_filters = []
            if device_filter:
                try:
                    device_filters = [DeviceType(d.strip()) for d in device_filter.split(",")]
                except ValueError:
                    pass
            
            locale_filters = []
            if locale_filter:
                try:
                    locale_filters = [LocaleType(l.strip()) for l in locale_filter.split(",")]
                except ValueError:
                    pass
            
            # Create request
            stats_request = ValidationStatsRequest(
                start_date=start_dt,
                end_date=end_dt,
                device_filter=device_filters if device_filters else None,
                locale_filter=locale_filters if locale_filters else None
            )
            
            # Get statistics
            stats_response = validation_manager.get_validation_stats(stats_request)
            
            return {
                "success": True,
                "stats": stats_response.model_dump(),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get validation stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.get("/api/validation-dashboard/")
    async def get_validation_dashboard():
        """
        Get comprehensive dashboard data for validation monitoring.
        
        Returns real-time metrics, charts, and system health information.
        """
        try:
            dashboard_data = validation_manager.get_dashboard_data()
            
            return ValidationDashboardResponse(
                success=True,
                dashboard_data=dashboard_data.model_dump(),
                generated_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.post("/api/validation-upload/")
    async def validation_upload(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        device_info: str = Form("{}"),
        ign: str = Form("Unknown Player")
    ):
        """
        Upload screenshot for validation with enhanced metadata collection.
        
        This endpoint creates validation entries that can be used for feedback collection.
        """
        try:
            # Parse device info
            device_data = json.loads(device_info) if device_info != "{}" else {}
            
            # Save uploaded file
            with NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_file_path = temp_file.name
            
            try:
                # Run AI analysis (using existing system)
                from core.ultimate_parsing_system import ultimate_parsing_system
                
                ultimate_result = ultimate_parsing_system.analyze_screenshot_ultimate(
                    image_path=temp_file_path,
                    ign=ign,
                    hero_override=None,
                    context="validation",
                    quality_threshold=70.0  # Lower threshold for validation
                )
                
                # Prepare AI result data
                ai_result = {
                    "parsed_data": ultimate_result.parsed_data,
                    "confidence_scores": {
                        "overall_confidence": ultimate_result.overall_confidence,
                        "component_scores": ultimate_result.confidence_breakdown.component_scores
                    },
                    "processing_time": ultimate_result.processing_time,
                    "warnings": ultimate_result.warnings,
                    "errors": []
                }
                
                # Create validation entry (async in background)
                background_tasks.add_task(
                    validation_manager.create_validation_entry,
                    temp_file_path,
                    ai_result,
                    device_data,
                    None  # session_id
                )
                
                return {
                    "success": True,
                    "message": "Screenshot uploaded for validation",
                    "ai_analysis": {
                        "confidence": ultimate_result.overall_confidence,
                        "parsed_data": ultimate_result.parsed_data,
                        "warnings": ultimate_result.warnings
                    },
                    "validation_eligible": ultimate_result.overall_confidence < 0.95  # Flag for manual review
                }
                
            finally:
                # Clean up temp file
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Failed to process validation upload: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.get("/api/validation-entries/")
    async def get_validation_entries(
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ):
        """
        Get validation entries for annotation or review.
        
        Used by annotation interface to show pending entries.
        """
        try:
            # This would need to be implemented in ValidationManager
            # For now, return a placeholder
            
            return {
                "success": True,
                "entries": [],
                "total_count": 0,
                "message": "Validation entries endpoint - implementation pending"
            }
            
        except Exception as e:
            logger.error(f"Failed to get validation entries: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.post("/api/batch-validate/")
    async def batch_validate(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...),
        batch_name: str = Form("Batch Validation"),
        device_info: str = Form("{}")
    ):
        """
        Process multiple screenshots for batch validation.
        
        Useful for testing edge cases or processing large datasets.
        """
        try:
            batch_results = []
            device_data = json.loads(device_info) if device_info != "{}" else {}
            
            for i, file in enumerate(files):
                # Process each file
                with NamedTemporaryFile(delete=False, suffix=f"_batch_{i}.png") as temp_file:
                    shutil.copyfileobj(file.file, temp_file)
                    temp_file_path = temp_file.name
                
                try:
                    # Quick analysis for batch processing
                    from core.ultimate_parsing_system import ultimate_parsing_system
                    
                    ultimate_result = ultimate_parsing_system.analyze_screenshot_ultimate(
                        image_path=temp_file_path,
                        ign="Batch User",
                        hero_override=None,
                        context="batch_validation",
                        quality_threshold=60.0  # Even lower threshold for batch
                    )
                    
                    batch_results.append({
                        "filename": file.filename,
                        "confidence": ultimate_result.overall_confidence,
                        "success": True,
                        "warnings": ultimate_result.warnings
                    })
                    
                    # Create validation entry in background
                    ai_result = {
                        "parsed_data": ultimate_result.parsed_data,
                        "confidence_scores": {"overall_confidence": ultimate_result.overall_confidence},
                        "processing_time": ultimate_result.processing_time,
                        "warnings": ultimate_result.warnings,
                        "errors": []
                    }
                    
                    background_tasks.add_task(
                        validation_manager.create_validation_entry,
                        temp_file_path,
                        ai_result,
                        device_data,
                        f"batch_{batch_name}"
                    )
                    
                except Exception as e:
                    batch_results.append({
                        "filename": file.filename,
                        "confidence": 0.0,
                        "success": False,
                        "error": str(e)
                    })
                finally:
                    os.unlink(temp_file_path)
            
            # Calculate batch statistics
            successful_files = [r for r in batch_results if r["success"]]
            avg_confidence = sum(r["confidence"] for r in successful_files) / len(successful_files) if successful_files else 0.0
            
            return {
                "success": True,
                "batch_name": batch_name,
                "total_files": len(files),
                "successful_files": len(successful_files),
                "average_confidence": avg_confidence,
                "results": batch_results
            }
            
        except Exception as e:
            logger.error(f"Failed to process batch validation: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# ============ EDGE CASE TESTING ENDPOINTS ============

def add_edge_case_routes(app: FastAPI):
    """Add edge case testing routes."""
    
    @app.post("/api/edge-case-test/")
    async def run_edge_case_test(
        background_tasks: BackgroundTasks,
        test_name: str = Form(...),
        test_description: str = Form(...),
        test_category: str = Form(...),
        test_files: List[UploadFile] = File(...)
    ):
        """
        Run a specific edge case test with provided screenshots.
        
        Tests system behavior under specific conditions.
        """
        try:
            test_results = []
            
            for file in test_files:
                with NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    shutil.copyfileobj(file.file, temp_file)
                    temp_file_path = temp_file.name
                
                try:
                    # Run analysis
                    from core.ultimate_parsing_system import ultimate_parsing_system
                    
                    ultimate_result = ultimate_parsing_system.analyze_screenshot_ultimate(
                        image_path=temp_file_path,
                        ign="Edge Case Test",
                        hero_override=None,
                        context="edge_case_test",
                        quality_threshold=30.0  # Very low threshold for edge case testing
                    )
                    
                    test_results.append({
                        "filename": file.filename,
                        "confidence": ultimate_result.overall_confidence,
                        "processing_time": ultimate_result.processing_time,
                        "warnings": ultimate_result.warnings,
                        "success": True
                    })
                    
                except Exception as e:
                    test_results.append({
                        "filename": file.filename,
                        "confidence": 0.0,
                        "processing_time": 0.0,
                        "warnings": [],
                        "error": str(e),
                        "success": False
                    })
                finally:
                    os.unlink(temp_file_path)
            
            # Calculate test statistics
            successful_tests = [r for r in test_results if r["success"]]
            success_rate = len(successful_tests) / len(test_results) if test_results else 0.0
            avg_confidence = sum(r["confidence"] for r in successful_tests) / len(successful_tests) if successful_tests else 0.0
            
            return {
                "success": True,
                "test_name": test_name,
                "test_category": test_category,
                "total_files": len(test_files),
                "success_rate": success_rate,
                "avg_confidence": avg_confidence,
                "results": test_results
            }
            
        except Exception as e:
            logger.error(f"Failed to run edge case test: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.get("/api/edge-case-summary/")
    async def get_edge_case_summary():
        """Get summary of all edge case test results."""
        try:
            # This would pull from validation database
            return {
                "success": True,
                "summary": {
                    "total_tests": 0,
                    "categories": {},
                    "success_rates": {},
                    "trending_failures": []
                },
                "message": "Edge case summary - implementation pending"
            }
            
        except Exception as e:
            logger.error(f"Failed to get edge case summary: {e}")
            raise HTTPException(status_code=500, detail=str(e)) 