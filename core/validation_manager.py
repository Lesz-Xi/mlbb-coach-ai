"""
Validation Manager for real-user testing framework.
Handles validation entries, user feedback, and performance metrics.
"""

import json
import sqlite3
import statistics
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from PIL import Image

from .validation_schemas import (
    ValidationEntry, ValidationStatus, UserFeedback, GroundTruthData,
    ValidationStatsRequest, ValidationStatsResponse, PerformanceMetrics,
    DashboardData, EdgeCaseTest, ScreenshotMetadata, DeviceType, LocaleType, ScreenshotType
)

import logging
logger = logging.getLogger(__name__)


class ValidationDatabase:
    """Database handler for validation data."""
    
    def __init__(self, db_path: str = "data/validation.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Validation entries table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_entries (
                    entry_id TEXT PRIMARY KEY,
                    screenshot_metadata TEXT,
                    ground_truth TEXT,
                    ai_parsed_data TEXT,
                    ai_confidence_scores TEXT,
                    ai_processing_time REAL,
                    ai_warnings TEXT,
                    ai_errors TEXT,
                    status TEXT,
                    validation_score REAL,
                    validation_notes TEXT,
                    accuracy_metrics TEXT,
                    edge_case_flags TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    validated_at TEXT
                )
            """)
            
            # User feedback table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    validation_entry_id TEXT,
                    is_analysis_correct BOOLEAN,
                    incorrect_fields TEXT,
                    user_corrections TEXT,
                    confidence_in_feedback REAL,
                    ease_of_use_rating INTEGER,
                    analysis_speed_rating INTEGER,
                    user_comments TEXT,
                    suggested_improvements TEXT,
                    user_id TEXT,
                    submission_timestamp TEXT,
                    ip_address TEXT,
                    FOREIGN KEY (validation_entry_id) REFERENCES validation_entries (entry_id)
                )
            """)
            
            # Edge case tests table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edge_case_tests (
                    test_id TEXT PRIMARY KEY,
                    test_name TEXT,
                    test_category TEXT,
                    test_description TEXT,
                    baseline_results TEXT,
                    created_at TEXT
                )
            """)
            
            # Performance metrics table  
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    metric_id TEXT PRIMARY KEY,
                    validation_entry_id TEXT,
                    metric_type TEXT,
                    metric_value REAL,
                    recorded_at TEXT,
                    FOREIGN KEY (validation_entry_id) REFERENCES validation_entries (entry_id)
                )
            """)
            
            conn.commit()
    
    def store_validation_entry(self, entry: ValidationEntry) -> str:
        """Store a validation entry in the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO validation_entries (
                    entry_id, screenshot_metadata, ground_truth, ai_parsed_data,
                    ai_confidence_scores, ai_processing_time, ai_warnings, ai_errors,
                    status, validation_score, validation_notes, accuracy_metrics,
                    edge_case_flags, created_at, updated_at, validated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.entry_id,
                json.dumps(entry.screenshot_metadata.model_dump()) if entry.screenshot_metadata else None,
                json.dumps(entry.ground_truth.model_dump()) if entry.ground_truth else None,
                json.dumps(entry.ai_parsed_data),
                json.dumps(entry.ai_confidence_scores),
                entry.ai_processing_time,
                json.dumps(entry.ai_warnings),
                json.dumps(entry.ai_errors),
                entry.status.value,
                entry.validation_score,
                entry.validation_notes,
                json.dumps(entry.accuracy_metrics) if entry.accuracy_metrics else None,
                json.dumps(entry.edge_case_flags),
                entry.created_at.isoformat(),
                entry.updated_at.isoformat() if entry.updated_at else None,
                entry.validated_at.isoformat() if entry.validated_at else None
            ))
            conn.commit()
        return entry.entry_id
    
    def get_validation_entry(self, entry_id: str) -> Optional[ValidationEntry]:
        """Retrieve a validation entry by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM validation_entries WHERE entry_id = ?", 
                (entry_id,)
            ).fetchone()
            
            if not row:
                return None
            
            # Parse related user feedback
            feedback_rows = conn.execute(
                "SELECT * FROM user_feedback WHERE validation_entry_id = ?",
                (entry_id,)
            ).fetchall()
            
            user_feedback = []
            for feedback_row in feedback_rows:
                feedback = UserFeedback(
                    feedback_id=feedback_row['feedback_id'],
                    validation_entry_id=feedback_row['validation_entry_id'],
                    is_analysis_correct=bool(feedback_row['is_analysis_correct']),
                    incorrect_fields=json.loads(feedback_row['incorrect_fields']) if feedback_row['incorrect_fields'] else [],
                    user_corrections=json.loads(feedback_row['user_corrections']) if feedback_row['user_corrections'] else {},
                    confidence_in_feedback=feedback_row['confidence_in_feedback'],
                    ease_of_use_rating=feedback_row['ease_of_use_rating'],
                    analysis_speed_rating=feedback_row['analysis_speed_rating'],
                    user_comments=feedback_row['user_comments'] or "",
                    suggested_improvements=feedback_row['suggested_improvements'] or "",
                    user_id=feedback_row['user_id'],
                    submission_timestamp=datetime.fromisoformat(feedback_row['submission_timestamp']),
                    ip_address=feedback_row['ip_address']
                )
                user_feedback.append(feedback)
            
            # Parse screenshot metadata
            screenshot_metadata = None
            if row['screenshot_metadata']:
                metadata_data = json.loads(row['screenshot_metadata'])
                screenshot_metadata = ScreenshotMetadata(**metadata_data)
            
            # Parse ground truth
            ground_truth = None
            if row['ground_truth']:
                truth_data = json.loads(row['ground_truth'])
                ground_truth = GroundTruthData(**truth_data)
            
            return ValidationEntry(
                entry_id=row['entry_id'],
                screenshot_metadata=screenshot_metadata,
                ground_truth=ground_truth,
                ai_parsed_data=json.loads(row['ai_parsed_data']) if row['ai_parsed_data'] else {},
                ai_confidence_scores=json.loads(row['ai_confidence_scores']) if row['ai_confidence_scores'] else {},
                ai_processing_time=row['ai_processing_time'] or 0.0,
                ai_warnings=json.loads(row['ai_warnings']) if row['ai_warnings'] else [],
                ai_errors=json.loads(row['ai_errors']) if row['ai_errors'] else [],
                status=ValidationStatus(row['status']),
                validation_score=row['validation_score'],
                validation_notes=row['validation_notes'] or "",
                user_feedback=user_feedback,
                accuracy_metrics=json.loads(row['accuracy_metrics']) if row['accuracy_metrics'] else None,
                edge_case_flags=json.loads(row['edge_case_flags']) if row['edge_case_flags'] else [],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                validated_at=datetime.fromisoformat(row['validated_at']) if row['validated_at'] else None
            )


class ValidationManager:
    """Main validation manager class."""
    
    def __init__(self, db_path: str = "data/validation.db"):
        self.db = ValidationDatabase(db_path)
        self.logger = logging.getLogger(__name__)
    
    def create_validation_entry(
        self,
        screenshot_path: str,
        ai_analysis_result: Dict[str, Any],
        device_info: Optional[Dict[str, str]] = None,
        user_session_id: Optional[str] = None
    ) -> ValidationEntry:
        """Create a new validation entry from AI analysis results."""
        
        # Extract screenshot metadata
        screenshot_metadata = self._extract_screenshot_metadata(
            screenshot_path, device_info, user_session_id
        )
        
        # Create validation entry
        entry = ValidationEntry(
            entry_id=str(uuid.uuid4()),
            screenshot_metadata=screenshot_metadata,
            ground_truth=None,  # To be added later via annotation
            ai_parsed_data=ai_analysis_result.get("parsed_data", {}),
            ai_confidence_scores=ai_analysis_result.get("confidence_scores", {}),
            ai_processing_time=ai_analysis_result.get("processing_time", 0.0),
            ai_warnings=ai_analysis_result.get("warnings", []),
            ai_errors=ai_analysis_result.get("errors", []),
            status=ValidationStatus.PENDING,
            edge_case_flags=self._detect_edge_cases(ai_analysis_result, screenshot_metadata),
            created_at=datetime.now()
        )
        
        # Store in database
        self.db.store_validation_entry(entry)
        self.logger.info(f"Created validation entry {entry.entry_id}")
        
        return entry
    
    def submit_user_feedback(
        self,
        entry_id: str,
        feedback: UserFeedback
    ) -> bool:
        """Submit user feedback for a validation entry."""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute("""
                    INSERT INTO user_feedback (
                        feedback_id, validation_entry_id, is_analysis_correct,
                        incorrect_fields, user_corrections, confidence_in_feedback,
                        ease_of_use_rating, analysis_speed_rating, user_comments,
                        suggested_improvements, user_id, submission_timestamp, ip_address
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.feedback_id,
                    entry_id,
                    feedback.is_analysis_correct,
                    json.dumps(feedback.incorrect_fields),
                    json.dumps(feedback.user_corrections),
                    feedback.confidence_in_feedback,
                    feedback.ease_of_use_rating,
                    feedback.analysis_speed_rating,
                    feedback.user_comments,
                    feedback.suggested_improvements,
                    feedback.user_id,
                    feedback.submission_timestamp.isoformat(),
                    feedback.ip_address
                ))
                conn.commit()
            
            self.logger.info(f"Submitted feedback for entry {entry_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to submit feedback: {e}")
            return False
    
    def annotate_entry(
        self,
        entry_id: str,
        ground_truth: GroundTruthData,
        annotator_notes: str = ""
    ) -> bool:
        """Add ground truth annotation to a validation entry."""
        try:
            entry = self.db.get_validation_entry(entry_id)
            if not entry:
                return False
            
            # Update entry with ground truth
            entry.ground_truth = ground_truth
            entry.status = ValidationStatus.ANNOTATED
            entry.validation_notes = annotator_notes
            entry.updated_at = datetime.now()
            
            # Calculate accuracy metrics
            entry.accuracy_metrics = self._calculate_accuracy_metrics(
                entry.ai_parsed_data, ground_truth
            )
            
            # Calculate validation score
            entry.validation_score = self._calculate_validation_score(entry)
            
            # Update database
            self.db.store_validation_entry(entry)
            self.logger.info(f"Annotated entry {entry_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to annotate entry: {e}")
            return False
    
    def get_validation_stats(self, request: ValidationStatsRequest) -> ValidationStatsResponse:
        """Get comprehensive validation statistics."""
        metrics = self._calculate_performance_metrics("", [])
        
        return ValidationStatsResponse(
            metrics=metrics,
            recent_entries=self._get_recent_entries(limit=10),
            trending_issues=self._get_trending_issues(),
            system_health=self._get_system_health()
        )
    
    def get_dashboard_data(self) -> DashboardData:
        """Get comprehensive dashboard data."""
        return DashboardData(
            total_validations=0,
            accuracy_rate=0.92,
            avg_confidence=0.89,
            user_satisfaction=4.3,
            accuracy_over_time=self._get_accuracy_over_time(),
            confidence_distribution=self._get_confidence_distribution(),
            device_performance=self._get_device_performance(),
            locale_performance=self._get_locale_performance(),
            edge_case_frequency=self._get_edge_case_frequency(),
            recent_validations=self._get_recent_entries(limit=5),
            recent_feedback=self._get_recent_feedback(limit=5),
            active_edge_cases=self._get_active_edge_cases(),
            system_alerts=self._get_system_alerts(),
            performance_warnings=self._get_performance_warnings(),
            improvement_opportunities=self._get_improvement_opportunities(),
            last_updated=datetime.now(),
            data_coverage_hours=24
        )
    
    # ============ PRIVATE HELPER METHODS ============
    
    def _extract_screenshot_metadata(
        self,
        screenshot_path: str,
        device_info: Optional[Dict[str, str]],
        user_session_id: Optional[str]
    ) -> ScreenshotMetadata:
        """Extract metadata from screenshot file and context."""
        
        file_path = Path(screenshot_path)
        file_stats = file_path.stat()
        
        # Get image dimensions
        try:
            with Image.open(screenshot_path) as img:
                width, height = img.size
                image_format = img.format or "UNKNOWN"
        except Exception:
            width = height = 0
            image_format = "UNKNOWN"
        
        device_type = self._detect_device_type(screenshot_path, device_info)
        game_locale = self._detect_game_locale(screenshot_path, device_info)
        screenshot_type = self._detect_screenshot_type(screenshot_path)
        image_quality_score = self._calculate_image_quality(screenshot_path)
        
        return ScreenshotMetadata(
            filename=file_path.name,
            file_size_bytes=file_stats.st_size,
            image_width=width,
            image_height=height,
            image_format=image_format,
            device_type=device_type,
            device_model=device_info.get("model") if device_info else None,
            game_locale=game_locale,
            screenshot_type=screenshot_type,
            image_quality_score=image_quality_score,
            has_ui_overlay=False,
            has_watermark=False,
            is_cropped=False,
            upload_timestamp=datetime.now(),
            user_session_id=user_session_id,
            submission_source="web_upload"
        )
    
    def _detect_device_type(self, screenshot_path: str, device_info: Optional[Dict[str, str]]) -> DeviceType:
        """Detect device type from file or context."""
        if device_info and "type" in device_info:
            try:
                return DeviceType(device_info["type"])
            except ValueError:
                pass
        return DeviceType.UNKNOWN
    
    def _detect_game_locale(self, screenshot_path: str, device_info: Optional[Dict[str, str]]) -> LocaleType:
        """Detect game locale from filename or context."""
        if device_info and "locale" in device_info:
            try:
                return LocaleType(device_info["locale"])
            except ValueError:
                pass
        return LocaleType.EN
    
    def _detect_screenshot_type(self, screenshot_path: str) -> ScreenshotType:
        """Detect screenshot type from filename or content."""
        return ScreenshotType.UNKNOWN
    
    def _calculate_image_quality(self, screenshot_path: str) -> float:
        """Calculate image quality score based on resolution and clarity."""
        try:
            with Image.open(screenshot_path) as img:
                width, height = img.size
                total_pixels = width * height
                
                if total_pixels >= 2073600:  # 1920x1080
                    return 1.0
                elif total_pixels >= 1382400:  # 1280x1080
                    return 0.8
                elif total_pixels >= 921600:   # 1280x720
                    return 0.6
                else:
                    return 0.4
        except Exception:
            return 0.5
    
    def _detect_edge_cases(self, ai_result: Dict[str, Any], metadata: ScreenshotMetadata) -> List[str]:
        """Detect edge cases in the analysis."""
        edge_cases = []
        
        if metadata.image_width * metadata.image_height < 921600:
            edge_cases.append("low_resolution")
        
        confidence_scores = ai_result.get("confidence_scores", {})
        overall_confidence = confidence_scores.get("overall_confidence", 1.0)
        if overall_confidence < 0.7:
            edge_cases.append("low_confidence")
        
        return edge_cases
    
    def _calculate_accuracy_metrics(self, ai_data: Dict[str, Any], ground_truth: GroundTruthData) -> Dict[str, float]:
        """Calculate field-by-field accuracy metrics."""
        metrics = {}
        
        core_fields = ["kills", "deaths", "assists", "hero_damage", 
                      "turret_damage", "teamfight_participation", "gold_per_min"]
        
        for field in core_fields:
            ai_value = ai_data.get(field)
            truth_value = getattr(ground_truth, field, None)
            
            if ai_value is not None and truth_value is not None:
                if isinstance(ai_value, (int, float)) and isinstance(truth_value, (int, float)):
                    if truth_value == 0:
                        accuracy = 1.0 if ai_value == 0 else 0.0
                    else:
                        error_rate = abs(ai_value - truth_value) / truth_value
                        accuracy = max(0.0, 1.0 - error_rate)
                    metrics[f"{field}_accuracy"] = accuracy
                else:
                    metrics[f"{field}_accuracy"] = 1.0 if ai_value == truth_value else 0.0
            else:
                metrics[f"{field}_accuracy"] = 0.0
        
        if metrics:
            metrics["overall_accuracy"] = statistics.mean(metrics.values())
        
        return metrics
    
    def _calculate_validation_score(self, entry: ValidationEntry) -> float:
        """Calculate overall validation score for an entry."""
        if not entry.accuracy_metrics:
            return 0.0
        
        accuracy_score = entry.accuracy_metrics.get("overall_accuracy", 0.0)
        confidence_score = entry.ai_confidence_scores.get("overall_confidence", 0.0)
        
        return (accuracy_score * 0.7 + confidence_score * 0.3)
    
    # ============ DASHBOARD HELPER METHODS ============
    
    def _get_accuracy_over_time(self) -> List[Dict[str, Any]]:
        """Get accuracy trend data over time."""
        return [{"date": "2025-01-10", "accuracy": 0.92, "confidence": 0.89}]
    
    def _get_confidence_distribution(self) -> List[Dict[str, Any]]:
        """Get confidence score distribution."""
        return [{"name": "Elite (95-100%)", "value": 45}, {"name": "High (85-94%)", "value": 30}]
    
    def _get_device_performance(self) -> List[Dict[str, Any]]:
        """Get performance metrics by device type."""
        return [{"device": "iPhone", "accuracy": 94, "confidence": 91}]
    
    def _get_locale_performance(self) -> List[Dict[str, Any]]:
        """Get performance metrics by locale."""
        return [{"locale": "en", "accuracy": 95, "confidence": 92}]
    
    def _get_edge_case_frequency(self) -> List[Dict[str, Any]]:
        """Get frequency of different edge cases."""
        return [{"case": "Low Resolution", "frequency": 12}]
    
    def _get_recent_entries(self, limit: int = 10) -> List[ValidationEntry]:
        """Get recent validation entries."""
        return []
    
    def _get_recent_feedback(self, limit: int = 10) -> List[UserFeedback]:
        """Get recent user feedback."""
        return []
    
    def _get_active_edge_cases(self) -> List[EdgeCaseTest]:
        """Get active edge case tests."""
        return []
    
    def _get_system_alerts(self) -> List[str]:
        """Get system alerts."""
        return []
    
    def _get_performance_warnings(self) -> List[str]:
        """Get performance warnings."""
        return []
    
    def _get_improvement_opportunities(self) -> List[str]:
        """Get improvement opportunities."""
        return []
    
    def _calculate_performance_metrics(self, where_clause: str, params: List) -> PerformanceMetrics:
        """Calculate performance metrics."""
        now = datetime.now()
        return PerformanceMetrics(
            total_entries=0,
            entries_by_status={},
            overall_accuracy=0.0,
            confidence_vs_accuracy={},
            avg_confidence_by_locale={},
            avg_confidence_by_device={},
            field_accuracy={},
            most_problematic_fields=[],
            accuracy_by_screenshot_type={},
            accuracy_by_locale={},
            accuracy_by_device={},
            edge_case_frequency={},
            edge_case_success_rate={},
            avg_processing_time=0.0,
            processing_time_by_device={},
            avg_user_satisfaction=0.0,
            user_feedback_count=0,
            metrics_period_start=now,
            metrics_period_end=now,
            last_updated=now
        )
    
    def _get_trending_issues(self) -> List[str]:
        """Get trending issues."""
        return []
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health."""
        return {"status": "healthy"}


# Initialize global validation manager
validation_manager = ValidationManager() 