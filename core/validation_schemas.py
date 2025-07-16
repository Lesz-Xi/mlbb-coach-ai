"""
Real-User Validation Schemas for MLBB Coach AI

Comprehensive schemas for tracking:
- User feedback and validation data
- Ground truth annotations
- Performance metrics across devices/locales
- Edge case analysis and failure modes
- Confidence correlation analysis
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import uuid


class DeviceType(str, Enum):
    """Device types for categorizing validation data."""
    IPHONE = "iPhone"
    ANDROID = "Android"
    IPAD = "iPad"
    TABLET = "Tablet"
    EMULATOR = "Emulator"
    UNKNOWN = "Unknown"


class LocaleType(str, Enum):
    """Supported game locales."""
    EN = "en"  # English
    ID = "id"  # Indonesian
    TH = "th"  # Thai
    VN = "vn"  # Vietnamese
    MY = "my"  # Malay
    PH = "ph"  # Filipino
    UNKNOWN = "unknown"


class ScreenshotType(str, Enum):
    """Types of post-match screenshots."""
    KDA_OVERVIEW = "kda_overview"
    DAMAGE_STATS = "damage_stats"
    TEAMFIGHT_PARTICIPATION = "teamfight_participation"
    ECONOMY_STATS = "economy_stats"
    MIXED_LAYOUT = "mixed_layout"
    UNKNOWN = "unknown"


class ValidationStatus(str, Enum):
    """Status of validation entries."""
    PENDING = "pending"
    ANNOTATED = "annotated"
    VERIFIED = "verified"
    DISPUTED = "disputed"
    REJECTED = "rejected"


class ConfidenceCategory(str, Enum):
    """Confidence categories for analysis results."""
    ELITE = "elite"        # 95-100%
    HIGH = "high"          # 85-94%
    MEDIUM = "medium"      # 70-84%
    LOW = "low"           # 50-69%
    UNRELIABLE = "unreliable"  # <50%


# ============ GROUND TRUTH SCHEMAS ============

class GroundTruthData(BaseModel):
    """Manual annotation of screenshot data for ground truth."""
    # Player identification
    player_ign: str
    hero_played: str
    
    # Core stats
    kills: int = Field(ge=0)
    deaths: int = Field(ge=0)
    assists: int = Field(ge=0)
    
    # Performance metrics
    hero_damage: int = Field(ge=0)
    turret_damage: int = Field(ge=0)
    damage_taken: int = Field(ge=0)
    teamfight_participation: int = Field(ge=0, le=100)
    gold_per_min: int = Field(ge=0)
    
    # Match context
    match_duration_minutes: int = Field(ge=1)
    match_result: Literal["Victory", "Defeat"]
    game_mode: str = "Classic"
    
    # Additional context
    team_composition: List[str] = []
    enemy_composition: List[str] = []
    
    # Annotation metadata
    annotator_id: str
    annotation_confidence: float = Field(ge=0.0, le=1.0)
    annotation_notes: str = ""
    annotation_timestamp: datetime
    
    # Quality markers
    is_verified: bool = False
    verification_source: Optional[str] = None


class ScreenshotMetadata(BaseModel):
    """Metadata about the screenshot and device context."""
    # File information
    filename: str
    file_size_bytes: int
    image_width: int
    image_height: int
    image_format: str
    
    # Device/context information
    device_type: DeviceType
    device_model: Optional[str] = None
    game_locale: LocaleType
    screenshot_type: ScreenshotType
    
    # Quality indicators
    image_quality_score: float = Field(ge=0.0, le=1.0)
    has_ui_overlay: bool = False
    has_watermark: bool = False
    is_cropped: bool = False
    compression_level: Optional[str] = None
    
    # Upload context
    upload_timestamp: datetime
    user_session_id: Optional[str] = None
    submission_source: str = "web_upload"


# ============ VALIDATION FEEDBACK SCHEMAS ============

class UserFeedback(BaseModel):
    """User feedback on analysis accuracy."""
    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    validation_entry_id: str
    
    # Feedback details
    is_analysis_correct: bool
    incorrect_fields: List[str] = []
    user_corrections: Dict[str, Any] = {}
    
    # User assessment
    confidence_in_feedback: float = Field(ge=0.0, le=1.0)
    ease_of_use_rating: int = Field(ge=1, le=5)
    analysis_speed_rating: int = Field(ge=1, le=5)
    
    # Free-form feedback
    user_comments: str = ""
    suggested_improvements: str = ""
    
    # Submission metadata
    user_id: Optional[str] = None
    submission_timestamp: datetime
    ip_address: Optional[str] = None


class ValidationEntry(BaseModel):
    """Complete validation entry combining AI results with ground truth."""
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Source data
    screenshot_metadata: ScreenshotMetadata
    ground_truth: Optional[GroundTruthData] = None
    
    # AI analysis results
    ai_parsed_data: Dict[str, Any]
    ai_confidence_scores: Dict[str, float]
    ai_processing_time: float
    ai_warnings: List[str] = []
    ai_errors: List[str] = []
    
    # Validation status
    status: ValidationStatus = ValidationStatus.PENDING
    validation_score: Optional[float] = None
    validation_notes: str = ""
    
    # User feedback
    user_feedback: List[UserFeedback] = []
    
    # Performance analysis
    accuracy_metrics: Optional[Dict[str, float]] = None
    edge_case_flags: List[str] = []
    
    # Timestamps
    created_at: datetime
    updated_at: Optional[datetime] = None
    validated_at: Optional[datetime] = None


# ============ METRICS AND ANALYTICS SCHEMAS ============

class PerformanceMetrics(BaseModel):
    """Performance metrics for validation analysis."""
    # Overall metrics
    total_entries: int
    entries_by_status: Dict[ValidationStatus, int]
    overall_accuracy: float = Field(ge=0.0, le=1.0)
    
    # Confidence correlation
    confidence_vs_accuracy: Dict[ConfidenceCategory, float]
    avg_confidence_by_locale: Dict[LocaleType, float]
    avg_confidence_by_device: Dict[DeviceType, float]
    
    # Field-specific accuracy
    field_accuracy: Dict[str, float]
    most_problematic_fields: List[str]
    
    # Performance by context
    accuracy_by_screenshot_type: Dict[ScreenshotType, float]
    accuracy_by_locale: Dict[LocaleType, float]
    accuracy_by_device: Dict[DeviceType, float]
    
    # Edge cases
    edge_case_frequency: Dict[str, int]
    edge_case_success_rate: Dict[str, float]
    
    # Processing metrics
    avg_processing_time: float
    processing_time_by_device: Dict[DeviceType, float]
    
    # User experience
    avg_user_satisfaction: float = Field(ge=0.0, le=5.0)
    user_feedback_count: int
    
    # Timestamps
    metrics_period_start: datetime
    metrics_period_end: datetime
    last_updated: datetime


class EdgeCaseTest(BaseModel):
    """Definition and results for edge case testing."""
    test_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    test_name: str
    test_description: str
    # Categories: "resolution", "locale", "device", "ui_variation", etc.
    test_category: str
    
    # Test parameters
    test_conditions: Dict[str, Any]
    expected_behavior: str
    
    # Test results
    test_status: Literal["pending", "running", "passed", "failed", "error"]
    success_rate: Optional[float] = None
    failure_modes: List[str] = []
    
    # Performance data
    avg_confidence_score: Optional[float] = None
    processing_time_impact: Optional[float] = None
    
    # Test execution
    last_run: Optional[datetime] = None
    run_count: int = 0
    automated: bool = False


class ValidationBatchRequest(BaseModel):
    """Request for batch validation processing."""
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    batch_name: str
    screenshots: List[str]  # List of file paths or URLs
    
    # Batch configuration
    auto_annotate: bool = False
    priority_level: int = Field(ge=1, le=5, default=3)
    expected_locale: Optional[LocaleType] = None
    expected_device_type: Optional[DeviceType] = None
    
    # Submission metadata
    submitted_by: Optional[str] = None
    submission_timestamp: datetime


class ValidationReport(BaseModel):
    """Comprehensive validation report."""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    report_type: Literal["daily", "weekly", "monthly", "custom"]
    
    # Report data
    performance_metrics: PerformanceMetrics
    top_issues: List[str]
    improvement_recommendations: List[str]
    confidence_trends: Dict[str, List[float]]
    
    # Edge case analysis
    edge_case_summary: Dict[str, Any]
    new_failure_modes: List[str]
    resolved_issues: List[str]
    
    # User feedback summary
    user_satisfaction_trend: List[float]
    common_user_complaints: List[str]
    feature_requests: List[str]
    
    # Report metadata
    generated_at: datetime
    report_period: Dict[str, datetime]
    total_data_points: int


# ============ API REQUEST/RESPONSE SCHEMAS ============

class FeedbackSubmissionRequest(BaseModel):
    """Request schema for submitting user feedback."""
    validation_entry_id: str
    feedback: UserFeedback


class AnnotationRequest(BaseModel):
    """Request schema for manual annotation."""
    entry_id: str
    ground_truth: GroundTruthData
    annotator_notes: str = ""


class ValidationStatsRequest(BaseModel):
    """Request schema for validation statistics."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    device_filter: Optional[List[DeviceType]] = None
    locale_filter: Optional[List[LocaleType]] = None
    status_filter: Optional[List[ValidationStatus]] = None


class ValidationStatsResponse(BaseModel):
    """Response schema for validation statistics."""
    metrics: PerformanceMetrics
    recent_entries: List[ValidationEntry]
    trending_issues: List[str]
    system_health: Dict[str, Any]


# ============ DASHBOARD DATA SCHEMAS ============

class DashboardData(BaseModel):
    """Complete dashboard data structure."""
    # Summary cards
    total_validations: int
    accuracy_rate: float
    avg_confidence: float
    user_satisfaction: float
    
    # Charts data
    accuracy_over_time: List[Dict[str, Any]]
    confidence_distribution: List[Dict[str, Any]]
    device_performance: List[Dict[str, Any]]
    locale_performance: List[Dict[str, Any]]
    edge_case_frequency: List[Dict[str, Any]]
    
    # Recent activity
    recent_validations: List[ValidationEntry]
    recent_feedback: List[UserFeedback]
    active_edge_cases: List[EdgeCaseTest]
    
    # Alerts and notifications
    system_alerts: List[str]
    performance_warnings: List[str]
    improvement_opportunities: List[str]
    
    # Data freshness
    last_updated: datetime
    data_coverage_hours: int 