import uuid
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ScreenshotType(Enum):
    """Types of screenshots we can analyze."""
    SCOREBOARD = "scoreboard"  # Post-match scoreboard with KDA
    STATS_PAGE = "stats_page"  # Detailed stats page
    UNKNOWN = "unknown"


@dataclass
class ScreenshotAnalysis:
    """Represents analysis results from a single screenshot."""
    screenshot_type: ScreenshotType
    raw_data: Dict[str, Any]
    confidence: float
    warnings: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AnalysisSession:
    """Represents a multi-screenshot analysis session."""
    session_id: str
    player_ign: str
    screenshots: List[ScreenshotAnalysis] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    is_complete: bool = False
    final_result: Optional[Dict[str, Any]] = None


class SessionManager:
    """Manages analysis sessions for multi-screenshot processing."""
    
    def __init__(self, session_timeout: int = 3600):  # 1 hour timeout
        self.sessions: Dict[str, AnalysisSession] = {}
        self.session_timeout = session_timeout
    
    def create_session(self, player_ign: str) -> str:
        """Create a new analysis session."""
        session_id = str(uuid.uuid4())
        session = AnalysisSession(
            session_id=session_id,
            player_ign=player_ign
        )
        self.sessions[session_id] = session
        logger.info(f"Created new session {session_id} for player {player_ign}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[AnalysisSession]:
        """Get an existing session by ID."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if session has expired
        if time.time() - session.last_updated > self.session_timeout:
            logger.info(f"Session {session_id} expired, removing")
            del self.sessions[session_id]
            return None
        
        return session
    
    def add_screenshot_analysis(
        self, 
        session_id: str, 
        analysis: ScreenshotAnalysis
    ) -> bool:
        """Add screenshot analysis to a session."""
        session = self.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return False
        
        session.screenshots.append(analysis)
        session.last_updated = time.time()
        
        logger.info(
            f"Added {analysis.screenshot_type.value} analysis to session {session_id}. "
            f"Total screenshots: {len(session.screenshots)}"
        )
        
        # Check if we can compile final result
        self._try_compile_result(session)
        return True
    
    def _try_compile_result(self, session: AnalysisSession):
        """Try to compile final result if we have enough data."""
        # Enhanced logic: wait for multiple screenshots OR very high confidence
        screenshot_count = len(session.screenshots)
        if screenshot_count > 0:
            best_analysis = max(session.screenshots, key=lambda x: x.confidence)
            
            # Complete if: (2+ screenshots AND confidence > 70%) OR (1 screenshot AND confidence > 90%)
            should_complete = (
                (screenshot_count >= 2 and best_analysis.confidence > 0.7) or
                (screenshot_count >= 1 and best_analysis.confidence > 0.9)
            )
            
            if should_complete:
                session.final_result = self._merge_screenshot_data(session.screenshots)
                session.is_complete = True
                logger.info(f"Session {session.session_id} marked as complete with {screenshot_count} screenshots (confidence: {best_analysis.confidence:.1%})")
    
    def _merge_screenshot_data(self, analyses: List[ScreenshotAnalysis]) -> Dict[str, Any]:
        """Merge data from multiple screenshot analyses."""
        merged_data = {}
        all_warnings = []
        
        # Find the best analysis for each data type
        scoreboard_analyses = [a for a in analyses if a.screenshot_type == ScreenshotType.SCOREBOARD]
        stats_analyses = [a for a in analyses if a.screenshot_type == ScreenshotType.STATS_PAGE]
        
        # Start with the highest confidence analysis
        best_analysis = max(analyses, key=lambda x: x.confidence)
        merged_data.update(best_analysis.raw_data)
        all_warnings.extend(best_analysis.warnings)
        
        # Merge additional data from other analyses
        for analysis in analyses:
            if analysis != best_analysis:
                # Merge non-conflicting data
                for key, value in analysis.raw_data.items():
                    if key not in merged_data or merged_data[key] in [None, 0, "unknown"]:
                        merged_data[key] = value
                
                all_warnings.extend(analysis.warnings)
        
        # Add metadata about the merge
        merged_data["_session_info"] = {
            "screenshot_count": len(analyses),
            "screenshot_types": [a.screenshot_type.value for a in analyses],
            "avg_confidence": sum(a.confidence for a in analyses) / len(analyses),
            "all_warnings": all_warnings
        }
        
        return merged_data
    
    def get_final_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the final compiled result for a session."""
        session = self.get_session(session_id)
        if not session:
            return None
        
        return session.final_result
    
    def is_session_complete(self, session_id: str) -> bool:
        """Check if a session has a complete result."""
        session = self.get_session(session_id)
        return session.is_complete if session else False
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions from memory."""
        current_time = time.time()
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if current_time - session.last_updated > self.session_timeout
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up expired session {session_id}")
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def list_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        self.cleanup_expired_sessions()
        return list(self.sessions.keys())


# Global session manager instance
session_manager = SessionManager()