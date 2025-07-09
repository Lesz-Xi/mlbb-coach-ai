import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np

from .data_collector import DataCollector, get_ocr_reader
from .session_manager import SessionManager, ScreenshotAnalysis, ScreenshotType, session_manager
from .screenshot_analyzer import screenshot_analyzer
from .hero_identifier import hero_identifier
from .advanced_hero_detector import advanced_hero_detector
from .ign_validator import IGNValidator

logger = logging.getLogger(__name__)


class EnhancedDataCollector(DataCollector):
    """
    Enhanced data collector with session management and improved analysis.
    Supports multi-screenshot processing and advanced hero identification.
    """
    
    def __init__(self):
        super().__init__()
        self.session_manager = session_manager
        self.ign_validator = IGNValidator()
    
    def analyze_screenshot_with_session(
        self,
        image_path: str,
        ign: str,
        session_id: Optional[str] = None,
        hero_override: Optional[str] = None,
        known_igns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze screenshot with session management for multi-screenshot processing.
        
        Args:
            image_path: Path to screenshot
            ign: Player's in-game name
            session_id: Existing session ID (optional)
            hero_override: Manual hero specification (optional)
            known_igns: List of known IGNs (optional)
            
        Returns:
            Dictionary with analysis results and session info
        """
        try:
            # Create or get session
            if session_id:
                session = self.session_manager.get_session(session_id)
                if not session:
                    logger.warning(f"Session {session_id} not found, creating new session")
                    session_id = self.session_manager.create_session(ign)
            else:
                session_id = self.session_manager.create_session(ign)
            
            # Analyze screenshot type
            screenshot_type, type_confidence, detected_keywords = (
                screenshot_analyzer.analyze_screenshot_type(image_path)
            )
            
            logger.info(
                f"Screenshot analysis: type={screenshot_type.value}, "
                f"confidence={type_confidence:.3f}, keywords={detected_keywords[:3]}"
            )
            
            # Perform enhanced OCR analysis
            analysis_result = self._enhanced_screenshot_analysis(
                image_path, ign, screenshot_type, hero_override, known_igns
            )
            
            # Create screenshot analysis object
            screenshot_analysis = ScreenshotAnalysis(
                screenshot_type=screenshot_type,
                raw_data=analysis_result.get("data", {}),
                confidence=analysis_result.get("overall_confidence", type_confidence),
                warnings=analysis_result.get("warnings", [])
            )
            
            # Add to session
            self.session_manager.add_screenshot_analysis(session_id, screenshot_analysis)
            
            # Check if we have a complete result
            session = self.session_manager.get_session(session_id)
            if session and session.is_complete:
                final_result = session.final_result
                logger.info(f"Session {session_id} complete with final result")
            else:
                final_result = analysis_result.get("data", {})
            
            return {
                "data": final_result,
                "session_id": session_id,
                "screenshot_type": screenshot_type.value,
                "type_confidence": type_confidence,
                "session_complete": session.is_complete if session else False,
                "warnings": analysis_result.get("warnings", []),
                "debug_info": {
                    "detected_keywords": detected_keywords,
                    "screenshot_count": len(session.screenshots) if session else 1,
                    "hero_suggestions": analysis_result.get("hero_suggestions", []),
                    "hero_debug": analysis_result.get("hero_debug", {})
                }
            }
            
        except Exception as e:
            logger.error(f"Enhanced screenshot analysis failed: {str(e)}")
            return {
                "data": {},
                "session_id": session_id,
                "error": str(e),
                "warnings": [f"Analysis failed: {str(e)}"]
            }
    
    def _enhanced_screenshot_analysis(
        self,
        image_path: str,
        ign: str,
        screenshot_type: ScreenshotType,
        hero_override: Optional[str] = None,
        known_igns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Enhanced analysis using new systems."""
        logger.info(f"Starting enhanced analysis for {screenshot_type.value} screenshot")
        
        # Get OCR results
        reader = get_ocr_reader()
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ocr_results = reader.readtext(gray, detail=1)
        
        warnings = []
        
        # Enhanced IGN validation
        validated_ign = self._validate_ign_enhanced(ign, ocr_results, warnings)
        
        # Advanced hero identification
        hero_name, hero_confidence, hero_debug = advanced_hero_detector.detect_hero_comprehensive(
            image_path, validated_ign, hero_override
        )
        
        logger.info(f"Hero identified: {hero_name} (confidence: {hero_confidence:.3f})")
        logger.info(f"Detection strategies used: {hero_debug.get('strategies_tried', [])}")
        
        # Extract hero suggestions from debug info
        hero_suggestions = hero_debug.get("hero_suggestions", [])
        
        if hero_confidence < 0.7 and hero_name != "unknown":
            warnings.append(f"Low hero identification confidence: {hero_confidence:.3f}")
            if hero_suggestions:
                top_suggestion = hero_suggestions[0]
                warnings.append(f"Top suggestion: {top_suggestion[0]} ({top_suggestion[1]:.3f})")
        
        # If hero is still unknown, provide helpful guidance
        if hero_name == "unknown":
            warnings.append("Hero could not be identified from screenshot")
            if hero_suggestions:
                top_3 = hero_suggestions[:3]
                suggestion_text = ", ".join([f"{h} ({c:.1%})" for h, c in top_3])
                warnings.append(f"Possible heroes: {suggestion_text}")
            warnings.append("Tip: Use manual hero override for better analysis")
        
        # Extract data based on screenshot type
        if screenshot_type == ScreenshotType.SCOREBOARD:
            parsed_data = self._parse_scoreboard_enhanced(validated_ign, ocr_results, warnings)
        elif screenshot_type == ScreenshotType.STATS_PAGE:
            parsed_data = self._parse_stats_page_enhanced(validated_ign, ocr_results, warnings)
        else:
            # Fallback to original parsing method
            parsed_data = self._parse_player_row(validated_ign, ocr_results, hero_override)
            if not parsed_data:
                warnings.append("Fallback parsing also failed")
        
        # Set hero name
        parsed_data["hero"] = hero_name
        
        # Add default values
        self._add_default_values(parsed_data)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            hero_confidence, len(warnings), len(parsed_data)
        )
        
        return {
            "data": parsed_data,
            "overall_confidence": overall_confidence,
            "warnings": warnings,
            "hero_suggestions": hero_suggestions,
            "hero_debug": hero_debug
        }
    
    def _validate_ign_enhanced(self, ign: str, ocr_results: List, warnings: List[str]) -> str:
        """Enhanced IGN validation with better matching."""
        for bbox, text, conf in ocr_results:
            validation_result = self.ign_validator.validate_mlbb_ign(text)
            if validation_result['is_valid']:
                # Use fuzzy matching for IGN comparison
                from fuzzywuzzy import fuzz
                similarity = fuzz.ratio(ign.lower(), validation_result['cleaned_ign'].lower())
                
                if similarity >= 80:  # 80% similarity threshold
                    logger.info(f"IGN validated: {validation_result['cleaned_ign']} (similarity: {similarity}%)")
                    if similarity < 95:
                        warnings.append(f"IGN similarity: {similarity}% - please verify")
                    return validation_result['cleaned_ign']
        
        warnings.append(f"IGN '{ign}' not found with high confidence in screenshot")
        return ign
    
    def _parse_scoreboard_enhanced(
        self, ign: str, ocr_results: List, warnings: List[str]
    ) -> Dict[str, Any]:
        """Enhanced parsing specifically for scoreboard screenshots."""
        logger.info("Using enhanced scoreboard parsing")
        
        # Find player row
        parsed_data = self._parse_player_row(ign, ocr_results)
        
        if not parsed_data:
            warnings.append("Could not locate player in scoreboard")
            return {}
        
        # Extract match duration and result
        text_content = " ".join([result[1] for result in ocr_results])
        
        # Match duration
        import re
        duration_match = re.search(r"duration\s+(\d{2}):(\d{2})", text_content.lower())
        if duration_match:
            minutes = int(duration_match.group(1))
            seconds = int(duration_match.group(2))
            parsed_data["match_duration"] = minutes + seconds / 60
        else:
            parsed_data["match_duration"] = 10  # Default
            warnings.append("Could not detect match duration")
        
        # Match result
        if "victory" in text_content.lower():
            parsed_data["match_result"] = "victory"
        elif "defeat" in text_content.lower():
            parsed_data["match_result"] = "defeat"
        
        return parsed_data
    
    def _parse_stats_page_enhanced(
        self, ign: str, ocr_results: List, warnings: List[str]
    ) -> Dict[str, Any]:
        """Enhanced parsing for detailed stats page screenshots."""
        logger.info("Using enhanced stats page parsing")
        
        parsed_data = {}
        text_content = " ".join([result[1] for result in ocr_results])
        
        # Extract damage values
        import re
        damage_values = [int(match) for match in re.findall(r"\b(\d{4,7})\b", text_content)]
        damage_values.sort(reverse=True)
        
        if len(damage_values) >= 1:
            parsed_data["hero_damage"] = damage_values[0]
        if len(damage_values) >= 2:
            parsed_data["damage_taken"] = damage_values[1]
        if len(damage_values) >= 3:
            parsed_data["turret_damage"] = damage_values[2]
        
        # Extract percentage values (teamfight participation)
        percentages = [int(match) for match in re.findall(r"\b(\d{1,3})%", text_content)]
        if percentages:
            parsed_data["teamfight_participation"] = max(percentages)
        
        # Try to find KDA if present
        kda_match = re.search(r"\b(\d{1,2})\s*/?\s*(\d{1,2})\s*/?\s*(\d{1,2})\b", text_content)
        if kda_match:
            parsed_data["kills"] = int(kda_match.group(1))
            parsed_data["deaths"] = max(1, int(kda_match.group(2)))
            parsed_data["assists"] = int(kda_match.group(3))
        
        return parsed_data
    
    def _add_default_values(self, parsed_data: Dict[str, Any]):
        """Add default values for missing fields."""
        defaults = {
            "kills": 0,
            "deaths": 1,
            "assists": 0,
            "gold": 0,
            "hero_damage": 0,
            "turret_damage": 0,
            "damage_taken": 0,
            "teamfight_participation": 0,
            "positioning_rating": "average",
            "ult_usage": "average",
            "match_duration": 10,
            "gold_per_min": 0
        }
        
        for key, value in defaults.items():
            parsed_data.setdefault(key, value)
        
        # Calculate GPM if we have gold and duration
        if parsed_data.get("gold", 0) > 0 and parsed_data.get("match_duration", 0) > 0:
            gpm = parsed_data["gold"] / parsed_data["match_duration"]
            parsed_data["gold_per_min"] = round(gpm)
    
    def _calculate_overall_confidence(
        self, hero_confidence: float, warning_count: int, data_field_count: int
    ) -> float:
        """Calculate overall confidence score."""
        # Start with hero confidence
        confidence = hero_confidence * 0.4
        
        # Add data completeness factor
        data_completeness = min(1.0, data_field_count / 10)  # Assume 10 is "complete"
        confidence += data_completeness * 0.4
        
        # Subtract for warnings
        warning_penalty = min(0.3, warning_count * 0.05)
        confidence -= warning_penalty
        
        # Add base confidence
        confidence += 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session."""
        session = self.session_manager.get_session(session_id)
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "player_ign": session.player_ign,
            "screenshot_count": len(session.screenshots),
            "is_complete": session.is_complete,
            "created_at": session.created_at,
            "last_updated": session.last_updated,
            "screenshot_types": [s.screenshot_type.value for s in session.screenshots]
        }


# Global enhanced data collector instance
enhanced_data_collector = EnhancedDataCollector()