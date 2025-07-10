import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import re
from difflib import SequenceMatcher

from .data_collector import DataCollector, get_ocr_reader
from .session_manager import (
    ScreenshotAnalysis, ScreenshotType, session_manager
)
from .screenshot_analyzer import screenshot_analyzer
from .advanced_hero_detector import advanced_hero_detector
from .ign_validator import IGNValidator
from .diagnostic_logger import diagnostic_logger
from .trophy_medal_detector_v2 import improved_trophy_medal_detector as trophy_medal_detector

logger = logging.getLogger(__name__)


class RobustIGNMatcher:
    """Advanced IGN matching with multiple fallback strategies."""
    
    def __init__(self):
        self.min_similarity = 0.6  # Minimum similarity threshold
        
    def find_ign_in_ocr(self, target_ign: str, ocr_results: List) -> Dict[str, Any]:
        """Find IGN using multiple robust strategies."""
        strategies = [
            ("exact_match", self._exact_match),
            ("case_insensitive", self._case_insensitive_match),
            ("fuzzy_match", self._fuzzy_match),
            ("partial_match", self._partial_match),
            ("cleaned_match", self._cleaned_match),
            ("token_match", self._token_match)
        ]
        
        best_result = {"found": False, "confidence": 0.0, "strategy": None, "bbox": None, "text": None}
        
        for strategy_name, strategy_func in strategies:
            result = strategy_func(target_ign, ocr_results)
            if result["found"] and result["confidence"] > best_result["confidence"]:
                result["strategy"] = strategy_name
                best_result = result
                
                # If we get a high-confidence match, return immediately
                if result["confidence"] >= 0.95:
                    break
        
        logger.debug(f"IGN matching result: {best_result['strategy']} "
                    f"confidence={best_result['confidence']:.3f}")
        return best_result
    
    def _exact_match(self, target_ign: str, ocr_results: List) -> Dict[str, Any]:
        """Exact string matching."""
        for bbox, text, conf in ocr_results:
            if target_ign == text:
                return {"found": True, "confidence": 1.0, "bbox": bbox, "text": text}
        return {"found": False, "confidence": 0.0}
    
    def _case_insensitive_match(self, target_ign: str, ocr_results: List) -> Dict[str, Any]:
        """Case-insensitive matching."""
        target_lower = target_ign.lower()
        for bbox, text, conf in ocr_results:
            if target_lower == text.lower():
                return {"found": True, "confidence": 0.95, "bbox": bbox, "text": text}
        return {"found": False, "confidence": 0.0}
    
    def _fuzzy_match(self, target_ign: str, ocr_results: List) -> Dict[str, Any]:
        """Fuzzy string matching using sequence similarity."""
        best_match = {"found": False, "confidence": 0.0}
        target_lower = target_ign.lower()
        
        for bbox, text, conf in ocr_results:
            # Calculate similarity ratio
            similarity = SequenceMatcher(None, target_lower, text.lower()).ratio()
            
            if similarity >= self.min_similarity and similarity > best_match["confidence"]:
                best_match = {
                    "found": True,
                    "confidence": similarity * 0.9,  # Slight penalty for fuzzy match
                    "bbox": bbox,
                    "text": text
                }
        
        return best_match
    
    def _partial_match(self, target_ign: str, ocr_results: List) -> Dict[str, Any]:
        """Partial matching - IGN contains or is contained in text."""
        target_lower = target_ign.lower()
        best_match = {"found": False, "confidence": 0.0}
        
        for bbox, text, conf in ocr_results:
            text_lower = text.lower()
            
            # Check if target is in text or text is in target
            if target_lower in text_lower:
                confidence = len(target_lower) / len(text_lower) * 0.8
            elif text_lower in target_lower:
                confidence = len(text_lower) / len(target_lower) * 0.8
            else:
                continue
                
            if confidence >= self.min_similarity and confidence > best_match["confidence"]:
                best_match = {
                    "found": True,
                    "confidence": confidence,
                    "bbox": bbox,
                    "text": text
                }
        
        return best_match
    
    def _cleaned_match(self, target_ign: str, ocr_results: List) -> Dict[str, Any]:
        """Matching after cleaning common OCR artifacts."""
        def clean_text(text):
            # Remove common OCR artifacts and normalize
            cleaned = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
            return cleaned.strip().lower()
        
        target_cleaned = clean_text(target_ign)
        best_match = {"found": False, "confidence": 0.0}
        
        for bbox, text, conf in ocr_results:
            text_cleaned = clean_text(text)
            
            if target_cleaned == text_cleaned:
                return {"found": True, "confidence": 0.85, "bbox": bbox, "text": text}
            
            # Fuzzy match on cleaned text
            similarity = SequenceMatcher(None, target_cleaned, text_cleaned).ratio()
            if similarity >= self.min_similarity and similarity > best_match["confidence"]:
                best_match = {
                    "found": True,
                    "confidence": similarity * 0.8,  # Penalty for cleaned match
                    "bbox": bbox,
                    "text": text
                }
        
        return best_match
    
    def _token_match(self, target_ign: str, ocr_results: List) -> Dict[str, Any]:
        """Token-based matching for multi-word IGNs."""
        target_tokens = set(target_ign.lower().split())
        if len(target_tokens) < 2:
            return {"found": False, "confidence": 0.0}
        
        best_match = {"found": False, "confidence": 0.0}
        
        for bbox, text, conf in ocr_results:
            text_tokens = set(text.lower().split())
            
            # Calculate token overlap
            common_tokens = target_tokens.intersection(text_tokens)
            if not common_tokens:
                continue
                
            # Calculate confidence based on token overlap
            overlap_ratio = len(common_tokens) / len(target_tokens)
            if overlap_ratio >= 0.5:  # At least half the tokens match
                confidence = overlap_ratio * 0.7  # Penalty for token-based match
                
                if confidence > best_match["confidence"]:
                    best_match = {
                        "found": True,
                        "confidence": confidence,
                        "bbox": bbox,
                        "text": text
                    }
        
        return best_match


class AnchorBasedLayoutParser:
    """Layout-aware parser using UI anchors for robust data extraction."""
    
    def __init__(self):
        self.ui_anchors = {
            # Match result anchors
            'DEFEAT': {'patterns': ['defeat', 'loss', 'lose'], 'region': (0.2, 0.0, 0.8, 0.4)},
            'VICTORY': {'patterns': ['victory', 'win', 'won'], 'region': (0.2, 0.0, 0.8, 0.4)},
            
            # Column header anchors
            'DURATION': {'patterns': ['duration', 'time'], 'region': (0.0, 0.0, 1.0, 0.4)},
            'KDA_HEADER': {'patterns': ['k/d/a', 'kda', 'kills'], 'identifies_column': True},
            'GOLD_HEADER': {'patterns': ['gold'], 'identifies_column': True},
            'DAMAGE_HEADER': {'patterns': ['damage', 'dmg'], 'identifies_column': True},
            
            # UI elements that help identify layout
            'PLAYER_LIST': {'patterns': ['blue', 'red', 'team'], 'region': (0.0, 0.3, 1.0, 0.9)},
        }
        
        self.column_tolerance = 50  # Pixels tolerance for column alignment
        
    def detect_anchors(self, ocr_results: List, image_shape: Tuple[int, int]) -> Dict[str, Dict]:
        """Detect UI anchors in the OCR results."""
        anchors = {}
        height, width = image_shape[:2]
        
        for bbox, text, conf in ocr_results:
            text_lower = text.lower().strip()
            if not text_lower:
                continue
                
            # Calculate text position
            text_center_x = sum(point[0] for point in bbox) / 4
            text_center_y = sum(point[1] for point in bbox) / 4
            
            # Normalize coordinates
            norm_x = text_center_x / width
            norm_y = text_center_y / height
            
            # Check each anchor type
            for anchor_name, anchor_info in self.ui_anchors.items():
                for pattern in anchor_info['patterns']:
                    if pattern in text_lower:
                        # Check if position matches expected region (if specified)
                        if 'region' in anchor_info:
                            x1, y1, x2, y2 = anchor_info['region']
                            if not (x1 <= norm_x <= x2 and y1 <= norm_y <= y2):
                                continue
                        
                        anchors[anchor_name] = {
                            'bbox': bbox,
                            'text': text,
                            'confidence': conf,
                            'position': (text_center_x, text_center_y),
                            'normalized_position': (norm_x, norm_y)
                        }
                        logger.debug(f"Found anchor {anchor_name}: '{text}' at ({norm_x:.2f}, {norm_y:.2f})")
                        break
        
        return anchors
    
    def identify_columns(self, ocr_results: List, anchors: Dict) -> Dict[str, List]:
        """Identify table columns based on header anchors and spatial alignment."""
        columns = {}
        
        # Find column headers
        column_headers = {}
        for anchor_name, anchor_data in anchors.items():
            if self.ui_anchors.get(anchor_name, {}).get('identifies_column'):
                header_x = anchor_data['position'][0]
                column_headers[anchor_name] = {
                    'x_position': header_x,
                    'column_data': []
                }
        
        # Group OCR results by column alignment
        for bbox, text, conf in ocr_results:
            text_center_x = sum(point[0] for point in bbox) / 4
            
            # Find which column this text belongs to
            for column_name, column_info in column_headers.items():
                column_x = column_info['x_position']
                if abs(text_center_x - column_x) <= self.column_tolerance:
                    column_info['column_data'].append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': conf,
                        'position': (text_center_x, sum(point[1] for point in bbox) / 4)
                    })
                    break
        
        # Sort column data by vertical position
        for column_name, column_info in column_headers.items():
            column_info['column_data'].sort(key=lambda x: x['position'][1])
            columns[column_name] = column_info['column_data']
        
        return columns
    
    def extract_player_row_data(self, ign: str, ocr_results: List, anchors: Dict, columns: Dict) -> Dict[str, Any]:
        """Extract player data using anchor-based layout understanding."""
        parsed_data = {}
        
        # First, find the player's row by IGN
        player_row_y = None
        ign_matcher = RobustIGNMatcher()
        ign_result = ign_matcher.find_ign_in_ocr(ign, ocr_results)
        
        if ign_result["found"]:
            ign_bbox = ign_result["bbox"]
            player_row_y = sum(point[1] for point in ign_bbox) / 4
            logger.info(f"Found player row at y={player_row_y:.1f}")
        else:
            logger.warning("Could not locate player IGN for row-based extraction")
            return self._fallback_extraction(ocr_results)
        
        # ENHANCED: Direct row extraction when columns are not identified
        if not columns:
            logger.info("No columns identified, using direct row extraction")
            parsed_data = self._extract_from_player_row_direct(ocr_results, player_row_y)
        else:
            # Extract data from each column at the player's row level
            row_tolerance = 50  # Increased vertical tolerance for same row alignment
            
            for column_name, column_data in columns.items():
                for item in column_data:
                    item_y = item['position'][1]
                    if abs(item_y - player_row_y) <= row_tolerance:
                        # This item is in the player's row
                        text = item['text']
                        
                        if column_name == 'KDA_HEADER':
                            self._extract_kda_from_text(text, parsed_data)
                        elif column_name == 'GOLD_HEADER':
                            self._extract_gold_from_text(text, parsed_data)
                        elif column_name == 'DAMAGE_HEADER':
                            self._extract_damage_from_text(text, parsed_data)
                        
                        break
        
        # Extract match-level data from anchors
        if 'VICTORY' in anchors:
            parsed_data['match_result'] = 'victory'
        elif 'DEFEAT' in anchors:
            parsed_data['match_result'] = 'defeat'
        
        if 'DURATION' in anchors:
            duration_text = anchors['DURATION']['text']
            self._extract_duration_from_text(duration_text, parsed_data)
        
        return parsed_data
    
    def _extract_from_player_row_direct(self, ocr_results: List, player_row_y: float) -> Dict[str, Any]:
        """Direct extraction from player row when column detection fails."""
        parsed_data = {}
        row_tolerance = 50  # Further increased tolerance for better row detection
        
        # Collect all text and bounding boxes in the player's row with spatial information
        row_items = []
        
        for bbox, text, conf in ocr_results:
            # Calculate center Y coordinate more accurately
            bbox_points = bbox if isinstance(bbox, list) else bbox.tolist()
            y_coords = [point[1] for point in bbox_points]
            text_y = sum(y_coords) / len(y_coords)
            
            if abs(text_y - player_row_y) <= row_tolerance:
                # Calculate center X coordinate for spatial ordering
                x_coords = [point[0] for point in bbox_points]
                text_x = sum(x_coords) / len(x_coords)
                
                row_items.append({
                    'text': text,
                    'x': text_x,
                    'y': text_y,
                    'confidence': conf,
                    'bbox': bbox
                })
        
        # Sort by X coordinate (left to right) for better field alignment
        row_items.sort(key=lambda item: item['x'])
        
        row_texts = [item['text'] for item in row_items]
        row_numbers = []
        
        # Extract numbers with their spatial positions
        import re
        for item in row_items:
            numbers = [(int(n), item['x']) for n in re.findall(r'\b(\d+)\b', item['text'])]
            row_numbers.extend(numbers)
        
        logger.debug(f"Direct row extraction - texts: {row_texts}")
        logger.debug(f"Direct row extraction - numbers: {[n[0] for n in row_numbers]}")
        
        # Extract KDA using pattern matching with improved spatial awareness
        combined_text = " ".join(row_texts)
        self._extract_kda_from_text_enhanced(combined_text, row_items, parsed_data)
        
        # Extract gold using simplified spatial positioning - Fixed threshold  
        gold_candidates = [(n, x) for n, x in row_numbers if 100 <= n <= 50000]
        if gold_candidates:
            # Sort by spatial position (X coordinate) for proper column alignment
            gold_candidates.sort(key=lambda item: item[1])  # Sort by X position
            
            # Use first gold candidate by spatial position (left to right)
            parsed_data['gold'] = gold_candidates[0][0]
            logger.debug(f"Extracted gold via spatial positioning: {parsed_data['gold']}")
        
        # Extract damage values using spatial positioning
        damage_candidates = [(n, x) for n, x in row_numbers if 1000 <= n <= 500000]
        if damage_candidates:
            # Sort by spatial position for column alignment
            damage_candidates.sort(key=lambda item: item[1])  # Sort by X position
            
            # Filter out gold value to avoid confusion
            gold_value = parsed_data.get('gold', 0)
            damage_only = [(n, x) for n, x in damage_candidates if n != gold_value]
            
            if damage_only:
                # Use spatial ordering (left to right)
                damage_values = [item[0] for item in damage_only]
                
                if len(damage_values) >= 1:
                    parsed_data['hero_damage'] = damage_values[0]
                    logger.debug(f"Extracted hero damage via spatial method: {damage_values[0]}")
                if len(damage_values) >= 2:
                    parsed_data['damage_taken'] = damage_values[1]
                    logger.debug(f"Extracted damage taken via spatial method: {damage_values[1]}")
        
        return parsed_data
    
    def _extract_kda_from_text(self, text: str, data: Dict[str, Any]):
        """Extract KDA values from text."""
        # Look for KDA pattern: X/Y/Z or X Y Z
        kda_patterns = [
            r'(\d+)\s*/\s*(\d+)\s*/\s*(\d+)',  # X/Y/Z format
            r'(\d+)\s+(\d+)\s+(\d+)',          # X Y Z format  
            r'(\d+)-(\d+)-(\d+)',              # X-Y-Z format
        ]
        
        for pattern in kda_patterns:
            match = re.search(pattern, text)
            if match:
                k, d, a = map(int, match.groups())
                if 0 <= k <= 50 and 0 <= d <= 50 and 0 <= a <= 50:  # Sanity check
                    data['kills'] = k
                    data['deaths'] = max(1, d)  # Deaths can't be 0
                    data['assists'] = a
                    logger.debug(f"Extracted KDA: {k}/{d}/{a}")
                    return

    def _extract_kda_from_text_enhanced(self, text: str, row_items: List, data: Dict[str, Any]):
        """Enhanced KDA extraction with spatial awareness."""
        import re
        
        # First try the standard text-based extraction
        self._extract_kda_from_text(text, data)
        
        # If that failed, try spatial extraction
        if not all(k in data for k in ['kills', 'deaths', 'assists']):
            # Look for individual numbers that could be KDA
            kda_candidates = []
            
            for item in row_items:
                # Look for small numbers (likely KDA values)
                numbers = [int(n) for n in re.findall(r'\b(\d+)\b', item['text'])]
                for num in numbers:
                    if 0 <= num <= 50:  # KDA range
                        kda_candidates.append((num, item['x']))
            
            # Sort by X position and take first 3 as K/D/A
            if len(kda_candidates) >= 3:
                kda_candidates.sort(key=lambda x: x[1])  # Sort by X position
                kills, deaths, assists = [k[0] for k in kda_candidates[:3]]
                
                data['kills'] = kills
                data['deaths'] = max(1, deaths)
                data['assists'] = assists
                logger.debug(f"Extracted KDA via spatial method: {kills}/{deaths}/{assists}")
    
    def _extract_gold_from_text(self, text: str, data: Dict[str, Any]):
        """Extract gold value from text."""
        # Look for 3-6 digit numbers (typical gold range) - Fixed threshold
        gold_match = re.search(r'\b(\d{3,6})\b', text)
        if gold_match:
            gold_value = int(gold_match.group(1))
            if 100 <= gold_value <= 50000:  # Reasonable gold range - Fixed threshold
                data['gold'] = gold_value
                logger.debug(f"Extracted gold: {gold_value}")
    
    def _extract_damage_from_text(self, text: str, data: Dict[str, Any]):
        """Extract damage values from text."""
        # Look for large numbers (damage values)
        damage_matches = re.findall(r'\b(\d{4,7})\b', text)
        if damage_matches:
            damage_values = [int(d) for d in damage_matches]
            damage_values.sort(reverse=True)
            
            # Assign largest as hero damage
            if damage_values and 1000 <= damage_values[0] <= 500000:
                data['hero_damage'] = damage_values[0]
                logger.debug(f"Extracted hero damage: {damage_values[0]}")
    
    def _extract_duration_from_text(self, text: str, data: Dict[str, Any]):
        """Extract match duration from text."""
        duration_match = re.search(r'(\d{1,2}):(\d{2})', text)
        if duration_match:
            minutes = int(duration_match.group(1))
            seconds = int(duration_match.group(2))
            data['match_duration'] = minutes + seconds / 60
            logger.debug(f"Extracted duration: {minutes}:{seconds:02d}")
    
    def _fallback_extraction(self, ocr_results: List) -> Dict[str, Any]:
        """Fallback extraction when anchor-based parsing fails."""
        logger.info("Using fallback extraction - anchor-based parsing failed")
        
        # Simple number extraction as fallback
        all_text = " ".join([result[1] for result in ocr_results])
        data = {}
        
        # Look for any KDA pattern in the text
        kda_match = re.search(r'(\d{1,2})\s*/?\s*(\d{1,2})\s*/?\s*(\d{1,2})', all_text)
        if kda_match:
            k, d, a = map(int, kda_match.groups())
            data.update({'kills': k, 'deaths': max(1, d), 'assists': a})
        
        # Look for gold values
        gold_values = [int(m) for m in re.findall(r'\b(\d{4,6})\b', all_text)]
        if gold_values:
            data['gold'] = max(gold_values)
        
        return data


class EnhancedDataCollector(DataCollector):
    """
    Enhanced data collector with session management and improved analysis.
    Supports multi-screenshot processing and advanced hero identification.
    """
    
    def __init__(self):
        super().__init__()
        self.session_manager = session_manager
        self.ign_validator = IGNValidator()
        self.ign_matcher = RobustIGNMatcher()
        self.anchor_parser = AnchorBasedLayoutParser()
    
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
                    logger.warning(
                        f"Session {session_id} not found, creating new session"
                    )
                    session_id = self.session_manager.create_session(ign)
                else:
                    logger.info(f"Reusing existing session {session_id} with {len(session.screenshots)} screenshots")
            else:
                session_id = self.session_manager.create_session(ign)
                logger.info(f"Created new session {session_id}")
            
            # Analyze screenshot type
            screenshot_type, type_confidence, detected_keywords = (
                screenshot_analyzer.analyze_screenshot_type(image_path)
            )
            
            logger.info(
                f"Screenshot analysis: type={screenshot_type.value}, "
                f"confidence={type_confidence:.3f}, "
                f"keywords={detected_keywords[:3]}"
            )
            
            # Perform enhanced OCR analysis
            analysis_result = self._enhanced_screenshot_analysis(
                image_path, ign, screenshot_type, hero_override, known_igns
            )
            
            # Create screenshot analysis object
            screenshot_analysis = ScreenshotAnalysis(
                screenshot_type=screenshot_type,
                raw_data=analysis_result.get("data", {}),
                confidence=analysis_result.get(
                    "overall_confidence", type_confidence
                ),
                warnings=analysis_result.get("warnings", [])
            )
            
            # Add to session
            self.session_manager.add_screenshot_analysis(
                session_id, screenshot_analysis
            )
            
            # Check if we have a complete result
            session = self.session_manager.get_session(session_id)
            if session and session.is_complete:
                final_result = session.final_result
                logger.info(
                    f"Session {session_id} complete with final result"
                )
            else:
                final_result = analysis_result.get("data", {})
            
            return {
                "data": final_result,
                "session_id": session_id,
                "screenshot_type": screenshot_type.value,
                "type_confidence": type_confidence,
                "session_complete": session.is_complete if session else False,
                "overall_confidence": analysis_result.get("overall_confidence", 0.0),
                "completeness_score": analysis_result.get("completeness_score", 0.0),
                "warnings": analysis_result.get("warnings", []),
                "debug_info": {
                    "detected_keywords": detected_keywords,
                    "screenshot_count": (
                        len(session.screenshots) if session else 1
                    ),
                    "hero_suggestions": analysis_result.get(
                        "hero_suggestions", []
                    ),
                    "hero_debug": analysis_result.get("hero_debug", {})
                }
            }
            
        except Exception as e:
            logger.error(f"Enhanced screenshot analysis failed: {str(e)}")
            return {
                "data": {},
                "session_id": session_id if 'session_id' in locals() else None,
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
        logger.info(
            f"Starting enhanced analysis for {screenshot_type.value} screenshot"
        )
        
        # Start diagnostic logging session
        session_id = diagnostic_logger.start_analysis(image_path, "enhanced")
        
        start_time = time.time()
        
        # Get OCR results with improved preprocessing
        reader = get_ocr_reader()
        image = self._preprocess_image_enhanced(image_path)
        
        # FORCE DEBUG: Save the actual preprocessed image for manual inspection
        debug_preprocessed_path = "debug_preprocessed_output.png"
        cv2.imwrite(debug_preprocessed_path, image)
        print(f"🔍 DEBUG: Preprocessed image saved to {debug_preprocessed_path}")
        print(f"🔍 DEBUG: Image shape after preprocessing: {image.shape}")
        print(f"🔍 DEBUG: Image dtype: {image.dtype}")
        print(f"🔍 DEBUG: Image min/max values: {image.min()}/{image.max()}")
        
        # Check if image is blank or corrupted
        if image.size == 0:
            print("❌ DEBUG: Preprocessed image is EMPTY!")
        elif image.max() == image.min():
            print(f"❌ DEBUG: Preprocessed image has uniform values ({image.max()}) - likely blank!")
        else:
            print("✅ DEBUG: Preprocessed image appears to have content")
        
        ocr_results = reader.readtext(image, detail=1)
        
        # FORCE DEBUG: Print OCR results
        print(f"🔍 DEBUG: OCR returned {len(ocr_results)} text detections")
        if ocr_results:
            print("🔍 DEBUG: OCR Results (first 5):")
            for i, (bbox, text, conf) in enumerate(ocr_results[:5]):
                print(f"  {i+1}. Text: '{text}' | Confidence: {conf:.3f}")
        else:
            print("❌ DEBUG: OCR returned EMPTY list - no text detected!")
        
        ocr_time = (time.time() - start_time) * 1000
        
        # Log OCR step
        diagnostic_logger.log_step(
            step_name="OCR_Processing",
            input_data={"image_path": image_path, "preprocessing": "enhanced"},
            output_data={
                "text_count": len(ocr_results),
                "avg_confidence": (
                    sum(r[2] for r in ocr_results) / len(ocr_results)
                    if ocr_results else 0
                )
            },
            confidence_score=(
                sum(r[2] for r in ocr_results) / len(ocr_results)
                if ocr_results else 0
            ),
            ocr_results=ocr_results,
            processing_time_ms=ocr_time
        )
        
        warnings = []
        
        # Enhanced IGN validation
        validated_ign = self._validate_ign_enhanced(ign, ocr_results, warnings)
        
        # Advanced hero identification
        hero_start = time.time()
        
        # ENHANCED: Use row-specific hero detection when player row is found
        if hasattr(self, 'anchor_parser') and self.anchor_parser:
            # Check if we found the player row
            ign_matcher = RobustIGNMatcher()
            ign_result = ign_matcher.find_ign_in_ocr(validated_ign, ocr_results)
            
            if ign_result["found"]:
                # Get player row Y coordinate
                ign_bbox = ign_result["bbox"]
                player_row_y = sum(point[1] for point in ign_bbox) / 4
                
                # Use row-specific hero detection
                from .row_specific_hero_detector import row_specific_hero_detector
                hero_name, hero_confidence, hero_debug = row_specific_hero_detector.detect_hero_in_player_row(
                    image_path, validated_ign, player_row_y, ocr_results, hero_override
                )
                logger.info(f"Row-specific hero detection used for player at y={player_row_y:.1f}")
            else:
                # Fallback to original method
                hero_name, hero_confidence, hero_debug = advanced_hero_detector.detect_hero_comprehensive(
                    image_path, validated_ign, hero_override
                )
                logger.warning("IGN not found, using full-image hero detection")
        else:
            # Fallback to original method
            hero_name, hero_confidence, hero_debug = advanced_hero_detector.detect_hero_comprehensive(
                image_path, validated_ign, hero_override
            )
        
        hero_time = (time.time() - hero_start) * 1000
        
        # Log hero detection step
        diagnostic_logger.log_step(
            step_name="Hero_Detection",
            input_data={"ign": validated_ign, "hero_override": hero_override},
            output_data={"hero": hero_name, "strategies_tried": hero_debug.get('strategies_tried', [])},
            confidence_score=hero_confidence,
            processing_time_ms=hero_time,
            warnings=[] if hero_confidence >= 0.7 else [f"Low hero confidence: {hero_confidence:.3f}"],
            errors=[] if hero_name != "unknown" else ["Hero detection failed"]
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
        
        # Extract data based on screenshot type with confidence validation
        if screenshot_type == ScreenshotType.SCOREBOARD:
            parsed_data = self._parse_scoreboard_enhanced(validated_ign, ocr_results, warnings)
        elif screenshot_type == ScreenshotType.STATS_PAGE:
            parsed_data = self._parse_stats_page_enhanced(validated_ign, ocr_results, warnings)
        else:
            # Fallback to original parsing method
            parsed_data = self._parse_player_row(validated_ign, ocr_results, hero_override)
            if not parsed_data:
                warnings.append("Fallback parsing also failed")
        
        # Set hero name - CRITICAL: Do this BEFORE completeness calculation
        if hero_name and hero_name != "unknown":
            parsed_data["hero"] = hero_name
            logger.info(f"Hero set in parsed data: {hero_name} (confidence: {hero_confidence:.3f})")
        else:
            parsed_data["hero"] = "unknown"
            logger.warning(f"Hero detection failed or returned unknown: {hero_name}")
        
        # Add default values BEFORE completeness calculation
        self._add_default_values(parsed_data)
        
        # PRIORITY 1: MVP Badge and Trophy Detection
        trophy_result = self._detect_trophy_and_performance(
            image_path, validated_ign, ocr_results, parsed_data, warnings
        )
        if trophy_result:
            parsed_data.update(trophy_result)
        
        # Validate parsed data completeness AFTER all data is set
        completeness_score = self._validate_data_completeness(parsed_data, warnings)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            hero_confidence, len(warnings), len(parsed_data), completeness_score
        )
        
        # Log data parsing step
        parsing_errors = []
        if not parsed_data.get("gold") or parsed_data.get("gold", 0) <= 0:
            parsing_errors.append("Gold/economy data missing or invalid")
        if not all(k in parsed_data for k in ["kills", "deaths", "assists"]):
            parsing_errors.append("KDA data incomplete")
        
        diagnostic_logger.log_step(
            step_name="Data_Parsing",
            input_data={"screenshot_type": screenshot_type.value, "data_fields": len(parsed_data)},
            output_data={"parsed_fields": list(parsed_data.keys()), "completeness": completeness_score},
            confidence_score=completeness_score,
            warnings=warnings,
            errors=parsing_errors
        )
        
        # Finish diagnostic session
        diagnostics = diagnostic_logger.finish_analysis(
            final_confidence=overall_confidence,
            final_warnings=warnings,
            final_errors=parsing_errors
        )
        
        return {
            "data": parsed_data,
            "overall_confidence": overall_confidence,
            "warnings": warnings,
            "hero_suggestions": hero_suggestions,
            "hero_debug": hero_debug,
            "completeness_score": completeness_score,
            "diagnostics": diagnostics
        }
    
    def _validate_ign_enhanced(self, ign: str, ocr_results: List, warnings: List[str]) -> str:
        """Enhanced IGN validation with robust multi-strategy matching."""
        # Use the robust IGN matcher
        ign_match_result = self.ign_matcher.find_ign_in_ocr(ign, ocr_results)
        
        if ign_match_result["found"]:
            confidence = ign_match_result["confidence"]
            matched_text = ign_match_result["text"]
            strategy = ign_match_result["strategy"]
            
            logger.info(f"IGN found using {strategy}: '{matched_text}' "
                       f"(confidence: {confidence:.3f})")
            
            # Validate the matched IGN using the MLBB validator
            validation_result = self.ign_validator.validate_mlbb_ign(matched_text)
            if validation_result['is_valid']:
                validated_ign = validation_result['cleaned_ign']
                
                # Add appropriate warnings based on confidence
                if confidence < 0.7:
                    warnings.append(f"Low IGN match confidence: {confidence:.1%}")
                elif confidence < 0.9:
                    warnings.append(f"IGN matched with {strategy} (confidence: {confidence:.1%})")
                
                logger.info(f"IGN validated and cleaned: '{validated_ign}'")
                return validated_ign
            else:
                # IGN found but doesn't validate as MLBB format
                warnings.append(f"Found '{matched_text}' but failed MLBB IGN validation")
                return matched_text
        
        # Fallback: Try original validation approach
        for bbox, text, conf in ocr_results:
            validation_result = self.ign_validator.validate_mlbb_ign(text)
            if validation_result['is_valid'] and ign.lower() in validation_result['cleaned_ign'].lower():
                warnings.append(f"IGN found via fallback validation: {validation_result['cleaned_ign']}")
                return validation_result['cleaned_ign']
        
        # No IGN found with any method
        warnings.append(f"IGN '{ign}' not found in screenshot using any matching strategy")
        logger.warning(f"IGN '{ign}' not found in OCR results despite robust matching")
        return ign
    
    def _parse_scoreboard_enhanced(
        self, ign: str, ocr_results: List, warnings: List[str]
    ) -> Dict[str, Any]:
        """Enhanced parsing specifically for scoreboard screenshots with anchor-based layout detection."""
        logger.info("Using enhanced scoreboard parsing with anchor-based layout detection")
        
        # Step 1: Try anchor-based parsing first (most robust)
        parsed_data = self._parse_with_anchors(ign, ocr_results, warnings)
        
        if not parsed_data or len(parsed_data) < 3:  # Need at least some basic data
            logger.info("Anchor-based parsing insufficient, trying enhanced row parsing")
            # Step 2: Fall back to enhanced player row parsing
            parsed_data = self._parse_player_row_enhanced(ign, ocr_results, warnings)
            
        if not parsed_data or len(parsed_data) < 2:
            logger.info("Enhanced row parsing insufficient, trying fallback strategy")
            warnings.append("Could not locate player using standard methods")
            # Step 3: Try alternative parsing methods
            parsed_data = self._parse_fallback_strategy(ocr_results, warnings)
            
        if not parsed_data:
            logger.warning("All parsing strategies failed")
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
            # Try alternative duration patterns
            alt_duration = re.search(r"(\d{1,2}):(\d{2})", text_content)
            if alt_duration:
                minutes = int(alt_duration.group(1))
                seconds = int(alt_duration.group(2))
                parsed_data["match_duration"] = minutes + seconds / 60
            else:
                parsed_data["match_duration"] = 10  # Default
                warnings.append("Could not detect match duration")
        
        # Match result - Enhanced detection
        text_lower = text_content.lower()
        if any(word in text_lower for word in ["victory", "win", "won"]):
            parsed_data["match_result"] = "victory"
        elif any(word in text_lower for word in ["defeat", "loss", "lose", "lost"]):
            parsed_data["match_result"] = "defeat"
        else:
            # Try to infer from UI elements or positioning
            if "blue" in text_lower and "red" in text_lower:
                # Look for score patterns like "4 26" for team kills
                score_pattern = re.search(r"(\d+)\s+(\d+)", text_content)
                if score_pattern:
                    team1_score = int(score_pattern.group(1))
                    team2_score = int(score_pattern.group(2))
                    # Basic heuristic: if one team has significantly fewer kills, they likely lost
                    if team1_score < team2_score * 0.6:
                        parsed_data["match_result"] = "defeat"
                    elif team2_score < team1_score * 0.6:
                        parsed_data["match_result"] = "victory"
                    else:
                        warnings.append("Could not determine match result from scores")
                else:
                    warnings.append("Could not detect match result")
            else:
                warnings.append("Could not detect match result")
        
        # Player rank detection
        rank_keywords = [
            "warrior", "elite", "master", "grandmaster", "epic", 
            "legend", "mythic", "mythical", "bronze", "silver", "gold"
        ]
        detected_rank = None
        for keyword in rank_keywords:
            if keyword in text_lower:
                detected_rank = keyword.title()
                break
        
        if detected_rank:
            parsed_data["player_rank"] = detected_rank
            logger.info(f"Detected player rank: {detected_rank}")
        else:
            # Default rank assumption for rating context
            parsed_data["player_rank"] = "Unknown"
            warnings.append("Could not detect player rank")
        
        # Enhanced context detection
        parsed_data["screenshot_confidence"] = self._calculate_screenshot_confidence(text_content, warnings)
        
        return parsed_data
    
    def _parse_with_anchors(self, ign: str, ocr_results: List, warnings: List[str]) -> Dict[str, Any]:
        """Parse using anchor-based layout detection for maximum robustness."""
        logger.info("Attempting anchor-based parsing")
        
        # We need image shape for anchor detection, let's get it from OCR results
        # Estimate image dimensions from OCR bounding boxes
        max_x = max_y = 0
        for bbox, _, _ in ocr_results:
            for point in bbox:
                max_x = max(max_x, point[0])
                max_y = max(max_y, point[1])
        
        estimated_image_shape = (int(max_y + 100), int(max_x + 100))  # Add padding
        
        try:
            # Step 1: Detect UI anchors
            anchors = self.anchor_parser.detect_anchors(ocr_results, estimated_image_shape)
            logger.debug(f"Found {len(anchors)} UI anchors: {list(anchors.keys())}")
            
            if not anchors:
                logger.warning("No UI anchors detected, skipping anchor-based parsing")
                return {}
            
            # Step 2: Identify table columns based on anchors
            columns = self.anchor_parser.identify_columns(ocr_results, anchors)
            logger.debug(f"Identified {len(columns)} table columns: {list(columns.keys())}")
            
            # Step 3: Extract player data using layout understanding
            parsed_data = self.anchor_parser.extract_player_row_data(ign, ocr_results, anchors, columns)
            
            if parsed_data:
                logger.info(f"Anchor-based parsing extracted {len(parsed_data)} fields")
                
                # Add parsing method info for debugging
                parsed_data['_parsing_method'] = 'anchor_based'
                parsed_data['_anchors_found'] = list(anchors.keys())
                parsed_data['_columns_found'] = list(columns.keys())
                
                return parsed_data
            else:
                warnings.append("Anchor-based parsing found layout but no player data")
                return {}
                
        except Exception as e:
            logger.error(f"Anchor-based parsing failed with error: {str(e)}")
            warnings.append(f"Anchor-based parsing error: {str(e)}")
            return {}
    
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
        """Add default values for missing fields, preserving extracted values."""
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
        
        # Only set defaults for missing fields - don't overwrite extracted data
        for key, default_value in defaults.items():
            if key not in parsed_data or parsed_data[key] is None:
                parsed_data[key] = default_value
        
        # Ensure deaths is never 0 (causes division by zero in KDA calculations)
        if parsed_data.get("deaths", 0) <= 0:
            parsed_data["deaths"] = 1
        
        # Calculate GPM if we have gold and duration
        gold = parsed_data.get("gold", 0)
        duration = parsed_data.get("match_duration", 0)
        if gold > 0 and duration > 0:
            gpm = gold / duration
            parsed_data["gold_per_min"] = round(gpm)
            logger.debug(f"Calculated GPM: {gpm:.1f} (Gold: {gold}, Duration: {duration})")
        
        # Log what we have after adding defaults
        data_summary = {k: v for k, v in parsed_data.items() 
                       if k in ["kills", "deaths", "assists", "hero", "gold", "hero_damage"]}
        logger.info(f"Final data after defaults: {data_summary}")
    
    def _preprocess_image_enhanced(self, image_path: str) -> np.ndarray:
        """Enhanced image preprocessing with quality detection for better OCR accuracy."""
        image = cv2.imread(image_path)
        
        # FORCE DEBUG: Check if original image loaded correctly
        if image is None:
            print(f"❌ DEBUG: Failed to load image from {image_path}")
            return np.array([])
        
        print(f"🔍 DEBUG: Original image shape: {image.shape}")
        print(f"🔍 DEBUG: Original image path: {image_path}")
        
        # Detect image quality
        quality_score = self._assess_image_quality(image)
        
        # FORCE DEBUG: Print quality score and processing type
        print(f"🔍 DEBUG: Image quality score: {quality_score:.3f}")
        processing_type = 'minimal' if quality_score > 0.7 else 'moderate' if quality_score > 0.5 else 'aggressive'
        print(f"🔍 DEBUG: Processing type: {processing_type}")
        
        logger.info(f"Image quality score: {quality_score:.3f}")
        
        if quality_score > 0.7:
            # High quality image - minimal preprocessing
            print("🔍 DEBUG: Applying MINIMAL preprocessing (grayscale only)")
            logger.info("High quality image detected - using minimal preprocessing")
            
            # Just convert to grayscale, no other processing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Save debug image
            debug_path = "temp/enhanced_preprocessed_image.png"
            cv2.imwrite(debug_path, gray)
            logger.info(f"Minimal preprocessed image saved to: {debug_path}")
            
            return gray
            
        elif quality_score > 0.5:
            # Medium quality - moderate preprocessing
            print("🔍 DEBUG: Applying MODERATE preprocessing (grayscale + denoising)")
            logger.info("Medium quality image detected - using moderate preprocessing")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Light denoising only
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            
            # Save debug image
            debug_path = "temp/enhanced_preprocessed_image.png"
            cv2.imwrite(debug_path, denoised)
            logger.info(f"Moderate preprocessed image saved to: {debug_path}")
            
            return denoised
            
        else:
            # Low quality - aggressive preprocessing
            print("🔍 DEBUG: Applying AGGRESSIVE preprocessing (full pipeline)")
            logger.info("Low quality image detected - using aggressive preprocessing")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding for better text clarity
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Noise reduction
            denoised = cv2.medianBlur(adaptive_thresh, 3)
            
            # Morphological operations to clean up text
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
            
            # Save debug image
            debug_path = "temp/enhanced_preprocessed_image.png"
            cv2.imwrite(debug_path, cleaned)
            logger.info(f"Aggressive preprocessed image saved to: {debug_path}")
            
            return cleaned
    
    def _assess_image_quality(self, image: np.ndarray) -> float:
        """
        Assess image quality based on various metrics.
        Returns a score between 0 (low quality) and 1 (high quality).
        """
        quality_factors = []
        
        # 1. Resolution factor
        height, width = image.shape[:2]
        resolution_score = float(min(1.0, (height * width) / (1920 * 1080)))  # Normalize to FHD
        quality_factors.append(resolution_score)
        
        # 2. Sharpness using Laplacian variance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = gray.astype(np.float64)  # Ensure float64 for calculations
        laplacian_var = float(cv2.Laplacian(gray, -1).var())
        sharpness_score = float(min(1.0, laplacian_var / 500.0))  # Normalize based on typical values
        quality_factors.append(sharpness_score)
        
        # 3. Contrast using standard deviation
        contrast_score = float(min(1.0, np.std(gray) / 80.0))  # Normalize based on typical values
        quality_factors.append(contrast_score)
        
        # 4. Brightness distribution
        mean_brightness = float(np.mean(gray))
        # Ideal brightness is around 120-140, penalize too dark or too bright
        if 100 <= mean_brightness <= 160:
            brightness_score = 1.0
        else:
            brightness_score = float(max(0, 1.0 - abs(mean_brightness - 130) / 130))
        quality_factors.append(brightness_score)
        
        # 5. Noise estimation using local variance
        kernel_size = 5
        local_mean = cv2.blur(gray, (kernel_size, kernel_size))
        # FIX: Ensure proper float operations to avoid numpy.bool_ issues
        gray_diff = gray.astype(np.float64) - local_mean.astype(np.float64)
        local_variance = cv2.blur(gray_diff ** 2, (kernel_size, kernel_size))
        noise_level = float(np.mean(np.sqrt(np.abs(local_variance))))  # abs() to handle any negative values
        noise_score = float(max(0, 1.0 - noise_level / 20.0))  # Lower noise = higher score
        quality_factors.append(noise_score)
        
        # Calculate weighted average - ensure all values are proper floats
        weights = [0.15, 0.25, 0.20, 0.20, 0.20]  # Resolution, Sharpness, Contrast, Brightness, Noise
        quality_score = float(sum(w * f for w, f in zip(weights, quality_factors)))
        
        # FORCE DEBUG: Print all quality metrics
        print(f"🔍 DEBUG: Quality metrics breakdown:")
        print(f"  Resolution: {resolution_score:.3f} (weight: 0.15)")
        print(f"  Sharpness:  {sharpness_score:.3f} (weight: 0.25)")
        print(f"  Contrast:   {contrast_score:.3f} (weight: 0.20)")
        print(f"  Brightness: {brightness_score:.3f} (weight: 0.20)")
        print(f"  Noise:      {noise_score:.3f} (weight: 0.20)")
        print(f"  FINAL:      {quality_score:.3f}")
        
        logger.debug(f"Quality metrics - Resolution: {resolution_score:.3f}, Sharpness: {sharpness_score:.3f}, "
                    f"Contrast: {contrast_score:.3f}, Brightness: {brightness_score:.3f}, Noise: {noise_score:.3f}")
        
        return quality_score
    
    def _detect_trophy_and_performance(
        self, 
        image_path: str, 
        ign: str, 
        ocr_results: List, 
        parsed_data: Dict[str, Any], 
        warnings: List[str]
    ) -> Dict[str, Any]:
        """
        PRIORITY 1: Detect MVP badge and medals for accurate performance rating.
        
        This implements the core requirement:
        - MVP Badge Detection → "Excellent" performance
        - Gold Medal → "Good" performance  
        - Silver Medal → "Average" performance
        - Bronze Medal → "Poor" performance
        """
        trophy_data = {}
        
        try:
            # Find player row location from IGN
            ign_matcher = RobustIGNMatcher()
            ign_result = ign_matcher.find_ign_in_ocr(ign, ocr_results)
            
            if not ign_result["found"]:
                logger.warning("Cannot detect trophy - IGN not found in screenshot")
                warnings.append("Trophy detection skipped - player row not located")
                return {}
            
            # Extract player row Y coordinate
            ign_bbox = ign_result["bbox"]
            player_row_y = sum(point[1] for point in ign_bbox) / 4
            player_name_x = sum(point[0] for point in ign_bbox) / 4
            
            logger.info(f"Trophy detection: Player row at y={player_row_y:.1f}, x={player_name_x:.1f}")
            
            # Run trophy detection
            trophy_result = trophy_medal_detector.detect_trophy_in_player_row(
                image_path=image_path,
                player_row_y=player_row_y,
                player_name_x=player_name_x,
                debug_mode=True  # Save debug images for analysis
            )
            
            # Log trophy detection results
            logger.info(f"Trophy detection result: {trophy_result.trophy_type.value} "
                       f"(confidence: {trophy_result.confidence:.1%})")
            
            # Update parsed data with trophy information
            trophy_data.update({
                "mvp_detected": trophy_result.trophy_type.value == "mvp_crown",
                "medal_type": trophy_result.trophy_type.value if trophy_result.trophy_type.value != "mvp_crown" else None,
                "trophy_confidence": trophy_result.confidence,
                "performance_label": trophy_result.performance_label.value,
                "trophy_detection_method": trophy_result.detection_method
            })
            
            # Apply performance rating based on trophy detection
            if trophy_result.confidence > 0.6:
                trophy_data["contextual_performance_rating"] = trophy_result.performance_label.value
                
                # Add specific performance indicators
                if trophy_result.trophy_type.value == "mvp_crown":
                    trophy_data["rating_boost_reasons"] = ["MVP Badge Detected"]
                    logger.info("🏆 MVP badge detected - applying 'Excellent' performance rating")
                    
                elif trophy_result.trophy_type.value == "gold_medal":
                    trophy_data["rating_boost_reasons"] = ["Gold Medal Detected"]
                    logger.info("🥇 Gold medal detected - applying 'Good' performance rating")
                    
                elif trophy_result.trophy_type.value == "silver_medal":
                    trophy_data["rating_boost_reasons"] = ["Silver Medal Detected"]
                    logger.info("🥈 Silver medal detected - applying 'Average' performance rating")
                    
                elif trophy_result.trophy_type.value == "bronze_medal":
                    trophy_data["rating_boost_reasons"] = ["Bronze Medal Detected"]
                    logger.info("🥉 Bronze medal detected - applying 'Poor' performance rating")
                    
                # Apply support role logic if applicable
                hero_name = parsed_data.get("hero", "unknown")
                if isinstance(hero_name, (list, tuple)):
                    hero_name = hero_name[0] if hero_name else "unknown"
                    
                if hero_name in ["estes", "mathilda", "angela", "rafaela", "kaja", "diggie"]:
                    # Support-specific performance criteria
                    tfp = parsed_data.get("teamfight_participation", 0)
                    if tfp >= 70 and trophy_result.trophy_type.value == "mvp_crown":
                        trophy_data["support_role_boost"] = True
                        trophy_data["rating_boost_reasons"].append("Support + TFP ≥ 70%")
                        logger.info("💫 Support role with high TFP - additional rating boost")
                
                # Apply victory bonus
                match_result = parsed_data.get("match_result", "unknown")
                if match_result == "victory" and trophy_result.confidence > 0.7:
                    if "rating_boost_reasons" not in trophy_data:
                        trophy_data["rating_boost_reasons"] = []
                    trophy_data["rating_boost_reasons"].append("Victory")
                    logger.info("🎊 Victory with high trophy confidence - additional boost")
                    
            else:
                # Low confidence trophy detection
                logger.warning(f"Low trophy detection confidence: {trophy_result.confidence:.1%}")
                warnings.append(f"Trophy detection confidence low: {trophy_result.confidence:.1%}")
            
            # Add debug information for analysis
            trophy_data["trophy_debug_info"] = {
                "detection_method": trophy_result.detection_method,
                "bounding_box": trophy_result.bounding_box,
                "color_analysis": trophy_result.color_analysis,
                "shape_analysis": trophy_result.shape_analysis,
                "search_region_y": player_row_y
            }
            
        except Exception as e:
            logger.error(f"Trophy detection failed: {str(e)}")
            warnings.append(f"Trophy detection error: {str(e)}")
            
        return trophy_data
    
    def _parse_player_row_enhanced(self, ign: str, ocr_results: List, warnings: List[str]) -> Dict[str, Any]:
        """Enhanced player row parsing with better gold extraction."""
        # Start with original method
        parsed_data = self._parse_player_row(ign, ocr_results)
        
        if not parsed_data:
            return {}
        
        # Enhanced gold parsing with multiple strategies
        if not parsed_data.get("gold") or parsed_data.get("gold", 0) <= 0:
            logger.info("Applying enhanced gold parsing strategies")
            
            # Strategy 1: Look for large numbers in proximity to player IGN
            gold_value = self._extract_gold_near_ign(ign, ocr_results)
            if gold_value:
                parsed_data["gold"] = gold_value
                logger.info(f"Gold found via IGN proximity: {gold_value}")
            
            # Strategy 2: Pattern-based gold extraction (common ranges)
            if not parsed_data.get("gold") or parsed_data.get("gold", 0) <= 0:
                gold_value = self._extract_gold_by_pattern(ocr_results)
                if gold_value:
                    parsed_data["gold"] = gold_value
                    logger.info(f"Gold found via pattern matching: {gold_value}")
            
            # Strategy 3: Statistical analysis of detected numbers
            if not parsed_data.get("gold") or parsed_data.get("gold", 0) <= 0:
                gold_value = self._extract_gold_statistical(ocr_results)
                if gold_value:
                    parsed_data["gold"] = gold_value
                    logger.info(f"Gold found via statistical analysis: {gold_value}")
                    warnings.append(f"Gold value estimated using statistical analysis: {gold_value}")
        
        return parsed_data
    
    def _parse_fallback_strategy(self, ocr_results: List, warnings: List[str]) -> Dict[str, Any]:
        """Fallback parsing when IGN-based parsing fails."""
        logger.info("Applying fallback parsing strategy")
        
        parsed_data = {}
        all_text = " ".join([result[1] for result in ocr_results])
        
        # Extract all numbers and try to categorize them
        import re
        all_numbers = [int(match) for match in re.findall(r'\b(\d+)\b', all_text)]
        
        if not all_numbers:
            warnings.append("No numeric data found in fallback parsing")
            return {}
        
        # Sort numbers for analysis
        all_numbers.sort()
        
        # Gold: Typically the largest number (> 1000)
        gold_candidates = [n for n in all_numbers if n >= 1000]
        if gold_candidates:
            # Take the median gold value to avoid outliers
            gold_index = len(gold_candidates) // 2
            parsed_data["gold"] = gold_candidates[gold_index]
        
        # KDA: Small numbers (0-50)
        kda_candidates = [n for n in all_numbers if 0 <= n <= 50]
        if len(kda_candidates) >= 3:
            # Take first three as KDA
            parsed_data["kills"] = kda_candidates[0]
            parsed_data["deaths"] = max(1, kda_candidates[1])  # Deaths can't be 0
            parsed_data["assists"] = kda_candidates[2]
        
        # Damage: Medium-large numbers (1000-100000)
        damage_candidates = [n for n in all_numbers if 1000 <= n <= 100000]
        if damage_candidates:
            parsed_data["hero_damage"] = max(damage_candidates)
        
        if parsed_data:
            warnings.append("Data extracted using fallback parsing - accuracy may be reduced")
        
        return parsed_data
    
    def _extract_gold_near_ign(self, ign: str, ocr_results: List) -> Optional[int]:
        """Extract gold by finding large numbers near the player IGN."""
        import re
        
        # Find IGN position in OCR results
        ign_positions = []
        for i, (bbox, text, conf) in enumerate(ocr_results):
            if ign.lower() in text.lower() or any(part in text.lower() for part in ign.lower().split()):
                ign_positions.append(i)
        
        if not ign_positions:
            return None
        
        # Look for gold values within 3 positions of IGN
        for ign_pos in ign_positions:
            for i in range(max(0, ign_pos - 3), min(len(ocr_results), ign_pos + 4)):
                bbox, text, conf = ocr_results[i]
                numbers = [int(match) for match in re.findall(r'\b(\d{4,6})\b', text)]
                
                for num in numbers:
                    if 1000 <= num <= 50000:  # Reasonable gold range
                        return num
        
        return None
    
    def _extract_gold_by_pattern(self, ocr_results: List) -> Optional[int]:
        """Extract gold using common MLBB gold value patterns."""
        import re
        
        gold_patterns = [
            r'\b(1[0-9]{3,4})\b',   # 1000-19999 range
            r'\b(2[0-9]{3,4})\b',   # 2000-29999 range  
            r'\b(3[0-9]{3,4})\b',   # 3000-39999 range
            r'\b([4-9][0-9]{3,4})\b' # 4000+ range
        ]
        
        all_text = " ".join([result[1] for result in ocr_results])
        
        for pattern in gold_patterns:
            matches = re.findall(pattern, all_text)
            if matches:
                # Return the most reasonable gold value
                gold_values = [int(match) for match in matches]
                # Filter for reasonable game gold amounts
                reasonable_values = [g for g in gold_values if 1000 <= g <= 30000]
                if reasonable_values:
                    return reasonable_values[0]
        
        return None
    
    def _extract_gold_statistical(self, ocr_results: List) -> Optional[int]:
        """Extract gold using statistical analysis of all detected numbers."""
        import re
        
        all_text = " ".join([result[1] for result in ocr_results])
        all_numbers = [int(match) for match in re.findall(r'\b(\d+)\b', all_text)]
        
        if not all_numbers:
            return None
        
        # Find numbers in typical gold range
        gold_candidates = [n for n in all_numbers if 1000 <= n <= 30000]
        
        if not gold_candidates:
            return None
        
        # Use median to avoid outliers
        gold_candidates.sort()
        median_index = len(gold_candidates) // 2
        estimated_gold = gold_candidates[median_index]
        
        # Sanity check: should be reasonable for match duration
        if 800 <= estimated_gold <= 35000:
            return estimated_gold
        
        return None
    
    def _validate_data_completeness(self, parsed_data: Dict[str, Any], warnings: List[str]) -> float:
        """Validate completeness of parsed data."""
        required_fields = ["kills", "deaths", "assists", "hero", "gold"]
        optional_fields = ["hero_damage", "turret_damage", "damage_taken", "teamfight_participation"]
        
        required_score = sum(1 for field in required_fields if parsed_data.get(field))
        optional_score = sum(1 for field in optional_fields if parsed_data.get(field))
        
        completeness = (required_score / len(required_fields)) * 0.7 + (optional_score / len(optional_fields)) * 0.3
        
        if completeness < 0.5:
            warnings.append(f"Low data completeness: {completeness:.1%}")
        
        return completeness
    
    def _calculate_overall_confidence(
        self, hero_confidence: float, warning_count: int, data_field_count: int, completeness_score: float
    ) -> float:
        """Calculate overall confidence score with data-first approach."""
        # NEW APPROACH: Data quality is primary factor, hero detection is secondary
        confidence = 0.0
        
        # 1. DATA COMPLETENESS: 50% weight - most important factor
        confidence += completeness_score * 0.50
        
        # 2. CRITICAL DATA PRESENCE: 25% weight - KDA + Gold = viable analysis
        critical_data_score = 0.0
        if data_field_count >= 4:  # KDA + Gold minimum
            critical_data_score = 0.8
        elif data_field_count >= 2:  # Partial data
            critical_data_score = 0.4
        elif data_field_count >= 1:  # Some data
            critical_data_score = 0.2
        confidence += critical_data_score * 0.25
        
        # 3. HERO DETECTION: Only 10% weight - nice to have, not essential
        confidence += hero_confidence * 0.10
        
        # 4. BASE CONFIDENCE: 15% - ensures viable minimum
        base_confidence = 0.15
        confidence += base_confidence
        
        # BONUSES for high-quality extraction
        quality_bonus = 0.0
        if completeness_score >= 0.8:  # Excellent data quality
            quality_bonus += 0.15
        elif completeness_score >= 0.6:  # Good data quality
            quality_bonus += 0.10
        
        if data_field_count >= 6:  # Rich data set
            quality_bonus += 0.10
        
        confidence += quality_bonus
        
        # WARNING PENALTIES (minimal impact)
        warning_penalty = min(0.20, warning_count * 0.05)  # Max 20% penalty
        confidence -= warning_penalty
        
        # SMART FLOOR: Based on actual data extracted
        if data_field_count >= 3 and completeness_score >= 0.3:
            minimum_confidence = 0.30  # We have usable data
        elif data_field_count >= 1:
            minimum_confidence = 0.15  # We have some data  
        else:
            minimum_confidence = 0.05  # Almost no data
        
        # Ensure valid range with smart boundaries
        final_confidence = max(minimum_confidence, min(1.0, confidence))
        
        logger.debug(f"Confidence calculation: completeness={completeness_score:.3f}, "
                    f"fields={data_field_count}, hero={hero_confidence:.3f}, "
                    f"warnings={warning_count}, final={final_confidence:.3f}")
        
        return float(final_confidence)
    
    def _calculate_screenshot_confidence(self, text_content: str, warnings: List[str]) -> float:
        """Calculate confidence score based on detected elements."""
        confidence_factors = []
        
        # Check for key UI elements
        ui_elements = ["duration", "gold", "damage", "kda", "kills", "deaths", "assists"]
        detected_elements = sum(1 for element in ui_elements if element in text_content.lower())
        confidence_factors.append(detected_elements / len(ui_elements))
        
        # Penalty for warnings
        warning_penalty = max(0, 1 - (len(warnings) * 0.1))
        confidence_factors.append(warning_penalty)
        
        # Check for match result clarity
        result_clarity = 1.0 if any(word in text_content.lower() for word in ["victory", "defeat"]) else 0.7
        confidence_factors.append(result_clarity)
        
        # Average confidence - ensure Python float return type
        overall_confidence = sum(confidence_factors) / len(confidence_factors)
        return float(round(overall_confidence, 2))
    
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