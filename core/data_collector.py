import json
import re
import logging
import threading
import time
from typing import Dict, Any, List, Tuple
from difflib import get_close_matches

import easyocr
import cv2
import numpy as np
from pydantic import ValidationError

from .schemas import Matches
from .ign_validator import IGNValidator

# --- Configuration ---
# Configure logging to provide insights into the OCR parsing process.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# PERFORMANCE OPTIMIZATION: Thread-safe OCR Reader Singleton
class OCRReaderSingleton:
    """
    Thread-safe singleton for EasyOCR reader with performance optimization.
    
    Features:
    - Lazy initialization (only create when first needed)
    - Thread-safe access using double-checked locking
    - Performance monitoring and caching
    - Memory-efficient reader reuse
    """
    
    _instance = None
    _lock = threading.Lock()
    _reader = None
    _initialization_time = None
    _access_count = 0
    
    def __new__(cls):
        # Double-checked locking pattern for thread safety
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(OCRReaderSingleton, cls).__new__(cls)
        return cls._instance
    
    def get_reader(self):
        """Get the OCR reader instance with lazy initialization."""
        if self._reader is None:
            with self._lock:
                if self._reader is None:
                    start_time = time.time()
                    logging.info("🔄 Initializing EasyOCR reader (one-time setup)...")
                    
                    # Initialize with optimized settings for performance
                    self._reader = easyocr.Reader(
                        ['en'],
                        gpu=False,      # CPU mode for stability
                        verbose=False,  # Reduce logging overhead
                        model_storage_directory=None,  # Use default caching
                        download_enabled=True,
                        detector=True,
                        recognizer=True,
                        width_ths=0.7,    # Optimized for MLBB screenshots
                        height_ths=0.7,   # Optimized for MLBB screenshots
                    )
                    
                    self._initialization_time = time.time() - start_time
                    logging.info(f"✅ OCR reader initialized in {self._initialization_time:.3f}s")
        
        # Track usage for performance analytics
        self._access_count += 1
        if self._access_count % 10 == 0:
            init_time = self._initialization_time
            logging.debug(f"📊 OCR reader accessed {self._access_count} times "
                          f"(init time: {init_time:.3f}s)")
        
        return self._reader
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the OCR reader."""
        return {
            "initialized": self._reader is not None,
            "initialization_time": self._initialization_time,
            "access_count": self._access_count,
            "memory_efficient": True,
            "thread_safe": True
        }
    
    def reset_reader(self):
        """Reset the reader (for testing or error recovery)."""
        with self._lock:
            if self._reader is not None:
                logging.info("🔄 Resetting OCR reader...")
                # EasyOCR doesn't have explicit cleanup, but we can reset
                self._reader = None
                self._initialization_time = None
                logging.info("✅ OCR reader reset complete")


# Global singleton instance
_ocr_singleton = OCRReaderSingleton()


def get_ocr_reader():
    """
    OPTIMIZED: Get OCR reader with singleton pattern and performance monitoring.
    
    Returns:
        EasyOCR reader instance (cached and reused)
    """
    return _ocr_singleton.get_reader()


def get_ocr_stats() -> Dict[str, Any]:
    """Get OCR reader performance statistics."""
    return _ocr_singleton.get_stats()


def reset_ocr_reader():
    """Reset OCR reader (for testing or error recovery)."""
    _ocr_singleton.reset_reader()


# A list of heroes for fuzzy matching. This improves detection accuracy.
HERO_LIST = [
    'miya', 'layla', 'franco', 'tigreal', 'kagura', 'eudora', 'lancelot',
    'fanny', 'estes', 'angela', 'chou', 'zilong', 'fredrinn', 'hayabusa',
    'unknown'
]


class DataCollector:
    """
    Handles data collection from various sources and validates it
    against the defined Pydantic schemas.
    """

    def from_json_upload(
        self, file_path: str
    ) -> List[Dict[str, Any]]:
        """
        Loads and validates a list of matches from a JSON file.

        Args:
            file_path: The path to the JSON file.

        Returns:
            A list of dictionaries, each a validated match.

        Raises:
            FileNotFoundError: If the file is not found.
            ValueError: If the file is not valid JSON or fails validation.
        """
        try:
            with open(file_path, 'r') as f:
                raw_data_list = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at path: {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {file_path}")

        try:
            # Validate the entire list using the Matches schema
            validated_matches = Matches(data=raw_data_list)
            return [match.model_dump() for match in validated_matches.data]
        except ValidationError as e:
            raise ValueError(f"JSON data failed validation: {e}")

    def from_manual_input(
        self, form_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validates a single match from a manual input form (e.g., a web form).

        Args:
            form_data: A dictionary containing the raw match data.

        Returns:
            A dictionary with the validated match data.

        Raises:
            ValueError: If the data fails validation.
        """
        try:
            # Use the Matches schema to validate a single match
            validated_matches = Matches(data=[form_data])
            return validated_matches.data[0].model_dump()
        except ValidationError as e:
            raise ValueError(f"Manual input data failed validation: {e}")

    def from_screenshot(
        self, ign: str, image_path: str, hero_override: str = None,
        known_igns: List[str] = None
    ) -> Dict[str, Any]:
        """
        Extracts match data from a single screenshot using EasyOCR.
        It locates the player by their In-Game Name (IGN) to ensure data
        is extracted for the correct user.

        Args:
            ign: The In-Game Name of the player to find.
            image_path: Path to the screenshot.
            hero_override: Manually specified hero name to bypass detection.
            known_igns: List of known IGNs for validation (optional).
        
        Returns:
            A dictionary with the validated match data, confidence scores,
            and any parsing warnings.
        """
        logging.info("Processing screenshot for IGN: %s", ign)
        
        # Initialize IGN validator
        validator = IGNValidator()
        
        logging.info("Preprocessing image for better OCR accuracy...")
        preprocessed_image = self._preprocess_image(image_path)
        
        reader = get_ocr_reader()
        warnings = []
        parsed_data = {}

        results = reader.readtext(preprocessed_image, detail=1)
        if not results:
            warnings.append("Screenshot OCR returned no results.")
        else:
            # Validate IGN using the new validator
            validated_ign = ign
            for bbox, text, conf in results:
                validation_result = validator.validate_mlbb_ign(text)
                if validation_result['is_valid'] and ign.lower() in validation_result['cleaned_ign'].lower():
                    validated_ign = validation_result['cleaned_ign']
                    logging.info(f"IGN validation successful: {validated_ign} (confidence: {validation_result['confidence']:.3f})")
                    if validation_result['confidence'] < 0.8:
                        warnings.append(f"IGN match confidence low: {validation_result['confidence']:.3f}")
                    break
            else:
                logging.warning(f"IGN validation failed for '{ign}'")
                warnings.append(f"IGN '{ign}' not found in OCR results.")
            
            # This unified parser will get all stats from the single image
            parsed_data = self._parse_player_row(validated_ign, results, hero_override)
            if not parsed_data:
                warnings.append(f"Could not find player '{validated_ign}' in screenshot.")

        # --- Final Data Assembly & Defaults ---
        if hero_override:
            parsed_data['hero'] = hero_override
            logging.info(f"Using manual hero override: {hero_override}")
        else:
            # Hero is now detected within _parse_player_row.
            if parsed_data.get("hero", "unknown") == "unknown":
                warnings.append("Could not confidently detect hero name.")

        # Match Duration & GPM
        duration_text = " ".join([res[1] for res in results])
        duration_match = re.search(r"Duration (\d{2}):(\d{2})", duration_text)
        if duration_match:
            minutes = int(duration_match.group(1))
            seconds = int(duration_match.group(2))
            parsed_data["match_duration"] = minutes + seconds / 60
        else:
            parsed_data["match_duration"] = 10

        if parsed_data.get("gold") and parsed_data["match_duration"] > 0:
            gpm = parsed_data["gold"] / parsed_data["match_duration"]
            parsed_data["gold_per_min"] = round(gpm)
        else:
            parsed_data["gold_per_min"] = 0
        
        # Add all other defaults
        parsed_data.setdefault("kills", 0)
        parsed_data.setdefault("deaths", 1)
        parsed_data.setdefault("assists", 0)
        parsed_data.setdefault("hero_damage", 0)
        parsed_data.setdefault("turret_damage", 0)
        parsed_data.setdefault("damage_taken", 0)
        parsed_data.setdefault("teamfight_participation", 0)
        parsed_data.setdefault("positioning_rating", "average")
        parsed_data.setdefault("ult_usage", "average")

        # Validate the parsed data before returning
        try:
            validated_matches = Matches(data=[parsed_data])
            final_data = validated_matches.data[0].model_dump()
            logging.info("OCR data parsed and validated successfully.")
            return {"data": final_data, "warnings": warnings}
        except ValidationError as e:
            logging.error(
                "OCR Parsing Validation Error: %s. Data: %s", e, parsed_data
            )
            warnings.append(f"Validation Error: {e.errors()[0]['msg']}")
            return {"data": parsed_data, "warnings": warnings}

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Applies pre-processing techniques to an image to improve OCR accuracy.
        """
        img = cv2.imread(image_path)
        
        # Convert to grayscale - this is often enough to improve contrast.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # --- Debugging: Save the processed image ---
        # This helps us visually inspect what the OCR engine is "seeing".
        output_path = "temp/preprocessed_image.png"
        cv2.imwrite(output_path, gray)
        logging.info(f"Saved pre-processed debug image to: {output_path}")
        
        return gray

    def _parse_player_row(
        self, ign: str, ocr_results: List[Tuple[Any, str, float]],
        hero_override: str = None
    ) -> Dict[str, Any]:
        """
        Finds the horizontal row for a given player IGN and extracts relevant
        stats from that single row.
        """
        player_row_y_center = -1
        ign_found = False

        for bbox, text, conf in ocr_results:
            if ign.lower() in text.lower():
                y_coords = [point[1] for point in bbox]
                player_row_y_center = (min(y_coords) + max(y_coords)) / 2
                ign_found = True
                logging.info(
                    f"Found IGN '{ign}' at y-center: {player_row_y_center}"
                )
                break
        
        if not ign_found:
            return {}

        row_texts = []
        for bbox, text, conf in ocr_results:
            y_coords = [point[1] for point in bbox]
            item_y_center = (min(y_coords) + max(y_coords)) / 2
            if abs(item_y_center - player_row_y_center) < 30:
                row_texts.append(text)
        
        logging.info(f"Texts found in player row: {row_texts}")
        
        data = {}

        # If not using an override, attempt to detect the hero from the row
        if not hero_override:
            player_row_text = " ".join(row_texts).lower()
            detected_heroes = [
                hero for hero in HERO_LIST if hero in player_row_text
            ]
            if detected_heroes:
                hero_name = detected_heroes[0]
            else:
                # Fallback to fuzzy match
                matches = get_close_matches(player_row_text, HERO_LIST, n=1, cutoff=0.8)
                hero_name = matches[0] if matches else "unknown"
            data['hero'] = hero_name
            logging.info(f"Parsed Hero: {data['hero']}")
        
        all_numbers = []
        for text in row_texts:
            all_numbers.extend([int(n) for n in re.findall(r'\b\d+\b', text)])

        # SPATIAL ALIGNMENT FIX: Extract data based on left-to-right position
        # Build list of (number, position) pairs from row texts
        spatial_data = []
        for i, text in enumerate(row_texts):
            numbers = [int(n) for n in re.findall(r'\b\d+\b', text)]
            for num in numbers:
                spatial_data.append((num, i))  # (value, position_index)
        
        # Sort by spatial position (left to right)
        spatial_data.sort(key=lambda x: x[1])
        
        # Extract KDA first (small numbers, typically first)
        kda_candidates = [(n, pos) for n, pos in spatial_data if n < 50]
        if len(kda_candidates) >= 3:
            # Take first 3 small numbers by position as K/D/A
            k, d, a = [n for n, pos in kda_candidates[:3]]
            data["kills"] = k
            data["deaths"] = max(1, d)  # Deaths can't be 0
            data["assists"] = a
            logging.info(f"Parsed KDA (spatial): {k}/{d}/{a}")
            
        # Extract gold (medium-large numbers, after KDA position)
        gold_candidates = [(n, pos) for n, pos in spatial_data 
                           if 500 <= n <= 50000]
        
        if gold_candidates:
            # Find KDA position to locate gold relative to it
            kda_positions = ([pos for n, pos in kda_candidates[:3]] 
                             if len(kda_candidates) >= 3 else [])
            
            if kda_positions:
                # Look for gold after KDA position
                max_kda_pos = max(kda_positions)
                gold_after_kda = [(n, pos) for n, pos in gold_candidates 
                                  if pos > max_kda_pos]
                
                if gold_after_kda:
                    data["gold"] = gold_after_kda[0][0]
                    logging.info(f"Parsed Gold (spatial, after KDA): "
                                 f"{data['gold']}")
                else:
                    # Fallback: use spatial ordering
                    data["gold"] = gold_candidates[0][0]
                    logging.info(f"Parsed Gold (spatial fallback): "
                                 f"{data['gold']}")
            else:
                # No KDA reference, use spatial ordering
                data["gold"] = gold_candidates[0][0] 
                logging.info(f"Parsed Gold (spatial, no KDA ref): "
                             f"{data['gold']}")
        else:
            logging.warning("No gold candidates found in range 500-50000")
            # FALLBACK: Try broader gold range or largest number approach
            broader_gold = [(n, pos) for n, pos in spatial_data if 1000 <= n <= 100000]
            if broader_gold:
                data["gold"] = broader_gold[0][0]  # First by position
                logging.info(f"FALLBACK: Using broader gold range: {data['gold']}")
            else:
                # Last resort: largest number in all_numbers
                large_numbers = [n for n in all_numbers if n >= 1000]
                if large_numbers:
                    data["gold"] = max(large_numbers)
                    logging.info(f"FALLBACK: Using largest number as gold: {data['gold']}")

        # For damage stats, extract large numbers (spatial approach)
        damage_candidates = [(n, pos) for n, pos in spatial_data if 1000 <= n <= 500000]
        if damage_candidates:
            # Sort by spatial position
            damage_candidates.sort(key=lambda x: x[1])
            # Filter out gold value to avoid duplication
            gold_value = data.get("gold", 0)
            damage_only = [(n, pos) for n, pos in damage_candidates if n != gold_value]
            
            if damage_only:
                # Use first 2-3 damage values by spatial position
                damage_values = [n for n, pos in damage_only]
                if len(damage_values) >= 1:
                    data["hero_damage"] = damage_values[0]
                if len(damage_values) >= 2:
                    data["damage_taken"] = damage_values[1]
                if len(damage_values) >= 3:
                    data["turret_damage"] = damage_values[2]
                logging.info(f"Parsed damage values (spatial): {damage_values[:3]}")
        
        # Teamfight is a percentage
        tfp_match = re.search(r"(\d{1,3})%", " ".join(row_texts))
        if tfp_match:
            data["teamfight_participation"] = int(tfp_match.group(1))

        # Add hero-specific default fields if needed
        if data.get("hero") == "franco":
            data.setdefault("hooks_landed", 0)
            data.setdefault("team_engages", 0)
            data.setdefault("vision_score", 0)

        return data

    def _find_nearest_value(self, keyword: str, lines: List[str]) -> str:
        """Finds the most likely numerical value associated with a keyword."""
        best_candidate = None
        for line in lines:
            if keyword in line:
                # Find all numbers in the rest of the line
                numbers = re.findall(r'\d+\.?\d*', line.split(keyword)[-1])
                if numbers:
                    # Simple heuristic: take the first number found after the keyword
                    best_candidate = numbers[0]
                    break
        return best_candidate

    def _parse_ocr_result_key_value(
        self, ocr_lines: List[Tuple[str, float]]
    ) -> Dict[str, Any]:
        """
        Parses OCR results using a combination of pattern matching and heuristics.
        """
        logging.info("Starting key-value parsing for %d lines.", len(ocr_lines))

        # Use logging.debug for verbose output, keeping terminal clean
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("=== Full OCR Text ===")
            for i, (text, confidence) in enumerate(ocr_lines):
                logging.debug(f"{i:2d}: {confidence:.2f} | {text}")
            logging.debug("=== End Full OCR Text ===")
        
        parsed_data = {}
        warnings = []

        text_lines = [line[0].lower() for line in ocr_lines]
        all_text_blob = " ".join(text_lines)

        # --- 1. Hero and Player Detection ---
        # Find the main player using IGN validation
        validator = IGNValidator()
        potential_igns = validator.extract_potential_igns([([], line, 0.8) for line in text_lines])
        player_name_found = any(ign.lower() in all_text_blob for ign in potential_igns)
        
        if player_name_found:
            logging.info("Primary player found in screenshot.")
            # If we find the main player, we can be more confident in hero detection
            # on the same line or nearby. This part can be enhanced later.
        
        hero_name = "unknown"
        for line in text_lines:
            match = get_close_matches(line, HERO_LIST, n=1, cutoff=0.7)
            if match:
                hero_name = match[0]
                logging.info(f"Detected hero: {hero_name} from line: '{line}'")
                break
        parsed_data['hero'] = hero_name
        if hero_name == 'unknown':
            warnings.append("Could not confidently detect hero name.")

        # --- 2. KDA Extraction ---
        kda_found = False
        for line in text_lines:
            # Pattern 1: "9 / 0 / 6" (with slashes)
            match = re.search(r"\b(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{1,2})\b", line)
            if not match:
                # Pattern 2: "9 0 6" (with spaces)
                match = re.search(r"\b(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\b", line)
            
            if match:
                kills, deaths, assists = (int(g) for g in match.groups())
                # Sanity check KDA values
                if kills <= 40 and deaths <= 40 and assists <= 60:
                    parsed_data["kills"] = kills
                    parsed_data["deaths"] = max(1, deaths)
                    parsed_data["assists"] = assists
                    logging.info(f"Parsed KDA: {kills}/{deaths}/{assists}")
                    kda_found = True
                    break
        if not kda_found:
            warnings.append("KDA not found.")

        # --- 3. Heuristic-based stat extraction ---
        # Find all numbers in the OCR output for heuristic analysis
        all_numbers = []
        for text, confidence in ocr_lines:
            numbers = re.findall(r'\d+', text)
            for num in numbers:
                all_numbers.append(int(num))
        all_numbers.sort(reverse=True)
        
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Found numbers for heuristics: %s", all_numbers)

        # Hero damage is typically the largest number
        damage_candidates = [n for n in all_numbers if 10000 <= n <= 200000]
        if damage_candidates:
            parsed_data['hero_damage'] = damage_candidates[0]
            logging.info(f"Assigned hero_damage = {damage_candidates[0]}")
        
        # --- 4. Fill in missing data for validation ---
        parsed_data.setdefault("hero", "unknown")
        parsed_data.setdefault("kills", 0)
        parsed_data.setdefault("deaths", 1)
        parsed_data.setdefault("assists", 0)
        parsed_data.setdefault("gold_per_min", 0)
        parsed_data.setdefault("hero_damage", 0)
        parsed_data.setdefault("turret_damage", 0)
        parsed_data.setdefault("damage_taken", 0)
        parsed_data.setdefault("teamfight_participation", 0)
        parsed_data.setdefault("positioning_rating", "average")
        parsed_data.setdefault("ult_usage", "average")
        parsed_data.setdefault("match_duration", 15)

        # Add hero-specific default fields if needed
        if parsed_data.get("hero") == "franco":
            parsed_data.setdefault("hooks_landed", 0)
            parsed_data.setdefault("team_engages", 0)
            parsed_data.setdefault("vision_score", 0)

        return {
            "data": parsed_data,
            "confidence": {},  # Confidence is now implicit in detection
            "warnings": warnings,
        }

    def _parse_ocr_result(
        self, ocr_lines: List[Tuple[str, float]]
    ) -> Dict[str, Any]:
        """
        This is the old, deprecated parser. It is kept for reference but
        should not be used. The new `_parse_ocr_result_key_value` is preferre d.
        """
        # All the old, problematic code is now isolated here.
        # We will leave it for now but it will be removed later.
        return {}