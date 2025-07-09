import json
import re
import logging
from typing import Dict, Any, List, Tuple
from difflib import get_close_matches

import easyocr
import cv2
import numpy as np
from pydantic import ValidationError

from .schemas import Matches
from .ign_validator import IGNValidator, IGNMatch

# --- Configuration ---
# Configure logging to provide insights into the OCR parsing process.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize the EasyOCR reader. 
# Using lazy loading pattern to avoid startup overhead.
ocr_reader = None


def get_ocr_reader():
    """Lazy initialization of OCR reader to improve startup time."""
    global ocr_reader
    if ocr_reader is None:
        ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return ocr_reader


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

        # Heuristic: KDA is 3 numbers, gold is a larger number.
        # This is more robust than simple sorting.
        # Gold is the max number > 1000.
        gold_candidates = [n for n in all_numbers if n > 1000]
        if gold_candidates:
            data["gold"] = max(gold_candidates)
            logging.info(f"Parsed Gold: {data['gold']}")

        # KDA numbers are all other numbers, typically < 50
        kda_numbers = [n for n in all_numbers if n < 50]
        if len(kda_numbers) >= 3:
            # We assume the three numbers appearing together are KDA
            # This is still a heuristic, but improved.
            # Example row text: ['Lesz XVII', '9', '0   6', '8920']
            # We need to find the "9 0 6" pattern.
            full_row_text = " ".join(row_texts)
            kda_match = re.search(r"(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})", full_row_text)
            if kda_match:
                 k, d, a = (int(g) for g in kda_match.groups())
                 if k in kda_numbers and d in kda_numbers and a in kda_numbers:
                    data["kills"] = k
                    data["deaths"] = d
                    data["assists"] = a
                    logging.info(f"Parsed KDA: {k}/{d}/{a}")

        # For damage stats, we'll pull from the same numbers. This is a fallback.
        remaining_numbers = sorted(
            [n for n in gold_candidates if n != data.get("gold")], 
            reverse=True
        )
        if len(remaining_numbers) >= 2:
            data["hero_damage"] = remaining_numbers[0]
            data["damage_taken"] = remaining_numbers[1]
            if len(remaining_numbers) >= 3:
                data["turret_damage"] = remaining_numbers[2]
        
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