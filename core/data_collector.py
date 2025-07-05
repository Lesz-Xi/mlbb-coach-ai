import json
import re
from typing import Dict, Any, List

from paddleocr import PaddleOCR
from pydantic import ValidationError
from .schemas import AnyMatch, Matches

# Initialize the PaddleOCR reader. 
# This is done once when the module is imported, which is efficient.
# use_angle_cls=True helps in handling rotated text.
# lang='en' sets the language to English.
ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')


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
            validated_model = AnyMatch.model_validate(form_data)
            return validated_model.model_dump()
        except ValidationError as e:
            raise ValueError(f"Manual input data failed validation: {e}")

    def from_screenshot(self, image_path: str) -> Dict[str, Any]:
        """
        Extracts match data from a screenshot using PaddleOCR.

        Args:
            image_path: The path to the screenshot image.
        
        Returns:
            A dictionary with the validated match data.
        """
        result = ocr_reader.ocr(image_path, cls=True)
        
        if result and result[0]:
            # result[0] is the first (and only) page/image processed
            text_lines = [line[1][0] for line in result[0]] # Extract just the text
            return self._parse_ocr_result(text_lines)
        
        return {}

    def _parse_ocr_result(self, text_lines: List[str]) -> Dict[str, Any]:
        """
        Parses a list of text lines from OCR into a structured match dictionary.
        """
        parsed_data = {}
        
        # Regex patterns for parsing specific stats
        kda_pattern = re.compile(r'(\d+)\s*/\s*(\d+)\s*/\s*(\d+)')
        percent_pattern = re.compile(r'(\d+\.?\d*)%')
        gpm_pattern = re.compile(r'(\d+)\s*GPM')

        # Lowercase all lines for case-insensitive matching
        lower_text_lines = [line.lower() for line in text_lines]

        for i, line in enumerate(lower_text_lines):
            # --- Player Info ---
            if "hero" in line:  # A simple heuristic
                # Assume hero name is the next significant word
                try:
                    parsed_data["hero"] = text_lines[i].split(':')[1].strip()
                except IndexError:
                    pass  # Or look at next line
            
            # --- Core Stats ---
            kda_match = kda_pattern.search(line)
            if kda_match:
                parsed_data["kills"] = int(kda_match.group(1))
                parsed_data["deaths"] = int(kda_match.group(2))
                parsed_data["assists"] = int(kda_match.group(3))

            gpm_match = gpm_pattern.search(line)
            if gpm_match:
                parsed_data["gpm"] = int(gpm_match.group(1))

            # --- Performance Metrics ---
            if "teamfight" in line:
                percent_match = percent_pattern.search(line)
                if percent_match:
                    parsed_data["teamfight_participation"] = float(
                        percent_match.group(1))

            if "damage" in line:
                percent_match = percent_pattern.search(line)
                if percent_match:
                    parsed_data["damage_dealt"] = float(percent_match.group(1))
            
            if "turret" in line:
                percent_match = percent_pattern.search(line)
                if percent_match:
                    parsed_data["turret_damage"] = float(percent_match.group(1))

        # We need to fill in some required dummy data for validation to pass
        # as not all fields can be reliably read from a screenshot.
        parsed_data.setdefault("match_id", "ocr_match")
        parsed_data.setdefault("player_id", "ocr_player")
        parsed_data.setdefault("win", True)  # Cannot determine from stats
        parsed_data.setdefault("duration_seconds", 900)  # Cannot determine

        # Set defaults for any values we failed to parse
        parsed_data.setdefault("hero", "Unknown")
        parsed_data.setdefault("kills", 0)
        parsed_data.setdefault("deaths", 0)
        parsed_data.setdefault("assists", 0)
        parsed_data.setdefault("gpm", 0)
        parsed_data.setdefault("teamfight_participation", 0.0)
        parsed_data.setdefault("damage_dealt", 0.0)
        parsed_data.setdefault("turret_damage", 0.0)

        # Validate the parsed data before returning
        try:
            validated_match = AnyMatch.model_validate(parsed_data)
            return validated_match.model_dump()
        except ValidationError as e:
            # If validation fails, we can return the error or an empty dict
            print(f"OCR Parsing Validation Error: {e}")
            return {} 