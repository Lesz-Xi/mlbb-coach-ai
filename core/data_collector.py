import json
from typing import Dict, Any, List

from pydantic import ValidationError
from .schemas import AnyMatch, Matches


class DataCollector:
    """
    Handles data collection from various sources and validates it
    against the defined Pydantic schemas.
    """

    def from_json_upload(
        self, file_path: str
    ) -> List[Dict[str, Any]]:
        """
        Loads a list of matches from a JSON file, validates them, and returns them.

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

    def from_screenshot(self, image_path: str):
        """
        (Placeholder) Extracts match data from a screenshot using OCR.

        Args:
            image_path: The path to the screenshot image.
        """
        raise NotImplementedError(
            "Screenshot OCR functionality is not yet implemented."
        ) 