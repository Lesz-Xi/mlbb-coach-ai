import unittest
import json
import os
import sys

# Ensure the app's root directory is in the path for imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

from core.data_collector import DataCollector


class TestDataCollectorNewParser(unittest.TestCase):
    """Unit tests for the new key-value parsing logic in DataCollector."""

    @classmethod
    def setUpClass(cls):
        """Load mock OCR data from the fixture file before tests run."""
        fixture_path = os.path.join(
            os.path.dirname(__file__), "fixtures", "mock_ocr_data.json"
        )
        with open(fixture_path, 'r') as f:
            cls.mock_ocr_data = json.load(f)
    
    def setUp(self):
        """Instantiate a new DataCollector for each test."""
        self.collector = DataCollector()

    def test_key_value_parser_happy_path(self):
        """
        Tests the key-value parser with a complete and high-confidence result.
        """
        # GIVEN a good OCR result
        good_result = self.mock_ocr_data['good_ocr_result']
        
        # WHEN the result is parsed using the new key-value method
        parsed_output = self.collector._parse_ocr_result_key_value(good_result)
        
        # THEN the data should be parsed correctly
        data = parsed_output['data']
        self.assertEqual(data['hero'], 'lancelot')
        self.assertEqual(data['kills'], 10)
        self.assertEqual(data['deaths'], 2)
        self.assertEqual(data['assists'], 8)
        self.assertEqual(data['gpm'], 850)
        self.assertEqual(data['hero_damage'], 95000)
        self.assertEqual(data['gold_earned'], 25000)
        
        # THEN there should be few or no warnings
        warnings = parsed_output['warnings']
        self.assertLessEqual(len(warnings), 3) # Allow for a few missing non-critical stats

    def test_key_value_parser_sad_path(self):
        """
        Tests the key-value parser with a partial and low-confidence result.
        """
        # GIVEN a bad OCR result
        bad_result = self.mock_ocr_data['bad_ocr_result']

        # WHEN the result is parsed
        parsed_output = self.collector._parse_ocr_result_key_value(bad_result)

        # THEN the hero should be 'unknown' due to low confidence
        data = parsed_output['data']
        self.assertEqual(data['hero'], 'unknown')

        # THEN the KDA and GPM should be parsed correctly
        self.assertEqual(data['kills'], 1)
        self.assertEqual(data['deaths'], 9)
        self.assertEqual(data['assists'], 3)
        self.assertEqual(data['gpm'], 420)
        # It should fall back to the default value for hero_damage
        self.assertEqual(data['hero_damage'], 0)

        # THEN there should be specific warnings
        warnings = parsed_output['warnings']
        self.assertIn("Hero match confidence low.", warnings)
        self.assertIn("Could not parse turret_damage.", warnings)
        self.assertIn("Could not parse healing_done.", warnings)


if __name__ == "__main__":
    unittest.main() 