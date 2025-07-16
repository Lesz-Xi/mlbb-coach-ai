"""
A/B Testing for ML Models
"""

from typing import Dict, Any
from .model_version import ModelVersion


class ABTester:
    """A/B testing for model versions"""
    
    def __init__(self):
        self.active_tests: Dict[str, Any] = {}
    
    def create_test(
        self,
        name: str,
        model_a: ModelVersion,
        model_b: ModelVersion,
        traffic_split: float = 0.5
    ):
        """Create an A/B test"""
        self.active_tests[name] = {
            "model_a": model_a,
            "model_b": model_b,
            "traffic_split": traffic_split,
            "results": {"a": [], "b": []}
        }
    
    def get_model_for_request(self, test_name: str) -> ModelVersion:
        """Get model for a request based on traffic split"""
        import random
        test = self.active_tests.get(test_name)
        if not test:
            return None
        
        if random.random() < test["traffic_split"]:
            return test["model_a"]
        return test["model_b"] 