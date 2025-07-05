from core.base_evaluator import BaseEvaluator
from typing import List, Dict, Tuple


class TigrealEvaluator(BaseEvaluator):
    """
    Tigreal-specific evaluation logic.
    Inherits from BaseEvaluator for common metrics and applies
    Tigreal-specific thresholds from the config.
    """

    def _evaluate_hero_specific(self, data: Dict, thresholds: Dict,
                                hero: str) -> List[Tuple[str, str]]:
        """
        Tigreal-specific checks. For now, we can add a simple check.
        In the future, this could include checks for successful multi-hero
        ultimates or peel effectiveness.
        """
        fb = []
        
        # Example hero-specific check
        if data.get("deaths", 0) > 8:
            fb.append(("critical",
                       "Over 8 deaths on Tigreal is very high. "
                       "Ensure your engages are followed up by your team."))
            
        return fb 