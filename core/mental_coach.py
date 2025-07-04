from typing import List, Dict, Any
import random


class MentalCoach:
    """
    Generates psychological and encouraging feedback based on performance
    trends.
    """

    def __init__(
        self, player_history: List[Dict[str, Any]], player_goal: str
    ):
        self.history = player_history
        self.goal = player_goal
        self.avg_kda = self._calculate_average_kda()

        self.messages = {
            "trending_down": [
                "Everyone has off games. What matters is that you're here to "
                "review and improve. Let's focus on what we can learn.",
                "A tough match, but every loss is a lesson in disguise. "
                "Let's look at the data, not the scoreboard.",
                "Don't let one result discourage you. Your overall progress "
                "is what counts. What's one thing you learned this game?"
            ],
            "consistent": [
                "Consistency is the foundation of mastery. You're building a "
                "strong baseline. Keep stacking good games.",
                "Another solid performance. This is your new normal. Now, "
                "let's look for those small edges to push even higher.",
                "This level of play is becoming second nature to you. That's a "
                "great sign of deep learning."
            ],
            "trending_up": [
                "You're trending upward. You may not feel it yet, but that's "
                "momentum. Let's keep it rolling.",
                "Excellent work. All those small adjustments are starting to "
                "pay off in a big way. What felt different for you this match?",
                "This is the result of your hard work. You're not just "
                "playing; you're actively improving. Great job."
            ]
        }

    def _calculate_average_kda(self) -> float:
        """Calculates the average KDA from the match history."""
        if not self.history:
            return 3.0  # Default KDA if no history
        
        total_kda = 0
        for match in self.history:
            k = match.get("kills", 0)
            d = match.get("deaths", 1)  # Avoid division by zero
            a = match.get("assists", 0)
            total_kda += (k + a) / d
        
        return total_kda / len(self.history)

    def get_mental_boost(self, current_match: Dict[str, Any]) -> str:
        """
        Provides contextual encouragement based on the most recent match.
        """
        k = current_match.get("kills", 0)
        d = current_match.get("deaths", 1)
        a = current_match.get("assists", 0)
        current_kda = (k + a) / d

        if current_kda < self.avg_kda * 0.8:
            category = "trending_down"
        elif current_kda > self.avg_kda * 1.2:
            category = "trending_up"
        else:
            category = "consistent"
            
        # Add goal-oriented feedback
        boost = random.choice(self.messages[category])
        gpm = current_match.get("gold_per_min", 0)
        if self.goal == "improve_early_game" and gpm < 600:
            boost += (
                " As we're focusing on your early game, let's pay special "
                "attention to GPM next match."
            )
        
        return boost 