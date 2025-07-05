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
        d = max(1, current_match.get("deaths", 1))  # Ensure d is at least 1
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

    def get_progress_journal(self) -> str:
        """
        Analyzes the entire match history to generate a reflective summary
        of a player's progress and tendencies.
        """
        if len(self.history) < 5:
            return (
                "Play a few more games to unlock your first Progress Journal entry!"
            )

        # --- Analysis ---
        roles = [self._get_role(m['hero']) for m in self.history]
        most_common_role = max(set(roles), key=roles.count)

        high_gpm_matches = sorted(
            self.history, key=lambda m: m.get('gold_per_min', 0), reverse=True
        )
        best_gpm_hero = high_gpm_matches[0]['hero']

        # --- Narrative Generation ---
        part1 = (
            f"Looking at your last {len(self.history)} games, a few patterns "
            f"are emerging. You seem to be gravitating towards the "
            f"**{most_common_role}** role, showing a real desire to "
            f"control the flow of the game."
        )
        part2 = (
            f"Your standout strength is farming efficiency, especially on "
            f"resource-hungry heroes like **{best_gpm_hero.title()}**. "
            f"This ability to build a gold lead is a massive asset."
        )
        part3 = (
            "A recurring theme is your proactive engagement in teamfights. "
            "This shows great map awareness, but let's ensure every fight "
            "is for a key objective."
        )

        # --- Goal-Oriented Feedback ---
        goal_feedback = self._generate_goal_focused_feedback()

        part4 = (
            "You're evolving into a decisive, impactful player. "
            "Keep focusing on that a tactical decision-making."
        )

        return f"{part1}\n\n{part2}\n\n{part3}\n\n{goal_feedback}\n\n{part4}"

    def _generate_goal_focused_feedback(self) -> str:
        """Analyzes history to provide feedback on the player's goal."""
        if self.goal == "improve_early_game" and len(self.history) >= 4:
            # Split history for trend analysis
            mid_point = len(self.history) // 2
            first_half = self.history[:mid_point]
            second_half = self.history[mid_point:]

            gpm_first_half = sum(
                m.get('gold_per_min', 0) for m in first_half
            ) / len(first_half)
            gpm_second_half = sum(
                m.get('gold_per_min', 0) for m in second_half
            ) / len(second_half)

            trend_narrative = ""
            # 5% increase in GPM
            if gpm_second_half > gpm_first_half * 1.05:
                trend_narrative = (
                    f"Your GPM is trending up from an average of "
                    f"**{gpm_first_half:.0f}** to **{gpm_second_half:.0f}** "
                    f"in recent games. This is great progress."
                )
            # 5% decrease
            elif gpm_second_half < gpm_first_half * 0.95:
                trend_narrative = (
                    f"Your average GPM has slightly dipped from "
                    f"**{gpm_first_half:.0f}** to **{gpm_second_half:.0f}**. "
                    f"Let's refocus on that early game farm."
                )
            else:
                trend_narrative = (
                    f"You are maintaining a consistent GPM around "
                    f"**{gpm_second_half:.0f}**. This stability is a strong "
                    "foundation to build on."
                )

            return (
                "**Progress on Your Goal (Improve Early Game):**\n"
                "You're focused on improving your early game, which hinges on "
                f"farming. {trend_narrative} To accelerate this, try to "
                "secure at least one jungle camp after clearing each minion "
                "wave in the first 5 minutes."
            )
        return ""

    def _get_role(self, hero: str) -> str:
        """Helper to determine role from hero name."""
        role_map = {
            'miya': 'marksman', 'layla': 'marksman', 'franco': 'tank',
            'tigreal': 'tank', 'kagura': 'mage', 'eudora': 'mage',
            'lancelot': 'assassin', 'fanny': 'assassin', 'estes': 'support',
            'angela': 'support', 'chou': 'fighter', 'zilong': 'fighter',
            'fredrinn': 'fighter', 'hayabusa': 'assassin'
        }
        return role_map.get(hero, 'fighter') 