# Base evaluator class that handles common coaching logic
import yaml
import os
from typing import List, Tuple, Dict, Any


class BaseEvaluator:
    """
    Base class for hero evaluation logic.
    Handles common patterns like KDA calculation, threshold loading,
    and match duration scaling.
    """
    
    def __init__(self):
        # Load configuration from YAML file
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "config", 
            "thresholds.yml"
        )
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Cache frequently used values
        self.defaults = self.config['defaults']
        self.roles = self.config['roles']
        self.heroes = self.config['heroes']
        self.severity_icons = self.config['severity_levels']
    
    def evaluate(self, data: Dict[str, Any], 
                 minutes: int = None) -> List[Tuple[str, str]]:
        """
        Main evaluation method. Returns list of (severity, message) tuples.
        
        Args:
            data: Match data dictionary
            minutes: Match duration in minutes (optional)
        
        Returns:
            List of tuples containing (severity_level, feedback_message)
        """
        fb = []  # Feedback list
        
        # Extract match duration if not provided
        if minutes is None:
            minutes = data.get('match_duration', 15)  # Default 15 min
            
        # Get hero and role info
        hero = data.get('hero', '').lower()
        role = self._get_role(hero)
        
        # Get thresholds for this hero/role
        thresholds = self._get_thresholds(hero, role)
        
        # --- Core evaluations that apply to all heroes ---
        
        # KDA evaluation
        kda_feedback = self._evaluate_kda(data, thresholds)
        if kda_feedback:
            fb.extend(kda_feedback)
            
        # GPM evaluation (scaled by match duration)
        gpm_feedback = self._evaluate_gpm(data, thresholds, minutes)
        if gpm_feedback:
            fb.extend(gpm_feedback)
            
        # Teamfight participation
        tfp_feedback = self._evaluate_teamfight_participation(
            data, thresholds)
        if tfp_feedback:
            fb.extend(tfp_feedback)
            
        # Hero damage (scaled by match duration)
        damage_feedback = self._evaluate_hero_damage(
            data, thresholds, minutes, role)
        if damage_feedback:
            fb.extend(damage_feedback)
            
        # Positioning check
        pos_feedback = self._evaluate_positioning(data)
        if pos_feedback:
            fb.extend(pos_feedback)
            
        # Add role-specific evaluations
        role_feedback = self._evaluate_role_specific(
            data, thresholds, minutes, role)
        if role_feedback:
            fb.extend(role_feedback)
            
        # Add hero-specific evaluations
        hero_feedback = self._evaluate_hero_specific(
            data, thresholds, hero)
        if hero_feedback:
            fb.extend(hero_feedback)
            
        return fb
    
    def _get_role(self, hero: str) -> str:
        """Determine role from hero name or data."""
        # This could be enhanced with a hero->role mapping
        role_map = {
            'miya': 'marksman',
            'layla': 'marksman',
            'franco': 'tank',
            'tigreal': 'tank',
            'kagura': 'mage',
            'eudora': 'mage',
            'lancelot': 'assassin',
            'fanny': 'assassin',
            'estes': 'support',
            'angela': 'support',
            'chou': 'fighter',
            'zilong': 'fighter'
        }
        return role_map.get(hero, 'fighter')  # Default to fighter
    
    def _get_thresholds(self, hero: str, role: str) -> Dict:
        """Get merged thresholds: defaults -> role -> hero."""
        # Start with defaults
        thresholds = self.defaults.copy()
        
        # Merge role-specific thresholds
        if role in self.roles:
            thresholds.update(self.roles[role])
            
        # Merge hero-specific overrides
        if hero in self.heroes:
            thresholds.update(self.heroes[hero])
            
        return thresholds
    
    def _evaluate_kda(self, data: Dict, 
                      thresholds: Dict) -> List[Tuple[str, str]]:
        """Evaluate KDA with contextual feedback."""
        fb = []
        
        # Calculate KDA
        kills = data.get("kills", 0)
        deaths = max(1, data.get("deaths", 1))  # Avoid division by zero
        assists = data.get("assists", 0)
        kda = (kills + assists) / deaths
        
        # Get thresholds
        kda_low = thresholds.get('kda', {}).get('low', 2.0)
        kda_high = thresholds.get('kda', {}).get('high', 4.0)
        
        if kda < kda_low:
            fb.append(("warning", 
                       f"KDA {kda:.1f} (< {kda_low}). Pick safer angles and "
                       f"exit with 2nd skill."))
        elif kda > kda_high:
            fb.append(("success", 
                       f"Great KDA {kda:.1f}! Keep pressuring their "
                       f"back-line."))
        else:
            # Neutral feedback for average performance
            fb.append(("info", 
                       f"Decent KDA ({kda:.1f}). You're on track—keep "
                       f"snowballing."))
                
        return fb
    
    def _evaluate_gpm(self, data: Dict, thresholds: Dict, 
                      minutes: int) -> List[Tuple[str, str]]:
        """Evaluate GPM with match duration scaling."""
        fb = []
        
        gpm = data.get("gold_per_min", 0)
        gpm_base = thresholds.get('gpm_base', 500)
        
        # Scale threshold based on match duration
        # Early game (< 10 min): lower expectations
        # Late game (> 20 min): higher expectations
        if minutes < 10:
            gpm_threshold = gpm_base * 0.8
        elif minutes > 20:
            gpm_threshold = gpm_base * 1.1
        else:
            gpm_threshold = gpm_base
            
        if gpm < gpm_threshold:
            fb.append(("warning", 
                       f"GPM {gpm} (< {gpm_threshold:.0f}). Clear side waves "
                       f"between ganks."))
                
        return fb
    
    def _evaluate_teamfight_participation(
            self, data: Dict, thresholds: Dict) -> List[Tuple[str, str]]:
        """Evaluate teamfight participation."""
        fb = []
        
        # Default to 0 if missing (more critical than defaulting to 100)
        tfp = data.get("teamfight_participation", 0)
        tfp_min = thresholds.get('teamfight_participation', 50)
        
        if tfp < tfp_min:
            fb.append(("warning", 
                       f"Low fight presence ({tfp}% < {tfp_min}%). Collapse "
                       f"faster on ally engages."))
                
        return fb
    
    def _evaluate_hero_damage(self, data: Dict, thresholds: Dict, 
                              minutes: int, 
                              role: str) -> List[Tuple[str, str]]:
        """Evaluate hero damage with scaling."""
        fb = []
        
        # Skip for roles that don't prioritize damage
        if role in ['tank', 'support']:
            return fb
            
        damage = data.get("hero_damage", 0)
        damage_base = thresholds.get('damage_base', 3000)
        
        # Scale by match duration
        damage_needed = damage_base * minutes
        
        # Also consider percentage of team damage for context
        team_damage_pct = data.get("damage_percentage", 0)
        
        if damage < damage_needed:
            fb.append(("warning", 
                       f"Damage {damage:,} (< {damage_needed:,}). Prioritise "
                       f"isolated kills."))
        elif team_damage_pct and team_damage_pct > 30:
            fb.append(("success", 
                       f"Excellent damage output! {team_damage_pct}% of team "
                       f"damage."))
                
        return fb
    
    def _evaluate_positioning(self, data: Dict) -> List[Tuple[str, str]]:
        """Evaluate positioning with clear ratings."""
        fb = []
        
        pos_rating = data.get("positioning_rating", "").lower()
        
        if pos_rating == "low":
            fb.append(("warning", 
                       "Positioning flagged low. Hold Retribution for wall "
                       "dashes."))
                
        return fb
    
    def _evaluate_role_specific(self, data: Dict, thresholds: Dict, 
                                minutes: int, 
                                role: str) -> List[Tuple[str, str]]:
        """Override this in subclasses for role-specific checks."""
        return []
    
    def _evaluate_hero_specific(self, data: Dict, thresholds: Dict, 
                                hero: str) -> List[Tuple[str, str]]:
        """Override this in subclasses for hero-specific checks."""
        return [] 