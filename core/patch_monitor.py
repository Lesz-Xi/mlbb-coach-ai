from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path
from dataclasses import dataclass

from .meta_analyzer import MetaAnalyzer
from .schemas import HeroMetaData, MetaLeaderboard


@dataclass
class PatchChange:
    """Represents a hero's change between patches."""
    hero: str
    old_ranking: int
    new_ranking: int
    old_win_rate: float
    new_win_rate: float
    old_pick_rate: float
    new_pick_rate: float
    old_ban_rate: float
    new_ban_rate: float
    old_tier: str
    new_tier: str
    impact_score: float
    change_type: str  # "buff", "nerf", "neutral", "rework"


@dataclass
class PatchSummary:
    """Summary of patch impact."""
    patch_version: str
    release_date: str
    total_heroes_changed: int
    major_changes: List[PatchChange]
    tier_shifts: Dict[str, List[str]]
    meta_stability: float
    affected_roles: Dict[str, int]


class PatchMonitor:
    """Monitor and analyze patch impacts on meta shifts."""
    
    def __init__(self, data_directory: str = "data/patches"):
        """Initialize with patch data directory."""
        self.data_directory = Path(data_directory)
        self.current_meta: Optional[MetaLeaderboard] = None
        
        # Create patches directory if it doesn't exist
        self.data_directory.mkdir(parents=True, exist_ok=True)
        
        # Hero roles for impact analysis
        self.hero_roles = {
            "Franco": "tank", "Tigreal": "tank", "Lolita": "tank", "Akai": "tank",
            "Atlas": "tank", "Belerick": "tank", "Gatotkaca": "tank", "Grock": "tank",
            "Hilda": "tank", "Hylos": "tank", "Johnson": "tank", "Khufra": "tank",
            "Minotaur": "tank", "Angela": "support", "Estes": "support", "Mathilda": "support",
            "Carmilla": "support", "Diggie": "support", "Faramis": "support", "Rafaela": "support",
            "Layla": "marksman", "Beatrix": "marksman", "Melissa": "marksman", "Yi Sun-shin": "marksman",
            "Popol and Kupa": "marksman", "Cyclops": "mage", "Bane": "fighter", "Ruby": "fighter",
            "Sun": "fighter"
        }
    
    def save_patch_snapshot(self, patch_version: str, meta_data: MetaLeaderboard) -> None:
        """Save current meta state as patch snapshot."""
        snapshot_file = self.data_directory / f"patch_{patch_version.replace('.', '_')}.json"
        
        # Convert to serializable format
        snapshot_data = {
            "patch_version": patch_version,
            "timestamp": datetime.now().isoformat(),
            "heroes": [
                {
                    "ranking": hero.ranking,
                    "hero": hero.hero,
                    "pick_rate": hero.pick_rate,
                    "win_rate": hero.win_rate,
                    "ban_rate": hero.ban_rate,
                    "counter_heroes": hero.counter_heroes,
                    "tier": hero.tier,
                    "meta_score": hero.meta_score
                }
                for hero in meta_data.data
            ]
        }
        
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot_data, f, indent=2)
        
        print(f"Saved patch snapshot: {snapshot_file}")
    
    def load_patch_snapshot(self, patch_version: str) -> Optional[MetaLeaderboard]:
        """Load patch snapshot from file."""
        snapshot_file = self.data_directory / f"patch_{patch_version.replace('.', '_')}.json"
        
        if not snapshot_file.exists():
            return None
        
        try:
            with open(snapshot_file, 'r') as f:
                snapshot_data = json.load(f)
            
            # Convert back to MetaLeaderboard
            heroes = []
            for hero_data in snapshot_data["heroes"]:
                hero = HeroMetaData(
                    ranking=hero_data["ranking"],
                    hero=hero_data["hero"],
                    pick_rate=hero_data["pick_rate"],
                    win_rate=hero_data["win_rate"],
                    ban_rate=hero_data["ban_rate"],
                    counter_heroes=hero_data["counter_heroes"]
                )
                heroes.append(hero)
            
            return MetaLeaderboard(
                data=heroes,
                patch_version=snapshot_data["patch_version"],
                last_updated=datetime.fromisoformat(snapshot_data["timestamp"])
            )
            
        except Exception as e:
            print(f"Error loading patch snapshot: {e}")
            return None
    
    def compare_patches(self, old_patch: str, new_patch: str) -> Optional[PatchSummary]:
        """Compare two patches and analyze changes."""
        old_meta = self.load_patch_snapshot(old_patch)
        new_meta = self.load_patch_snapshot(new_patch)
        
        if not old_meta or not new_meta:
            return None
        
        # Create lookup dictionaries
        old_heroes = {hero.hero: hero for hero in old_meta.data}
        new_heroes = {hero.hero: hero for hero in new_meta.data}
        
        changes = []
        
        # Analyze changes for each hero
        for hero_name in set(old_heroes.keys()) | set(new_heroes.keys()):
            old_hero = old_heroes.get(hero_name)
            new_hero = new_heroes.get(hero_name)
            
            if old_hero and new_hero:
                change = self._analyze_hero_change(old_hero, new_hero)
                if change.impact_score > 0.1:  # Only include significant changes
                    changes.append(change)
        
        # Sort by impact score
        changes.sort(key=lambda x: x.impact_score, reverse=True)
        
        # Generate summary
        summary = PatchSummary(
            patch_version=new_patch,
            release_date=new_meta.last_updated.strftime("%Y-%m-%d") if new_meta.last_updated else "Unknown",
            total_heroes_changed=len(changes),
            major_changes=changes[:10],  # Top 10 changes
            tier_shifts=self._analyze_tier_shifts(changes),
            meta_stability=self._calculate_meta_stability(changes),
            affected_roles=self._analyze_affected_roles(changes)
        )
        
        return summary
    
    def get_patch_history(self) -> List[str]:
        """Get list of available patch versions."""
        patches = []
        for file in self.data_directory.glob("patch_*.json"):
            patch_version = file.stem.replace("patch_", "").replace("_", ".")
            patches.append(patch_version)
        
        return sorted(patches)
    
    def track_hero_evolution(self, hero_name: str, patches: List[str]) -> Dict[str, List[Dict[str, float]]]:
        """Track a hero's evolution across multiple patches."""
        evolution = {
            "win_rate": [],
            "pick_rate": [],
            "ban_rate": [],
            "ranking": [],
            "meta_score": []
        }
        
        for patch_version in patches:
            meta_data = self.load_patch_snapshot(patch_version)
            if meta_data:
                hero_data = next((h for h in meta_data.data if h.hero == hero_name), None)
                if hero_data:
                    evolution["win_rate"].append({"patch": patch_version, "value": hero_data.win_rate})
                    evolution["pick_rate"].append({"patch": patch_version, "value": hero_data.pick_rate})
                    evolution["ban_rate"].append({"patch": patch_version, "value": hero_data.ban_rate})
                    evolution["ranking"].append({"patch": patch_version, "value": hero_data.ranking})
                    evolution["meta_score"].append({"patch": patch_version, "value": hero_data.meta_score})
        
        return evolution
    
    def identify_meta_trends(self, patches: List[str]) -> Dict[str, any]:
        """Identify meta trends across multiple patches."""
        trends = {
            "dominant_heroes": [],
            "rising_stars": [],
            "falling_powers": [],
            "stable_picks": [],
            "role_meta_shifts": {}
        }
        
        if len(patches) < 2:
            return trends
        
        # Analyze first and last patches
        first_meta = self.load_patch_snapshot(patches[0])
        last_meta = self.load_patch_snapshot(patches[-1])
        
        if not first_meta or not last_meta:
            return trends
        
        first_heroes = {h.hero: h for h in first_meta.data}
        last_heroes = {h.hero: h for h in last_meta.data}
        
        # Analyze changes
        for hero_name in set(first_heroes.keys()) & set(last_heroes.keys()):
            first_hero = first_heroes[hero_name]
            last_hero = last_heroes[hero_name]
            
            rank_change = first_hero.ranking - last_hero.ranking
            wr_change = last_hero.win_rate - first_hero.win_rate
            
            if rank_change > 10 and wr_change > 5:
                trends["rising_stars"].append({
                    "hero": hero_name,
                    "rank_change": rank_change,
                    "wr_change": wr_change
                })
            elif rank_change < -10 and wr_change < -5:
                trends["falling_powers"].append({
                    "hero": hero_name,
                    "rank_change": rank_change,
                    "wr_change": wr_change
                })
            elif abs(rank_change) < 5 and abs(wr_change) < 2:
                trends["stable_picks"].append({
                    "hero": hero_name,
                    "consistency": 1.0 - (abs(rank_change) + abs(wr_change)) / 20
                })
            
            # Check for dominant heroes (consistently high tier)
            if last_hero.tier in ["S", "A"] and first_hero.tier in ["S", "A"]:
                trends["dominant_heroes"].append({
                    "hero": hero_name,
                    "avg_tier": (ord(first_hero.tier) + ord(last_hero.tier)) / 2
                })
        
        return trends
    
    def generate_patch_notes_analysis(self, patch_summary: PatchSummary) -> Dict[str, any]:
        """Generate analysis suitable for patch notes."""
        analysis = {
            "headline_changes": [],
            "role_impact": {},
            "ban_priority_shifts": [],
            "recommendation_updates": []
        }
        
        # Headline changes
        for change in patch_summary.major_changes[:5]:
            if change.change_type == "buff":
                analysis["headline_changes"].append(f"🔥 {change.hero} rises to tier {change.new_tier}")
            elif change.change_type == "nerf":
                analysis["headline_changes"].append(f"⚠️ {change.hero} drops to tier {change.new_tier}")
            elif change.change_type == "rework":
                analysis["headline_changes"].append(f"🔄 {change.hero} reworked - new meta impact")
        
        # Role impact
        for role, count in patch_summary.affected_roles.items():
            if count > 2:
                analysis["role_impact"][role] = f"Major changes affecting {count} heroes"
        
        # Ban priority shifts
        high_impact_bans = [c for c in patch_summary.major_changes if c.new_ban_rate > c.old_ban_rate + 10]
        for change in high_impact_bans:
            analysis["ban_priority_shifts"].append(f"{change.hero} ban rate increased to {change.new_ban_rate:.1f}%")
        
        # Recommendation updates
        if patch_summary.meta_stability < 0.7:
            analysis["recommendation_updates"].append("High meta volatility - recommendations may change rapidly")
        
        return analysis
    
    def _analyze_hero_change(self, old_hero: HeroMetaData, new_hero: HeroMetaData) -> PatchChange:
        """Analyze change between two hero states."""
        # Calculate impact score
        wr_change = abs(new_hero.win_rate - old_hero.win_rate)
        pr_change = abs(new_hero.pick_rate - old_hero.pick_rate)
        br_change = abs(new_hero.ban_rate - old_hero.ban_rate)
        rank_change = abs(new_hero.ranking - old_hero.ranking)
        
        impact_score = (wr_change * 0.4) + (pr_change * 0.3) + (br_change * 0.2) + (rank_change * 0.1)
        
        # Determine change type
        change_type = "neutral"
        if new_hero.win_rate > old_hero.win_rate + 3:
            change_type = "buff"
        elif new_hero.win_rate < old_hero.win_rate - 3:
            change_type = "nerf"
        elif abs(new_hero.pick_rate - old_hero.pick_rate) > 5:
            change_type = "rework"
        
        return PatchChange(
            hero=old_hero.hero,
            old_ranking=old_hero.ranking,
            new_ranking=new_hero.ranking,
            old_win_rate=old_hero.win_rate,
            new_win_rate=new_hero.win_rate,
            old_pick_rate=old_hero.pick_rate,
            new_pick_rate=new_hero.pick_rate,
            old_ban_rate=old_hero.ban_rate,
            new_ban_rate=new_hero.ban_rate,
            old_tier=old_hero.tier,
            new_tier=new_hero.tier,
            impact_score=impact_score,
            change_type=change_type
        )
    
    def _analyze_tier_shifts(self, changes: List[PatchChange]) -> Dict[str, List[str]]:
        """Analyze tier shifts from changes."""
        shifts = {"promoted": [], "demoted": []}
        
        for change in changes:
            if change.new_tier < change.old_tier:  # S < A < B < C < D
                shifts["promoted"].append(f"{change.hero}: {change.old_tier} → {change.new_tier}")
            elif change.new_tier > change.old_tier:
                shifts["demoted"].append(f"{change.hero}: {change.old_tier} → {change.new_tier}")
        
        return shifts
    
    def _calculate_meta_stability(self, changes: List[PatchChange]) -> float:
        """Calculate meta stability score (0-1, higher = more stable)."""
        if not changes:
            return 1.0
        
        # Calculate average impact
        avg_impact = sum(c.impact_score for c in changes) / len(changes)
        
        # Normalize to 0-1 scale (lower impact = higher stability)
        stability = max(0, 1 - (avg_impact / 50))
        
        return stability
    
    def _analyze_affected_roles(self, changes: List[PatchChange]) -> Dict[str, int]:
        """Analyze which roles are most affected."""
        role_counts = {}
        
        for change in changes:
            role = self.hero_roles.get(change.hero, "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
        
        return role_counts