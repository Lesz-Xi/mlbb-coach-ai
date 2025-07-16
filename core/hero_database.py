"""
Enhanced hero database with comprehensive hero data and detection capabilities.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import re
try:
    from fuzzywuzzy import fuzz, process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class HeroData:
    """Enhanced hero data structure."""
    name: str
    role: str
    aliases: List[str] = field(default_factory=list)
    detection_keywords: List[str] = field(default_factory=list)
    skill_names: List[str] = field(default_factory=list)
    counters: List[str] = field(default_factory=list)
    countered_by: List[str] = field(default_factory=list)
    synergies: List[str] = field(default_factory=list)
    meta_tier: str = "B"
    win_rate: float = 50.0
    pick_rate: float = 5.0
    ban_rate: float = 2.0
    image_url: str = ""
    wiki_url: str = ""
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Add normalized name to aliases
        normalized_name = self.name.lower().replace(" ", "").replace("-", "")
        if normalized_name not in [alias.lower() for alias in self.aliases]:
            self.aliases.append(normalized_name)
        
        # Auto-generate detection keywords from name
        name_words = re.findall(r'\w+', self.name.lower())
        for word in name_words:
            if word not in [kw.lower() for kw in self.detection_keywords]:
                self.detection_keywords.append(word)


@dataclass
class HeroMatchResult:
    """Result of hero matching operation."""
    hero: str
    confidence: float
    match_type: str  # "exact", "alias", "fuzzy", "keyword"
    source: str = ""  # What triggered the match
    

class HeroDatabase:
    """Enhanced hero database with intelligent detection."""
    
    def __init__(self, data_path: Optional[str] = None):
        """Initialize hero database."""
        self.heroes: Dict[str, HeroData] = {}
        self.role_mapping: Dict[str, Set[str]] = {}
        self.search_index: Dict[str, str] = {}  # normalized_name -> actual_name
        
        if data_path:
            self.load_from_file(data_path)
        else:
            # Try to load from the existing heroes file first
            heroes_file = "data/mlbb-heroes-corrected.json"
            if os.path.exists(heroes_file):
                self.load_from_file(heroes_file)
            else:
                self._load_default_heroes()
    
    def load_from_file(self, file_path: str):
        """Load heroes from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for hero_data in data:
                hero = self._parse_hero_data(hero_data)
                self.add_hero(hero)
                
            logger.info(f"Loaded {len(self.heroes)} heroes from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load heroes from {file_path}: {e}")
            self._load_default_heroes()
    
    def _parse_hero_data(self, data: Dict) -> HeroData:
        """Parse hero data from JSON."""
        hero_name = data.get("hero_name", "Unknown")
        
        # Infer role from hero name using simple heuristics
        role = self._infer_role(hero_name)
        
        # Generate basic detection keywords from name
        keywords = [hero_name.lower()]
        if " " in hero_name:
            keywords.extend(hero_name.lower().split())
        
        return HeroData(
            name=hero_name,
            role=role,
            aliases=data.get("aliases", []),
            detection_keywords=keywords,
            skill_names=data.get("skills", []),
            counters=data.get("counters", []),
            countered_by=data.get("countered_by", []),
            synergies=data.get("synergies", []),
            meta_tier=data.get("tier", "B"),
            win_rate=data.get("win_rate", 50.0),
            pick_rate=data.get("pick_rate", 5.0),
            ban_rate=data.get("ban_rate", 2.0),
            image_url=data.get("image_url", ""),
            wiki_url=data.get("correct_hero_url", "")
        )
    
    def _infer_role(self, hero_name: str) -> str:
        """Infer hero role from name using basic heuristics."""
        # Basic role mapping based on common hero names
        tank_heroes = ["franco", "tigreal", "atlas", "akai", "belerick", "gatotkaca", 
                      "grock", "hilda", "hylos", "johnson", "khufra", "minotaur", "lolita"]
        marksman_heroes = ["layla", "miya", "bruno", "clint", "hanabi", "karrie", "kimmy", 
                          "lesley", "moskov", "roger", "wanwan", "granger", "claude", "brody"]
        mage_heroes = ["eudora", "alice", "nana", "cyclops", "kagura", "aurora", "gord", 
                      "vale", "harith", "chang'e", "lunox", "zhask", "faramis", "cecilion"]
        assassin_heroes = ["alucard", "saber", "karina", "fanny", "natalia", "hayabusa", 
                          "lancelot", "gusion", "helcurt", "hanzo", "selena", "ling"]
        support_heroes = ["angela", "estes", "rafaela", "diggie", "mathilda", "floryn"]
        
        name_lower = hero_name.lower()
        
        if any(name in name_lower for name in tank_heroes):
            return "tank"
        elif any(name in name_lower for name in marksman_heroes):
            return "marksman"
        elif any(name in name_lower for name in mage_heroes):
            return "mage"
        elif any(name in name_lower for name in assassin_heroes):
            return "assassin"
        elif any(name in name_lower for name in support_heroes):
            return "support"
        else:
            return "fighter"  # Default to fighter
    
    def add_hero(self, hero: HeroData):
        """Add hero to database."""
        self.heroes[hero.name] = hero
        
        # Update role mapping
        if hero.role not in self.role_mapping:
            self.role_mapping[hero.role] = set()
        self.role_mapping[hero.role].add(hero.name)
        
        # Update search index
        normalized_name = hero.name.lower().replace(" ", "").replace("-", "")
        self.search_index[normalized_name] = hero.name
        
        # Add aliases to search index
        for alias in hero.aliases:
            normalized_alias = alias.lower().replace(" ", "").replace("-", "")
            self.search_index[normalized_alias] = hero.name
    
    def find_hero(self, query: str, min_confidence: float = 0.5) -> Optional[HeroMatchResult]:
        """Find hero with intelligent matching."""
        if not query:
            return None
        
        query_lower = query.lower()
        results = []
        
        # 1. Exact name match
        for hero_name, hero in self.heroes.items():
            if hero_name.lower() == query_lower:
                results.append(HeroMatchResult(
                    hero=hero_name,
                    confidence=1.0,
                    match_type="exact",
                    source="name"
                ))
        
        # 2. Alias match
        for hero_name, hero in self.heroes.items():
            for alias in hero.aliases:
                if alias.lower() == query_lower:
                    results.append(HeroMatchResult(
                        hero=hero_name,
                        confidence=0.95,
                        match_type="alias",
                        source=f"alias:{alias}"
                    ))
        
        # 3. Keyword match
        for hero_name, hero in self.heroes.items():
            for keyword in hero.detection_keywords:
                if keyword.lower() in query_lower:
                    confidence = min(0.9, len(keyword) / len(query))
                    results.append(HeroMatchResult(
                        hero=hero_name,
                        confidence=confidence,
                        match_type="keyword",
                        source=f"keyword:{keyword}"
                    ))
        
        # 4. Fuzzy match
        if FUZZYWUZZY_AVAILABLE:
            hero_names = list(self.heroes.keys())
            fuzzy_matches = process.extract(query, hero_names, limit=3)
            for match_name, score in fuzzy_matches:
                if score >= min_confidence * 100:
                    results.append(HeroMatchResult(
                        hero=match_name,
                        confidence=score / 100,
                        match_type="fuzzy",
                        source=f"fuzzy:{score}"
                    ))
        
        # Return best match
        if results:
            best_match = max(results, key=lambda r: r.confidence)
            if best_match.confidence >= min_confidence:
                return best_match
        
        return None
    
    def find_heroes_by_role(self, role: str) -> List[str]:
        """Find all heroes by role."""
        return list(self.role_mapping.get(role.lower(), set()))
    
    def get_hero_counters(self, hero_name: str) -> List[str]:
        """Get heroes that counter the given hero."""
        hero = self.heroes.get(hero_name)
        return hero.countered_by if hero else []
    
    def get_hero_victims(self, hero_name: str) -> List[str]:
        """Get heroes that this hero counters."""
        hero = self.heroes.get(hero_name)
        return hero.counters if hero else []
    
    def get_hero_synergies(self, hero_name: str) -> List[str]:
        """Get heroes that synergize well with the given hero."""
        hero = self.heroes.get(hero_name)
        return hero.synergies if hero else []
    
    def get_meta_tier_heroes(self, tier: str) -> List[str]:
        """Get all heroes in a specific meta tier."""
        return [name for name, hero in self.heroes.items() if hero.meta_tier == tier]
    
    def get_top_meta_heroes(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get top meta heroes by win rate."""
        sorted_heroes = sorted(
            self.heroes.items(),
            key=lambda x: (x[1].win_rate, x[1].pick_rate),
            reverse=True
        )
        return [(name, hero.win_rate) for name, hero in sorted_heroes[:limit]]
    
    def get_ban_worthy_heroes(self, limit: int = 10) -> List[str]:
        """Get heroes worth banning based on pick/ban/win rates."""
        ban_score = lambda hero: (hero.ban_rate * 0.4 + hero.pick_rate * 0.3 + hero.win_rate * 0.3)
        sorted_heroes = sorted(
            self.heroes.items(),
            key=lambda x: ban_score(x[1]),
            reverse=True
        )
        return [name for name, _ in sorted_heroes[:limit]]
    
    def suggest_counter_picks(self, enemy_heroes: List[str], allied_heroes: List[str] = None) -> List[Tuple[str, float]]:
        """Suggest counter picks against enemy team."""
        if not enemy_heroes:
            return []
        
        allied_heroes = allied_heroes or []
        counter_suggestions = {}
        
        for enemy_hero in enemy_heroes:
            counters = self.get_hero_counters(enemy_hero)
            for counter in counters:
                if counter not in allied_heroes and counter not in enemy_heroes:
                    counter_suggestions[counter] = counter_suggestions.get(counter, 0) + 1
        
        # Sort by counter effectiveness and meta strength
        suggestions = []
        for hero_name, counter_count in counter_suggestions.items():
            hero = self.heroes.get(hero_name)
            if hero:
                # Calculate suggestion score
                score = (counter_count / len(enemy_heroes)) * 0.6 + (hero.win_rate / 100) * 0.4
                suggestions.append((hero_name, score))
        
        return sorted(suggestions, key=lambda x: x[1], reverse=True)
    
    def _load_default_heroes(self):
        """Load default hero set if file loading fails."""
        default_heroes = [
            HeroData(
                name="Franco",
                role="tank",
                aliases=["franco", "hook"],
                detection_keywords=["franco", "hook", "tank"],
                skill_names=["iron hook", "fury shock", "bloody hunt"],
                counters=["Fanny", "Hayabusa", "Gusion"],
                countered_by=["Diggie", "Khufra", "Mathilda"],
                synergies=["Odette", "Aurora", "Eudora"],
                meta_tier="A",
                win_rate=52.5
            ),
            HeroData(
                name="Kagura",
                role="mage",
                aliases=["kagura", "umbrella"],
                detection_keywords=["kagura", "umbrella", "mage"],
                skill_names=["umbrella", "yin yang", "link"],
                counters=["Fanny", "Gusion", "Ling"],
                countered_by=["Hayabusa", "Natalia", "Helcurt"],
                synergies=["Franco", "Tigreal", "Atlas"],
                meta_tier="S",
                win_rate=54.2
            ),
            HeroData(
                name="Hayabusa",
                role="assassin",
                aliases=["hayabusa", "haya", "ninja"],
                detection_keywords=["hayabusa", "haya", "ninja", "assassin"],
                skill_names=["phantom", "shadow", "ultimate"],
                counters=["Kagura", "Eudora", "Aurora"],
                countered_by=["Saber", "Chou", "Kaja"],
                synergies=["Angela", "Estes", "Mathilda"],
                meta_tier="A",
                win_rate=53.8
            ),
            HeroData(
                name="Layla",
                role="marksman",
                aliases=["layla", "mm"],
                detection_keywords=["layla", "marksman", "mm"],
                skill_names=["malefic", "void", "destruction"],
                counters=["Tanks", "Supports"],
                countered_by=["Hayabusa", "Fanny", "Gusion"],
                synergies=["Franco", "Tigreal", "Angela"],
                meta_tier="B",
                win_rate=48.5
            ),
            HeroData(
                name="Chou",
                role="fighter",
                aliases=["chou", "bruce", "lee"],
                detection_keywords=["chou", "bruce", "fighter"],
                skill_names=["jeet", "kick", "dragon"],
                counters=["Hayabusa", "Fanny", "Gusion"],
                countered_by=["Esmeralda", "Uranus", "Belerick"],
                synergies=["Angela", "Estes", "Mathilda"],
                meta_tier="A",
                win_rate=51.2
            )
        ]
        
        for hero in default_heroes:
            self.add_hero(hero)
        
        logger.info(f"Loaded {len(default_heroes)} default heroes")
    
    def get_hero_info(self, hero_name: str) -> Optional[HeroData]:
        """Get complete hero information."""
        return self.heroes.get(hero_name)
    
    def search_heroes(self, query: str, limit: int = 5) -> List[HeroMatchResult]:
        """Search for heroes with multiple results."""
        results = []
        query_lower = query.lower()
        
        for hero_name, hero in self.heroes.items():
            # Check name similarity
            if FUZZYWUZZY_AVAILABLE:
                similarity = fuzz.ratio(query_lower, hero_name.lower())
                if similarity >= 50:
                    results.append(HeroMatchResult(
                        hero=hero_name,
                        confidence=similarity / 100,
                        match_type="fuzzy",
                        source=f"name_similarity:{similarity}"
                    ))
                
                # Check aliases
                for alias in hero.aliases:
                    similarity = fuzz.ratio(query_lower, alias.lower())
                    if similarity >= 70:
                        results.append(HeroMatchResult(
                            hero=hero_name,
                            confidence=similarity / 100,
                            match_type="alias",
                            source=f"alias:{alias}"
                        ))
            else:
                # Fallback to simple string matching
                if query_lower in hero_name.lower():
                    results.append(HeroMatchResult(
                        hero=hero_name,
                        confidence=0.8,
                        match_type="partial",
                        source="name_contains"
                    ))
                
                for alias in hero.aliases:
                    if query_lower in alias.lower():
                        results.append(HeroMatchResult(
                            hero=hero_name,
                            confidence=0.7,
                            match_type="alias",
                            source=f"alias:{alias}"
                        ))
        
        # Sort by confidence and return top results
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:limit]
    
    def export_to_json(self, file_path: str):
        """Export hero database to JSON file."""
        export_data = []
        for hero in self.heroes.values():
            export_data.append({
                "hero_name": hero.name,
                "role": hero.role,
                "aliases": hero.aliases,
                "keywords": hero.detection_keywords,
                "skills": hero.skill_names,
                "counters": hero.counters,
                "countered_by": hero.countered_by,
                "synergies": hero.synergies,
                "tier": hero.meta_tier,
                "win_rate": hero.win_rate,
                "pick_rate": hero.pick_rate,
                "ban_rate": hero.ban_rate,
                "image_url": hero.image_url,
                "correct_hero_url": hero.wiki_url
            })
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(export_data)} heroes to {file_path}")


# Global hero database instance
hero_database = HeroDatabase()