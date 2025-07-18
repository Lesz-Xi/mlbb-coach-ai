#!/usr/bin/env python3
"""
Create Strategic MLBB YOLO Dataset Directory Structure

This script creates a complete, organized directory structure for collecting
80 matches × 2 views = 160 images according to the strategic collection plan.
"""

import os
from pathlib import Path


def create_strategic_structure():
    """Create the complete strategic directory structure."""
    
    base_dir = Path(".")
    
    # Strategic collection structure
    structure = {
        "1_RANKED_MATCHES": {
            "excellent_performance": {
                "target": "6 matches (12 images)",
                "description": "Ranked MVP wins, gold medals, outstanding KDA"
            },
            "good_performance": {
                "target": "10 matches (20 images)", 
                "description": "Ranked silver medals, solid KDA, good wins"
            },
            "average_performance": {
                "target": "10 matches (20 images)",
                "description": "Ranked bronze medals, moderate KDA, typical games"
            },
            "poor_performance": {
                "target": "6 matches (12 images)",
                "description": "Ranked losses, low KDA, difficult matches"
            }
        },
        "2_CLASSIC_MATCHES": {
            "excellent_performance": {
                "target": "6 matches (12 images)",
                "description": "Classic MVP wins, gold medals, outstanding KDA"
            },
            "good_performance": {
                "target": "10 matches (20 images)",
                "description": "Classic silver medals, solid KDA, good wins"
            },
            "average_performance": {
                "target": "10 matches (20 images)", 
                "description": "Classic bronze medals, moderate KDA, typical games"
            },
            "poor_performance": {
                "target": "6 matches (12 images)",
                "description": "Classic losses, low KDA, difficult matches"
            }
        },
        "3_OTHER_MODES": {
            "excellent_performance": {
                "target": "4 matches (8 images)",
                "description": "Brawl/Custom MVP wins, outstanding performance"
            },
            "good_performance": {
                "target": "4 matches (8 images)",
                "description": "Brawl/Custom good performance, silver medals"
            },
            "average_performance": {
                "target": "4 matches (8 images)",
                "description": "Brawl/Custom average performance, bronze medals"
            },
            "poor_performance": {
                "target": "4 matches (8 images)",
                "description": "Brawl/Custom poor performance, losses"
            }
        }
    }
    
    print("🎯 Creating Strategic MLBB YOLO Dataset Structure")
    print("=" * 60)
    
    # Create main collection directories
    for match_type, performance_levels in structure.items():
        match_dir = base_dir / "collection" / match_type
        match_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 {match_type}")
        
        for performance, info in performance_levels.items():
            perf_dir = match_dir / performance
            perf_dir.mkdir(parents=True, exist_ok=True)
            
            # Create damage and kda subdirectories
            (perf_dir / "damage_view").mkdir(exist_ok=True)
            (perf_dir / "kda_view").mkdir(exist_ok=True)
            
            # Create info file
            info_file = perf_dir / "README.md"
            with open(info_file, 'w') as f:
                f.write(f"""# {performance.replace('_', ' ').title()}

## Target: {info['target']}

### Description
{info['description']}

### Collection Instructions
1. **Play matches** that fit this category
2. **After each match**: Screenshot BOTH views
3. **Damage view**: Save to `damage_view/` folder
4. **KDA view**: Save to `kda_view/` folder

### File Naming Convention
- **Damage**: `{match_type.lower()}_{performance}_match01_damage.jpg`
- **KDA**: `{match_type.lower()}_{performance}_match01_kda.jpg`

### What to Look For
- **Excellent**: MVP badges, gold medals, high KDA (10+ kills)
- **Good**: Silver medals, solid KDA (5-10 kills, low deaths)
- **Average**: Bronze medals, moderate KDA (3-8 kills)
- **Poor**: No medals, low KDA (0-3 kills, high deaths)

### Progress Tracking
- [ ] Match 1: _____ (Hero: ______, KDA: ______)
- [ ] Match 2: _____ (Hero: ______, KDA: ______)
- [ ] Match 3: _____ (Hero: ______, KDA: ______)
- [ ] Match 4: _____ (Hero: ______, KDA: ______)
- [ ] Match 5: _____ (Hero: ______, KDA: ______)
""")
            
            print(f"  ✅ {performance}: {info['target']}")
    
    # Create hero role tracking
    create_hero_tracking()
    
    # Create master tracking file
    create_master_tracking()
    
    # Create quick start guide
    create_quick_start_guide()
    
    print(f"\n📊 Structure Summary:")
    print(f"  🏅 Ranked: 32 matches (64 images)")
    print(f"  🎯 Classic: 32 matches (64 images)")
    print(f"  🎊 Other: 16 matches (32 images)")
    print(f"  📈 Total: 80 matches (160 images)")


def create_hero_tracking():
    """Create hero role tracking system."""
    
    hero_dir = Path("collection/HERO_TRACKING")
    hero_dir.mkdir(parents=True, exist_ok=True)
    
    roles = {
        "tank": {
            "target": 12,
            "meta": ["Tigreal", "Franco", "Atlas", "Khufra"],
            "rare": ["Belerick", "Minotaur", "Akai"]
        },
        "fighter": {
            "target": 15,
            "meta": ["Chou", "Paquito", "Fredrinn", "Phoveus"], 
            "rare": ["Argus", "Lapu-Lapu", "Martis"]
        },
        "assassin": {
            "target": 12,
            "meta": ["Hayabusa", "Lancelot", "Ling", "Fanny"],
            "rare": ["Helcurt", "Natalia", "Saber"]
        },
        "mage": {
            "target": 15,
            "meta": ["Kagura", "Xavier", "Lylia", "Cecilion"],
            "rare": ["Vexana", "Faramis", "Alice"]
        },
        "marksman": {
            "target": 13,
            "meta": ["Beatrix", "Brody", "Wanwan", "Melissa"],
            "rare": ["Layla", "Miya", "Bruno"]
        },
        "support": {
            "target": 13,
            "meta": ["Mathilda", "Estes", "Floryn", "Diggie"],
            "rare": ["Angela", "Nana", "Rafaela"]
        }
    }
    
    for role, info in roles.items():
        role_file = hero_dir / f"{role}_tracking.md"
        with open(role_file, 'w') as f:
            f.write(f"""# {role.title()} Role Collection Tracking

## Target: {info['target']} matches

### Meta Heroes (70% priority)
{chr(10).join([f"- [ ] {hero} (_____ matches)" for hero in info['meta']])}

### Rare Heroes (30% priority)  
{chr(10).join([f"- [ ] {hero} (_____ matches)" for hero in info['rare']])}

### Performance Distribution Target
- Excellent: {info['target'] // 4 + (1 if info['target'] % 4 >= 3 else 0)} matches
- Good: {info['target'] // 4 + (1 if info['target'] % 4 >= 1 else 0)} matches
- Average: {info['target'] // 4 + (1 if info['target'] % 4 >= 2 else 0)} matches  
- Poor: {info['target'] // 4} matches

### Collection Log
| Date | Hero | Match Type | Performance | KDA | Notes |
|------|------|------------|-------------|-----|-------|
|      |      |            |             |     |       |
|      |      |            |             |     |       |
""")


def create_master_tracking():
    """Create master progress tracking file."""
    
    master_file = Path("collection/MASTER_PROGRESS.md")
    with open(master_file, 'w') as f:
        f.write("""# MLBB YOLO Dataset Collection Progress

## 🎯 Overall Target: 80 Matches × 2 Views = 160 Images

### Match Type Progress
- [ ] Ranked: 0/32 matches (0/64 images)
- [ ] Classic: 0/32 matches (0/64 images)  
- [ ] Other Modes: 0/16 matches (0/32 images)

### Performance Progress
- [ ] Excellent: 0/16 matches (0/32 images)
- [ ] Good: 0/24 matches (0/48 images)
- [ ] Average: 0/24 matches (0/48 images)
- [ ] Poor: 0/16 matches (0/32 images)

### Hero Role Progress
- [ ] Tank: 0/12 matches
- [ ] Fighter: 0/15 matches
- [ ] Assassin: 0/12 matches
- [ ] Mage: 0/15 matches
- [ ] Marksman: 0/13 matches
- [ ] Support: 0/13 matches

### Weekly Collection Log

#### Week 1 (Target: 10 matches)
- Monday: _____ 
- Tuesday: _____
- Wednesday: _____
- Thursday: _____
- Friday: _____

#### Week 2 (Target: 10 matches)
- Monday: _____
- Tuesday: _____
- Wednesday: _____
- Thursday: _____
- Friday: _____

### Recent Collections
| Date | Match Type | Performance | Hero | Role | KDA | Files |
|------|------------|-------------|------|------|-----|-------|
|      |            |             |      |      |     |       |

### Notes & Observations
- 
- 
- 

### Next Priority
1. Focus on: _____
2. Need more: _____
3. Complete soon: _____
""")


def create_quick_start_guide():
    """Create quick start collection guide."""
    
    guide_file = Path("collection/QUICK_START_GUIDE.md")
    with open(guide_file, 'w') as f:
        f.write("""# 🚀 MLBB YOLO Collection Quick Start Guide

## 📱 After Every Match Workflow

### Step 1: Finish Match
- Wait for post-match results screen to fully load

### Step 2: Screenshot Damage View  
- Take screenshot of default view (usually damage stats)
- Save as: `[matchtype]_[performance]_match##_damage.jpg`

### Step 3: Switch to KDA View
- Tap "Data" button to switch to KDA/Items view
- Take screenshot of KDA/Equipment view  
- Save as: `[matchtype]_[performance]_match##_kda.jpg`

### Step 4: Determine Category
**Match Type:**
- Ranked: Look for rank indicators
- Classic: Standard match interface
- Other: Brawl/Custom/Special modes

**Performance Level:**
- Excellent: MVP badge, gold medal, 10+ kills
- Good: Silver medal, 5-10 kills, low deaths
- Average: Bronze medal, 3-8 kills, moderate performance
- Poor: No medal, 0-3 kills, high deaths

### Step 5: File Organization
Navigate to correct folder:
```
collection/
├── 1_RANKED_MATCHES/[performance]/
├── 2_CLASSIC_MATCHES/[performance]/  
└── 3_OTHER_MODES/[performance]/
```

Place files in:
- `damage_view/` folder for damage screenshots
- `kda_view/` folder for KDA screenshots

### Step 6: Update Tracking
- Mark progress in folder README.md
- Update MASTER_PROGRESS.md
- Log hero role in HERO_TRACKING/

## 🎯 Priority Collection Order

### Week 1-2: Foundation
1. Focus on **good** and **average** performance
2. Mix of ranked and classic matches
3. Cover all hero roles

### Week 3-4: Excellence  
1. Target **excellent** performance (MVP hunting)
2. Continue role diversity
3. Add other modes (brawl/custom)

### Week 5-6: Completion
1. Fill gaps in performance distribution
2. Complete hero role requirements
3. Add **poor** performance edge cases

## ❗ Important Reminders

### Always Capture Both Views
- Damage view: `damage_dealt`, `participation_rate` classes
- KDA view: `kda_display`, `item_icon_1-6` classes
- **Both needed** for complete training data!

### File Naming Examples
- `ranked_excellent_match01_damage.jpg`
- `ranked_excellent_match01_kda.jpg`  
- `classic_good_match05_damage.jpg`
- `classic_good_match05_kda.jpg`

### Quality Standards
- Clear UI elements, readable text
- Complete scoreboard visible
- No blurry or cut-off screenshots
- Both views from same match

---

**Ready to start collecting! 📸 Remember: 80 matches × 2 views = 160 images total**
""")


if __name__ == "__main__":
    create_strategic_structure()
    
    print(f"\n" + "=" * 60)
    print("✅ Strategic Directory Structure Created!")
    print("\n📋 Next Steps:")
    print("1. Read: collection/QUICK_START_GUIDE.md")
    print("2. Start with: 1_RANKED_MATCHES/good_performance/")
    print("3. Track progress in: MASTER_PROGRESS.md")
    print("4. Remember: Screenshot BOTH views every match!")
    print("\n🎯 Goal: 80 matches × 2 views = 160 images")
    print("🚀 Start collecting today!") 