#!/usr/bin/env python3
"""
Create MVP Loss Category for MLBB YOLO Dataset

This script adds the MVP Loss performance category to the existing
strategic collection structure for matches where player gets MVP
but the team loses.
"""

from pathlib import Path


def create_mvp_loss_structure():
    """Create the MVP Loss category structure."""
    
    base_dir = Path(".")
    
    # Match types to add MVP Loss category to
    match_types = [
        "1_RANKED_MATCHES",
        "2_CLASSIC_MATCHES", 
        "3_OTHER_MODES"
    ]
    
    # MVP Loss category info
    mvp_loss_info = {
        "target_ranked": "4 matches (8 images)",
        "target_classic": "4 matches (8 images)", 
        "target_other": "2 matches (4 images)",
        "description_ranked": ("Ranked MVP loss - excellent individual "
                              "performance but team defeat"),
        "description_classic": ("Classic MVP loss - excellent individual "
                               "performance but team defeat"),
        "description_other": ("Brawl/Custom MVP loss - excellent individual "
                             "performance but team defeat")
    }
    
    print("🎯 Creating MVP Loss Category Structure")
    print("=" * 50)
    
    for match_type in match_types:
        match_dir = base_dir / "collection" / match_type
        
        # Ensure base match directory exists
        match_dir.mkdir(parents=True, exist_ok=True)
        
        # Create MVP Loss performance directory
        mvp_loss_dir = match_dir / "mvp_loss_performance"
        mvp_loss_dir.mkdir(parents=True, exist_ok=True)
        
        # Create damage and kda subdirectories
        (mvp_loss_dir / "damage_view").mkdir(exist_ok=True)
        (mvp_loss_dir / "kda_view").mkdir(exist_ok=True)
        
        # Determine target and description based on match type
        if "RANKED" in match_type:
            target = mvp_loss_info["target_ranked"]
            description = mvp_loss_info["description_ranked"]
        elif "CLASSIC" in match_type:
            target = mvp_loss_info["target_classic"]
            description = mvp_loss_info["description_classic"]
        else:
            target = mvp_loss_info["target_other"]
            description = mvp_loss_info["description_other"]
        
        # Create README file
        readme_file = mvp_loss_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(f"""# MVP Loss Performance

## Target: {target}

### Description
{description}

### Collection Instructions
1. **Play matches** where you get MVP but your team loses
2. **After each match**: Screenshot BOTH views
3. **Damage view**: Save to `damage_view/` folder
4. **KDA view**: Save to `kda_view/` folder

### File Naming Convention
- **Damage**: `{match_type.lower()}_mvp_loss_performance_match01_damage.jpg`
- **KDA**: `{match_type.lower()}_mvp_loss_performance_match01_kda.jpg`

### What to Look For
- **MVP Badge**: Silver MVP badge (losing team MVP)
- **High KDA**: Usually 8+ kills, low deaths
- **High Damage**: Top damage on your team
- **Team Defeat**: Red "DEFEAT" screen despite your excellent performance

### Criteria
- Individual performance: Excellent (MVP-worthy)
- Team outcome: Loss/Defeat
- Badge type: Silver MVP badge (not gold)
- Your stats: Top performer on losing team

### Progress Tracking
- [ ] Match 1: _____ (Hero: ______, KDA: ______, MVP: Yes/No)
- [ ] Match 2: _____ (Hero: ______, KDA: ______, MVP: Yes/No)
- [ ] Match 3: _____ (Hero: ______, KDA: ______, MVP: Yes/No)
- [ ] Match 4: _____ (Hero: ______, KDA: ______, MVP: Yes/No)

### Notes
- These matches are rare but valuable for training
- Focus on clear MVP badge visibility in screenshots
- Capture both the defeat screen AND MVP recognition
- Document if MVP badge is silver vs gold
""")
        
        print(f"✅ {match_type}/mvp_loss_performance: {target}")
    
    # Update master progress tracking
    update_master_progress()
    
    # Update hero tracking to include MVP loss
    update_hero_tracking()
    
    print(f"\n📊 MVP Loss Structure Summary:")
    print(f"  🏅 Ranked MVP Loss: 4 matches (8 images)")
    print(f"  🎯 Classic MVP Loss: 4 matches (8 images)")
    print(f"  🎊 Other MVP Loss: 2 matches (4 images)")
    print(f"  📈 Total MVP Loss: 10 matches (20 images)")


def update_master_progress():
    """Update master progress tracking to include MVP Loss."""
    
    master_file = Path("collection/MASTER_PROGRESS.md")
    
    if master_file.exists():
        # Read existing content
        with open(master_file, 'r') as f:
            content = f.read()
        
        # Add MVP Loss tracking
        mvp_loss_section = """
### MVP Loss Progress (NEW)
- [ ] Ranked MVP Loss: 0/4 matches (0/8 images)
- [ ] Classic MVP Loss: 0/4 matches (0/8 images)
- [ ] Other MVP Loss: 0/2 matches (0/4 images)
"""
        
        # Insert after Performance Progress section
        if "### Hero Role Progress" in content:
            content = content.replace("### Hero Role Progress", mvp_loss_section + "\n### Hero Role Progress")
        
        # Write back
        with open(master_file, 'w') as f:
            f.write(content)
        
        print("✅ Updated MASTER_PROGRESS.md with MVP Loss tracking")


def update_hero_tracking():
    """Update hero tracking files to include MVP Loss performance."""
    
    hero_tracking_dir = Path("collection/HERO_TRACKING")
    
    if not hero_tracking_dir.exists():
        return
    
    # Update each role tracking file
    for role_file in hero_tracking_dir.glob("*_tracking.md"):
        with open(role_file, 'r') as f:
            content = f.read()
        
        # Add MVP Loss to performance distribution
        if "### Performance Distribution Target" in content:
            # Find the section and add MVP Loss
            lines = content.split('\n')
            new_lines = []
            
            for line in lines:
                new_lines.append(line)
                if line.startswith("- Poor:"):
                    new_lines.append("- MVP Loss: 1-2 matches")
            
            content = '\n'.join(new_lines)
        
        # Add MVP Loss column to collection log
        if "| Date | Hero | Match Type | Performance | KDA | Notes |" in content:
            content = content.replace(
                "| Date | Hero | Match Type | Performance | KDA | Notes |",
                "| Date | Hero | Match Type | Performance | KDA | MVP | Notes |"
            )
            content = content.replace(
                "|------|------|------------|-------------|-----|-------|",
                "|------|------|------------|-------------|-----|-----|-------|"
            )
        
        with open(role_file, 'w') as f:
            f.write(content)
    
    print("✅ Updated hero tracking files with MVP Loss category")


def create_quick_collection_guide():
    """Create a quick guide for collecting MVP Loss screenshots."""
    
    guide_file = Path("collection/MVP_LOSS_COLLECTION_GUIDE.md")
    
    with open(guide_file, 'w') as f:
        f.write("""# 🥈 MVP Loss Collection Guide

## What is MVP Loss?
- You get the **MVP badge** (silver, not gold)
- Your team **loses the match** (red DEFEAT screen)
- Your individual performance was **excellent** despite the loss

## How to Identify MVP Loss Matches
1. **During Match**: Play your best, even if team is losing
2. **Post-Match**: Look for:
   - ❌ Red "DEFEAT" screen
   - 🥈 Silver MVP badge on your hero
   - 📊 Top stats on your team (damage, KDA, etc.)

## Screenshot Workflow
1. **Wait** for post-match screen to fully load
2. **Screenshot 1**: Damage view (shows defeat + MVP badge)
3. **Tap "Data"**: Switch to KDA/Items view  
4. **Screenshot 2**: KDA view (shows stats + MVP badge)

## File Organization
```
collection/
├── 1_RANKED_MATCHES/mvp_loss_performance/
│   ├── damage_view/
│   └── kda_view/
├── 2_CLASSIC_MATCHES/mvp_loss_performance/
│   ├── damage_view/
│   └── kda_view/
└── 3_OTHER_MODES/mvp_loss_performance/
    ├── damage_view/
    └── kda_view/
```

## Target Collection
- **Ranked**: 4 matches (8 images)
- **Classic**: 4 matches (8 images)  
- **Other**: 2 matches (4 images)
- **Total**: 10 matches (20 images)

## Tips for Finding MVP Loss Matches
- **Solo queue ranked**: Higher chance of team coordination issues
- **Play carry roles**: Marksman, Mage, Assassin more likely to get MVP
- **Late game heroes**: Heroes that scale can carry despite early disadvantage
- **Don't give up**: Even in losing matches, play for individual performance

## What Makes Good MVP Loss Screenshots
✅ Clear silver MVP badge visible
✅ Red DEFEAT text clearly shown
✅ Your stats are highlighted/prominent
✅ Both damage and KDA views captured
✅ High quality, readable UI elements

❌ Blurry or cut-off MVP badge
❌ Unclear defeat indication
❌ Missing individual stat highlights
❌ Poor image quality

## Validation Checklist
- [ ] MVP badge clearly visible (silver)
- [ ] Defeat screen clearly shown (red)
- [ ] Your hero is highlighted as MVP
- [ ] Both views captured (damage + KDA)
- [ ] Files named correctly
- [ ] Progress updated in README

---

**Remember**: MVP Loss matches are rare but extremely valuable for training the model to distinguish between individual excellence and team outcomes! 🎯
""")
    
    print("✅ Created MVP_LOSS_COLLECTION_GUIDE.md")


if __name__ == "__main__":
    create_mvp_loss_structure()
    create_quick_collection_guide()
    
    print(f"\n" + "=" * 50)
    print("✅ MVP Loss Structure Created!")
    print("\n📋 Next Steps:")
    print("1. Read: collection/MVP_LOSS_COLLECTION_GUIDE.md")
    print("2. Start collecting MVP Loss matches in ranked/classic")
    print("3. Update progress in each folder's README.md")
    print("4. Remember: Screenshot BOTH views for every MVP Loss!")
    print("\n🎯 New Goal: 90 matches × 2 views = 180 images total")
    print("   (Original 80 + New 10 MVP Loss matches)")
    print("🚀 Happy hunting for those rare MVP Loss gems! 🥈") 