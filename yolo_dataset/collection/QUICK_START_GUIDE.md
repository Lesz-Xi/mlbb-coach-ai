# 🚀 MLBB YOLO Collection Quick Start Guide

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
