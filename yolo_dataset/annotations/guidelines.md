# MLBB YOLO Dataset Annotation Guidelines

## 🎯 Overview

This document provides comprehensive guidelines for annotating MLBB (Mobile Legends: Bang Bang) post-match screenshots. Following these guidelines ensures consistent, high-quality annotations that will produce an accurate YOLO detection model.

## 📏 General Annotation Principles

### Bounding Box Standards

- **Tight Fit**: Boxes should closely follow object boundaries
- **Complete Coverage**: Include entire visible object area
- **No Overlap**: Avoid overlapping annotations when possible
- **Consistent**: Use same approach across similar objects

### Quality Requirements

- **Minimum Size**: Objects must be at least 20x20 pixels
- **Clarity**: Only annotate clearly visible, recognizable objects
- **Completeness**: Object should be at least 80% visible
- **Focus**: Prioritize objects relevant to game analysis

## 🦸 Hero Icon Detection Guidelines

### Bounding Box Rules

#### What to Include ✅

- **Complete Hero Portrait**: Entire character image within the icon
- **Icon Background**: The circular/square background of the hero icon
- **Hero Name Text**: Text label below the hero portrait (if visible)

#### What to Exclude ❌

- **Level Indicators**: Small numbers showing hero level
- **Border Elements**: Decorative frames around icons
- **Selection Highlights**: Blue/gold selection borders
- **Overlapping UI**: Other elements covering the icon

### Position-Based Naming Convention

```
ALLY TEAM (Top Section)
┌─────────────────────────┐
│ ally_1  ally_2  ally_3  │  ← Top row (positions 1-3)
│ ally_4  ally_5         │  ← Bottom row (positions 4-5)
└─────────────────────────┘

ENEMY TEAM (Bottom Section)
┌─────────────────────────┐
│ enemy_1 enemy_2 enemy_3 │  ← Top row (positions 1-3)
│ enemy_4 enemy_5        │  ← Bottom row (positions 4-5)
└─────────────────────────┘
```

### Technical Specifications

- **Aspect Ratio**: 0.9 - 1.1 (nearly square)
- **Minimum Size**: 50x50 pixels
- **Maximum Size**: 150x150 pixels
- **Preferred Size**: 80x80 pixels

### Common Scenarios

#### Standard Post-Match Screen

- Heroes clearly visible in team sections
- Use standard ally_1-5, enemy_1-5 naming
- Annotate all 10 heroes when visible

#### Partial Views/Scrolled Screens

- Only annotate fully visible hero icons
- Maintain position numbering based on layout
- Skip cut-off or partially visible icons

## 🏅 Medal Detection Guidelines

### Medal Types and Classes

1. **MVP Badges**

   - `mvp_badge`: Gold MVP crown/badge
   - `mvp_loss_badge`: Silver MVP badge (losing team)

2. **Performance Medals**
   - `medal_gold`: Gold performance medal
   - `medal_silver`: Silver performance medal
   - `medal_bronze`: Bronze performance medal

### Bounding Box Rules

#### What to Include ✅

- **Complete Medal Icon**: Entire medal/badge graphic
- **Glow Effects**: Surrounding glow or shine effects
- **Medal Text**: Any text within the medal (MVP, etc.)

#### What to Exclude ❌

- **Background Elements**: Game background behind medal
- **Overlapping UI**: Other interface elements
- **Player Stats**: Numbers or text outside medal area

### Quality Thresholds

#### High Confidence (Required)

- Medal clearly visible and recognizable
- No obstruction by other UI elements
- Standard size and positioning

#### Medium Confidence (Include with Notes)

- Partially visible but identifiable
- Slight UI overlap but medal type clear
- Non-standard positioning but recognizable

#### Low Confidence (Skip)

- Heavily obscured or cut off
- Unclear medal type
- Distorted or corrupted appearance

### Technical Specifications

- **Aspect Ratio**: 0.8 - 1.2 (slightly flexible)
- **Minimum Size**: 30x30 pixels
- **Maximum Size**: 100x100 pixels
- **Preferred Size**: 60x60 pixels

## 🎮 Match Type Detection Guidelines

### Match Type Classes

1. **Game Modes**

   - `match_type_classic`: Classic match indicator
   - `match_type_ranked`: Ranked match indicator
   - `match_type_brawl`: Brawl mode indicator
   - `match_type_custom`: Custom game indicator

2. **Match Outcomes**
   - `victory_text`: "Victory" text display
   - `defeat_text`: "Defeat" text display

### Detection Areas

#### Mode Indicators

- Usually located in top section of screen
- May appear as text or icon badges
- Include surrounding UI context if needed

#### Result Text

- Large, prominent text in center/top area
- Include complete text string
- Exclude decorative elements around text

### Technical Specifications

- **Text Elements**: Minimum 100x20 pixels
- **Mode Badges**: 50x25 to 150x75 pixels
- **Result Text**: Minimum 200x50 pixels

## 📊 Statistics Detection Guidelines

### Core Statistics Classes

1. **KDA Display**

   - `kda_display`: Kill/Death/Assist format (e.g., "5/2/8")

2. **Economic Metrics**

   - `gold_amount`: Total gold earned
   - `gpm_display`: Gold per minute

3. **Combat Statistics**

   - `damage_dealt`: Damage to heroes
   - `damage_taken`: Damage received
   - `healing_done`: Healing provided

4. **Performance Metrics**
   - `participation_rate`: Team fight participation %
   - `turret_damage`: Damage to structures
   - `match_duration`: Game length

### Bounding Box Guidelines

#### Numerical Values

- Include complete number string
- Include unit indicators (%, K, M)
- Exclude descriptive labels when separate

#### Combined Labels

- Include both label and value when integrated
- Use tight bounding box around text area
- Maintain readability standards

## 🎯 UI Elements Detection

### Player Identification

1. **Player Names**

   - `player_name`: In-game name display
   - Include complete name string
   - Exclude clan tags if separately displayed

2. **Team Indicators**
   - `team_indicator_ally`: Ally team section marker
   - `team_indicator_enemy`: Enemy team section marker

### Interface Elements

1. **Container Areas**

   - `scoreboard_container`: Main statistics area
   - Use for full scoreboard sections

2. **Quality Markers** (for validation)
   - `ui_complete`: Interface fully loaded
   - `text_readable`: Text clearly legible
   - `icons_clear`: Icons properly rendered

## 🛠️ Equipment & Items Detection

### Item Categories

1. **Equipment Slots**

   - `item_icon_1` through `item_icon_6`: Equipment items
   - Use position-based numbering (left to right)

2. **Battle Spells**
   - `battle_spell_1`: First battle spell
   - `battle_spell_2`: Second battle spell

### Guidelines

- Include complete item icon area
- Exclude empty slots unless specifically needed
- Maintain consistent slot numbering across images

## 🎊 Achievement Detection

### Special Indicators

1. **Achievement Badges**

   - `savage_indicator`: Savage kill achievement
   - `maniac_indicator`: Maniac kill achievement
   - `legendary_indicator`: Legendary kill streak

2. **Role Information**
   - `role_indicator`: Player role badge/text
   - `position_rank`: Ranking within role

### Technical Notes

- These elements may be rare in screenshots
- Prioritize clear, prominent displays
- Document unusual placements or styles

## ✅ Quality Control Checklist

### Before Starting Annotation

- [ ] Screenshot is clear and high resolution (1080p+)
- [ ] Post-match screen is completely loaded
- [ ] No major UI glitches or loading states
- [ ] At least 80% of game interface visible

### During Annotation

- [ ] Bounding boxes are tight and accurate
- [ ] Class labels match visible objects exactly
- [ ] Position-based numbering is consistent
- [ ] No duplicate annotations for same object
- [ ] All clearly visible target objects annotated

### After Completing Image

- [ ] All hero icons properly labeled (ally_1-5, enemy_1-5)
- [ ] Performance indicators identified correctly
- [ ] Text elements include complete readable area
- [ ] No overlapping bounding boxes without reason
- [ ] Annotation count matches expected objects

## 🚨 Common Mistakes to Avoid

### Hero Icons

❌ Including level numbers in bounding box  
❌ Missing partially visible heroes  
❌ Inconsistent position numbering  
✅ Clean icon boundaries, systematic numbering

### Medals & Badges

❌ Cutting off glow effects  
❌ Including too much background  
❌ Annotating unclear/corrupted medals  
✅ Complete medal area, quality standards

### Statistics

❌ Splitting integrated label-value pairs  
❌ Including column headers with values  
❌ Missing unit indicators (%, K, M)  
✅ Complete stat displays, proper grouping

### General

❌ Annotations too large or too small  
❌ Skipping objects due to minor occlusion  
❌ Inconsistent quality standards  
✅ Systematic approach, consistent quality

## 📈 Annotation Workflow

### 1. Initial Assessment

- Load screenshot in annotation tool
- Identify all visible target objects
- Note any quality issues or unusual layouts

### 2. Systematic Annotation

- Start with hero icons (ally team, then enemy team)
- Add performance indicators (MVP, medals)
- Include match information (type, result)
- Annotate statistics and UI elements

### 3. Quality Review

- Check all bounding boxes for accuracy
- Verify class labels are correct
- Ensure consistent naming conventions
- Remove or fix any problematic annotations

### 4. Final Validation

- Run validation script (see validation.py)
- Address any flagged issues
- Document any unusual cases or edge conditions

## 🔧 Tools and Shortcuts

### Label Studio Tips

- Use keyboard shortcuts for faster annotation
- Create custom color schemes for object types
- Use templates for recurring annotation patterns
- Regular save/backup of annotation progress

### Validation Integration

- Run validation script after each session
- Address dimension and ratio warnings
- Maintain annotation quality logs
- Regular peer review of annotations

---

**Target Quality**: 95%+ accuracy on validation set  
**Minimum Annotations**: 200+ per high-priority class  
**Review Process**: Peer validation + automated checks
