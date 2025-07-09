# Dataset Structure for Screenshot Classifier

## Directory Organization

```
dataset/
├── train/
│   ├── mlbb/
│   │   ├── mlbb_screenshot_001.jpg
│   │   ├── mlbb_screenshot_002.jpg
│   │   └── ...
│   ├── honor_of_kings/
│   │   ├── hok_screenshot_001.jpg
│   │   ├── hok_screenshot_002.jpg
│   │   └── ...
│   ├── wild_rift/
│   │   ├── wr_screenshot_001.jpg
│   │   ├── wr_screenshot_002.jpg
│   │   └── ...
│   ├── dota2/
│   │   ├── dota2_screenshot_001.jpg
│   │   ├── dota2_screenshot_002.jpg
│   │   └── ...
│   └── other/
│       ├── other_game_001.jpg
│       ├── other_game_002.jpg
│       └── ...
├── val/
│   ├── mlbb/
│   ├── honor_of_kings/
│   ├── wild_rift/
│   ├── dota2/
│   └── other/
└── test/
    ├── mlbb/
    ├── honor_of_kings/
    ├── wild_rift/
    ├── dota2/
    └── other/
```

## Data Collection Guidelines

### For Each Game Category:

**MLBB (Mobile Legends: Bang Bang)**
- Collect in-game screenshots during matches
- Include UI elements, minimap, heroes, abilities
- Capture different game phases (early, mid, late game)
- Recommended: 500-1000 images per category

**Honor of Kings**
- Similar to MLBB but with distinct UI/visual style
- Include Chinese characters if applicable
- Different hero designs and map layouts

**Wild Rift (League of Legends: Wild Rift)**
- Distinctive Riot Games UI style
- Different item icons and champion designs
- Unique jungle layout and objectives

**Dota 2**
- Steam/Valve UI elements
- Distinctive hero designs and abilities
- Different map structure and visual style

**Other**
- Screenshots from other games (non-MOBA)
- Generic game screenshots
- Non-game images that might be mistakenly uploaded

## Training Recommendations

1. **Minimum per class**: 200-500 images
2. **Optimal per class**: 1000+ images
3. **Image quality**: High resolution preferred
4. **Variety**: Different devices, screen sizes, game modes
5. **Balance**: Similar number of images per class

## Split Ratios
- Training: 70% of data
- Validation: 15% of data  
- Testing: 15% of data