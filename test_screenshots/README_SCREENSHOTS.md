# Real Screenshot Testing Instructions

To enable comprehensive production validation, add MLBB screenshots to this directory:

## Required Screenshots:

1. **victory_scoreboard_clear.jpg** - Clear victory screen with full data
2. **defeat_scoreboard_normal.jpg** - Standard defeat screen
3. **hero_selection_screen.jpg** - Hero selection/pick screen
4. **scoreboard_low_light.jpg** - Screenshot in poor lighting
5. **scoreboard_motion_blur.jpg** - Slightly blurry screenshot
6. **partial_scoreboard.jpg** - Partially covered UI
7. **scoreboard_multilang.jpg** - Non-English interface (optional)
8. **hires_complete_scoreboard.jpg** - High-resolution perfect screenshot

## Screenshot Guidelines:

- **Format**: JPG, PNG, or BMP
- **Resolution**: 720p or higher recommended
- **Content**: Post-match scoreboards, hero selection screens
- **Quality**: Mix of perfect, average, and challenging conditions
- **Privacy**: Remove or blur player names if needed

## Testing Categories:

- **High Quality**: Clear, well-lit screenshots (Target: 95%+ confidence)
- **Average Quality**: Normal mobile screenshots (Target: 85%+ confidence)  
- **Challenging**: Poor lighting, blur, partial UI (Target: 75%+ confidence)

Add screenshots and run: `python test_real_screenshot_validation.py`
