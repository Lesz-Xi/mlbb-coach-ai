# Generative Coach Implementation Progress

## Current Status
- [x] Core MLBB Coach working
- [x] IGN Validator
- [x] Video Reader Module
- [ ] LLM Integration


## TODOS
1. Screenshot Classifier (√)
• Input: Raw screenshot image
• Model: CNN or CLIP-based classifier
• Classes:
  - mlbb/
  - honor_of_kings/
  - wild_rift/
  - dota2/
  - other/
• Output: Game type label
• Logic:
  - If MLBB → run stat ingestor
  - Else → reject or handle gracefully

2. IGN Validator for better OCR Accuracy (√)
• Improves existing OCR
• Pure Python logic

• Integration Points:

  • Replace hardcoded "Lesz XVII" in from_screenshot
  • Add validation before match processing

• Benefits:
  • Reduces "player not found" errors
  • Handles special characters better

3. Video Reader Module (√)
• Purpose: Extract stats from gameplay recordings
• New Dependencies:
  • opencv-python==4.8.1
  • moviepy==1.0.3
  • ffmpeg-python==0.2.0
• Integration
  • New endpoint /analyze-video
  • Reuse existing OCR pipeline for frame analysis

4. LLM Integration
• High impact feature
• Start with OpenAI
• Add fallback logic 