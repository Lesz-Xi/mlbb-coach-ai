#!/bin/bash
# MLBB YOLO Annotation Startup Script

echo "🎯 Starting MLBB YOLO Annotation Environment..."

# Set environment variables
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/Users/lesz/Documents/Project-AI/skillshift-ai/yolo_dataset/images

# Create data directory if it doesn't exist
mkdir -p ~/.local/share/label-studio

echo "📁 Project Directory: /Users/lesz/Documents/Project-AI/skillshift-ai/yolo_dataset"
echo "🖼️  Images Directory: /Users/lesz/Documents/Project-AI/skillshift-ai/yolo_dataset/images"
echo "🏷️  Config File: /Users/lesz/Documents/Project-AI/skillshift-ai/yolo_dataset/configs/label_studio_config.xml"

# Start Label Studio
echo "🚀 Starting Label Studio..."
echo "   💻 Open http://localhost:8080 in your browser"
echo "   📧 Create account or login"
echo "   🎯 Import config from: /Users/lesz/Documents/Project-AI/skillshift-ai/yolo_dataset/configs/label_studio_config.xml"

label-studio start --port 8080 \
    --data-dir ~/.local/share/label-studio \
    --log-level INFO

echo "👋 Label Studio session ended"
