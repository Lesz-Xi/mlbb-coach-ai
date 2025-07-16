#!/usr/bin/env python3
"""
YOLO Model Deployment Script for MLBB Coach AI
============================================

This script handles the deployment of the trained YOLOv8 model
for immediate operational readiness.
"""

import logging
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def deploy_yolo_model():
    """Deploy the YOLO model for immediate use."""
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "mlbb_yolo_best.pt"
    
    # Option 1: Use pre-trained YOLOv8 as temporary fallback
    if not model_path.exists():
        logger.info("🔄 No trained MLBB model found. Using YOLOv8 nano "
                   "as temporary fallback...")
        
        try:
            # Download YOLOv8 nano as fallback
            model = YOLO('yolov8n.pt')
            
            # Save to expected location
            fallback_path = models_dir / "yolov8n_fallback.pt"
            model.save(str(fallback_path))
            
            logger.info(f"✅ Fallback model deployed to {fallback_path}")
            logger.warning("⚠️  This is a generic model. Train MLBB-specific "
                          "model for optimal performance.")
            
            return str(fallback_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to deploy fallback model: {e}")
            return None
    
    else:
        logger.info(f"✅ MLBB-specific model found at {model_path}")
        return str(model_path)

def validate_model_deployment():
    """Validate that the model is working correctly."""
    try:
        from core.services.yolo_detection_service import get_yolo_detection_service
        
        service = get_yolo_detection_service()
        health = service.health_check()
        
        if health['status'] == 'healthy':
            logger.info("🎯 YOLO deployment validation: SUCCESSFUL")
            logger.info(f"📊 Model: {health['model_path']}")
            logger.info(f"🔧 Device: {health['device']}")
            return True
        else:
            logger.error(f"❌ YOLO deployment validation failed: {health.get('error', 'Unknown')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Validation error: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    logger.info("🚀 Starting YOLO model deployment...")
    
    model_path = deploy_yolo_model()
    if model_path:
        if validate_model_deployment():
            logger.info("🎯 YOLO deployment complete and validated!")
        else:
            logger.error("❌ Deployment validation failed")
    else:
        logger.error("❌ Model deployment failed") 