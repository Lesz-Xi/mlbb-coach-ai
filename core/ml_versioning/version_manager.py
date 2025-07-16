"""
Version Manager
Manages model versions, rollbacks, and deployments
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

from .model_version import ModelVersion

logger = logging.getLogger(__name__)


class VersionManager:
    """Manages ML model versions"""
    
    def __init__(self, models_dir: Path, backups_dir: Path):
        self.models_dir = Path(models_dir)
        self.backups_dir = Path(backups_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        
        # Active models per type
        self.active_models: Dict[str, ModelVersion] = {}
        
        # All registered models
        self.registry: Dict[str, ModelVersion] = {}
        
        # Load existing models
        self._load_registry()
    
    def register_model(
        self,
        model: ModelVersion,
        activate: bool = False
    ) -> bool:
        """Register a new model version"""
        try:
            model_id = model.model_id
            
            # Check if already exists
            if model_id in self.registry:
                logger.warning(f"Model {model_id} already registered")
                return False
            
            # Save model metadata
            model.save_metadata(self.models_dir)
            
            # Add to registry
            self.registry[model_id] = model
            
            # Activate if requested
            if activate:
                self.activate_model(model_id)
            
            logger.info(f"Registered model: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            return False
    
    def activate_model(self, model_id: str) -> bool:
        """Activate a model version"""
        if model_id not in self.registry:
            logger.error(f"Model {model_id} not found in registry")
            return False
        
        model = self.registry[model_id]
        
        # Deactivate current active model of same type
        current_active = self.active_models.get(model.model_type)
        if current_active:
            current_active.status = "inactive"
            logger.info(
                f"Deactivated model: {current_active.model_id}"
            )
        
        # Activate new model
        model.status = "active"
        self.active_models[model.model_type] = model
        
        # Save updated registry
        self._save_registry()
        
        logger.info(f"Activated model: {model_id}")
        return True
    
    def rollback_model(
        self,
        model_type: str,
        version: Optional[str] = None
    ) -> bool:
        """Rollback to a previous model version"""
        # Get model history for type
        history = self.get_model_history(model_type)
        
        if not history:
            logger.error(f"No models found for type: {model_type}")
            return False
        
        if version:
            # Rollback to specific version
            target_model = None
            for model in history:
                if model.version == version:
                    target_model = model
                    break
            
            if not target_model:
                logger.error(
                    f"Version {version} not found for {model_type}"
                )
                return False
        else:
            # Rollback to previous version
            current_active = self.active_models.get(model_type)
            if not current_active or len(history) < 2:
                logger.error("No previous version available")
                return False
            
            # Find previous version
            target_model = None
            for i, model in enumerate(history):
                if model.model_id == current_active.model_id and i > 0:
                    target_model = history[i - 1]
                    break
        
        if target_model:
            return self.activate_model(target_model.model_id)
        
        return False
    
    def get_active_model(
        self, model_type: str
    ) -> Optional[ModelVersion]:
        """Get currently active model for a type"""
        return self.active_models.get(model_type)
    
    def get_model_history(
        self, model_type: str
    ) -> List[ModelVersion]:
        """Get version history for a model type"""
        history = [
            model for model in self.registry.values()
            if model.model_type == model_type
        ]
        
        # Sort by creation date (newest first)
        history.sort(key=lambda m: m.created_at, reverse=True)
        
        return history
    
    def compare_models(
        self,
        model_id1: str,
        model_id2: str
    ) -> Dict[str, Any]:
        """Compare two model versions"""
        model1 = self.registry.get(model_id1)
        model2 = self.registry.get(model_id2)
        
        if not model1 or not model2:
            return {"error": "One or both models not found"}
        
        comparison = {
            "model1": model1.model_id,
            "model2": model2.model_id,
            "metrics_diff": {},
            "hyperparameters_diff": {},
            "tags_diff": {
                "added": list(set(model2.tags) - set(model1.tags)),
                "removed": list(set(model1.tags) - set(model2.tags))
            }
        }
        
        # Compare metrics
        if model1.metrics and model2.metrics:
            metrics1 = model1.metrics.__dict__
            metrics2 = model2.metrics.__dict__
            
            for key in metrics1:
                if key != "custom_metrics":
                    val1 = metrics1[key]
                    val2 = metrics2.get(key, 0)
                    comparison["metrics_diff"][key] = {
                        "model1": val1,
                        "model2": val2,
                        "change": val2 - val1,
                        "change_pct": (
                            ((val2 - val1) / val1 * 100) if val1 != 0 else 0
                        )
                    }
        
        # Compare hyperparameters
        all_params = set(model1.hyperparameters.keys()) | set(
            model2.hyperparameters.keys()
        )
        for param in all_params:
            val1 = model1.hyperparameters.get(param)
            val2 = model2.hyperparameters.get(param)
            if val1 != val2:
                comparison["hyperparameters_diff"][param] = {
                    "model1": val1,
                    "model2": val2
                }
        
        return comparison
    
    def cleanup_old_models(
        self,
        model_type: str,
        keep_versions: int = 5
    ):
        """Clean up old model versions"""
        history = self.get_model_history(model_type)
        
        if len(history) <= keep_versions:
            return
        
        # Keep active model and most recent versions
        active_model = self.active_models.get(model_type)
        models_to_keep = set()
        
        if active_model:
            models_to_keep.add(active_model.model_id)
        
        # Add most recent versions
        for model in history[:keep_versions]:
            models_to_keep.add(model.model_id)
        
        # Archive old models
        for model in history[keep_versions:]:
            if model.model_id not in models_to_keep:
                self._archive_model(model)
    
    def _archive_model(self, model: ModelVersion):
        """Archive a model version"""
        try:
            # Create archive directory
            archive_dir = self.backups_dir / model.model_type
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Move model files
            if model.model_path.exists():
                archive_path = archive_dir / model.model_path.name
                shutil.move(str(model.model_path), str(archive_path))
                model.model_path = archive_path
            
            # Update status
            model.status = "archived"
            model.save_metadata(archive_dir)
            
            # Remove from active registry
            del self.registry[model.model_id]
            
            logger.info(f"Archived model: {model.model_id}")
            
        except Exception as e:
            logger.error(f"Failed to archive model: {str(e)}")
    
    def _load_registry(self):
        """Load model registry from disk"""
        # Load from models directory
        for metadata_file in self.models_dir.glob("*_metadata.json"):
            try:
                model = ModelVersion.load_metadata(metadata_file)
                self.registry[model.model_id] = model
                
                # Set active models
                if model.status == "active":
                    self.active_models[model.model_type] = model
                    
            except Exception as e:
                logger.error(
                    f"Failed to load model metadata {metadata_file}: {str(e)}"
                )
    
    def _save_registry(self):
        """Save registry state"""
        registry_file = self.models_dir / "registry.json"
        registry_data = {
            "active_models": {
                model_type: model.model_id
                for model_type, model in self.active_models.items()
            },
            "models": [
                model.model_id for model in self.registry.values()
            ]
        }
        
        with open(registry_file, "w") as f:
            json.dump(registry_data, f, indent=2)
    
    def get_metrics_summary(
        self, model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get metrics summary for models"""
        models = self.registry.values()
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if not models:
            return {}
        
        # Calculate averages
        metrics_sum = {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "inference_time_ms": 0,
            "memory_usage_mb": 0
        }
        
        count = 0
        for model in models:
            if model.metrics:
                count += 1
                for key in metrics_sum:
                    metrics_sum[key] += getattr(model.metrics, key, 0)
        
        if count > 0:
            for key in metrics_sum:
                metrics_sum[key] /= count
        
        return {
            "model_count": len(models),
            "active_count": len(
                [m for m in models if m.status == "active"]
            ),
            "average_metrics": metrics_sum
        } 