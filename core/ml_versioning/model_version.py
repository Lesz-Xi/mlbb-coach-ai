"""
Model Version
Represents a specific version of an ML model
"""

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class ModelMetrics:
    """Performance metrics for a model version"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time_ms: float
    memory_usage_mb: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelVersion:
    """Represents a specific version of an ML model"""
    name: str
    version: str
    model_type: str  # e.g., "hero_detection", "ocr", "trophy_detection"
    created_at: datetime
    created_by: str
    
    # Model artifacts
    model_path: Path
    config_path: Optional[Path] = None
    weights_path: Optional[Path] = None
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_data_version: Optional[str] = None
    
    # Performance
    metrics: Optional[ModelMetrics] = None
    status: str = "inactive"  # inactive, active, deprecated, testing
    
    # Deployment info
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    
    # Tracking
    usage_count: int = 0
    last_used: Optional[datetime] = None
    
    @property
    def model_id(self) -> str:
        """Generate unique model ID"""
        return f"{self.name}-{self.version}"
    
    @property
    def checksum(self) -> str:
        """Calculate model checksum for integrity"""
        if not self.model_path.exists():
            return ""
        
        hash_md5 = hashlib.md5()
        with open(self.model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "model_path": str(self.model_path),
            "config_path": (
                str(self.config_path) if self.config_path else None
            ),
            "weights_path": (
                str(self.weights_path) if self.weights_path else None
            ),
            "description": self.description,
            "tags": self.tags,
            "hyperparameters": self.hyperparameters,
            "training_data_version": self.training_data_version,
            "metrics": {
                "accuracy": self.metrics.accuracy,
                "precision": self.metrics.precision,
                "recall": self.metrics.recall,
                "f1_score": self.metrics.f1_score,
                "inference_time_ms": self.metrics.inference_time_ms,
                "memory_usage_mb": self.metrics.memory_usage_mb,
                "custom_metrics": self.metrics.custom_metrics
            } if self.metrics else None,
            "status": self.status,
            "deployment_config": self.deployment_config,
            "dependencies": self.dependencies,
            "usage_count": self.usage_count,
            "last_used": (
                self.last_used.isoformat() if self.last_used else None
            ),
            "checksum": self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary"""
        metrics = None
        if data.get("metrics"):
            metrics = ModelMetrics(**data["metrics"])
        
        return cls(
            name=data["name"],
            version=data["version"],
            model_type=data["model_type"],
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data["created_by"],
            model_path=Path(data["model_path"]),
            config_path=(
                Path(data["config_path"]) if data.get("config_path") else None
            ),
            weights_path=(
                Path(data["weights_path"]) if data.get("weights_path") else None
            ),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            hyperparameters=data.get("hyperparameters", {}),
            training_data_version=data.get("training_data_version"),
            metrics=metrics,
            status=data.get("status", "inactive"),
            deployment_config=data.get("deployment_config", {}),
            dependencies=data.get("dependencies", []),
            usage_count=data.get("usage_count", 0),
            last_used=(
                datetime.fromisoformat(data["last_used"]) 
                if data.get("last_used") else None
            )
        )
    
    def save_metadata(self, path: Path):
        """Save model metadata to file"""
        metadata_path = path / f"{self.model_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_metadata(cls, path: Path) -> "ModelVersion":
        """Load model metadata from file"""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def update_usage(self):
        """Update usage statistics"""
        self.usage_count += 1
        self.last_used = datetime.now()
    
    def is_compatible(self, requirements: Dict[str, Any]) -> bool:
        """Check if model meets requirements"""
        # Check model type
        if requirements.get("model_type") and self.model_type != requirements["model_type"]:
            return False
        
        # Check minimum metrics
        if self.metrics and requirements.get("min_metrics"):
            min_metrics = requirements["min_metrics"]
            if self.metrics.accuracy < min_metrics.get("accuracy", 0):
                return False
            if self.metrics.f1_score < min_metrics.get("f1_score", 0):
                return False
            if self.metrics.inference_time_ms > min_metrics.get("max_inference_time_ms", float('inf')):
                return False
        
        # Check tags
        if requirements.get("required_tags"):
            required_tags = set(requirements["required_tags"])
            if not required_tags.issubset(set(self.tags)):
                return False
        
        return True 