import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import logging
from pathlib import Path
from typing import Tuple, Union
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScreenshotClassifier:
    """
    A PyTorch-based classifier for game screenshots using ResNet50.
    
    Classifies screenshots into game categories: mlbb, honor_of_kings, wild_rift, dota2, other
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the screenshot classifier.
        
        Args:
            model_path: Path to pre-trained model weights (optional)
        """
        self.games = ['mlbb', 'honor_of_kings', 'wild_rift', 'dota2', 'other']
        self.num_classes = len(self.games)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize model
        self.model = self._build_model()
        
        # Load pre-trained weights if available
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        else:
            logger.warning("No pre-trained model found. Using randomly initialized weights.")
    
    def _build_model(self) -> nn.Module:
        """Build ResNet50 model with custom classifier head."""
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Freeze early layers for transfer learning
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace classifier head
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        # Unfreeze classifier head
        for param in model.fc.parameters():
            param.requires_grad = True
        
        return model.to(self.device)
    
    def _load_model(self, model_path: str):
        """Load pre-trained model weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
            
        Raises:
            ValueError: If image cannot be loaded or processed
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise ValueError(f"Image file not found: {image_path}")
            
            # Load and convert image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transformations
            image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise ValueError(f"Failed to preprocess image: {str(e)}")
    
    def classify(self, image_path: Union[str, Path]) -> Tuple[str, float]:
        """
        Classify a game screenshot.
        
        Args:
            image_path: Path to the screenshot image
            
        Returns:
            Tuple of (game_name, confidence_score)
            
        Raises:
            ValueError: If image cannot be processed
            RuntimeError: If model inference fails
        """
        try:
            # Validate input
            if not image_path:
                raise ValueError("Image path cannot be empty")
            
            # Preprocess image
            image_tensor = self._preprocess_image(image_path)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence_score, predicted_idx = torch.max(probabilities, 1)
                
                # Convert to Python types
                predicted_game = self.games[predicted_idx.item()]
                confidence = confidence_score.item()
                
                logger.info(f"Classified {image_path} as {predicted_game} with confidence {confidence:.3f}")
                
                return predicted_game, confidence
                
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error during classification: {str(e)}")
            raise RuntimeError(f"Model inference failed: {str(e)}")
    
    def get_all_predictions(self, image_path: Union[str, Path]) -> dict:
        """
        Get prediction probabilities for all game classes.
        
        Args:
            image_path: Path to the screenshot image
            
        Returns:
            Dictionary with game names as keys and confidence scores as values
        """
        try:
            image_tensor = self._preprocess_image(image_path)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Convert to dictionary
                predictions = {}
                for i, game in enumerate(self.games):
                    predictions[game] = probabilities[0][i].item()
                
                return predictions
                
        except Exception as e:
            logger.error(f"Error getting predictions: {str(e)}")
            raise RuntimeError(f"Failed to get predictions: {str(e)}")
    
    def save_model(self, save_path: str):
        """Save model weights to file."""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'games': self.games,
                'num_classes': self.num_classes
            }, save_path)
            
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise


def main():
    """Example usage of the ScreenshotClassifier."""
    # Initialize classifier
    classifier = ScreenshotClassifier()
    
    # Example classification
    try:
        # Replace with actual image path
        image_path = "screenshot.png"
        
        if Path(image_path).exists():
            game_name, confidence = classifier.classify(image_path)
            print(f"Game: {game_name}, Confidence: {confidence:.3f}")
            
            # Get all predictions
            all_predictions = classifier.get_all_predictions(image_path)
            print("\nAll predictions:")
            for game, score in sorted(all_predictions.items(), key=lambda x: x[1], reverse=True):
                print(f"  {game}: {score:.3f}")
        else:
            print(f"Image file not found: {image_path}")
            
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()