import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from pathlib import Path
import logging
from screenshot_classifier import ScreenshotClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_classifier(dataset_path: str = "dataset", 
                    epochs: int = 20,
                    batch_size: int = 32,
                    learning_rate: float = 0.001,
                    save_path: str = "models/screenshot_classifier.pth"):
    """
    Train the screenshot classifier on organized dataset.
    
    Args:
        dataset_path: Path to dataset directory with train/val/test folders
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        save_path: Path to save trained model
    """
    
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_path, 'train'),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_path, 'val'),
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    classifier = ScreenshotClassifier()
    model = classifier.model
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    device = classifier.device
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Calculate accuracies
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        logger.info(f'Epoch {epoch+1}/{epochs}:')
        logger.info(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Create directory if it doesn't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'games': classifier.games,
                'class_to_idx': train_dataset.class_to_idx
            }, save_path)
            logger.info(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        scheduler.step()
    
    logger.info(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')
    return best_val_acc

def evaluate_model(model_path: str, test_dataset_path: str):
    """Evaluate trained model on test set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    classifier = ScreenshotClassifier()
    classifier.model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test transform
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    test_dataset = datasets.ImageFolder(
        root=test_dataset_path,
        transform=test_transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate
    classifier.model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = classifier.model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_acc = 100 * correct / total
    logger.info(f'Test Accuracy: {test_acc:.2f}%')
    return test_acc

if __name__ == "__main__":
    # Check if dataset exists
    if not os.path.exists("dataset/train"):
        print("Dataset not found. Please organize your images in the dataset/ directory structure.")
        print("See dataset/README.md for details.")
        exit(1)
    
    # Train the model
    print("Starting training...")
    best_acc = train_classifier(
        dataset_path="dataset",
        epochs=20,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Evaluate on test set
    if os.path.exists("dataset/test"):
        print("Evaluating on test set...")
        test_acc = evaluate_model("models/screenshot_classifier.pth", "dataset/test")
        print(f"Final test accuracy: {test_acc:.2f}%")