# Resnet50-Fintetune
# Cat vs Dog Image Classifier

A PyTorch implementation of a binary image classifier using ResNet50 with transfer learning. This model distinguishes between images of cats and dogs using feature extraction from a pre-trained ResNet50 model.

Dataset: https://www.kaggle.com/datasets/tongpython/cat-and-dog

## Setup

1. Create a virtual environment and install dependencies:
```bash
python3.11 -m venv myenv
source myenv/bin/activate
pip install torch torchvision Pillow numpy matplotlib
```

## Project Structure
```
project/
│
├── data/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── test/
│       ├── cats/
│       └── dogs/
│
└── cat_dog_classifier.pth    # Saved model
```

## Model Details
- Base Model: ResNet50 (pretrained on ImageNet)
- Training Approach: Feature Extraction (frozen base layers, trained classifier)
- Final Layer: Binary classifier (2 output classes)
- Input Size: 224x224 pixels

## Training Results
- Training Loss: 0.082 (final)
- Validation Loss: 0.057 (final)
- Validation Accuracy: 97.68% (final)

## Usage

To train the model:
```python
# Load and prepare the model
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = torch.nn.Linear(model.fc.in_features, 2)

# Train
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5)
```

To save the model:
```python
torch.save(model.state_dict(), 'cat_dog_classifier.pth')
```

To load the trained model:
```python
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('cat_dog_classifier.pth'))
model.eval()  # Set to evaluation mode
```

## Hardware Notes
- Optimized for Apple Silicon (M-series) using MPS
- Can be adapted for NVIDIA GPUs by changing device to CUDA

## Future Improvements
- Implement data augmentation
- Try full fine-tuning
- Add model inference examples
- Add confusion matrix visualization
