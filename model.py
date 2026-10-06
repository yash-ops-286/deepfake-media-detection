import os
import torch
import torch.nn as nn
from torchvision import models

class DeepfakeDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepfakeDetector, self).__init__()
        # Load standard ResNet50 backbone
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # Replace the final classification head for binary classification (Real vs Fake)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # Index 0: Real, Index 1: Fake
        )

    def forward(self, x):
        return self.backbone(x)

def get_detector(weights_path=None, device="cpu"):
    """
    Instantiates the model and loads custom weights if provided.
    """
    has_weights = weights_path is not None and os.path.isfile(weights_path)
    model = DeepfakeDetector(pretrained=not has_weights)
    if has_weights:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model