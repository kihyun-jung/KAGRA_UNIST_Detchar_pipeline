import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    KAGRA Glitch Classification을 위한 간단한 CNN 모델
    Input: (3, 128, 128) RGB Images
    Output: Number of Classes (e.g., 3)
    """
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        # Convolution Layer 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Convolution Layer 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Fully Connected Layers
        # 128x128 -> pool -> 64x64 -> pool -> 32x32
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
