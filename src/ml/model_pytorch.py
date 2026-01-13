import torch
import torch.nn as nn
from torchvision import models

class GlitchClassifier(nn.Module):
    """
    KAGRA Glitch Classification Model (ResNet18 based)
    Input: (3, 224, 224) RGB Images
    Output: Logits for N classes
    """
    def __init__(self, num_classes):
        super(GlitchClassifier, self).__init__()
        # 전이 학습(Transfer Learning)을 위해 ImageNet으로 사전 학습된 가중치 사용
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 마지막 Fully Connected Layer를 우리 데이터셋의 클래스 개수에 맞게 교체
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.base_model(x)
