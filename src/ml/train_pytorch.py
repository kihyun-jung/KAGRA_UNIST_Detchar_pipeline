#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
from src.ml.model_pytorch import SimpleCNN  # <-- 변경됨

def train_model(epochs=5):
    print("🔵 [PyTorch] Starting Training Pipeline...")
    data_dir = BASE_DIR / "data" / "training_set"
    # 모델명 구분 저장
    model_save_path = BASE_DIR / "results" / "ml" / "models" / "glitch_classifier_pytorch.pth"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    try:
        train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    except Exception as e:
        print(f"[!] Error loading dataset: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"    Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': train_dataset.classes
    }, model_save_path)
    print(f"✅ [PyTorch] Model saved to {model_save_path}")

if __name__ == "__main__":
    train_model()
