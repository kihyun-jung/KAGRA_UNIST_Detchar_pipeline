#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mock ML Training Data Generator
===============================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = BASE_DIR / "data" / "training_set"

# ▼▼▼ 이 부분을 수정했습니다 ▼▼▼
CLASSES = [
    "Blip", 
    "Dot", 
    "Helix", 
    "Line", 
    "Scratchy", 
    "Scattering", 
    "Not_Classified"
]
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

def create_mock_dataset(num_samples=10):
    print(f"[*] Generating mock training dataset in {DATASET_DIR}...")
    
    # (이하 코드는 동일합니다)
    for cls in CLASSES:
        cls_dir = DATASET_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_samples):
            # 랜덤 노이즈 이미지 생성
            data = np.random.rand(100, 100)
            
            plt.figure(figsize=(2, 2))
            plt.imshow(data, cmap='viridis', aspect='auto')
            plt.axis('off')
            
            filename = f"{cls}_{i:03d}.png"
            plt.savefig(cls_dir / filename, bbox_inches='tight', pad_inches=0)
            plt.close()
            
    print(f"✅ Created {num_samples} images per class for: {CLASSES}")

if __name__ == "__main__":
    create_mock_dataset()
