#!/usr/bin/env python3
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys
import pandas as pd

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
from src.ml.model_pytorch import SimpleCNN

def predict_qscans():
    print("🔵 [PyTorch] Starting Inference Pipeline...")
    model_path = BASE_DIR / "results" / "ml" / "models" / "glitch_classifier_pytorch.pth"
    target_root_dir = BASE_DIR / "results" / "qscan"
    output_csv = BASE_DIR / "results" / "ml" / "predictions" / "pytorch_classification_results.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"[!] Model not found: {model_path}")
        return

    # 모델 로드
    checkpoint = torch.load(model_path)
    classes = checkpoint['classes']
    model = SimpleCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 전처리
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # -------------------------------------------------------------
    # [수정됨] 메인 채널 이미지("Main-")만 필터링
    # -------------------------------------------------------------
    all_images = list(target_root_dir.rglob("*.png"))
    image_paths = [p for p in all_images if p.name.startswith("Main-")]
    
    if not image_paths:
        print(f"[!] No Main Channel Q-scan images found in {target_root_dir}")
        print("    (Note: Skipping Aux channel images)")
        return

    print(f"[*] Found {len(image_paths)} Main-Channel images (out of {len(all_images)} total files).")

    # 추론 수행
    results = []
    with torch.no_grad():
        for img_path in image_paths:
            try:
                # RGBA -> RGB 변환
                image = Image.open(img_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0)
                
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                
                pred_class = classes[predicted.item()]
                
                results.append({
                    "date": img_path.parent.parent.name,
                    "category": img_path.parent.name, # over8 or under8
                    "filename": img_path.name,
                    "prediction": pred_class
                })
            except Exception as e:
                print(f"[!] Error predicting {img_path.name}: {e}")

    # 결과 저장
    if results:
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"✅ [PyTorch] Results saved to {output_csv}")
        print(f"   (Classified {len(results)} Main channel glitches)")

if __name__ == "__main__":
    predict_qscans()
