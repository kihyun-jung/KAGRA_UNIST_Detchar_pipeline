import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys
import shutil
import pandas as pd
import argparse
import matplotlib.pyplot as plt

try:
    from model_pytorch import GlitchClassifier
except ImportError:
    from src.ml.model_pytorch import GlitchClassifier

def predict_and_sort(args):
    print("[PyTorch] Starting Inference with Full Probabilities...")
    
    model_path = Path(args.model_path)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    csv_path = Path(args.csv_path)
    
    if not model_path.exists():
        print(f"[!] Model not found: {model_path}")
        sys.exit(1)

    # 1. 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    classes = checkpoint['classes']
    model = GlitchClassifier(num_classes=len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"[*] Loaded Model. Classes: {classes}")

    # 2. 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_paths = sorted(list(input_dir.glob("*.png")))
    if not image_paths:
        print(f"[!] No images found in {input_dir}")
        return

    print(f"[*] Found {len(image_paths)} images to classify.")
    
    results = []
    
    # 3. 추론 루프
    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert('RGB')
                input_tensor = transform(img).unsqueeze(0).to(device)
                
                # 예측
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1)[0] # 배치 차원 제거 [C]
                
                # Top-1 클래스 및 확률
                max_prob, predicted_idx = torch.max(probs, 0)
                pred_class = classes[predicted_idx.item()]
                confidence = max_prob.item() * 100.0 
                
                # [핵심 수정] 기본 정보 저장
                row = {
                    "filename": img_path.name,
                    "predicted_class": pred_class,
                    "confidence": f"{confidence:.2f}%"
                }
                
                # [핵심 수정] 모든 클래스 확률 추가 (Loop)
                for idx, class_name in enumerate(classes):
                    prob_percent = probs[idx].item() * 100.0
                    row[class_name] = f"{prob_percent:.2f}%"

                results.append(row)
                
                # 이미지 복사
                dest_dir = output_dir / pred_class
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dest_dir / img_path.name)
                
                if (i+1) % 10 == 0:
                    sys.stdout.write(f"\r    -> Processed {i+1}/{len(image_paths)}")
                    sys.stdout.flush()
                    
            except Exception as e:
                print(f"\n[!] Error processing {img_path.name}: {e}")

    print("\n[*] Inference completed.")

    # 4. CSV 저장 (컬럼 순서 정렬: 기본정보 -> 클래스별 확률)
    if results:
        df = pd.DataFrame(results)
        
        # 컬럼 순서 예쁘게 정리
        cols = ['filename', 'predicted_class', 'confidence'] + classes
        # 혹시 모를 에러 방지를 위해 존재하는 컬럼만 선택
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        
        df.to_csv(csv_path, index=False)
        print(f"Detailed predictions saved to {csv_path}")
    
    # 5. 요약 그래프
    if results:
        df = pd.DataFrame(results)
        if not df.empty:
            class_counts = df['predicted_class'].value_counts()
            plt.figure(figsize=(8, 8))
            class_counts.plot.pie(autopct='%1.1f%%', startangle=90, cmap='Pastel1')
            plt.title('Glitch Classification Distribution')
            plt.ylabel('')
            plt.savefig(output_dir / "classification_summary_pytorch.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    # data_dir는 사용되지 않지만 인터페이스 호환성을 위해 남겨둠
    parser.add_argument("--data_dir", type=str, default="") 
    args = parser.parse_args()
    
    predict_and_sort(args)
