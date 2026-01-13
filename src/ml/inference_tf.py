import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import json
import shutil
import sys
import matplotlib.pyplot as plt
from pathlib import Path

def predict_and_sort(args):
    print("[TensorFlow] Starting Inference with Full Probabilities...")
    
    model_path = Path(args.model_path)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    csv_path = Path(args.csv_path)
    
    classes_path = model_path.parent / "classes_tf.json"

    if not model_path.exists():
        print(f"[!] Model not found: {model_path}")
        sys.exit(1)
    if not classes_path.exists():
        print(f"[!] Class info not found: {classes_path}")
        sys.exit(1)

    # 1. 모델 및 클래스 로드
    model = tf.keras.models.load_model(model_path)
    with open(classes_path, 'r') as f:
        class_names = json.load(f)
    print(f"[*] Loaded Model. Classes: {class_names}")

    image_paths = sorted(list(input_dir.glob("*.png")))
    if not image_paths:
        print(f"[!] No images found in {input_dir}")
        return

    print(f"[*] Found {len(image_paths)} images to classify.")

    results = []
    
    # 3. 추론 루프
    for i, img_path in enumerate(image_paths):
        try:
            img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            predictions = model.predict(img_array, verbose=0)
            score = tf.nn.softmax(predictions[0]).numpy() # numpy array로 변환
            
            predicted_idx = np.argmax(score)
            pred_class = class_names[predicted_idx]
            confidence = 100 * np.max(score)

            # [핵심 수정] 기본 정보
            row = {
                "filename": img_path.name,
                "predicted_class": pred_class,
                "confidence": f"{confidence:.2f}%"
            }

            # [핵심 수정] 모든 클래스 확률 추가
            for idx, class_name in enumerate(class_names):
                prob_percent = score[idx] * 100.0
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

    # 4. CSV 저장 (컬럼 정렬)
    if results:
        df = pd.DataFrame(results)
        cols = ['filename', 'predicted_class', 'confidence'] + class_names
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        
        df.to_csv(csv_path, index=False)
        print(f"Detailed predictions saved to {csv_path}")

    # 5. 그래프 저장
    if results:
        df = pd.DataFrame(results)
        if not df.empty:
            class_counts = df['predicted_class'].value_counts()
            plt.figure(figsize=(8, 8))
            class_counts.plot.pie(autopct='%1.1f%%', startangle=90, cmap='Pastel1')
            plt.title('Glitch Classification Distribution (TensorFlow)')
            plt.ylabel('')
            plt.savefig(output_dir / "classification_summary_tensorflow.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    args = parser.parse_args()
    
    predict_and_sort(args)
