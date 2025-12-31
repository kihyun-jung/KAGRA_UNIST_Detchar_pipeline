#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def predict_qscans():
    print("🟠 [TensorFlow] Starting Inference Pipeline...")
    model_dir = BASE_DIR / "results" / "ml" / "models"
    model_path = model_dir / "glitch_classifier_tf.h5"
    classes_path = model_dir / "classes_tf.json"
    
    target_root_dir = BASE_DIR / "results" / "qscan"
    output_csv = BASE_DIR / "results" / "ml" / "predictions" / "tf_classification_results.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists() or not classes_path.exists():
        print("[!] Model or classes file not found. Train first.")
        return

    # 모델 및 클래스 로드
    model = tf.keras.models.load_model(model_path)
    with open(classes_path, 'r') as f:
        class_names = json.load(f)

    # -------------------------------------------------------------
    # [수정됨] 메인 채널 이미지("Main-")만 필터링
    # -------------------------------------------------------------
    all_images = list(target_root_dir.rglob("*.png"))
    image_paths = [p for p in all_images if p.name.startswith("Main-")]

    if not image_paths:
        print(f"[!] No Main Channel Q-scan images found in {target_root_dir}")
        return

    print(f"[*] Found {len(image_paths)} Main-Channel images (out of {len(all_images)} total files).")

    results = []
    # 추론 루프
    for img_path in image_paths:
        try:
            img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            predictions = model.predict(img_array, verbose=0)
            score = tf.nn.softmax(predictions[0])
            pred_class = class_names[np.argmax(score)]
            confidence = 100 * np.max(score)

            results.append({
                "date": img_path.parent.parent.name,
                "category": img_path.parent.name,
                "filename": img_path.name,
                "prediction": pred_class,
                "confidence": f"{confidence:.2f}%"
            })
        except Exception: pass

    # 결과 저장
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"✅ [TensorFlow] Results saved to {output_csv}")
        print(f"   (Classified {len(results)} Main channel glitches)")

if __name__ == "__main__":
    predict_qscans()
