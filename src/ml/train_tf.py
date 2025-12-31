#!/usr/bin/env python3
import os
import json
import tensorflow as tf
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
from src.ml.model_tf import create_model # <-- 변경됨

def train_model(epochs=10):
    print("🟠 [TensorFlow] Starting Training Pipeline...")
    data_dir = BASE_DIR / "data" / "training_set"
    model_dir = BASE_DIR / "results" / "ml" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 모델명 구분 저장
    model_save_path = model_dir / "glitch_classifier_tf.h5"
    classes_save_path = model_dir / "classes_tf.json"

    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir, labels='inferred', label_mode='int',
            image_size=(128, 128), batch_size=8, shuffle=True
        )
    except Exception as e:
        print(f"[!] Error: {e}")
        return

    class_names = train_ds.class_names
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    model = create_model(num_classes=len(class_names))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_ds, epochs=epochs)

    model.save(model_save_path)
    with open(classes_save_path, 'w') as f:
        json.dump(class_names, f)
    
    print(f"✅ [TensorFlow] Model saved to {model_save_path}")

if __name__ == "__main__":
    train_model()
