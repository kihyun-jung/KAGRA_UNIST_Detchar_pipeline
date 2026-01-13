import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers
from pathlib import Path
import sys
import argparse
import matplotlib.pyplot as plt
import json

# 모델 임포트
try:
    from model_tf import create_model
except ImportError:
    from src.ml.model_tf import create_model

def save_plots(history, save_path):
    """학습 곡선 저장"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, 'b-', label='Train Loss')
    plt.plot(epochs_range, val_loss, 'r-', label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, acc, 'b-', label='Train Acc')
    plt.plot(epochs_range, val_acc, 'r-', label='Val Acc')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[*] Learning curves saved to {save_path}")

def train(args):
    # GPU 설정 확인
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Starting TensorFlow Training on {gpus if gpus else 'CPU'}...")
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[!] Data directory not found: {data_dir}")
        sys.exit(1)

    # 1. 데이터 로드 (Keras Utility 사용)
    # Validation Split 20%
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(224, 224),
        batch_size=args.batch_size,
        label_mode='int' # SparseCategoricalCrossentropy 사용
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(224, 224),
        batch_size=args.batch_size,
        label_mode='int'
    )

    # 클래스 이름 추출 및 저장 (Inference 때 필수)
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"[*] Detected Classes ({num_classes}): {class_names}")

    # 성능 최적화
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 2. 모델 생성 및 컴파일
    model = create_model(num_classes=num_classes)
    
    model.compile(optimizer='adam',
                  loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # 3. 학습
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs
    )

    # 4. 모델 저장 (.h5)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    # 5. 클래스 정보 저장 (JSON) - 중요!
    # TF 모델 파일에는 label string 정보가 없으므로 별도 저장해야 함
    json_path = save_path.parent / "classes_tf.json"
    with open(json_path, 'w') as f:
        json.dump(class_names, f)
    print(f"Class names saved to {json_path}")

    # 6. 그래프 저장
    save_plots(history, args.plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--plot_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    train(args)
