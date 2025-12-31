import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape=(128, 128, 3), num_classes=7):
    """
    KAGRA Glitch Classification을 위한 TensorFlow/Keras 모델
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # 1. 전처리: 0~255 값을 0~1로 정규화
        layers.Rescaling(1./255),
        
        # 2. Convolution Layer 1
        layers.Conv2d(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # 3. Convolution Layer 2
        layers.Conv2d(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # 4. Convolution Layer 3
        layers.Conv2d(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # 5. Classification
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # 과적합 방지
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
