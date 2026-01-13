import tensorflow as tf
from tensorflow.keras import layers, models, applications

def create_model(input_shape=(224, 224, 3), num_classes=3):
    """
    KAGRA Glitch Classification Model (ResNet50 Transfer Learning)
    """
    # 1. Base Model (ResNet50) - ImageNet 가중치 사용
    base_model = applications.ResNet50(
        include_top=False, # 마지막 분류 레이어 제외
        weights='imagenet',
        input_shape=input_shape
    )
    
    # 2. Base Model 동결 (Feature Extractor로 사용)
    base_model.trainable = False 
    
    # 3. 새로운 분류 헤드 부착
    inputs = tf.keras.Input(shape=input_shape)
    
    # 전처리 (ResNet50 전용 preprocess_input 사용 권장)
    x = applications.resnet50.preprocess_input(inputs)
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x) # 과적합 방지
    outputs = layers.Dense(num_classes)(x) # from_logits=True를 위해 activation 없음
    
    model = tf.keras.Model(inputs, outputs)
    
    return model
