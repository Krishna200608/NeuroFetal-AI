import tensorflow as tf
from tensorflow.keras import layers, models, Input

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    
    # First Conv
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Second Conv
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Adjust shortcut if shapes stick out (due to stride or filter change)
    if x.shape[-1] != shortcut.shape[-1] or stride != 1:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_fusion_resnet(input_shape_ts=(1200, 1), input_shape_tab=(3,)):
    # --- Branch 1: Time Series (ResNet 1D) ---
    input_ts = Input(shape=input_shape_ts, name='input_fhr')
    
    # Initial Conv to expand features?
    # Paper implementation detail might vary. 
    # Usually start with a Conv layer to get some filters.
    x1 = layers.Conv1D(64, 7, strides=2, padding='same')(input_ts)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.MaxPooling1D(3, strides=2, padding='same')(x1)
    
    # 3 Residual Blocks
    # User said "3 Residual Blocks (Conv1D + BatchNorm + ReLU)"
    # I'll stack 3 blocks. Increasing filters is common.
    # To match fusion dimension (which is likely 128), I'll target 128 filters at the end.
    
    x1 = residual_block(x1, 64)
    x1 = residual_block(x1, 128, stride=2) 
    x1 = residual_block(x1, 128)
    
    # Global Average Pooling
    x1 = layers.GlobalAveragePooling1D()(x1) # Output shape: (Batch, 128)
    
    # --- Branch 2: Tabular ---
    input_tab = Input(shape=input_shape_tab, name='input_tabular')
    
    # Dense 10 -> 128 with Dropout
    x2 = layers.Dense(10, activation='relu')(input_tab)
    x2 = layers.Dropout(0.3)(x2) # Assuming 0.3 dropout
    x2 = layers.Dense(128, activation='relu')(x2)
    x2 = layers.Dropout(0.3)(x2)
    
    # --- Fusion ---
    # Element-wise Multiplication
    # x1: (Batch, 128)
    # x2: (Batch, 128)
    fusion = layers.Multiply()([x1, x2])
    
    # --- Head ---
    # Final Dense + Sigmoid
    output = layers.Dense(1, activation='sigmoid', name='output')(fusion)
    
    model = models.Model(inputs=[input_ts, input_tab], outputs=output)
    return model

if __name__ == "__main__":
    model = build_fusion_resnet()
    model.summary()
