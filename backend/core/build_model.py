import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def build_model():
    """Defines the dense, cascading Deep Neural Network (DNN) architecture."""
    model = models.Sequential([
        layers.InputLayer(input_shape=(1024,)),
        
        # Hidden Layer 1 — Expansion (2048 neurons)
        layers.Dense(2048),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Hidden Layer 2 — Compression (1024 neurons)
        layers.Dense(1024),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Hidden Layer 3 — Compression (512 neurons)
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output Layer — Binary decision
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
