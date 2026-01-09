import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, 
    MultiHeadAttention, GlobalAveragePooling1D, Add
)
from tensorflow.keras. callbacks import EarlyStopping, ReduceLROnPlateau
from src.constant import (
    PROCESSED_FEATURES,
    PROCESSED_OUTPUTS_TRAIN,
    ROOT_PATH
)
from src.utils.io import read_multi_csv


# ==== Custom Transformer Block ====
class TransformerBlock(tf.keras. layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self. dropout_rate = dropout_rate
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self. dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self. num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate":  self.dropout_rate,
        })
        return config


# ==== Positional Encoding (Optional) ====
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        # Create positional encoding matrix
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embed_dim, 2) * -(np.log(10000.0) / self.embed_dim))
        
        pos_encoding = np.zeros((self.max_len, self.embed_dim))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[: , 1::2] = np.cos(position * div_term)
        
        self.pos_encoding = tf.cast(pos_encoding[np.newaxis, ... ], dtype=tf.float32)
        
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "embed_dim": self. embed_dim,
        })
        return config


# ==== Đọc và chuẩn bị dữ liệu ====
print("Loading training data...")
df_train = read_multi_csv(PROCESSED_OUTPUTS_TRAIN)
df_train = df_train[PROCESSED_FEATURES]

X_train = df_train.drop('label', axis=1).values
y_train = df_train['label'].values

# Reshape cho Transformer:  (samples, timesteps, features)
# Timesteps = 1, features = số lượng features
X_train_transformer = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

print(f"X_train_transformer shape: {X_train_transformer.shape}")
print(f"y_train shape: {y_train.shape}")

for i in range(3):
    print(f"X_train_transformer sample {i}: {X_train_transformer[i]}")
    print(f"y_train sample {i}: {y_train[i]}")


# ==== Xây dựng Transformer Model ====
print("\nBuilding Transformer model...")

# Hyperparameters
embed_dim = X_train_transformer.shape[2]  # Số features
num_heads = 4  # Số attention heads
ff_dim = 128   # Feed-forward network dimension
num_transformer_blocks = 2  # Số lượng Transformer blocks
dropout_rate = 0.3

# Input layer
inputs = Input(shape=(X_train_transformer.shape[1], X_train_transformer.shape[2]))

# Positional Encoding (Optional)
x = PositionalEncoding(max_len=X_train_transformer.shape[1], embed_dim=embed_dim)(inputs)

# Stack Transformer Blocks
for _ in range(num_transformer_blocks):
    x = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, 
                         ff_dim=ff_dim, dropout_rate=dropout_rate)(x)

# Global Average Pooling
x = GlobalAveragePooling1D()(x)

# Dense layers
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)

# Output layer
outputs = Dense(1, activation='sigmoid')(x)

# Create model
model = Model(inputs=inputs, outputs=outputs)

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Model summary
print("\n" + "="*60)
model.summary()
print("="*60 + "\n")


# ==== Callbacks ====
es = EarlyStopping(
    monitor='loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)


# ==== Training ====
print("Starting training...")
history = model.fit(
    X_train_transformer, y_train,
    epochs=50,
    batch_size=256,
    callbacks=[es, reduce_lr],
    verbose=2
)


# ==== Lưu model ====
model_save_path = ROOT_PATH + 'models/transformer/cicddos_transformer_model.h5'
features_save_path = ROOT_PATH + 'models/transformer/cicddos_features_transformer.joblib'

model.save(model_save_path)
joblib.dump(PROCESSED_FEATURES, features_save_path)

print(f"\nTraining completed!")
print(f"Model saved to: {model_save_path}")
print(f"Features saved to: {features_save_path}")
print("Ready for testing phase after this.")


# ==== Hiển thị training history ====
print("\n" + "="*60)
print("TRAINING HISTORY")
print("="*60)
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Total Epochs Trained: {len(history.history['loss'])}")
