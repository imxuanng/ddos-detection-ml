import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dense, Dropout
from src.constant import PROCESSED_OUTPUTS_TEST, PROCESSED_FEATURES, ROOT_PATH
from src.utils.io import read_multi_csv
from src.utils.visualization import show_confusion_matrix


# === Custom Layers (BẮT BUỘC phải có để load model) ===
class TransformerBlock(tf.keras. layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self. dropout_rate = dropout_rate
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras. Sequential([
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
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads":  self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np. arange(0, self.embed_dim, 2) * -(np.log(10000.0) / self.embed_dim))
        
        pos_encoding = np.zeros((self.max_len, self.embed_dim))
        pos_encoding[:, 0::2] = np. sin(position * div_term)
        pos_encoding[: , 1::2] = np.cos(position * div_term)
        
        self.pos_encoding = tf.cast(pos_encoding[np.newaxis, ... ], dtype=tf.float32)
        
    def call(self, inputs):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "embed_dim": self.embed_dim,
        })
        return config


# === Đọc dữ liệu test đã xử lý ===
df_test = read_multi_csv(PROCESSED_OUTPUTS_TEST)
df_test = df_test[PROCESSED_FEATURES]

X_test = df_test.drop('label', axis=1).values
y_test = df_test['label'].values

# Định hình cho Transformer (timesteps=1)
X_test_transformer = X_test. reshape((X_test.shape[0], 1, X_test.shape[1]))

# === Load mô hình đã train ===
model_path = ROOT_PATH + "models/transformer/cicddos_transformer_model.h5"

# ⚠️ QUAN TRỌNG: Phải thêm custom_objects
custom_objects = {
    'TransformerBlock': TransformerBlock,
    'PositionalEncoding': PositionalEncoding
}
model = load_model(model_path, custom_objects=custom_objects)

# === Dự đoán ===
y_pred_prob = model.predict(X_test_transformer)
y_pred = (y_pred_prob >= 0.5).astype(int).reshape(-1)

# === Đánh giá trực quan & in báo cáo ===
show_confusion_matrix(y_test, y_pred, class_names=["Benign", "Attack"])
