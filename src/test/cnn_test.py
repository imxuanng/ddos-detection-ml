import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from src.constant import PROCESSED_OUTPUTS_TEST, PROCESSED_FEATURES, ROOT_PATH
from src.utils.io import read_multi_csv
from src.utils.visualization import show_confusion_matrix

# === Đọc dữ liệu test đã xử lý ===
df_test = read_multi_csv(PROCESSED_OUTPUTS_TEST)
df_test = df_test[PROCESSED_FEATURES]

X_test = df_test.drop('label', axis=1).values
y_test = df_test['label'].values

# Định hình cho CNN 1D:  (samples, features, channels)
# Khác với LSTM, CNN reshape theo dạng (samples, features, 1)
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"Shape of X_test_cnn: {X_test_cnn.shape}")
print(f"Shape of y_test: {y_test.shape}")

# === Load mô hình CNN đã train ===
model_path = ROOT_PATH + "models/cnn/cicddos_cnn_model.h5"
model = load_model(model_path)

print(f"\nModel loaded from: {model_path}")
model.summary()

# === Dự đoán ===
print("\nPredicting on test data...")
y_pred_prob = model.predict(X_test_cnn, batch_size=256, verbose=1)
y_pred = (y_pred_prob >= 0.5).astype(int).reshape(-1)

# === In một số mẫu dự đoán ===
print("\nSample predictions:")
for i in range(5):
    print(f"Sample {i}: True={y_test[i]}, Predicted={y_pred[i]}, Probability={y_pred_prob[i][0]:.4f}")

# === Đánh giá trực quan & in báo cáo ===
print("\n" + "="*50)
print("CONFUSION MATRIX & CLASSIFICATION REPORT")
print("="*50)
show_confusion_matrix(y_test, y_pred, class_names=["Benign", "Attack"])
