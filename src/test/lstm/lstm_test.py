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

# Định hình cho LSTM (timesteps=1)
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# === Load mô hình đã train ===
model_path = ROOT_PATH + "models/lstm/cicddos_lstm_model.h5"
model = load_model(model_path)

# === Dự đoán ===
y_pred_prob = model.predict(X_test_lstm)
y_pred = (y_pred_prob >= 0.5).astype(int).reshape(-1)

# === Đánh giá trực quan & in báo cáo ===
show_confusion_matrix(y_test, y_pred, class_names=["Benign", "Attack"])
