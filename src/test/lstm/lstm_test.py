# import pandas as pd
# import numpy as np
# from typing import List
# from tensorflow.keras.models import load_model
# from sklearn.metrics import classification_report, confusion_matrix
# import joblib
# import sys
# import os
# from src.constant import PROCESSED_OUTPUTS_TEST, PROCESSED_FEATURES
# from src.utils.io import read_multi_csv

# # === Nếu cần import constant từ src.constant ===
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))  # Thêm root project vào PYTHONPATH
# from src.constant import PROCESSED_OUTPUTS_TEST, PROCESSED_FEATURES



# # === Đọc dữ liệu test đã XỬ LÝ ===
# df_test = read_multi_csv(PROCESSED_OUTPUTS_TEST)
# df_test = df_test[PROCESSED_FEATURES]

# X_test = df_test.drop('label', axis=1).values
# y_test = df_test['label'].values

# # Định hình lại cho LSTM (timesteps=1)
# X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# # === Load model đã train ===
# model_path = 'src/models/lstm/cicddos_lstm_model.h5'
# model = load_model(model_path)

# # === Predict ===
# y_pred_prob = model.predict(X_test_lstm)
# y_pred = (y_pred_prob >= 0.5).astype(int).reshape(-1)

# # === Report ===
# print('Classification report:')
# print(classification_report(y_test, y_pred, target_names=["Benign", "Attack"]))

# print('Confusion matrix:')
# print(confusion_matrix(y_test, y_pred))

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