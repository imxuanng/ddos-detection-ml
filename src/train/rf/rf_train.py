import os
import joblib
from typing import List, Optional
from sklearn.ensemble import RandomForestClassifier

from src.constant import (
    PROCESSED_FEATURES,
    PROCESSED_OUTPUTS_TRAIN,
    ROOT_PATH
)
from src.utils.io import read_multi_csv

# ==== Đọc & chọn dữ liệu ====
df_train = read_multi_csv(PROCESSED_OUTPUTS_TRAIN)
df_train = df_train[PROCESSED_FEATURES]

# ==== Tách X, y ====
X_train = df_train.drop('label', axis=1).values
y_train = df_train['label'].values

# ==== Khởi tạo và train Random Forest ====
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    n_jobs=-1,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Lưu model ra thư mục models/rf/
model_save_dir = ROOT_PATH + "models/rf/"
os.makedirs(model_save_dir, exist_ok=True)
model_save_file = os.path.join(model_save_dir, "cicddos_rf_model.joblib")
joblib.dump(rf_model, model_save_file)

# Lưu lại danh sách feature
joblib.dump(PROCESSED_FEATURES, os.path.join(model_save_dir, "cicddos_features.joblib"))

print("Training completed and model is saved. Ready for testing phase after this.")
