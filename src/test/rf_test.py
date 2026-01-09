import joblib
from src.constant import PROCESSED_OUTPUTS_TEST, PROCESSED_FEATURES, ROOT_PATH
from src.utils.io import read_multi_csv
from src.utils.visualization import show_confusion_matrix

# === Đọc dữ liệu test đã xử lý ===
df_test = read_multi_csv(PROCESSED_OUTPUTS_TEST)
df_test = df_test[PROCESSED_FEATURES]

X_test = df_test.drop('label', axis=1).values
y_test = df_test['label'].values

# === Load mô hình Random Forest đã train ===
model_path = ROOT_PATH + "models/rf/cicddos_rf_model.joblib"
rf_model = joblib.load(model_path)

# === Dự đoán ===
y_pred = rf_model.predict(X_test)

# === Đánh giá trực quan & in báo cáo ===
show_confusion_matrix(y_test, y_pred, class_names=["Benign", "Attack"])
