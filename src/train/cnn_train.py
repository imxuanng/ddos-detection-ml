import pandas as pd
import numpy as np
import joblib
from typing import List, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras. models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow. keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.constant import (
    PROCESSED_FEATURES,
    PROCESSED_OUTPUTS_TRAIN,
    ROOT_PATH
)
from src.utils.io import (
    read_multi_csv,    
)


df_train = read_multi_csv(PROCESSED_OUTPUTS_TRAIN)
df_train = df_train[PROCESSED_FEATURES]

# ==== Tách ra X, y và reshape cho CNN ====
X_train = df_train.drop('label', axis=1).values
y_train = df_train['label']. values

# Reshape cho CNN 1D:  (samples, timesteps, features)
# Với CNN, có thể giữ shape tương tự LSTM hoặc điều chỉnh
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

for i in range(3):
    print(f"X_train_cnn sample {i}: {X_train_cnn[i]}")
    print(f"y_train sample {i}: {y_train[i]}")

# ==== Xây dựng và train CNN ====
model = Sequential()

# Conv Block 1
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', 
                 input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2])))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# Conv Block 2
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# Conv Block 3 (optional - có thể bỏ nếu số features ít)
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Flatten và Dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-7)

# Model summary
model.summary()

# Training
history = model.fit(
    X_train_cnn, y_train,
    epochs=50,
    batch_size=256,
    callbacks=[es, reduce_lr],
    verbose=2
)

# ==== Lưu model ====
model.save(ROOT_PATH + 'models/cnn/cicddos_cnn_model.h5')
joblib.dump(PROCESSED_FEATURES, ROOT_PATH + 'models/cnn/cicddos_features_cnn.joblib')
print("Training completed and CNN model is saved. Ready for testing phase after this.")
