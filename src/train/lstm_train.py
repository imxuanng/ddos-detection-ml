
import os
import pandas as pd
import numpy as np
import joblib
from typing import List, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
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

# ==== Tách ra X, y và reshape cho LSTM ====
X_train = df_train.drop('label', axis=1).values
y_train = df_train['label'].values
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

for i in range(3):
    print(f"X_train_lstm sample {i}: {X_train_lstm[i]}")
    print(f"y_train sample {i}: {y_train[i]}")

# ==== Xây dựng và train LSTM ====
model = Sequential()
model.add(LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train_lstm, y_train,
    epochs=20,
    batch_size=256,
    callbacks=[es],
    verbose=2
)


# ==== Lưu model ====
model.save(ROOT_PATH +'models/lstm/cicddos_lstm_model.h5')
joblib.dump(PROCESSED_FEATURES, 'cicddos_features.joblib')
print("Training completed and model is saved. Ready for testing phase after this.")
