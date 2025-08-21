# voice_emotion_disorder_prediction.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# -----------------------------
# 1. Load Synthetic Voice Data
# -----------------------------
data = pd.read_csv("synthetic_voice_data.csv")

X = data.drop("label", axis=1)
y = data["label"]

# -----------------------------
# 2. Preprocess Data
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -----------------------------
# 3. Random Forest Model
# -----------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("Random Forest Results:")
print(classification_report(y_test, rf_preds))
print("Accuracy:", accuracy_score(y_test, rf_preds))

# -----------------------------
# 4. XGBoost Model
# -----------------------------
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

print("\nXGBoost Results:")
print(classification_report(y_test, xgb_preds))
print("Accuracy:", accuracy_score(y_test, xgb_preds))

# -----------------------------
# 5. LSTM Model (Deep Learning)
# -----------------------------
X_train_lstm = np.expand_dims(X_train, axis=1)
X_test_lstm = np.expand_dims(X_test, axis=1)

y_train_cat = to_categorical(y_train, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)

lstm = Sequential([
    LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), activation="tanh"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(2, activation="softmax")
])

lstm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = lstm.fit(X_train_lstm, y_train_cat, epochs=20, batch_size=16, validation_data=(X_test_lstm, y_test_cat), verbose=0)

lstm_loss, lstm_acc = lstm.evaluate(X_test_lstm, y_test_cat, verbose=0)
print("\nLSTM Accuracy:", lstm_acc)

# -----------------------------
# 6. Confusion Matrix Plot
# -----------------------------
cm = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Healthy", "Disorder"], yticklabels=["Healthy", "Disorder"])
plt.title("Random Forest Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()
