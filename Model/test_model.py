import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

window_size = 5

# Get the current directory of the script to load the data
current_dir = os.path.dirname(os.path.realpath(__file__))

model = tf.keras.models.load_model("lstm", compile=False)
scaler = joblib.load("scaler.pkl")

# Load CSV data into a DataFrame from the data folder
file_path = os.path.join(current_dir, "../data", "system_metrics_test_data.csv")
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print("File loaded successfully.")
else:
    print(f"File not found at: {file_path}")
    exit(1)  # Exit the script if the file is not found

# # Handle Timestamps
# df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Drop PID of 0
df = df.drop(df[df["Process ID"] == 0].index)

# Normalize numerical columns
df[["CPU Usage (%)", "Memory Usage (%)", "Disk Usage (%)"]] = scaler.transform(
    df[["CPU Usage (%)", "Memory Usage (%)", "Disk Usage (%)"]]
)

unique_pids = df["Process ID"].unique()
X, y = [], []
for pid in unique_pids:
    pid_data = df[df["Process ID"] == pid][
        ["CPU Usage (%)", "Memory Usage (%)", "Disk Usage (%)", "Anomaly", "Timestamp"]
    ]
    pid_data = pid_data.sort_values(by="Timestamp")
    pid_data = pid_data.drop("Timestamp", axis=1)  # COmment if Timestamp is required
    X.append(pid_data.drop("Anomaly", axis=1).values)
    # y.append(pid_data['Anomaly'].values)
    anomaly = pid_data["Anomaly"].values[-1]
    y.append(np.array([anomaly]))

X = pad_sequences(
    X, maxlen=window_size, dtype="float32", padding="post", truncating="post", value=0.0
)
y_true = np.array(y)

y_pred = np.round(model.predict(X))

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

# Calculate F1-score
f1 = f1_score(y_true, y_pred)
print("F1-score:", f1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", conf_matrix)

# Calculate precision
precision = precision_score(y_true, y_pred)
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_true, y_pred)
print("Recall:", recall)
