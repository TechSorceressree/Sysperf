import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Get the current directory of the script to load the data
current_dir = os.path.dirname(os.path.realpath(__file__))

# Load CSV data into a DataFrame from the data folder
file_path = os.path.join(current_dir, "../data", "system_metrics.csv")
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
scaler = MinMaxScaler()
df[["CPU Usage (%)", "Memory Usage (%)", "Disk Usage (%)"]] = scaler.fit_transform(
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
    anomaly = pid_data["Anomaly"].values[-1]
    y.append(pd.DataFrame({"Anomaly": [anomaly]}))

window_size = 0
for v in X:
    l = len(v)
    if l > window_size:
        window_size = l

X = pad_sequences(
    X, maxlen=window_size, dtype="float32", padding="post", truncating="post", value=0.0
)
# y = pad_sequences(y, maxlen=window_size, dtype='int', padding='post', truncating='post', value=0)


# # Function to create input sequences and labels

# def create_sequences(data, window_size):
#     X, y = [], []
#     for i in range(len(data) - window_size):
#         X.append(data[i : i + window_size])
#         y.append(data[i + window_size])
#     return np.array(X), np.array(y)


# # Create input sequences and labels
# X, y = create_sequences(
#     df[["CPU Usage (%)", "Memory Usage (%)", "Disk Usage (%)"]].values, window_size
# )

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# print("Shapes")
# print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
# Define the LSTM model
model = tf.keras.Sequential(
    [
        tf.keras.layers.LSTM(64, input_shape=(window_size, X_train.shape[2])),
        tf.keras.layers.Dense(X_train.shape[2]),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy")

# Train the model# Train the model and capture the training history
history = model.fit(
    X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32
)

# Save the training history to a CSV file
history_df = pd.DataFrame(history.history)
history_df.to_csv(
    os.path.join(current_dir, "data", "training_history.csv"), index=False
)
