import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Define your F1-score function
def f1(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)

    true_positives = tf.reduce_sum(y_true * y_pred, axis=0)
    predicted_positives = tf.reduce_sum(y_pred, axis=0)
    possible_positives = tf.reduce_sum(y_true, axis=0)

    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return tf.reduce_mean(f1)


# Get the current directory of the script to load the data
current_dir = os.path.dirname(os.path.realpath(__file__))

# Load CSV data into a DataFrame from the data folder
file_path = os.path.join(current_dir, "../data", "system_metrics_train_data.csv")
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
    # y.append(pid_data['Anomaly'].values)
    anomaly = pid_data["Anomaly"].values[-1]
    y.append(np.array([anomaly]))

window_size = 0
for v in X:
    l = len(v)
    if l > window_size:
        window_size = l

X = pad_sequences(
    X, maxlen=window_size, dtype="float32", padding="post", truncating="post", value=0.0
)
y = np.array(y)
# y = pad_sequences(y, maxlen=window_size, dtype='int', padding='post', truncating='post', value=0)

print("Anomaly count")
print(np.unique(y, return_counts=True))


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)


print("Shapes")
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

# Define the NN model
model = tf.keras.Sequential(
    [
        tf.keras.layers.LSTM(128, input_shape=(window_size, X_train.shape[2])),
        # tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(32, activation="tanh"),
        # tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(y_train.shape[1], activation="sigmoid"),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", f1])

# Train the model# Train the model and capture the training history
history = model.fit(
    X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32
)

# Save the training history to a CSV file
history_df = pd.DataFrame(history.history)
history_df.to_csv(
    os.path.join(current_dir, "../data", "training_history.csv"), index=False
)

model.save("lstm")
joblib.dump(scaler, "scaler.pkl")
print("Window Size:", window_size)

print(
    "########################################################################################################################"
)
print(
    "################################################# Metric for Train Data ################################################"
)
print(
    "########################################################################################################################"
)

y_true = y
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

print(
    "########################################################################################################################"
)
print(
    "################################################# Metric for Test Data #################################################"
)
print(
    "########################################################################################################################"
)

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
