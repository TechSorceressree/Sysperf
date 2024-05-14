import os
import pandas as pd
import matplotlib.pyplot as plt

# Get the current directory of the script
current_dir = os.path.dirname(os.path.realpath(__file__))

# Load the training history from the CSV file in the Data folder
history_df = pd.read_csv(os.path.join(current_dir, "Data", "training_history.csv"))

# Plot the training and validation loss
plt.plot(history_df["loss"], label="Training Loss")
plt.plot(history_df["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()
