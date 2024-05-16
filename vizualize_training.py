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

# Plot the training and validation loss
plt.plot(history_df["accuracy"], label="Training Accuracy")
plt.plot(history_df["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Validation Accuracy")
plt.show()

# Plot the training and validation loss
plt.plot(history_df["f1"], label="Training Accuracy")
plt.plot(history_df["val_f1"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("F1")
plt.legend()
plt.title("Training and Validation F1")
plt.show()



#loss tried to converge to 0 
# There is a steep increase in accuracy and it is consistent after some epochs I used early stopping to stop the training and get a more accurate model but the loss does not converge to 0

 