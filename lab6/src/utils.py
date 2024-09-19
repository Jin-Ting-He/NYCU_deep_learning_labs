# Correcting the issue with the plot
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Define the directory containing the files
directory = "src/figure/DDPM"  # Update with the correct path if different

# Define a regular expression pattern to extract dataset, epoch, and accuracy
pattern = r"test_(\d+)_(\d+\.\d+)\.png"

# Initialize a list to store the extracted information
data = []

# Loop through the files in the directory
for filename in os.listdir(directory):
    match = re.match(pattern, filename)
    if match:
        dataset, accuracy = match.groups()
        epoch = int(dataset)
        accuracy = float(accuracy)
        data.append((epoch, accuracy))

# Create a DataFrame from the extracted data
df = pd.DataFrame(data, columns=["Epoch", "Accuracy"])

# Sort the DataFrame by epoch
df = df.sort_values(by="Epoch")

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df["Epoch"], df["Accuracy"], linestyle='-', color='b')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy per Epoch")
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.savefig("ddpm_validation_loss_curve_test.png")

# import ace_tools as tools; tools.display_dataframe_to_user(name="Epoch and Accuracy Data", dataframe=df)
