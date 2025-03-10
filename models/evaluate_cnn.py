import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
from store_results import save_results  # Import result-saving function

# Define file paths
dataset_csv = r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\data\scores\training_data.csv"
models_dir = r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\models"
plots_folder = r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\results\plots"

# Ensure plots folder exists
os.makedirs(plots_folder, exist_ok=True)

# Load dataset
df = pd.read_csv(dataset_csv)
df["MOS"] = df["MOS"] / 100.0  # Normalize MOS

X = df.drop(columns=["Image", "MOS"]).values  # Features
y = df["MOS"].values  # Labels

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Create test dataset (20% of data)
test_size = int(0.2 * len(X))
test_dataset = TensorDataset(X_tensor[-test_size:], y_tensor[-test_size:])
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define CNN Model
class BIQA_CNN(nn.Module):
    def __init__(self, input_dim):
        super(BIQA_CNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Evaluate all trained models
for model_file in os.listdir(models_dir):
    if model_file.endswith(".pth"):
        model_path = os.path.join(models_dir, model_file)

        # Load trained model
        input_dim = X.shape[1]
        model = BIQA_CNN(input_dim)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Evaluate Model
        actual, predicted = [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                actual.extend(batch_y.numpy().flatten())
                predicted.extend(predictions.numpy().flatten())

        # Reverse normalization
        predicted = np.array(predicted) * 100
        actual = np.array(actual) * 100

        # Compute MAE & Pearson Correlation
        mae = np.mean(np.abs(actual - predicted))
        pearson_corr, _ = pearsonr(actual, predicted)

        # Extract model parameters correctly
        try:
            _, epochs, batch_size, learning_rate = model_file.replace(".pth", "").split("_")
            epochs, batch_size, learning_rate = int(epochs), int(batch_size), float(learning_rate)
        except ValueError:
            print(f"⚠️ Skipping model {model_file} due to incorrect filename format.")
            continue  # Skip this model if the filename does not match the expected format

        model_name = f"CNN_{epochs}_{batch_size}_{learning_rate}"

        # Save results
        save_results(model_name, epochs, batch_size, learning_rate, mae, pearson_corr)

        # Save plot
        plot_path = os.path.join(plots_folder, f"{model_name}.png")

        plt.figure(figsize=(8, 6))
        plt.scatter(actual, predicted, alpha=0.7)
        plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r', linestyle="--")
        plt.xlabel("Actual MOS")
        plt.ylabel("Predicted MOS")
        plt.title(f"{model_name} (Pearson Corr: {pearson_corr:.2f})")
        plt.grid()
        plt.savefig(plot_path)  # Save the plot
        plt.close()

        print(f"✅ Model {model_name} evaluated & results saved")
