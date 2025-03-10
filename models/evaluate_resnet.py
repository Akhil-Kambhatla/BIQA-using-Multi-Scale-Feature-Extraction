import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
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


# Define the ResNetFeatureExtractor class (same as training)
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, input_dim, resnet_version=18):
        super(ResNetFeatureExtractor, self).__init__()

        # Load Pretrained ResNet Model
        if resnet_version == 18:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif resnet_version == 50:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            raise ValueError("Unsupported ResNet version. Use 18 or 50.")

        # Replace the first convolution layer to accept feature vectors
        self.resnet.conv1 = nn.Linear(input_dim, 512)  # Map feature input to CNN layer

        # Replace the final layer to predict MOS
        self.resnet.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.resnet.conv1(x)  # Pass through modified first layer
        x = self.resnet.relu(x)   # Apply activation
        x = self.resnet.fc(x)     # Predict MOS
        return x


# Evaluate all trained models
for model_file in os.listdir(models_dir):
    if model_file.startswith("ResNet") and model_file.endswith(".pth"):
        model_path = os.path.join(models_dir, model_file)

        # Extract model parameters correctly
        try:
            parts = model_file.replace(".pth", "").split("_")
            resnet_version = int(parts[0].replace("ResNet", ""))  # Extract ResNet version
            epochs, batch_size, learning_rate = int(parts[1]), int(parts[2]), float(parts[3])
        except ValueError:
            print(f"⚠️ Skipping model {model_file} due to incorrect filename format.")
            continue  # Skip this model if the filename does not match the expected format

        # Initialize Model (using the correct class)
        input_dim = X.shape[1]
        model = ResNetFeatureExtractor(input_dim, resnet_version)

        # Load trained model
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

        model_name = f"ResNet{resnet_version}_{epochs}_{batch_size}_{learning_rate}"

        # Save results
        save_results(model_name, epochs, batch_size, learning_rate, mae, pearson_corr)

        # Save plot
        plot_path = os.path.join(plots_folder, f"{model_name}.png")
        plt.scatter(actual, predicted, alpha=0.7)
        plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r', linestyle="--")
        plt.xlabel("Actual MOS")
        plt.ylabel("Predicted MOS")
        plt.title(f"{model_name} (Pearson Corr: {pearson_corr:.2f})")
        plt.grid()
        plt.savefig(plot_path)
        plt.close()

        print(f"✅ Model {model_name} evaluated & results saved")
