import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.models as models
import os

# Define file paths
dataset_csv = r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\data\scores\training_data.csv"
models_dir = r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\models"

# Ensure models directory exists
os.makedirs(models_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(dataset_csv)
df["MOS"] = df["MOS"] / 100.0  # Normalize MOS

X = df.drop(columns=["Image", "MOS"]).values  # Features
y = df["MOS"].values  # Labels

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Split into train/test sets
train_size = int(0.8 * len(X))
test_size = len(X) - train_size
dataset = TensorDataset(X_tensor, y_tensor)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


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


def train_resnet(num_epochs, batch_size, learning_rate, resnet_version=18):
    """ Train a ResNet model for BIQA. """

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model
    input_dim = X.shape[1]
    model = ResNetFeatureExtractor(input_dim, resnet_version)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

    # Save trained model
    model_name = f"ResNet{resnet_version}_{num_epochs}_{batch_size}_{learning_rate}.pth"
    model_save_path = os.path.join(models_dir, model_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model trained and saved at {model_save_path}")

    return model_name


# Train models with different hyperparameters
hyperparameter_sets = [
    (100, 32, 0.0005, 18),  # (epochs, batch_size, learning_rate, resnet_version)
    (150, 64, 0.0003, 50)
]

for params in hyperparameter_sets:
    train_resnet(*params)
