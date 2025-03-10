import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
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


def train_model(num_epochs, batch_size, learning_rate):
    """ Train a CNN model with specified hyperparameters. """
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
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

    # Initialize model
    input_dim = X.shape[1]
    model = BIQA_CNN(input_dim)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

    # Save trained model
    model_name = f"CNN_{num_epochs}_{batch_size}_{learning_rate}.pth"
    model_save_path = os.path.join(models_dir, model_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model trained and saved at {model_save_path}")

    return model_name


# Train models with different hyperparameters
hyperparameter_sets = [
    (100, 32, 0.0005),  # (epochs, batch_size, learning_rate)
    (150, 64, 0.0003),
    (200, 16, 0.001)
]

for params in hyperparameter_sets:
    train_model(*params)
