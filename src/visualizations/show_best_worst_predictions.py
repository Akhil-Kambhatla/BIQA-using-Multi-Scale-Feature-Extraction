import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# === Load Data ===
df = pd.read_csv(r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\data\scores\training_data.csv")
df["MOS"] = df["MOS"] / 100.0

X = df.drop(columns=["Image", "MOS"]).values.astype(np.float32)
y = df["MOS"].values.astype(np.float32)
image_names = df["Image"].values

# === Define CNN ===
class SimpleCNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# === Prepare tensors ===
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y).view(-1, 1)
dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
_, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
test_indices = test_dataset.indices

# === Load model ===
model = SimpleCNN(input_dim=9)
model.load_state_dict(torch.load(
    r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\models\CNN_150_64_0.0003.pth",
    map_location='cpu'
))
model.eval()

# === Predict ===
with torch.no_grad():
    y_pred = model(X_tensor[test_indices]).squeeze().numpy() * 100
    y_true = y_tensor[test_indices].squeeze().numpy() * 100
    errors = np.abs(y_true - y_pred)
    image_test = image_names[test_indices]

# === Pick best & worst 3 ===
best_idxs = np.argsort(errors)[:3]
worst_idxs = np.argsort(errors)[-3:]

# === Show predictions ===
def plot_images(idxs, title, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, idx in enumerate(idxs):
        img_path = os.path.join(
            r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\data\images",
            image_test[idx]
        )
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
            axes[i].set_title(f"Pred: {y_pred[idx]:.1f} / True: {y_true[idx]:.1f}")
        else:
            axes[i].text(0.5, 0.5, "Image not found", ha="center")
        axes[i].axis("off")
    
    fig.suptitle(title)
    plt.tight_layout()
    
    # ✅ Ensure the save folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.show()

# === Save plots ===
plot_images(
    best_idxs,
    "Best Predictions (Lowest Error)",
    r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\results\plots\best_predictions.png"
)

plot_images(
    worst_idxs,
    "Worst Predictions (Highest Error)",
    r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\results\plots\worst_predictions.png"
)
