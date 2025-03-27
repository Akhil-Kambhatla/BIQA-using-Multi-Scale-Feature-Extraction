import torch
import torch.nn as nn
import numpy as np
import cv2
import os

# Path to model and test image
MODEL_PATH = r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\models\CNN_150_64_0.0003.pth"
TEST_IMAGE_PATH = r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\data\images\img1.bmp"

# -------- Feature Extraction (inline from extract_features.py) --------
def laplacian_pyramid(image, levels=3):
    pyramid = []
    temp = image.copy()
    
    for _ in range(levels):
        down = cv2.pyrDown(temp)
        up = cv2.pyrUp(down, dstsize=(temp.shape[1], temp.shape[0]))
        laplacian = cv2.subtract(temp, up)
        pyramid.append(laplacian)
        temp = down

    return pyramid

def extract_features_from_image_path(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"❌ Could not load image: {image_path}")

    image = cv2.resize(image, (256, 256))
    pyramid = laplacian_pyramid(image, levels=3)

    features = []
    for level in pyramid:
        sharpness = np.var(level)
        contrast = np.std(level)
        edges = np.mean(cv2.Canny(level, 50, 150))
        features.extend([sharpness, contrast, edges])

    return np.array(features, dtype=np.float32).reshape(1, -1)

# -------- CNN Model Definition --------
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
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x

# -------- Predict Score --------
def predict_mos(image_path):
    features = extract_features_from_image_path(image_path)
    input_dim = features.shape[1]

    model = SimpleCNN(input_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        input_tensor = torch.tensor(features)
        pred = model(input_tensor).item()
        mos = pred * 100  # Unnormalize
        return mos

# -------- Run Prediction --------
if __name__ == "__main__":
    predicted_mos = predict_mos(TEST_IMAGE_PATH)
    print(f"\n📸 Predicted Quality Score (MOS): {predicted_mos:.2f} / 100")
