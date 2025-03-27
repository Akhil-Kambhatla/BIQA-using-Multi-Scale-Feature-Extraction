import tkinter as tk
from tkinter import filedialog, Label, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import numpy as np
import cv2
import os

# === Global model instance ===
model = None

# === CNN Class Matching Your Training Model ===
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

# === Feature Extraction ===
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
    image = cv2.resize(image, (256, 256))
    pyramid = laplacian_pyramid(image, levels=3)

    features = []
    for level in pyramid:
        sharpness = np.var(level)
        contrast = np.std(level)
        edges = np.mean(cv2.Canny(level, 50, 150))
        features.extend([sharpness, contrast, edges])

    return np.array(features, dtype=np.float32).reshape(1, -1)

# === Load Model ===
def load_model_from_path(model_path):
    input_dim = 9
    m = SimpleCNN(input_dim)
    m.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    m.eval()
    return m

# === Predict MOS ===
def predict_mos(image_path):
    global model
    if model is None:
        messagebox.showerror("Error", "Please select a model before predicting.")
        return None

    features = extract_features_from_image_path(image_path)
    with torch.no_grad():
        input_tensor = torch.tensor(features)
        prediction = model(input_tensor).item()
        return prediction * 100

# === Browse for Model File ===
def select_model():
    global model
    model_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("PyTorch Model", "*.pth")])
    if model_path:
        try:
            model = load_model_from_path(model_path)
            model_label.config(text=f"✅ Model Loaded:\n{os.path.basename(model_path)}", fg="green")
        except Exception as e:
            messagebox.showerror("Model Load Error", str(e))
            model = None

# === Browse and Predict Image ===
def browse_image():
    if model is None:
        messagebox.showwarning("Model Required", "Please select a model first.")
        return

    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.bmp *.png")])
    if filepath:
        image = Image.open(filepath)
        image.thumbnail((250, 250))
        photo = ImageTk.PhotoImage(image)

        image_label.config(image=photo)
        image_label.image = photo

        score = predict_mos(filepath)
        if score is not None:
            result_label.config(text=f"📊 Predicted MOS: {score:.2f} / 100")

# === Setup UI ===
root = tk.Tk()
root.title("BIQA - Image Quality Estimator")
root.geometry("430x500")
root.resizable(False, False)

title_label = Label(root, text="📷 Blind Image Quality Assessment (CNN)", font=("Arial", 14))
title_label.pack(pady=10)

model_button = tk.Button(root, text="📂 Select Model", command=select_model, font=("Arial", 11))
model_button.pack(pady=5)

model_label = Label(root, text="❌ No model loaded", font=("Arial", 10), fg="red")
model_label.pack(pady=5)

image_button = tk.Button(root, text="🖼️ Select Image", command=browse_image, font=("Arial", 11))
image_button.pack(pady=10)

image_label = Label(root)
image_label.pack()

result_label = Label(root, text="", font=("Arial", 13), fg="blue")
result_label.pack(pady=20)

root.mainloop()
