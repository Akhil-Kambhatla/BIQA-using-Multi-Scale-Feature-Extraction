# Blind Image Quality Assessment (BIQA) Project

This project is focused on developing a **Blind Image Quality Assessment (BIQA)** system using **Laplacian Pyramid Networks (LPN)** for feature extraction and deep learning models (CNN and ResNet) for prediction. The system predicts the **Mean Opinion Score (MOS)** for an image without needing a reference image.

---

## 🔁 **Overall Project Flow Outline**

### **1. Data Collection & Preprocessing**

**Raw Subjective Scores**:
- Raw subjective scores are collected from `.txt` files.
- The data is processed using the script **`process_subject_scores.py`**, which:
  - Removes outliers using a threshold of ±2.5 standard deviations.
  - Computes Z-scores for normalization.
  - Rescales the scores to a 1-100 scale.
- The processed subjective scores are stored in `processed_subject_scores.csv`.

**Objective Scores from .mat files (LIVE IQA Database)**:
- The objective scores are extracted from `.mat` files using **`convert_mat_to_csv.py`**.
- The script extracts the **Mean MOS** (`mmt`), **Standard Deviation** (`mst`), and **Bitrate** (`br`) for each image.
- The processed objective scores are saved in `processed_mat_scores.csv`.

**Merging Both Scores**:
- The subjective and objective scores are merged using **`merge_scores.py`** to create a final dataset.
- The combined dataset is saved as `final_scores.csv`, which includes both the MOS and the associated features for each image.

---

### **2. Feature Extraction (Multi-Scale using LPN)**

**Technique Used**: 
- We utilize **Laplacian Pyramid Networks (LPN)** with **3 levels** of decomposition for each image.

For each image:
- The following features are extracted at each level of the Laplacian Pyramid:
  - **Sharpness** (calculated using variance of pixel intensities).
  - **Contrast** (calculated using standard deviation of pixel intensities).
  - **Edge Intensity** (calculated using the Canny edge detection).
  
Each image has a total of **9 features** (3 features per level: sharpness, contrast, and edge intensity). The features are extracted using **`extract_features.py`** and stored in `features.csv`.

---

### **3. Dataset Preparation**

**Feature + MOS Data**:
- The features from **`features.csv`** are combined with the MOS values from **`final_scores.csv`**.
- This is done using **`prepare_dataset.py`**.
- The final dataset, `training_data.csv`, is prepared for training the models, containing the multi-scale features and their associated MOS values.

---

### **4. Model Training**

#### **a. CNN Model**

- The CNN model is implemented in **`train_cnn.py`**.
- The architecture consists of **4 fully connected layers** (256 → 128 → 64 → 1).
- **Loss Function**: Mean Squared Error (MSE).
- **Optimizer**: Adam optimizer.
- Multiple hyperparameter configurations (epochs, batch size, learning rate) were tested.
- The model is saved as `CNN_{epochs}_{batch_size}_{lr}.pth`.

#### **b. ResNet-based Model**

- The **ResNet18** and **ResNet50** models are implemented in **`train_resnet.py`**.
- Pretrained ResNet18/50 models are fine-tuned:
  - The first convolutional layer is replaced with a linear input layer.
  - The final fully connected layer is customized to output a single MOS value.
- The models are saved as `ResNet{version}_{epochs}_{batch_size}_{lr}.pth`.

---

### **5. Evaluation**

- Evaluation is done using **`evaluate_resnet.py`**.
- The models are tested on a **20% hold-out data**.
- **Metrics Used**:
  - **MAE (Mean Absolute Error)**: Measures the average error between predicted and true MOS values.
  - **PCC (Pearson Correlation Coefficient)**: Measures the correlation between predicted and true MOS values.
- The results are saved in `model_results.csv` and scatter plots are saved in the `results/plots/` directory.

---

### **6. Results Visualization**

- The script **`show_best_worst_predictions.py`** predicts MOS values for the test set using the CNN model.
- The top **best** (lowest error) and **worst** (highest error) predictions are selected.
- These predictions are plotted and saved in `results/plots/`.

---

### **7. UI for Real-Time Prediction**

- The **real-time prediction** UI is built using **Tkinter** in **`app_cnn_ui.py`**.
- The UI allows users to:
  - Load trained models (`.pth` files).
  - Select and upload images (`.jpg/.bmp/.png`).
  - Display the predicted MOS value on the screen.
- The UI uses the same **Laplacian Pyramid-based feature extractor** for preprocessing the image before passing it to the model.

---

### **8. Inference (CLI version)**

- The script **`predict_cnn_score.py`** allows inference from the **Command Line Interface (CLI)**.
- It predicts the MOS score of a single image by loading the CNN model and performing the same feature extraction steps.

---

## Install the dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Project

### **Step 1: Install Dependencies**
- Install the necessary Python packages:
  ```bash
  pip install -r requirements.txt
  ```

### **Step 2: Data Collection and Preprocessing**
- Run the pre-processing steps:
  ```bash
  python process_subject_scores.py
  python convert_mat_to_csv.py
  python merge_scores.py
  ```

### **Step 3: Feature Extraction**
- Run the feature extraction script:
  ```bash
  python extract_features.py
  ```

### **Step 4: Dataset Preparation**
- Prepare the dataset:
  ```bash
  python prepare_dataset.py
  ```

### **Step 5: Model Training**
- Train the CNN model:
  ```bash
  python train_cnn.py
  ```
- Train the ResNet models::
  ```bash
  python train_resnet.py
  ```

### **Step 6: Model Evaluation**
- Evaluate the models:
  ```bash
  python evaluate_cnn.py
  python evaluate_resnet.py
  ```

### **Step 7: Visualizing Predictions**
- Visualize the best and worst predictions:
  ```bash
  python show_best_worst_predictions.py
  ```

### **Step 8: Real-Time Prediction UI**
- Run the Tkinter UI for real-time prediction (developed only for CNN models):
  ```bash
  python app_cnn_ui.py
  ```

### **Step 9: Inference (CLI)**
- Run the inference for a single image via CLI:
  ```bash
  python predict_cnn_score.py
  ```

---

## Conclusion
This Blind Image Quality Assessment (BIQA) system predicts image quality without the need for reference images. The combination of Laplacian Pyramid Networks for feature extraction and deep learning models like CNN and ResNet provides a robust solution for real-world image quality assessment tasks. This project demonstrates the power of multi-scale feature extraction and deep learning in computer vision applications.
