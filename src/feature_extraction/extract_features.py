import cv2
import numpy as np
import pandas as pd
import os
from multiprocessing import Pool, cpu_count

# Define file paths
image_folder = r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\data\images"
final_scores_csv = r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\data\scores\final_scores.csv"
output_features_csv = r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\data\scores\features.csv"

# Number of CPU processes to use
NUM_PROCESSES = cpu_count()  

def laplacian_pyramid(image, levels=3):
    """ Compute Laplacian Pyramid features with 3 levels. """
    pyramid = []
    temp = image.copy()
    
    for _ in range(levels):
        down = cv2.pyrDown(temp)  # Downsample
        up = cv2.pyrUp(down, dstsize=(temp.shape[1], temp.shape[0]))  # Upsample
        laplacian = cv2.subtract(temp, up)  # Compute Laplacian
        pyramid.append(laplacian)
        temp = down  # Move to the next level

    return pyramid

def extract_features_for_image(img_name):
    """ Extract Laplacian Pyramid features for a single image. """
    img_path = os.path.join(image_folder, img_name)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"⚠️ Warning: Image {img_name} not found, skipping...")
        return None  # Skip missing images

    # Resize image to speed up processing (256x256 pixels)
    image = cv2.resize(image, (256, 256))
    
    # Compute Laplacian Pyramid features
    pyramid = laplacian_pyramid(image, levels=3)  # Now using 3 levels

    # Extract sharpness, contrast, and edge intensity at each level
    features = []
    for level in pyramid:
        sharpness = np.var(level)  # Variance for sharpness
        contrast = np.std(level)  # Standard deviation for contrast
        edges = np.mean(cv2.Canny(level, 50, 150))  # Edge intensity
        features.extend([sharpness, contrast, edges])

    return [img_name] + features

if __name__ == "__main__":
    # Load list of images from final_scores.csv
    df_scores = pd.read_csv(final_scores_csv)
    image_list = df_scores["Image"].tolist()

    # Define feature column names
    columns = ["Image"]
    for i in range(3):  # Now extracting 3 pyramid levels
        columns.extend([f"Sharpness_L{i+1}", f"Contrast_L{i+1}", f"Edges_L{i+1}"])

    # Run feature extraction in parallel (Fix for Windows)
    print(f"🔄 Extracting features using {NUM_PROCESSES} CPU cores...")
    with Pool(NUM_PROCESSES) as pool:
        feature_data = pool.map(extract_features_for_image, image_list)

    # Remove None values (failed images)
    feature_data = [f for f in feature_data if f]

    # Convert to DataFrame & Save
    df_features = pd.DataFrame(feature_data, columns=columns)
    df_features.to_csv(output_features_csv, index=False)

    print(f"✅ Feature extraction completed! Results saved to {output_features_csv}")
