import pandas as pd
import numpy as np
import os

# Define file paths (Update according to your directory structure)
score_files = {
    "subjectscores1.txt": r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\data\raw_subject_scores\subjectscores1.txt",
    "subjectscores2.txt": r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\data\raw_subject_scores\subjectscores2.txt"
}
output_csv = r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\data\scores\processed_subject_scores.csv"

# Outlier removal threshold (same as MATLAB: 2.5 standard deviations)
SCORE_DEVIATION = 2.5

def read_subjective_scores(file_path):
    """ Reads the raw subjective scores from a text file and returns a dictionary. """
    scores_dict = {}

    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            img_name = parts[0]  # Extract image filename
            scores = list(map(int, parts[1:]))  # Convert scores to integers
            scores_dict[img_name] = scores

    return scores_dict

def remove_outliers(scores_dict):
    """ Removes outliers based on the standard deviation method. """
    images = list(scores_dict.keys())
    scores_matrix = np.array(list(scores_dict.values()))

    # Compute mean and standard deviation per image
    mean_scores = np.mean(scores_matrix, axis=1)
    std_scores = np.std(scores_matrix, axis=1)

    # Identify outliers (values beyond 2.5 standard deviations)
    valid_mask = np.abs((scores_matrix - mean_scores[:, None]) / std_scores[:, None]) < SCORE_DEVIATION

    # Remove subjects with >3 bad rankings
    valid_subjects = np.sum(~valid_mask, axis=0) < 3
    filtered_scores = scores_matrix[:, valid_subjects]

    return images, filtered_scores

def convert_to_z_scores(filtered_scores):
    """ Converts scores to Z-scores and rescales to range 1-100. """
    mean_per_subject = np.mean(filtered_scores, axis=0, keepdims=True)
    std_per_subject = np.std(filtered_scores, axis=0, keepdims=True)

    # Normalize using Z-score transformation
    z_scores = (filtered_scores - mean_per_subject) / std_per_subject

    # Rescale to 1-100
    min_val, max_val = np.min(z_scores), np.max(z_scores)
    scaled_scores = (z_scores - min_val) / (max_val - min_val) * 99 + 1

    return np.nan_to_num(scaled_scores)  # Replace NaNs with 0

# Process both score files
final_scores = {}
for file_name, score_file in score_files.items():
    print(f"🔍 Processing {file_name}...")

    scores_dict = read_subjective_scores(score_file)
    images, filtered_scores = remove_outliers(scores_dict)
    scaled_scores = convert_to_z_scores(filtered_scores)

    # Compute Mean Opinion Score (MOS) and standard deviation
    mean_mos = np.mean(scaled_scores, axis=1)
    std_dev = np.std(scaled_scores, axis=1)

    # Store results in dictionary
    for i, img in enumerate(images):
        final_scores[img] = [mean_mos[i], std_dev[i]]

# Convert to DataFrame
df_scores = pd.DataFrame.from_dict(final_scores, orient="index", columns=["MOS", "StdDev"])
df_scores.reset_index(inplace=True)
df_scores.rename(columns={"index": "Image"}, inplace=True)

# Save processed scores to CSV
df_scores.to_csv(output_csv, index=False)
print(f"✅ Processed subjective scores saved to {output_csv}")
