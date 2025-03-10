import pandas as pd
import os

# Define file paths
mat_scores_path = r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\data\scores\processed_mat_scores.csv"
subject_scores_path = r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\data\scores\processed_subject_scores.csv"
output_csv = r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\data\scores\final_scores.csv"

# Load processed .mat scores
df_mat = pd.read_csv(mat_scores_path)

# Load processed subjective scores
df_subject = pd.read_csv(subject_scores_path)

# Merge on 'Image' column
df_final = pd.merge(df_mat, df_subject, on="Image", how="inner", suffixes=("_mat", "_subject"))

# Save the final merged dataset
df_final.to_csv(output_csv, index=False)

print(f"✅ Final merged dataset saved to {output_csv}")
