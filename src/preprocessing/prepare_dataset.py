import pandas as pd
import os

# Define file paths
features_csv = r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\data\scores\features.csv"
final_scores_csv = r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\data\scores\final_scores.csv"
output_csv = r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\data\scores\training_data.csv"

# Load features and final scores
df_features = pd.read_csv(features_csv)
df_scores = pd.read_csv(final_scores_csv)

# Merge datasets on "Image" column
df_final = pd.merge(df_features, df_scores[['Image', 'MOS_subject']], on="Image", how="inner")

# Rename MOS column for clarity
df_final.rename(columns={'MOS_subject': 'MOS'}, inplace=True)

# Save the final dataset
df_final.to_csv(output_csv, index=False)

print(f"✅ Training dataset saved to {output_csv}")
