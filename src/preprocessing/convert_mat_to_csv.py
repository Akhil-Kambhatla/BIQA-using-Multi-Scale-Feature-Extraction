import scipy.io
import pandas as pd
import os

# Define file paths
mat_files = {
    "scores1.mat": r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\data\mat_files\scores1.mat",
    "scores2.mat": r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\data\mat_files\scores2.mat",
}
output_csv = r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\data\scores\processed_mat_scores.csv"

# Initialize an empty DataFrame
df_list = []

for file_name, file_path in mat_files.items():
    print(f"🔍 Processing {file_name}...")

    # Load the .mat file
    mat_data = scipy.io.loadmat(file_path)

    # Extract variables
    images = [f"img{i+1}.bmp" for i in range(mat_data["mmt"].shape[1])]  # Generate image names
    mmt = mat_data["mmt"].flatten()  # Mean opinion scores
    mst = mat_data["mst"].flatten()  # Standard deviation
    br = mat_data["br"].flatten()  # Bit rates

    # Create DataFrame
    df = pd.DataFrame({"Image": images, "MOS": mmt, "StdDev": mst, "BitRate": br})
    df_list.append(df)

# Combine both dataframes
df_final = pd.concat(df_list, ignore_index=True)

# Save to CSV
df_final.to_csv(output_csv, index=False)
print(f"✅ Processed .mat scores saved to {output_csv}")
