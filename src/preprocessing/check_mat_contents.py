import scipy.io
import os

# Define paths to .mat files
mat_files = ["C:\\Users\\akhil\\OneDrive\\Desktop\\BIQA_LIVE\\data\\mat_files\\scores1.mat", 
             "C:\\Users\\akhil\\OneDrive\\Desktop\\BIQA_LIVE\\data\\mat_files\\scores2.mat"]


for file_path in mat_files:
    if os.path.exists(file_path):
        print(f"🔍 Checking contents of {file_path}...\n")
        mat_data = scipy.io.loadmat(file_path)

        # Print all variables inside the .mat file
        for key in mat_data.keys():
            if not key.startswith("__"):  # Ignore internal MATLAB keys
                print(f"📂 Variable: {key} - Type: {type(mat_data[key])} - Shape: {mat_data[key].shape}")

        print("\n" + "="*50 + "\n")
    else:
        print(f"⚠️ File {file_path} not found. Please check the path.")
