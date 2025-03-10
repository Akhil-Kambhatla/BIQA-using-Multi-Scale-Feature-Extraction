import pandas as pd
import os

# Define results file path
results_csv = r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\results\model_results.csv"

def save_results(model_name, num_epochs, batch_size, learning_rate, mae, pearson_corr):
    """ Saves model performance metrics to a CSV file in a structured format. """
    
    # Create a new entry with all fields
    new_entry = pd.DataFrame([{
        "Model": model_name,
        "Epochs": int(num_epochs),
        "Batch Size": int(batch_size),
        "Learning Rate": float(learning_rate),
        "MAE": float(mae),
        "Pearson Correlation": float(pearson_corr)
    }])

    # Check if the file exists and has valid content
    if not os.path.exists(results_csv) or os.stat(results_csv).st_size == 0:
        new_entry.to_csv(results_csv, index=False)
    else:
        try:
            df = pd.read_csv(results_csv)

            # Ensure column order consistency
            expected_columns = ["Model", "Epochs", "Batch Size", "Learning Rate", "MAE", "Pearson Correlation"]
            df = df[expected_columns] if all(col in df.columns for col in expected_columns) else df

            # Append new entry and save
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv(results_csv, index=False)

        except pd.errors.EmptyDataError:
            new_entry.to_csv(results_csv, index=False)

    print(f"✅ Results saved to {results_csv}")
