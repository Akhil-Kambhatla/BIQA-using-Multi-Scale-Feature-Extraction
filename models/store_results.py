import pandas as pd
import os

# Define results file path
results_csv = r"C:\Users\akhil\OneDrive\Desktop\BIQA_LIVE\results\model_results.csv"

def save_results(model_name, num_epochs, batch_size, learning_rate, mae, pearson_corr):
    """ Saves model performance metrics to a CSV file. """
    new_entry = pd.DataFrame([{
        "Model": model_name,
        "Epochs": num_epochs,
        "Batch Size": batch_size,
        "Learning Rate": learning_rate,
        "MAE": mae,
        "Pearson Correlation": pearson_corr
    }])

    # Check if file exists and is not empty
    if not os.path.exists(results_csv) or os.stat(results_csv).st_size == 0:
        new_entry.to_csv(results_csv, index=False)
    else:
        try:
            df = pd.read_csv(results_csv)
            df = pd.concat([df, new_entry], ignore_index=True)  # Use concat instead of append
            df.to_csv(results_csv, index=False)
        except pd.errors.EmptyDataError:
            new_entry.to_csv(results_csv, index=False)

    print(f"✅ Results saved to {results_csv}")
