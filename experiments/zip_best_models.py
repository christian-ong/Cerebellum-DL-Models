import os
import pandas as pd
import zipfile
import glob

def zip_best_models(source_csv_dir, output_zip="best_models_backup.zip"):
    # 1. Find all best_runs.csv files
    csv_files = glob.glob(os.path.join(source_csv_dir, "*", "best_runs.csv"))
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            
            # Sort by your metric (e.g., best_val_rollout_rmse_h100)
            # Adjust 'best_val_rollout_rmse_h100' if you use a different column
            df = df.sort_values(by="best_val_rollout_rmse_h100")
            
            # Keep top 5
            top_5 = df.head(5)
            
            for _, row in top_5.iterrows():
                # Reconstruct the path to the model folder
                # Adjust this path logic based on your actual file structure
                run_id = row.get("run_id", row.get("run_name"))
                model_name = row["model_name"]
                system = row["system_name"]
                
                model_folder = os.path.join("data", "models", model_name, system, run_id)
                
                if os.path.exists(model_folder):
                    print(f"Adding to zip: {model_folder}")
                    # Walk through the directory and add files
                    for root, _, files in os.walk(model_folder):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zipf.write(file_path)
                else:
                    print(f"Warning: Folder not found: {model_folder}")

    print(f"Successfully created {output_zip} with top models.")

if __name__ == "__main__":
    # Point this to the folder where extract_wandb.py saved the best_runs.csv files
    zip_best_models("experiments/wandb")