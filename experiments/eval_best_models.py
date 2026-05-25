import argparse
import pandas as pd
import subprocess
import os
from pathlib import Path


MODE_VISUALIZATION_MODELS = {"ml_dmd", "regression_dmd"}


def resolve_model_checkpoint(model_name, system, run_name):
    """Return the first checkpoint path that exists for this sweep row."""
    folder_candidates = [model_name]
    if model_name == "ml_linear_dynamics":
        folder_candidates.append("ml_lineardynamics")
    elif model_name == "ml_lineardynamics":
        folder_candidates.append("ml_linear_dynamics")
    elif model_name == "ml_dmd_free":
        folder_candidates.append("ml_dmd")
    elif model_name == "ml_dmd_band":
        folder_candidates.append("ml_dmd")

    if model_name == "regression_dmd":
        file_candidates = ["model.npz"]
    elif model_name == "hardcoded_dmd":
        file_candidates = ["model.pt", "model_best.pt"]
    else:
        file_candidates = ["model_best.pt", "model.pt", "model.npz"]

    for folder_name in dict.fromkeys(folder_candidates):
        base_dir = Path("data/models") / folder_name / system / run_name
        for file_name in file_candidates:
            candidate = base_dir / file_name
            if candidate.exists():
                return str(candidate)

    return None

def run_evaluations(csv_path, dt_val, target_time, target_metric):
    print(f"\n=========================================================")
    print(f" PROCESSING DATASET: {csv_path} (dt={dt_val})")
    print(f"=========================================================\n")
    
    if not os.path.exists(csv_path):
        print(f"❌ File {csv_path} not found. Skipping.")
        return

    # 1. Load dataframe
    df = pd.read_csv(csv_path)
    
    # 2. Isolate the best combinations
    if target_metric in df.columns:
        best_idx = df.groupby(['system_name', 'model_name', 'expansion_type'])[target_metric].idxmin()
        best_df = df.loc[best_idx.dropna()].reset_index(drop=True)
        print(f"✅ Found {len(best_df)} unique best model configurations based on '{target_metric}'.")
    else:
        print(f"⚠️ Metric '{target_metric}' not found. Evaluating all {len(df)} rows.")
        best_df = df

    # 3. Calculate exactly how many steps to simulate for the given physical time
    num_steps = int(target_time / dt_val)

    # 4. Loop through the best models
    for idx, row in best_df.iterrows():
        run_name = row['run_name']
        model_name = row['model_name']
        system = row['system_name']
        data_path = row['data_path']
        
        print(f"\n[{idx+1}/{len(best_df)}] Evaluating: {system} | {model_name} ({row['expansion_type']}) -> Run: {run_name}")
        
        # Resolve the actual checkpoint file saved for this model family.
        model_path = resolve_model_checkpoint(model_name, system, run_name)
        if model_path is None:
            print(f"    ⚠️ Warning: no checkpoint found for {model_name}/{system}/{run_name}. Skipping.")
            continue

        supports_mode_visualization = model_name in MODE_VISUALIZATION_MODELS
        
        # The 3 evaluation commands
        commands = [
            # Script 1: Trajectory Rollout Plotter
            [
                "python", "-m", "experiments.eval_trajectory_rollout",
                "--model_name", model_name,
                "--custom_name", run_name,
                "--data_path", data_path,
                "--num_steps", str(num_steps)
            ],
            # Script 2: Visualize Dynamic Modes
            [
                "python", "-m", "experiments.visualize_dynamic_modes",
                "--model_name", model_name,
                "--custom_name", run_name,
                "--data_path", data_path
            ],
        ] if supports_mode_visualization else []

        commands += [
            # Script 3: Core Eval Script
            [
                "python", "-m", "scripts.eval",
                "--model", model_name,
                "--data_path", data_path,
                "--model_path", model_path,
                "--name", run_name,
                "--steps", str(num_steps)
            ]
        ]
        
        for cmd in commands:
            script_name = cmd[2].split('.')[-1]
            print(f"  > Executing: {script_name}...") 
            
            # Execute the script
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Print standard errors if the script crashed (useful for skipping incompatible modes)
            if result.returncode != 0:
                error_snippet = "\n    ".join(result.stderr.strip().split('\n')[-3:])
                print(f"    ⚠️ Warning: {script_name} failed. Error snippet:\n    {error_snippet}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate best models from CSV sweeps.")
    parser.add_argument("--csv_01", type=str, default="experiments/wandb_runs_dt_0.01.csv", help="Path to dt=0.01 CSV")
    parser.add_argument("--csv_05", type=str, default="experiments/wandb_runs_dt_0.05.csv", help="Path to dt=0.05 CSV")
    parser.add_argument("--target_time", type=float, default=3.0, help="Physical time (s) to simulate.")
    args = parser.parse_args()

    # Process dt=0.01 models (Optimizing for the longest horizon: h100)
    run_evaluations(args.csv_01, dt_val=0.01, target_time=args.target_time, target_metric="best_val_rollout_rmse_h100")
    
    # Process dt=0.05 models (Optimizing for the longest horizon: h20)
    run_evaluations(args.csv_05, dt_val=0.05, target_time=args.target_time, target_metric="best_val_rollout_rmse_h20")