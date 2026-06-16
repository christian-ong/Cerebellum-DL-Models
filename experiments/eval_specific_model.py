import argparse
import pandas as pd
import numpy as np
import os
from pathlib import Path

from experiments.eval_best_models import run_evaluations

def main():
    parser = argparse.ArgumentParser(description="Evaluate the best models With vs Without L1.")
    parser.add_argument("--model", type=str, required=True, help="e.g., ml_dmd, regression_dmd, sindy_baseline")
    parser.add_argument("--expansion", type=str, required=True, help="e.g., specific, general, rbf, hankel_svd")
    parser.add_argument("--system", type=str, required=True, help="e.g., closed_large, vanderpol, lorenz")
    parser.add_argument("--dt", type=float, default=0.01, help="Timestep to filter for (e.g., 0.01 or 0.05). Default: 0.01")
    parser.add_argument("--top_k", type=int, default=1, help="Number of top models to evaluate. Default: 1")
    
    # Standard evaluation arguments
    parser.add_argument("--target_time", type=float, default=1.0, help="Physical time (s) to simulate.")
    parser.add_argument("--noisy_data_path", type=str, default=None, help="Optional noisy data path.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if summary exists.")
    parser.add_argument("--force", action="store_true", help="Force evaluation.")
    
    # Path to the master processed CSV
    parser.add_argument("--master_csv", type=str, default="experiments/wandb/wandb_all_runs_processed.csv", help="Path to processed W&B CSV")
    
    args = parser.parse_args()

    print(f"\n🔍 Searching for Top {args.top_k}: Model={args.model} | Expansion={args.expansion} | System={args.system} | dt={args.dt}")
    
    if not os.path.exists(args.master_csv):
        print(f"❌ Master CSV not found at {args.master_csv}. Please run extract_wandb.py first.")
        return

    # 1. Load the master dataframe
    df = pd.read_csv(args.master_csv, low_memory=False)

    # 2. Filter the dataframe based on user specifications
    mask = (df["model_name"] == args.model) & (df["system_name"] == args.system)
    
    # Handle SINDy's unique expansion column name
    if args.model == "sindy_baseline":
        mask &= (df["sindy_library_type"] == args.expansion)
    else:
        mask &= (df["expansion_type"] == args.expansion)

    # Filter for the correct timestep in the data_path
    dt_str = f"dt_{args.dt}"
    mask &= df["data_path"].str.contains(dt_str, na=False)

    filtered_df = df[mask].copy()

    if filtered_df.empty:
        print(f"❌ No runs found matching those exact criteria.")
        return
    
    print(f"✅ Found {len(filtered_df)} total runs matching the configuration.")

    # 3. Determine the correct metric to sort by based on dt
    if args.dt == 0.05:
        target_metric = "best_val_rollout_rmse_h20"
    else:
        target_metric = "best_val_rollout_rmse_h100"

    # Fallback to standard metric name if 'best_' prefix is missing
    if target_metric not in filtered_df.columns:
        target_metric = target_metric.replace("best_", "")

    master_df = filtered_df.copy()

    # 1. Ensure l1_weight exists and handle missing values as 0 (No L1)
    if "l1_weight" not in master_df.columns:
        print("⚠️ Warning: 'l1_weight' column not found. Treating all as l1_weight=0.")
        master_df["l1_weight"] = 0.0
    else:
        master_df["l1_weight"] = master_df["l1_weight"].fillna(0.0)

    # 2. Split into two groups
    # Group A: No L1 (l1_weight == 0)
    df_no_l1 = master_df[master_df["l1_weight"] == 0].copy()
    # Group B: With L1 (l1_weight > 0)
    df_with_l1 = master_df[master_df["l1_weight"] > 0].copy()

    selected_runs = pd.DataFrame()

    # 3. Find best in No L1
    if not df_no_l1.empty:
        best_no_l1 = df_no_l1.loc[df_no_l1[target_metric].idxmin()]
        selected_runs = pd.concat([selected_runs, best_no_l1.to_frame().T])
        print(f"🏆 Best 'No L1' Run: {best_no_l1['run_name']} (RMSE: {best_no_l1[target_metric]:.5f})")
    else:
        print("ℹ️ No runs found without L1 weight.")

    # 4. Find best in With L1
    if not df_with_l1.empty:
        best_with_l1 = df_with_l1.loc[df_with_l1[target_metric].idxmin()]
        selected_runs = pd.concat([selected_runs, best_with_l1.to_frame().T])
        print(f"🏆 Best 'With L1' Run: {best_with_l1['run_name']} (RMSE: {best_with_l1[target_metric]:.5f})")
    else:
        print("ℹ️ No runs found with L1 weight.")

    if selected_runs.empty:
        print("❌ No models found to evaluate.")
        return

    # 5. Save the selected runs to the temporary CSV
    tmp_dir = Path("experiments/wandb/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_csv_path = tmp_dir / f"compare_l1_{args.model}_{args.system}_dt{args.dt}.csv"
    selected_runs.to_csv(tmp_csv_path, index=False)

    # 6. Pass it to your robust evaluation pipeline
    overview_path = f"experiments/wandb/{args.model}/test_results_dt_{args.dt}.csv"
    os.makedirs(os.path.dirname(overview_path), exist_ok=True)

    run_evaluations(
        csv_path=str(tmp_csv_path),
        dt_val=args.dt,
        target_time=args.target_time,
        target_metric=target_metric,
        skip_existing=args.skip_existing,
        force=args.force,
        noisy_data_path=args.noisy_data_path,
        overview_csv_path=overview_path
    )

if __name__ == "__main__":
    main()