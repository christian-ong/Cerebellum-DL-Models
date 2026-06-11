import wandb
import pandas as pd
import argparse
import os
import glob
import numpy as np

def main():
    # 1. FETCH & PROCESS
    api = wandb.Api(timeout=60)
    project_path = "DeepLearningP4Destruction/koopman-operator-learning"
    runs = list(api.runs(project_path, per_page=200, lazy=False, include_sweeps=False))
    
    # We fetch all runs and will export one CSV per model (no chunking)

    rows = []
    for run in runs:
        row = dict(run._attrs)
        row["run_id"] = run.id
        row["wandb_name"] = run.name
        row["config"] = run.config
        # Match the notebook: also keep the raw config payload
        try:
            row["rawconfig"] = run.rawconfig
        except Exception:
            row["rawconfig"] = None
        row["summaryMetrics"] = run.summary_metrics
        row["systemMetrics"] = run.system_metrics
        row["state"] = run.state
        row["url"] = run.url
        rows.append(row)

    # 2. FLATTEN & PROCESS
    df = pd.json_normalize(rows, sep="_")
    # Save raw export for reference
    os.makedirs("experiments/wandb", exist_ok=True)
    df.to_csv("experiments/wandb/wandb_all_runs.csv", index=False)
    print(f"Exported {len(rows)} runs with {len(df.columns)} columns to experiments/wandb/wandb_all_runs.csv")

    # Inline the assemble/rename logic and write one CSV per model into `experiments/wand/`
    # (keeps a single-step, easy-to-inspect export without chunk files)
    rename_map = {
        "run_id": "run_name", "wandb_name": "wandb_name",
        "config_data_path": "data_path", "config_model_name": "model_name",
        "config_system_name": "system_name", "config_expansion_type": "expansion_type",
        "config_expansion_degree": "expansion_degree", "config_rbf_bandwidth_mode": "rbf_bandwidth_mode",
        "config_lr": "lr", "config_rollout_horizon": "rollout_horizon",
        "config_l1_weight": "l1_weight", "config_biorth_weight": "biorth_weight",
        "config_bias": "bias", "config_batch_size": "batch_size", "config_delay_depth": "delay_depth",
        "config_epochs": "epochs", "config_hankel_rank": "hankel_rank",
        "config_rbf_center_selection": "rbf_center_selection", "config_rbf_knn_k": "rbf_knn_k",
        "config_rbf_n_centers": "rbf_n_centers", "config_sine_cosine_expansion": "sine_cosine_expansion",
        "config_hidden_dim": "hidden_dim", "config_num_layers": "num_layers",
        "config_normalize_state": "normalize_state", "config_normalize_lifted": "normalize_lifted",
        "config_rank": "rank", "config_ridge": "ridge",
        "config_regression_rollout_mode": "regression_rollout_mode", "config_load_rbf_from": "load_rbf_from",
        "config_expander_fit_samples": "expander_fit_samples", "config_num_workers": "num_workers",
        "config_print_every_batch": "print_every_batch", "config_seed": "seed",
        "config_outdir": "outdir", "config_eval_every": "eval_every",
        "config_max_val_rollout_trajs": "max_val_rollout_trajs",
        "config_eval_horizon_divisor": "eval_horizon_divisor",
        "config_dataset_rollout_reserve": "dataset_rollout_reserve",
        "config_log_phi_every": "log_phi_every", "config_phi_print_max_dim": "phi_print_max_dim",
        "config_sindy_discrete_time": "sindy_discrete_time", "config_sindy_poly_order": "sindy_poly_order",
        "config_sindy_threshold": "sindy_threshold", "config_sindy_alpha": "sindy_alpha",
        "config_sindy_include_bias": "sindy_include_bias",
        "config_sindy_include_interaction": "sindy_include_interaction",
        "config_sindy_diff_method": "sindy_diff_method", "config_sindy_library_type": "sindy_library_type",
        "config_sindy_fourier_n_frequencies": "sindy_fourier_n_frequencies",
        "config_sindy_specific_basis_size": "sindy_specific_basis_size",
        "summaryMetrics_best_train_loss": "best_train_loss",
        "summaryMetrics_best_val_loss": "best_val_loss",
        "summaryMetrics_best_val_onestep_rmse": "best_val_onestep_rmse",
        "summaryMetrics_best_val_rollout_rmse_h2": "best_val_rollout_rmse_h2",
        "summaryMetrics_best_val_rollout_rmse_h4": "best_val_rollout_rmse_h4",
        "summaryMetrics_best_val_rollout_rmse_h10": "best_val_rollout_rmse_h10",
        "summaryMetrics_best_val_rollout_rmse_h20": "best_val_rollout_rmse_h20",
        "summaryMetrics_best_val_rollout_rmse_h100": "best_val_rollout_rmse_h100",
    }

    for col in df.columns:
        if col.startswith("config_") and col not in rename_map: rename_map[col] = col.removeprefix("config_")
        elif col.startswith("summaryMetrics_") and col not in rename_map: rename_map[col] = col.removeprefix("summaryMetrics_")

    df = df.rename(columns=rename_map)
    
    # ADD THIS LINE: Keep only the first instance of any duplicated column
    df = df.loc[:, ~df.columns.duplicated()] 
    
    os.makedirs("experiments/wandb", exist_ok=True)
    df.to_csv('experiments/wandb/wandb_all_runs_processed.csv', index=False)

    # Write one CSV per model — detect model column robustly
    candidates = ["config_model_name", "model_name", "config.model_name", "config.model", "config_model", "config_model.name"]
    model_col = None
    for c in candidates:
        if c in df.columns:
            model_col = c
            break
    if model_col is None:
        # Fallback: pick first column name containing 'model'
        for c in df.columns:
            if "model" in c.lower():
                model_col = c
                break

    if model_col is None:
        print("Warning: could not determine model column; wrote full CSV only.")
        return

    # Ensure numeric columns for sorting
    df["best_val_rollout_rmse_h100"] = pd.to_numeric(df.get("best_val_rollout_rmse_h100", np.nan), errors="coerce")
    df["l1_weight"] = pd.to_numeric(df.get("l1_weight", 0.0), errors="coerce").fillna(0.0)
    df["data_path"] = df.get("data_path", "unknown").fillna("unknown").astype(str)
    df["expansion_type"] = df.get("expansion_type", "none").fillna("none").astype(str)

    # Dynamically pick the 1.0s physical time metric for sorting
    def get_target_metric(row):
        if 'dt_0.01' in str(row.get('data_path', '')):
            return row.get('best_val_rollout_rmse_h100', np.nan)  # 100 * 0.01 = 1.0s
        elif 'dt_0.05' in str(row.get('data_path', '')):
            return row.get('best_val_rollout_rmse_h20', np.nan)   # 20 * 0.05 = 1.0s
        return row.get('best_val_rollout_rmse_h100', np.nan)
    
    df["sort_metric"] = df.apply(get_target_metric, axis=1)

    # Drop runs that crashed or failed to log the 1.0s metric (instead of strictly h100)
    df_valid = df.dropna(subset=["sort_metric"])

    best_runs = []
    
    # Group by Model
    for model_val, grp_model in df_valid.groupby(df_valid[model_col].fillna("unknown")):
        model_name = str(model_val).strip()
        if not model_name: model_name = "unknown"
        safe_model_name = model_name.replace(os.sep, "_")
        
        # Rule for ML_DMD: Keep best per expansion type, splitting L1=0 vs L1>0
        if safe_model_name == "ml_dmd":
            # ---> CHANGED: Group by system_name instead of data_path <---
            for (sys_name, exp_type), grp_sub in grp_model.groupby(["system_name", "expansion_type"]):
                grp_zero = grp_sub[grp_sub["l1_weight"] == 0.0]
                grp_pos = grp_sub[grp_sub["l1_weight"] > 0.0]
                
                if not grp_zero.empty:
                    best_runs.append(grp_zero.loc[grp_zero["sort_metric"].idxmin()])
                if not grp_pos.empty:
                    best_runs.append(grp_pos.loc[grp_pos["sort_metric"].idxmin()])

        # Rule for ML_DMD_DROP: Keep absolute best L1=0 and best L1>0 overall per system
        elif safe_model_name == "ml_dmd_drop":
            # ---> CHANGED: Group by system_name instead of data_path <---
            for sys_name, grp_sub in grp_model.groupby("system_name"):
                grp_zero = grp_sub[grp_sub["l1_weight"] == 0.0]
                grp_pos = grp_sub[grp_sub["l1_weight"] > 0.0]
                
                if not grp_zero.empty:
                    best_runs.append(grp_zero.loc[grp_zero["sort_metric"].idxmin()])
                if not grp_pos.empty:
                    best_runs.append(grp_pos.loc[grp_pos["sort_metric"].idxmin()])
                    
        # Standard Rule for ALL other models: Just keep the single absolute best run per system
        else:
            # ---> CHANGED: Group by system_name instead of data_path <---
            for sys_name, grp_sub in grp_model.groupby("system_name"):
                best_runs.append(grp_sub.loc[grp_sub["sort_metric"].idxmin()])
                
    best_df = pd.DataFrame(best_runs)
    
    # Save the filtered runs into clean subfolders
    for model_val, grp in best_df.groupby(model_col):
        safe_model_name = str(model_val).strip().replace(os.sep, "_")
        if not safe_model_name: safe_model_name = "unknown"
        
        # Create subfolder: experiments/wandb/{model_name}/
        out_dir = os.path.join("experiments", "wandb", safe_model_name)
        os.makedirs(out_dir, exist_ok=True)
        
        out_file = os.path.join(out_dir, "best_runs.csv")
        grp.to_csv(out_file, index=False)
        print(f"Saved {len(grp)} best configs for '{safe_model_name}' -> {out_file}")

if __name__ == "__main__":
    main()