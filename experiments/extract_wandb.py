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
    
    rows = []
    for run in runs:
        row = dict(run._attrs)
        row["run_id"] = run.id
        row["wandb_name"] = run.name
        row["config"] = run.config
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
    
    # =====================================================================
    # --- ADD THIS BLOCK TO RENAME THE MODEL IN THE EXTRACTED DATA ---
    # 1. Replace exact cell matches (fixes 'config_model' etc.)
    df = df.replace({"ml_dmd_drop": "ml_dmd"})
    
    # 2. Replace substrings inside strings and lists (fixes 'group', 'tags', 'run_name')
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].apply(
            lambda x: x.replace("ml_dmd_drop", "ml_dmd") if isinstance(x, str) else 
                      [i.replace("ml_dmd_drop", "ml_dmd") if isinstance(i, str) else i for i in x] if isinstance(x, list) else x
        )
    # =====================================================================

    os.makedirs("experiments/wandb", exist_ok=True)
    df.to_csv("experiments/wandb/wandb_all_runs.csv", index=False)
    print(f"Exported {len(rows)} runs with {len(df.columns)} columns to experiments/wandb/wandb_all_runs.csv")

    # Comprehensive rename map capturing all possible model/expansion parameters
    rename_map = {
        "run_id": "run_name", "wandb_name": "wandb_name",
        "config_data_path": "data_path", "config_model_name": "model_name",
        "config_system_name": "system_name", "config_expansion_type": "expansion_type",
        "config_expansion_degree": "expansion_degree", "config_rbf_bandwidth_mode": "rbf_bandwidth_mode",
        "config_lr": "lr", "config_weight_decay": "weight_decay", "config_rollout_horizon": "rollout_horizon",
        "config_l1_weight": "l1_weight", "config_biorth_weight": "biorth_weight",
        "config_bias": "bias", "config_batch_size": "batch_size", "config_delay_depth": "delay_depth",
        "config_epochs": "epochs", "config_hankel_rank": "hankel_rank",
        "config_rbf_center_selection": "rbf_center_selection", "config_rbf_knn_k": "rbf_knn_k",
        "config_rbf_n_centers": "rbf_n_centers", "config_sine_cosine_expansion": "sine_cosine_expansion",
        "config_hidden_dim": "hidden_dim", "config_num_layers": "num_layers",
        "config_normalize_state": "normalize_state", "config_normalize_lifted": "normalize_lifted",
        "config_rank": "rank", "config_ridge": "ridge", "config_dt": "dt", "config_state_dim": "state_dim",
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
        "summaryMetrics_best_val_rollout_rmse_h20": "best_val_rollout_rmse_h20",
        "summaryMetrics_best_val_rollout_rmse_h100": "best_val_rollout_rmse_h100",
    }

    for col in df.columns:
        if col.startswith("config_") and col not in rename_map: rename_map[col] = col.removeprefix("config_")
        elif col.startswith("summaryMetrics_") and col not in rename_map: rename_map[col] = col.removeprefix("summaryMetrics_")

    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()] 
    df.to_csv('experiments/wandb/wandb_all_runs_processed.csv', index=False)

    # Robust model column detection
    candidates = ["config_model_name", "model_name", "config.model_name", "config.model", "config_model", "config_model.name"]
    model_col = next((c for c in candidates if c in df.columns), None)
    if model_col is None:
        model_col = next((c for c in df.columns if "model" in c.lower()), None)
    if model_col is None:
        print("Warning: could not determine model column; wrote full CSV only.")
        return

    # Clean metrics for sorting
    df["best_val_rollout_rmse_h100"] = pd.to_numeric(df.get("best_val_rollout_rmse_h100", np.nan), errors="coerce")
    df["best_val_rollout_rmse_h20"] = pd.to_numeric(df.get("best_val_rollout_rmse_h20", np.nan), errors="coerce")
    df["l1_weight"] = pd.to_numeric(df.get("l1_weight", 0.0), errors="coerce").fillna(0.0)
    df["data_path"] = df.get("data_path", "unknown").fillna("unknown").astype(str)

    # Unify dt=0.01 (h100) and dt=0.05 (h20) into a single 1.0s target metric column
    def get_target_metric(row):
        if 'dt_0.01' in row['data_path']:
            return row.get('best_val_rollout_rmse_h100', np.nan)
        elif 'dt_0.05' in row['data_path']:
            return row.get('best_val_rollout_rmse_h20', np.nan)
        return row.get('best_val_rollout_rmse_h100', np.nan)
    
    df["sort_metric"] = df.apply(get_target_metric, axis=1)

    # Drop any runs that failed or didn't reach the 1.0s evaluation horizon
    df_valid = df.dropna(subset=["sort_metric"])

    best_runs = []
    nn_edmd_models = {"ml_dmd"}
    
    # ---------------------------------------------------------
    # CORE LOGIC: Find the Absolute Best Model Per System
    # ---------------------------------------------------------
    for model_val, grp_model in df_valid.groupby(df_valid[model_col].fillna("unknown")):
        safe_model_name = str(model_val).strip().replace(os.sep, "_")
        if not safe_model_name: safe_model_name = "unknown"
        
        # Group by System Name (Pools dt=0.01 and dt=0.05 together!)
        for sys_name, grp_sys in grp_model.groupby("system_name"):
            
            # RULE 1: NN-EDMD gets exactly two models (L1=0 and L1>0)
            if safe_model_name in nn_edmd_models:
                grp_zero = grp_sys[grp_sys["l1_weight"] == 0.0]
                grp_pos = grp_sys[grp_sys["l1_weight"] > 0.0]
                
                if not grp_zero.empty:
                    best_runs.append(grp_zero.loc[grp_zero["sort_metric"].idxmin()])
                if not grp_pos.empty:
                    best_runs.append(grp_pos.loc[grp_pos["sort_metric"].idxmin()])
                    
            # RULE 2: ALL other models get exactly 1 absolute best model
            else:
                best_runs.append(grp_sys.loc[grp_sys["sort_metric"].idxmin()])

    # ---------------------------------------------------------
    # SAVE CLEAN DATA
    # ---------------------------------------------------------
    best_df = pd.DataFrame(best_runs)
    
    for model_val, grp in best_df.groupby(model_col):
        safe_model_name = str(model_val).strip().replace(os.sep, "_")
        if not safe_model_name: safe_model_name = "unknown"
        
        out_dir = os.path.join("experiments", "wandb", safe_model_name)
        os.makedirs(out_dir, exist_ok=True)
        
        out_file = os.path.join(out_dir, "best_runs.csv")
        grp.to_csv(out_file, index=False)
        print(f"Saved {len(grp)} best configs for '{safe_model_name}' -> {out_file}")

if __name__ == "__main__":
    main()