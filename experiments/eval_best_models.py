import argparse
import pandas as pd
import subprocess
import os
import time
from pathlib import Path

import torch


MODE_VISUALIZATION_MODELS = {"ml_dmd", "regression_dmd", "ml_dmd_drop"}
# Allow the noise-robustness sweep for all common model types so we can compare
# modal and non-modal models side-by-side.
NOISE_ROBUSTNESS_MODELS = {
    "regression_dmd",
    "ml_dmd",
    "ml_dmd_drop",
    "ml_linear_dynamics",
    "ml_lineardynamics",
    "mlp_baseline",
}

import numpy as _np
import glob


def _pretty_model_name(model_name: str) -> str:
    names = {
        "ml_dmd": "NN-EDMD",
        "ml_dmd_drop": "NN-EDMD",
        "ml_linear_dynamics": "NN-LINOP",
        "ml_lineardynamics": "NN-LINOP",
        "mlp_baseline": "MLP",
        "linear_baseline": "REG",
        "dmd_baseline": "DMD",
        "regression_dmd": "EDMD",
        "sindy_baseline": "SINDy",
    }
    return names.get(model_name, model_name.replace("_", " ").title())


def _subtitle_from_row(row, evaluation_rollout_mode: str | None = None) -> str:
    pieces = [_pretty_model_name(str(row.get("model_name", "model")))]

    system_name = row.get("system_name", None)
    if system_name is not None and not pd.isna(system_name):
        pieces.append(str(system_name).replace("_", " ").title())

    model_name = str(row.get("model_name", ""))
    
    # --- FIX: Skip expansion formatting for MLP ---
    if model_name != "mlp_baseline":
        expansion_type = row.get("expansion_type", None)
        if expansion_type is not None and not pd.isna(expansion_type):
            expansion_type = str(expansion_type)
            expansion_type_lc = expansion_type.lower()
            pretty_expansion_type = {
                "general": "General",
                "specific": "Specific",
                "rbf": "RBF",
                "hankel": "Hankel",
                "hankel_svd": "Hankel",
            }.get(expansion_type_lc, expansion_type.replace("_", " ").title())
            pieces.append(pretty_expansion_type)
    
            if expansion_type_lc in {"general", "specific"}:
                degree = row.get("expansion_degree", None)
                if degree is not None and not pd.isna(degree):
                    pieces.append(f"Degree {int(degree)}")
            elif expansion_type_lc == "rbf":
                bandwidth_mode = row.get("rbf_bandwidth_mode", None)
                if bandwidth_mode is not None and not pd.isna(bandwidth_mode):
                    bandwidth_mode = str(bandwidth_mode)
                    if bandwidth_mode.strip().lower() == "global":
                        pieces.append("Global")
                    elif bandwidth_mode.strip().lower() == "knn":
                        knn_k = row.get("rbf_knn_k", None)
                        if knn_k is not None and not pd.isna(knn_k):
                            pieces.append(f"KNN K{int(knn_k)}")
                        else:
                            pieces.append("KNN")
                    else:
                        pieces.append(bandwidth_mode.replace("_", " ").title())
                centers = row.get("rbf_n_centers", None)
                if centers is not None and not pd.isna(centers):
                    pieces.append(f"N Centers {int(centers)}")
            elif expansion_type_lc in {"hankel", "hankel_svd"}:
                depth = row.get("delay_depth", None)
                if depth is not None and not pd.isna(depth):
                    pieces.append(f"Depth {int(depth)}")
                hankel_rank = row.get("hankel_rank", None)
                if hankel_rank is not None and not pd.isna(hankel_rank):
                    pieces.append(f"Rank {int(hankel_rank)}")

    model_name = str(row.get("model_name", ""))
    if model_name == "regression_dmd":
        rollout_mode = row.get("regression_rollout_mode", row.get("rollout_mode", None))
        if rollout_mode is not None and not pd.isna(rollout_mode):
            pieces.append(f"Rollout mode {rollout_mode}")

    # Update _subtitle_from_row() (Around line 74)
    if model_name in {"ml_dmd", "ml_dmd_drop"}: # <-- CHANGE TO INCLUDE ml_dmd_drop
        l1_weight = row.get("l1_weight", None)
        if l1_weight is not None and not pd.isna(l1_weight):
            try:
                pieces.append(f"L1 Weight {float(l1_weight):.3g}")
            except Exception:
                pieces.append(f"L1 Weight {l1_weight}")
                
        # Optional: Append the biorth_weight to your plot subtitles!
        biorth_weight = row.get("biorth_weight", None)
        if biorth_weight is not None and not pd.isna(biorth_weight):
            pieces.append(f"Biorth Weight {float(biorth_weight):.3g}")

    if model_name == "mlp_baseline":
        hidden_dim = row.get("hidden_dim", None)
        num_layers = row.get("num_layers", None)
        if hidden_dim is not None and not pd.isna(hidden_dim):
            pieces.append(f"Hidden Dim {int(hidden_dim)}")
        if num_layers is not None and not pd.isna(num_layers):
            pieces.append(f"Num Layers {int(num_layers)}")

    if model_name == "sindy_baseline":
        library_type = row.get("sindy_library_type", None)
        if library_type is not None and not pd.isna(library_type):
            pieces.append(f"Library type {library_type}")
        if str(expansion_type) == "specific":
            specific_basis_size = row.get("sindy_specific_basis_size", None)
            if specific_basis_size is not None and not pd.isna(specific_basis_size):
                pieces.append(f"Basis size {int(specific_basis_size)}")

    if evaluation_rollout_mode is not None:
        pieces.append(f"eval_rollout_mode={evaluation_rollout_mode}")

    return " | ".join(pieces)


def _mode_specific_run_base(run_base: str, model_name: str, rollout_mode: str | None) -> str:
    if model_name == "regression_dmd" and rollout_mode:
        # Use the rollout mode as a direct subfolder (e.g. 'DMD' or 'linear_dynamics')
        return os.path.join(run_base, str(rollout_mode))
    return run_base


def _infer_mode_count_from_pt_checkpoint(model_path: str):
    """Best-effort mode-count inference for torch checkpoints.

    This is used only to decide whether to add the mode-visualization command.
    It intentionally avoids constructing the full model.
    """
    try:
        ckpt = torch.load(model_path, map_location="cpu")
    except Exception:
        return None

    if not isinstance(ckpt, dict):
        return None

    train_args = ckpt.get("train_args", {})
    if hasattr(train_args, "item"):
        try:
            train_args = train_args.item()
        except Exception:
            pass
    if not isinstance(train_args, dict):
        try:
            train_args = dict(train_args)
        except Exception:
            train_args = {}

    for key in ("expanded_dim", "latent_dim", "rank"):
        value = train_args.get(key, None)
        try:
            if value is not None and int(value) > 0:
                return int(value)
        except Exception:
            continue

    expansion_type = str(train_args.get("expansion_type", ""))
    if expansion_type in {"general", "specific"}:
        value = train_args.get("expansion_degree", None)
        try:
            if value is not None and int(value) > 0:
                return int(value)
        except Exception:
            pass

    if expansion_type == "rbf":
        value = train_args.get("rbf_n_centers", None)
        try:
            if value is not None and int(value) > 0:
                return int(value)
        except Exception:
            pass

    if expansion_type == "hankel_svd":
        value = train_args.get("hankel_rank", None)
        try:
            if value is not None and int(value) > 0:
                return int(value)
        except Exception:
            pass

    return None


def _behavior_commands(
    model_name,
    run_name,
    data_path,
    model_path,
    base_figdir,
    *,
    mode_subset_thresholds=None,
    num_steps=100,  # <--- NEW: Accept num_steps as an argument
):
    """Return a single eval_behavior invocation."""
    final_outdir = os.path.join(base_figdir, "behavior")

    # --- NEW: Dynamically pick 3 horizons to plot for the heatmaps ---
    mid_h = max(1, num_steps // 2)
    grid_horizons = sorted(list(set([1, mid_h, num_steps])))
    grid_str = ",".join(str(h) for h in grid_horizons)

    cmd = [
        "python", "-m", "scripts.eval_behavior",
        "--model", model_name,
        "--data_path", data_path,
        "--model_path", model_path,
        "--name", run_name,
        "--split", "test",
        "--metric_horizons", str(num_steps),         # <--- FIXED
        "--rollout_metric_horizons", str(num_steps), # <--- FIXED
        "--metric_cap", "0",
        "--run_true_grid_heatmap",
        "--true_grid_horizons", grid_str,            # <--- FIXED (e.g., "1,10,20")
        "--grid_resolution", "100",
        "--grid_overlay_n_trajs", "0",
        "--outdir", final_outdir,
    ]

    if mode_subset_thresholds:
        thresholds_arg = ",".join(
            str(int(round(float(t)))) if float(t).is_integer() else str(float(t)) 
            for t in mode_subset_thresholds
        )
        cmd += ["--mode_subset_thresholds", thresholds_arg]

    return [cmd]


def _mode_visualization_commands(model_name, run_name, data_path, base_figdir, num_steps=None):
    """Return the mode-visualization command sweep for supported models."""
    if model_name not in MODE_VISUALIZATION_MODELS:
        return []

    mode_orders = ["contribution","mse", "time_int_energy", "magnitude", "original"]

    commands = []
    for mode_order in mode_orders:
        cmd = [
            "python", "-m", "experiments.visualize_dynamic_modes",
            "--model_name", model_name,
            "--custom_name", run_name,
            "--data_path", data_path,
            "--mode_order", mode_order,
            "--outdir", os.path.join(base_figdir, "modes", mode_order),
        ]
        # --- NEW: Forward the steps if provided ---
        if num_steps is not None:
            cmd.extend(["--num_steps", str(num_steps)])
            
        commands.append(cmd)

    return commands


def _noise_robustness_commands(
    model_name,
    run_name,
    clean_data_path,
    noisy_data_path,
    model_path,
    base_figdir,
    num_steps,
    *,
    mode_subset_thresholds=None,
    plot_mode_subsets=False,
):
    """Return the noise-robustness command for supported models when a noisy dataset is available."""
    if model_name not in NOISE_ROBUSTNESS_MODELS:
        return []

    # If no explicit noisy path provided, try to infer a matching noisy dataset root
    if not noisy_data_path:
        inferred = _infer_noisy_data_root_from_clean_path(clean_data_path)
        noisy_data_path = inferred

    if not noisy_data_path:
        return []

    cmd = [
        "python", "-m", "experiments.eval_noise_robustness",
        "--model", model_name,
        "--model_path", model_path,
        "--clean_data_path", clean_data_path,
        "--noisy_data_path", noisy_data_path,
        "--split", "test",
        "--traj_index", "0",
        "--plot_traj_indices", "0,1,2,3",
        "--steps", str(int(num_steps)),
        "--feedback_noise_std", "0.001",
        "--feedback_rollout_mode", "DMD",
        "--max_pairs", "5000",
        "--name", run_name,
        "--outdir", os.path.join(base_figdir, "noise_robustness"),
    ]

    if mode_subset_thresholds:
        thresholds_arg = ",".join(str(int(round(float(t)))) if float(t).is_integer() else str(float(t)) for t in mode_subset_thresholds)
        cmd += ["--mode_subset_thresholds", thresholds_arg]
        if plot_mode_subsets:
            cmd.append("--plot_mode_subsets")

    return [cmd]


def _overview_horizons_for_dt(dt_val, target_time):
    horizons = [
        1,
        int(round(0.1 / float(dt_val))),
        int(round(1.0 / float(dt_val))),
        int(round(float(target_time) / float(dt_val))),
    ]
    return sorted({int(h) for h in horizons if int(h) >= 1})


def _overview_metadata_columns():
    return [
        "id",
        "run_name",
        "wandb_name",
        "name",
        "url",
        "state",
        "group",
        "group_name",
        "projectId",
        "model",
        "model_name",
        "system",
        "system_name",
        "data_path",
        "expansion_type",
        "expansion_degree",
        "expansion_bucket",
        "delay_depth",
        "hankel_rank",
        "rollout_horizon",
        "regression_rollout_mode",
        "l1_weight",
        "biorth_weight",
        "lr",
        "weight_decay",
        "batch_size",
        "epochs",
        "eval_every",
        "seed",
        "subset",
        "state_dim",
        "num_workers",
        "load_rbf_from",
        "log_phi_every",
        "rbf_n_centers",
        "rbf_knn_k",
        "rbf_bandwidth_mode",
        "rbf_center_selection",
        "expander_fit_samples",
        "phi_print_max_dim",
        "print_every_batch",
        "sine_cosine_expansion",
        "dataset_rollout_reserve",
        "max_val_rollout_trajs",
        "normalize_state",
        "normalize_lifted",
        "eval_horizon_divisor",
        "best_train_loss",
        "best_train_loss_epoch",
        "best_val_loss",
        "best_val_loss_epoch",
        "best_val_onestep_rmse",
        "best_val_onestep_rmse_epoch",
        "best_val_rollout_failed",
        "best_val_rollout_failed_epoch",
        "best_val_rollout_rmse_h2",
        "best_val_rollout_rmse_h2_epoch",
        "best_val_rollout_rmse_h4",
        "best_val_rollout_rmse_h4_epoch",
        "best_val_rollout_rmse_h10",
        "best_val_rollout_rmse_h10_epoch",
        "best_val_rollout_rmse_h20",
        "best_val_rollout_rmse_h20_epoch",
        "best_val_rollout_rmse_h100",
        "best_val_rollout_rmse_h100_epoch",
        "val_loss",
        "val_onestep_rmse",
        "val_rollout_failed",
        "val_rollout_rmse_h2",
        "val_rollout_rmse_h4",
        "val_rollout_rmse_h10",
        "val_rollout_rmse_h20",
        "val_rollout_rmse_h100",
    ]


def _to_float_or_none(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def resolve_model_checkpoint(model_name, system, run_name):
    """Return the first checkpoint path that exists for this sweep row."""
    folder_candidates = [model_name]
    if model_name == "ml_linear_dynamics":
        folder_candidates.append("ml_lineardynamics")
    elif model_name == "ml_lineardynamics":
        folder_candidates.append("ml_linear_dynamics") 
    elif model_name == "ml_dmd":
        folder_candidates.append("ml_dmd")
    elif model_name == "ml_dmd_drop":
        folder_candidates.append("ml_dmd_drop")
    elif model_name == "linear_baseline":
        folder_candidates.append("linear_baseline")
    elif model_name == "dmd_baseline":
        folder_candidates.append("dmd_baseline")
    elif model_name == "sindy_baseline":
        folder_candidates.append("sindy_baseline")
    elif model_name == "mlp_baseline":
        folder_candidates.append("mlp_baseline")

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


def _infer_noisy_data_root_from_clean_path(clean_data_path: str):
    """Try common heuristics to locate a matching noisy-data root for a given clean data path.

    Returns the inferred noisy-data path string if it appears to exist on disk, otherwise None.
    """
    if not isinstance(clean_data_path, str) or clean_data_path.strip() == "":
        return None

    # Common explicit replacement
    if "data/trajectories" in clean_data_path:
        candidate = clean_data_path.replace("data/trajectories", "data/noisy_trajectories")
        if os.path.exists(candidate):
            return candidate

    # Fallback: if path contains '/trajectories/' replace with '/noisy_trajectories/'
    if "/trajectories/" in clean_data_path:
        candidate = clean_data_path.replace("/trajectories/", "/noisy_trajectories/")
        if os.path.exists(candidate):
            return candidate

    # Generic sibling under data/noisy_trajectories preserving the remainder after 'data/'
    if "data/" in clean_data_path:
        suffix = clean_data_path.split("data/", 1)[1]
        candidate = os.path.join("data", "noisy_trajectories", suffix)
        if os.path.exists(candidate):
            return candidate

    return None


def _infer_state_dim_from_data_path(data_path: str):
    """Try to infer the state's dimensionality from a dataset path.

    Returns an int state_dim if found, otherwise None.
    """
    try:
        if not isinstance(data_path, str) or data_path.strip() == "":
            return None

        # If a directory is provided, look for common split/npz files inside it.
        if os.path.isdir(data_path):
            candidates = [
                os.path.join(data_path, "test.npz"),
                os.path.join(data_path, "split_test.npz"),
                os.path.join(data_path, "trajectories.npz"),
                os.path.join(data_path, "data.npz"),
            ]
            data_file = None
            for c in candidates:
                if os.path.exists(c):
                    data_file = c
                    break
            if data_file is None:
                matches = glob.glob(os.path.join(data_path, "*.npz"))
                if matches:
                    data_file = matches[0]
                else:
                    return None
        else:
            data_file = data_path

        loaded = _np.load(data_file)
        # loaded can be an array or an NpzFile-like mapping
        if hasattr(loaded, "files") and len(loaded.files) > 0:
            # Prefer common keys
            for key in ("X", "x", "data", "trajectories", "traj", "states"):
                if key in loaded:
                    X = loaded[key]
                    break
            else:
                X = loaded[loaded.files[0]]
        else:
            X = loaded

        if hasattr(X, "ndim"):
            # Typical shapes: (n_samples, seq_len, state_dim) or (seq_len, state_dim)
            if X.ndim >= 2:
                return int(X.shape[-1])

        return None
    except Exception:
        return None

def run_evaluations(
    csv_path,
    dt_val,
    target_time,
    target_metric,
    *,
    skip_existing=False,
    force=False,
    noisy_data_path=None,
    overview_csv_path=None,
):
    print(f"\n=========================================================", flush=True)
    print(f" PROCESSING DATASET: {csv_path} (dt={dt_val})", flush=True)
    print(f"=========================================================\n", flush=True)
    
    if not os.path.exists(csv_path):
        print(f"❌ File {csv_path} not found. Skipping.", flush=True)
        return

    # 1. Load dataframe
    df = pd.read_csv(csv_path, low_memory=False)
    
    # 2. The CSV is already pre-filtered by extract_wandb.py to only contain the best runs.
    # We just pass it through directly.
    best_df = df.copy()
    
    print(f"✅ Loaded {len(best_df)} pre-filtered model configurations.", flush=True)

    # Resolve checkpoints and metadata
    best_df["resolved_checkpoint"] = best_df.apply(
        lambda row: resolve_model_checkpoint(
            row["model_name"],
            row["system_name"],
            row.get("run_id", row.get("run_name")),
        ),
        axis=1,
    )
    best_df["selected_target_metric"] = target_metric
    best_df["selected_dt"] = dt_val
    best_df["selected_num_steps"] = int(target_time / dt_val)
    best_df["selection_role"] = "best"
    print(f"✅ Found {len(best_df)} selected model configurations based on '{target_metric}' within requested combinations.", flush=True)

    best_df = best_df.copy()
    best_df["resolved_checkpoint"] = best_df.apply(
        lambda row: resolve_model_checkpoint(
            row["model_name"],
            row["system_name"],
            row.get("run_id", row.get("run_name")),
        ),
        axis=1,
    )
    best_df["selected_target_metric"] = target_metric
    best_df["selected_dt"] = dt_val
    best_df["selected_num_steps"] = int(target_time / dt_val)
    best_df["selection_role"] = "best"

    # Track skipping status so repeated runs can avoid re-evaluating the same checkpoints.
    best_df["skipped"] = False
    best_df["skip_reason"] = ""

    if "model_name" in best_df.columns and "l1_weight" in best_df.columns:
        l1_values = best_df["l1_weight"].apply(_to_float_or_none)
        best_df.loc[(best_df["model_name"] == "ml_dmd") & (l1_values == 0.0), "selection_role"] = "ml_dmd_l1_0"
        best_df.loc[(best_df["model_name"] == "ml_dmd") & (l1_values.isin([1e-3, 1e-2])), "selection_role"] = "ml_dmd_l1_companion"

    # 3. Calculate exactly how many steps to simulate for the given physical time
    num_steps = int(target_time / dt_val)
    overview_horizons = _overview_horizons_for_dt(dt_val, target_time)
    overview_metric_horizons = overview_horizons[1:]
    overview_rows = []

    def _collect_overview_row(summary_path, *, row, run_name, evaluation_rollout_mode):
        if overview_csv_path is None:
            return
        if not os.path.exists(summary_path):
            print(f"    ⚠️ Warning: overview summary not found at {summary_path}; skipping overview row.", flush=True)
            return

        try:
            data = _np.load(summary_path, allow_pickle=True)
        except Exception as exc:
            print(f"    ⚠️ Warning: failed to load overview summary {summary_path}: {exc}", flush=True)
            return

        horizons = _np.asarray(data["horizons"], dtype=int) if "horizons" in data else _np.array([], dtype=int)
        horizon_rmse = _np.asarray(data["horizon_rmse"], dtype=float) if "horizon_rmse" in data else _np.array([], dtype=float)

        overview_row = {
            "dt": float(dt_val),
            "target_time": float(target_time),
            "model_name": row["model_name"],
            "system_name": row["system_name"],
            "expansion_type": row.get("expansion_type", None),
            "run_name": run_name,
            "evaluation_rollout_mode": evaluation_rollout_mode or "default",
            "one_step_rmse": float(data["one_step_rmse"]) if "one_step_rmse" in data else _np.nan,
            "composite_score": float(data["composite_score"]) if "composite_score" in data else _np.nan,
            "summary_path": summary_path,
        }

        for key in _overview_metadata_columns():
            if key in overview_row:
                continue
            if key in row.index:
                overview_row[key] = row[key]

        for h in overview_metric_horizons:
            key = f"h{int(h)}_rmse"
            match = _np.where(horizons == int(h))[0]
            overview_row[key] = float(horizon_rmse[int(match[0])]) if len(match) > 0 else _np.nan

        overview_rows.append(overview_row)

    # 4. Loop through the best models
    for idx, row in best_df.iterrows():
        run_name = row.get('run_id', row['run_name'])
        wandb_name = row.get('wandb_name', row.get('name', run_name))
        model_name = row['model_name']
        system = row['system_name']
        data_path = row['data_path']
        
        print(f"\n[{idx+1}/{len(best_df)}] Evaluating: {system} | {model_name} ({row['expansion_type']}) -> Run: {run_name} ({wandb_name})", flush=True)
        
        # Resolve the actual checkpoint file saved for this model family.
        model_path = resolve_model_checkpoint(model_name, system, run_name)
        if model_path is None:
            print(f"    ⚠️ Warning: no checkpoint found for {model_name}/{system}/{run_name}. Skipping.", flush=True)
            continue

        # Canonical output grouping for this row. If the run root already exists,
        # treat the whole evaluation as done and move on immediately.
        expansion_type = str(row.get("expansion_type", "none")) if row.get("expansion_type", None) is not None else "none"
        if model_name == "sindy_baseline":
            expansion_type = str(row.get("sindy_library_type", expansion_type))

        # Base root should be model/system — the specific expansion folder is
        # composed below to avoid duplicating the expansion type twice.
        base_root = os.path.join("experiments", "figures", model_name, system)


        # Compose expansion folder with special handling for RBF and Hankel
        expansion_folder = str(expansion_type) if expansion_type is not None else "none"
        if expansion_type == "rbf":
            bandwidth = row.get("rbf_bandwidth_mode", None)
            bw = str(bandwidth).strip().lower() if bandwidth is not None and not pd.isna(bandwidth) else "global"
            expansion_folder = os.path.join("rbf", "global" if bw == "global" else "knn")
        if expansion_type in {"hankel", "hankel_svd"}:
            expansion_folder = "hankel_svd"

        # For SINDy-specific libraries, keep the actual basis size visible in the path.
        if model_name == "sindy_baseline" and expansion_folder == "specific":
            specific_basis_size = row.get("sindy_specific_basis_size")
            if specific_basis_size is not None and not pd.isna(specific_basis_size):
                expansion_folder = os.path.join(expansion_folder, f"basis_{int(specific_basis_size)}")

        # For ml_dmd, add an l1_weight subfolder (e.g., '0.0' or '1e-03'/'1e-02')
        l1_folder = None
        if model_name in {"ml_dmd", "ml_dmd_drop"}:
            l1_val = _to_float_or_none(row.get("l1_weight")) if "l1_weight" in row else None
            if l1_val is not None:
                if float(l1_val) == 0.0:
                    l1_folder = "l1_0.0"
                else:
                    try:
                        l1_folder = "l1_{:.0e}".format(float(l1_val))
                    except Exception:
                        l1_folder = str(l1_val)

        # Place l1 folder under the expansion folder for ml_dmd models
        final_root = os.path.join(base_root, expansion_folder)
        if l1_folder:
            final_root = os.path.join(final_root, l1_folder)

        dt_folder = f"dt_{dt_val:.2f}"
        run_base = os.path.join(final_root, dt_folder, run_name)

        if model_name != "regression_dmd" and skip_existing and not force and os.path.isdir(run_base):
            print(f"    ⏭ Skipping evaluation — found existing run folder: {run_base}", flush=True)
            best_df.loc[idx, "skipped"] = True
            best_df.loc[idx, "skip_reason"] = "existing_run_folder"
            _collect_overview_row(
                os.path.join(run_base, "data", "test_summary.npz"),
                row=row,
                run_name=run_name,
                evaluation_rollout_mode=None,
            )
            continue

        # Lightweight metadata inference: avoid instantiating full models here
        # (those are loaded again in subprocesses). Try to extract modal info
        # from a nearby .npz metadata file when available.
        mode_count = None
        inferred_state_dim = None
        try:
            inferred_state_dim = _infer_state_dim_from_data_path(data_path)
            if inferred_state_dim is None:
                state_dim_arg = 2
                print(f"    ℹ️ Could not infer state_dim from data_path; falling back to 2", flush=True)
            else:
                state_dim_arg = int(inferred_state_dim)
                print(f"    ℹ️ Inferred state_dim={state_dim_arg} from data_path", flush=True)

            # If the checkpoint is a .npz, read it directly for metadata.
            metadata_candidates = []
            try:
                p = Path(model_path)
                if str(model_path).endswith('.npz'):
                    metadata_candidates.append(str(model_path))
                # Common sibling names
                metadata_candidates.extend([
                    str(p.with_suffix('.npz')),
                    str(p.parent / 'model.npz'),
                    str(p.parent / 'model_best.npz'),
                ])
            except Exception:
                metadata_candidates = []

            for meta in metadata_candidates:
                if not meta:
                    continue
                if os.path.exists(meta):
                    try:
                        md = _np.load(meta, allow_pickle=True)
                        # Prefer explicit Phi matrix
                        if 'Phi' in md:
                            try:
                                phi = _np.asarray(md['Phi'])
                                if phi.ndim == 2:
                                    mode_count = int(phi.shape[1])
                                    break
                            except Exception:
                                pass
                        if mode_count is None and 'Lambda' in md:
                            try:
                                lam = _np.asarray(md['Lambda'])
                                mode_count = int(lam.size)
                                break
                            except Exception:
                                pass
                        # fallback to rank-like entries
                        for key in ('rank', 'expanded_dim', 'latent_dim'):
                            if key in md:
                                try:
                                    mode_count = int(_np.asarray(md[key]).item())
                                    break
                                except Exception:
                                    continue
                        if mode_count is not None:
                            break
                    except Exception:
                        continue

        except Exception as e:
            print(f"    ⚠️ Light-weight metadata inference failed: {e}", flush=True)
            mode_count = None

        if mode_count is None and model_name in MODE_VISUALIZATION_MODELS and str(model_path).endswith(".pt"):
            mode_count = _infer_mode_count_from_pt_checkpoint(model_path)
            if mode_count is not None:
                print(f"    ℹ️ Inferred mode_count={mode_count} from torch checkpoint metadata", flush=True)

        # The evaluation commands
        # Arrange figure outputs under:
        #   model_name / system / expansion_type-or-library_type / [extra subfolder] / run_name / ...
        # For SINDy, the meaningful bucket is `sindy_library_type` rather than the generic
        # `expansion_type` column used by the other model families.

        evaluation_rollout_modes = [None]
        if model_name == "regression_dmd":
            evaluation_rollout_modes = ["DMD"]

        row_skipped_all = True

        for evaluation_rollout_mode in evaluation_rollout_modes:
            # For regression_dmd we want the rollout-mode to appear before the dt/run_name
            # and also as a subfolder under the run_name (matches requested layout):
            # final_root/<RolloutMode>/dt_<...>/<run_name>/<RolloutMode>/...
            if model_name == "regression_dmd" and evaluation_rollout_mode:
                # Place the rollout mode as a top-level folder under the expansion
                # (before dt/run_name). Avoid duplicating the rollout-mode inside
                # the run folder to prevent redundant nesting.
                mode_run_base = os.path.join(final_root, str(evaluation_rollout_mode), dt_folder, run_name)
            else:
                mode_run_base = _mode_specific_run_base(run_base, model_name, evaluation_rollout_mode)

            if skip_existing and not force and os.path.isdir(mode_run_base):
                print(f"    ⏭ Skipping evaluation mode {evaluation_rollout_mode or 'default'} — found existing run folder: {mode_run_base}", flush=True)
                _collect_overview_row(
                    os.path.join(mode_run_base, "data", "test_summary.npz"),
                    row=row,
                    run_name=run_name,
                    evaluation_rollout_mode=evaluation_rollout_mode,
                )
                continue

            row_skipped_all = False

            # --- NEW: Check our skipping conditions ---
            is_lorenz = "lorenz" in system.lower()
            is_sindy = (model_name == "sindy_baseline")

            commands = []

            # 1. Trajectory Rollout Plotter
            commands.append(
                [
                    "python", "-m", "experiments.eval_trajectory_rollout",
                    "--model_name", model_name,
                    "--custom_name", run_name,
                    "--data_path", data_path,
                    "--num_steps", str(num_steps),
                    "--model_path", model_path,
                    "--outdir", os.path.join(mode_run_base, "rollout"),
                ]
            )

            # 2. Mode visualization for modal models
            subset_thresholds = None
            if not is_sindy and mode_count is not None and mode_count > 1:
                commands += _mode_visualization_commands(
                    model_name=model_name,
                    run_name=run_name,
                    data_path=data_path,
                    base_figdir=mode_run_base,
                    num_steps=num_steps,
                )

            # 3. Core Eval Script (ALWAYS RUN - this evaluates the test set and calculates metrics)
            commands.append(
                [
                    "python", "-m", "scripts.eval",
                    "--model", model_name,
                    "--data_path", data_path,
                    "--model_path", model_path,
                    "--name", run_name,
                    "--steps", str(num_steps),
                    "--horizons", ",".join(str(h) for h in overview_horizons),
                    "--rollout_horizons", ",".join(str(h) for h in overview_metric_horizons),
                    "--outdir", os.path.join(mode_run_base, "data"),
                ]
            )

            # 4. Behavior (Heatmaps)
            if not is_sindy and not is_lorenz:
                commands += _behavior_commands(
                    model_name=model_name,
                    run_name=run_name,
                    data_path=data_path,
                    model_path=model_path,
                    base_figdir=mode_run_base,
                    mode_subset_thresholds=subset_thresholds,
                    num_steps=num_steps,
                )

            # 5. Noise robustness
            if not is_sindy:
                if mode_count is not None and mode_count > 1 and evaluation_rollout_mode != "linear_dynamics":
                    commands += _noise_robustness_commands(
                        model_name=model_name,
                        run_name=run_name,
                        clean_data_path=data_path,
                        noisy_data_path=noisy_data_path,
                        model_path=model_path,
                        base_figdir=mode_run_base,
                        num_steps=num_steps,
                        mode_subset_thresholds=[1, 5, 10, 25, 50, 100],
                        plot_mode_subsets=True,
                    )
                else:
                    commands += _noise_robustness_commands(
                        model_name=model_name,
                        run_name=run_name,
                        clean_data_path=data_path,
                        noisy_data_path=noisy_data_path,
                        model_path=model_path,
                        base_figdir=mode_run_base,
                        num_steps=num_steps,
                    )

            for command_idx, cmd in enumerate(commands, start=1):
                script_name = cmd[2].split('.')[-1]
                print(f"  > [{command_idx}/{len(commands)}] Executing: {script_name}...", flush=True)
                start_time = time.perf_counter()

                # Execute the script; set EVAL_BASE_DIR so the called scripts save under experiments/figures.
                env = dict(os.environ)
                env["EVAL_BASE_DIR"] = os.path.join("experiments", "figures")
                env["PYTHONUNBUFFERED"] = "1"
                if evaluation_rollout_mode is not None:
                    env["EVAL_REGRESSION_ROLLOUT_MODE"] = evaluation_rollout_mode
                # Pass lightweight inferred state-dim to subprocesses to avoid repeated dataset loads
                try:
                    if inferred_state_dim is not None:
                        env["EVAL_INFERRED_STATE_DIM"] = str(int(inferred_state_dim))
                except Exception:
                    pass

                result = subprocess.run(cmd, env=env)

                if result.returncode != 0:
                    print(f"    ⚠️ Warning: {script_name} failed with exit code {result.returncode}.", flush=True)
                else:
                    elapsed = time.perf_counter() - start_time
                    print(f"    ✓ Finished {script_name} in {elapsed:.1f}s", flush=True)

            _collect_overview_row(
                os.path.join(mode_run_base, "data", "test_summary.npz"),
                row=row,
                run_name=run_name,
                evaluation_rollout_mode=evaluation_rollout_mode,
            )

        if row_skipped_all:
            best_df.loc[idx, "skipped"] = True
            best_df.loc[idx, "skip_reason"] = "existing_run_folder"

    if overview_csv_path is not None and len(overview_rows) > 0:
        overview_df = pd.DataFrame(overview_rows)
        if os.path.exists(overview_csv_path):
            try:
                existing_overview = pd.read_csv(overview_csv_path)
                overview_df = pd.concat([existing_overview, overview_df], ignore_index=True, sort=False)
                dedupe_cols = [
                    col
                    for col in [
                        "dt",
                        "target_time",
                        "model_name",
                        "system_name",
                        "expansion_type",
                        "run_name",
                        "evaluation_rollout_mode",
                    ]
                    if col in overview_df.columns
                ]
                if dedupe_cols:
                    overview_df = overview_df.drop_duplicates(subset=dedupe_cols, keep="last")
            except Exception as exc:
                print(f"⚠️ Warning: could not merge existing overview CSV {overview_csv_path}: {exc}", flush=True)
        overview_df.to_csv(overview_csv_path, index=False)
        print(f"💾 Saved RMSE overview CSV to {overview_csv_path}", flush=True)

    skipped = int(best_df["skipped"].sum()) if "skipped" in best_df.columns else 0
    print(f"\n✅ Evaluation run complete. Skipped {skipped} already-evaluated models.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate best models from processed wandb model subfolders.")
    parser.add_argument("--csv", type=str, default="experiments/wandb", help="Directory containing the model subfolders")
    parser.add_argument("--target_time", type=float, default=1.0, help="Physical time (s) to simulate.")
    parser.add_argument(
        "--noisy_data_path",
        type=str,
        default=None,
        help="Optional root path for a noisy trajectory dataset used by the noise-robustness sweep.",
    )
    parser.add_argument("--skip_existing", action="store_true", help="Skip evaluation if a summary file already exists for the resolved checkpoint.")
    parser.add_argument("--force", action="store_true", help="Force evaluation even if an existing summary is found (overrides --skip_existing).")
    args = parser.parse_args()

    # Find all best_runs.csv files in the model subdirectories
    best_run_files = glob.glob(os.path.join(args.csv, "*", "best_runs.csv"))
    
    if not best_run_files:
        print(f"No best_runs.csv files found in {args.csv}/*/")
        exit(0)

    # We still use a tmp folder to pass the filtered data to run_evaluations, 
    # but the final output goes cleanly into the model folder.
    tmp_dir = Path("experiments/wandb/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for master_csv in best_run_files:
        master_df = pd.read_csv(master_csv, low_memory=False)
        model_dir = Path(master_csv).parent
        model_folder_name = model_dir.name
        
        # Partition by substring in data_path
        df_001 = master_df[master_df.get("data_path", "").str.contains("dt_0.01", na=False)].copy()
        df_005 = master_df[master_df.get("data_path", "").str.contains("dt_0.05", na=False)].copy()

        csv_001 = str(tmp_dir / f"{model_folder_name}_best_dt_0.01.csv")
        csv_005 = str(tmp_dir / f"{model_folder_name}_best_dt_0.05.csv")

        if not df_001.empty:
            df_001.to_csv(csv_001, index=False)
            run_evaluations(
                csv_001,
                dt_val=0.01,
                target_time=args.target_time,
                target_metric="best_val_rollout_rmse_h100",
                skip_existing=args.skip_existing,
                force=args.force,
                noisy_data_path=args.noisy_data_path,
                overview_csv_path=str(model_dir / "test_results_dt_0.01.csv"), # Saves inside the model folder
            )

        if not df_005.empty:
            df_005.to_csv(csv_005, index=False)
            run_evaluations(
                csv_005,
                dt_val=0.05,
                target_time=args.target_time,
                target_metric="best_val_rollout_rmse_h20",
                skip_existing=args.skip_existing,
                force=args.force,
                noisy_data_path=args.noisy_data_path,
                overview_csv_path=str(model_dir / "test_results_dt_0.05.csv"), # Saves inside the model folder
            )