import argparse
import csv
import os
import numpy as np
import torch

from src.eval.model_io import load_model, infer_run_name
from src.data_generation.load_data import resolve_split_npz_path
from src.eval.noise_robustness import run_noise_robustness_suite


def parse_int_list_maybe(s):
    if s is None or str(s).strip() == "":
        return None
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_float_list_maybe(s):
    if s is None or str(s).strip() == "":
        return None
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def append_rows_to_csv(csv_path, rows, extra):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    flat_rows = []
    for row in rows:
        r = dict(extra)
        r.update(row)
        flat_rows.append(r)

    if len(flat_rows) == 0:
        return

    fieldnames = list(flat_rows[0].keys())
    exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists:
            writer.writeheader()

        for row in flat_rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Run noise robustness experiments for regression_dmd.")

    parser.add_argument("--model", type=str, default="regression_dmd")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--clean_data_path", type=str, required=True)
    parser.add_argument("--noisy_data_path", type=str, required=True)

    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--traj_index", type=int, default=0)
    parser.add_argument("--plot_traj_indices", type=str, default="0,1,2,3")

    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--feedback_noise_std", type=float, default=0.001)
    parser.add_argument(
        "--feedback_rollout_mode",
        type=str,
        default="DMD",
        choices=["DMD", "projected_DMD", "linear_dynamics"],
    )

    parser.add_argument("--max_pairs", type=int, default=5000)
    parser.add_argument(
        "--mode_subset_thresholds",
        type=str,
        default="1,5,10,25,50,100",
        help="Fractions of the sorted mode list (percent) for noise-robustness subset plots.",
    )
    parser.add_argument(
        "--plot_mode_subsets",
        action="store_true",
        help="If set, also save A/B/C/D plots for contribution-selected mode subsets.",
    )
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv_path", type=str, default=None)

    args = parser.parse_args()

    # Allow baseline and learned models alike; modal subset plots are skipped automatically
    # when the model does not expose Phi/Lambda-style coordinates.

    device = "cuda" if torch.cuda.is_available() else "cpu"

    split_path = resolve_split_npz_path(args.clean_data_path, args.split)
    data = np.load(split_path, allow_pickle=True)

    X = data["X"]
    state_dim = X.shape[-1]
    system = str(data["system"])
    dt = float(np.asarray(data["dt"]).item()) if "dt" in data else np.nan

    model, extras = load_model(
        model_name=args.model,
        model_path=args.model_path,
        data_path=split_path,
        state_dim=state_dim,
        system=system,
        device=device,
    )

    run_name = args.name or infer_run_name(args.model_path)

    outdir = os.path.join(
        "data",
        "figures",
        args.model,
        system,
        "noise_robustness",
        run_name,
        f"split_{args.split}",
        f"steps_{args.steps}",
        f"feedback_{args.feedback_rollout_mode}_std_{str(args.feedback_noise_std).replace('.', 'p')}",
    )

    plot_traj_indices = parse_int_list_maybe(args.plot_traj_indices)
    mode_subset_thresholds = parse_float_list_maybe(args.mode_subset_thresholds)

    primary_metrics, rows = run_noise_robustness_suite(
        model=model,
        model_name=args.model,
        extras=extras,
        clean_data_path=args.clean_data_path,
        noisy_data_path=args.noisy_data_path,
        outdir=outdir,
        split=args.split,
        traj_index=args.traj_index,
        plot_traj_indices=plot_traj_indices,
        steps=args.steps,
        noise_std_for_feedback=args.feedback_noise_std,
        feedback_rollout_mode=args.feedback_rollout_mode,
        max_pairs=args.max_pairs,
        mode_subset_thresholds=mode_subset_thresholds,
        plot_mode_subsets=args.plot_mode_subsets,
        seed=args.seed,
    )

    print("\n--- Primary noise robustness metrics ---")
    for k, v in primary_metrics.items():
        print(f"{k:45s}: {v}")

    print("\n--- Variant summary ---")
    for row in rows:
        print(
            f"{row['variant']:25s} | "
            f"modes={row['n_modes_used']:>3} | "
            f"B={row['modal_output_rmse_projected_vs_clean']:.3e} | "
            f"A={row['one_step_rmse_pred_vs_clean_next']:.3e} | "
            f"Cfrac={row['feedback_completed_fraction_mean']:.3f} | "
            f"Cnoise={row.get('feedback_perturbation_rmse_mean', np.nan):.3e} | "
            f"D={row['noisy_initial_free_rollout_rmse']:.3e}"
        )

    csv_path = args.csv_path
    if csv_path is None:
        csv_path = os.path.join(
            "data",
            "figures",
            args.model,
            system,
            "noise_robustness",
            "summary.csv",
        )

    extra = {
        "run_name": run_name,
        "system": system,
        "dt": dt,
        "model_path": args.model_path,
        "clean_data_path": args.clean_data_path,
        "noisy_data_path": args.noisy_data_path,
        "split": args.split,
        "steps": args.steps,
        "feedback_rollout_mode": args.feedback_rollout_mode,
        "feedback_noise_std": args.feedback_noise_std,
    }

    append_rows_to_csv(csv_path, rows, extra)

    print(f"\nSaved figures and metrics to: {outdir}")
    print(f"Appended summary CSV      : {csv_path}")


if __name__ == "__main__":
    main()