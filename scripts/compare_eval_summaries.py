import argparse
import glob
import os

import numpy as np
'''
# Compare all saved validation summaries for saddle_point across models/runs
python -m scripts.compare_eval_summaries --pattern "data/figures/*/saddle_point/*/diagnostics/diagnostics_summary.npz"

# Print the main validation numbers from one saved diagnostics summary
python -c "import numpy as np; d=np.load(r'data/figures/manual_expansion_manual_dmd/saddle_point/default/diagnostics/diagnostics_summary.npz', allow_pickle=True); print('score=', float(d['composite_validation_score'])); print('one_step_nrmse=', float(d['one_step_nrmse'])); print('mean_horizon_nrmse=', float(d['horizon_nrmse'].mean())); print('mean_rollout_nrmse=', float(d['rollout_nrmse'].mean()))"
'''

def scalar(x):
    arr = np.asarray(x)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(()).item()
    return arr


def main():
    parser = argparse.ArgumentParser(description="Compare saved diagnostics_summary.npz files.")
    parser.add_argument(
        "--pattern",
        type=str,
        required=True,
        help="Glob pattern, e.g. data/figures/*/lorenz/*/diagnostics/diagnostics_summary.npz",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default="composite_validation_score",
        choices=[
            "composite_validation_score",
            "one_step_nrmse",
            "one_step_rmse",
        ],
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise ValueError(f"No files matched pattern: {args.pattern}")

    rows = []
    for path in paths:
        data = np.load(path, allow_pickle=True)

        rows.append({
            "model": str(scalar(data["model_name"])),
            "system": str(scalar(data["system"])),
            "run": str(scalar(data["run_name"])),
            "score": float(scalar(data["composite_validation_score"])),
            "one_step_nrmse": float(scalar(data["one_step_nrmse"])),
            "one_step_rmse": float(scalar(data["one_step_rmse"])),
            "mean_horizon_nrmse": float(np.mean(data["horizon_nrmse"])),
            "mean_rollout_nrmse": float(np.mean(data["rollout_nrmse"])),
            "path": path,
        })

    key_map = {
        "composite_validation_score": "score",
        "one_step_nrmse": "one_step_nrmse",
        "one_step_rmse": "one_step_rmse",
    }

    rows = sorted(rows, key=lambda r: r[key_map[args.sort_by]])

    print()
    print(f"{'model':30s} {'run':20s} {'score':>12s} {'1step_nrmse':>14s} {'mean_h_nrmse':>14s} {'mean_roll_nrmse':>16s}")
    print("-" * 120)
    for r in rows:
        print(
            f"{r['model'][:30]:30s} "
            f"{r['run'][:20]:20s} "
            f"{r['score']:12.6e} "
            f"{r['one_step_nrmse']:14.6e} "
            f"{r['mean_horizon_nrmse']:14.6e} "
            f"{r['mean_rollout_nrmse']:16.6e}"
        )

    print("\nBest file:")
    print(rows[0]["path"])


if __name__ == "__main__":
    main()