"""
Core evaluation script for a trained model.

Computes one-step, horizon, and rollout metrics on a chosen split,
saves a lightweight summary and metadata, and generates one standard rollout plot.

Global options (defaults):
    --model {
        linear_baseline,
        dmd_baseline,
        regression_dmd,
        ml_lineardynamics,
        ml_dmd_free,
        ml_dmd_band,
        sindy_baseline}
    --data_path data/trajectories/{linear|nonlinear}/{system}
    --model_path data/models/{model}/{system}/{run_name}/model.{npz|pt}
    --steps 500
    --traj_index 0
    --name optional_suffix
    --horizons 1,5
    --rollout_horizons 5
"""
import argparse
import os
import numpy as np

from src.eval.eval_runner import (
    MODEL_CHOICES,
    parse_int_list,
    prepare_eval_context,
    compute_core_metrics_bundle,
    save_summary,
    save_metadata_json,
)
from src.eval.rollout_eval import compute_single_rollout
from src.eval.plot_rollout import plot_time_series, plot_phase_space


def print_core_summary(one_step_metrics, horizon_metrics, rollout_metrics, composite_score):
    print("\n--- Test metric summary ---")
    print(f"One-step MSE              : {float(one_step_metrics['one_step_mse']):.6e}")
    print(f"One-step RMSE             : {float(one_step_metrics['one_step_rmse']):.6e}")
    print(f"One-step NRMSE            : {float(one_step_metrics['one_step_nrmse']):.6e}")

    print(f"Mean horizon RMSE         : {float(np.mean(horizon_metrics['horizon_rmse'])):.6e}")
    print(f"Mean horizon NRMSE        : {float(np.mean(horizon_metrics['horizon_nrmse'])):.6e}")
    for h, rmse, nrmse in zip(
        horizon_metrics["horizons"],
        horizon_metrics["horizon_rmse"],
        horizon_metrics["horizon_nrmse"],
    ):
        print(f"  Horizon h={int(h):>3d}        : RMSE={float(rmse):.6e}, NRMSE={float(nrmse):.6e}")

    print(f"Mean rollout RMSE         : {float(np.mean(rollout_metrics['rollout_rmse'])):.6e}")
    print(f"Mean rollout NRMSE        : {float(np.mean(rollout_metrics['rollout_nrmse'])):.6e}")
    for h, rmse, nrmse in zip(
        rollout_metrics["rollout_horizons"],
        rollout_metrics["rollout_rmse"],
        rollout_metrics["rollout_nrmse"],
    ):
        print(f"  Rollout h={int(h):>3d}        : RMSE={float(rmse):.6e}, NRMSE={float(nrmse):.6e}")

    print(f"Composite test score      : {composite_score:.6e}")


def save_rollout_example_npz(out_path, *, X_true, X_hat, traj_index, steps, model_name, system):
    np.savez(
        out_path,
        X_true=X_true,
        X_hat=X_hat,
        traj_index=np.array(traj_index),
        steps=np.array(steps),
        model_name=np.array(model_name),
        system=np.array(system),
    )


def save_optional_cache_npz(out_path, rollout_cache):
    if rollout_cache is None:
        return

    payload = {}
    traj_ids = sorted(list(rollout_cache.keys()))
    payload["traj_ids"] = np.asarray(traj_ids, dtype=int)

    starts_obj = np.empty(len(traj_ids), dtype=object)
    rollouts_obj = np.empty(len(traj_ids), dtype=object)

    for i, traj_id in enumerate(traj_ids):
        starts_obj[i] = np.asarray(rollout_cache[traj_id]["starts"])
        rollouts_obj[i] = np.asarray(rollout_cache[traj_id]["rollouts"], dtype=object)

    payload["starts"] = starts_obj
    payload["rollouts"] = rollouts_obj
    np.savez(out_path, **payload)


def main():
    parser = argparse.ArgumentParser(description="Core evaluation for trained models.")

    parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES, help="Model type to evaluate.")
    parser.add_argument("--data_path", type=str, required=True, help="Base path to the dataset (without split suffix).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved trained model checkpoint.")
    parser.add_argument("--name", type=str, default=None, help="Optional run name override for output folders.")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Dataset split to evaluate on.")
    parser.add_argument("--horizons", type=str, default="1,10,50", help="Comma-separated terminal prediction horizons.")
    parser.add_argument("--rollout_horizons", type=str, default="10,50", help="Comma-separated rollout horizons.")
    parser.add_argument("--steps", type=int, default=200, help="Steps for the standard rollout plot.")
    parser.add_argument("--traj_index", type=int, default=0, help="Which trajectory to plot.")


    parser.add_argument("--metric_cap",type=int,default=64,help="Cap on sampled start points per trajectory for metrics. Use 0 for all.")
    parser.add_argument("--use_cache",action="store_true",help="Reuse rollout cache across metric computations in memory.",)
    parser.add_argument("-save_rollout_cache",action="store_true",help="Optionally save rollout cache to disk for debugging or reuse.")
    parser.add_argument("--save_rollout_arrays",action="store_true",help="Save example rollout arrays as NPZ in addition to the plots.")

    args = parser.parse_args()

    horizons = parse_int_list(args.horizons)
    rollout_horizons = parse_int_list(args.rollout_horizons)
    max_needed = max(max(horizons), max(rollout_horizons))

    ctx = prepare_eval_context(
        args=args,
        split=args.split,
        subdir=None,
        need_cache=args.use_cache,
        max_horizon_for_cache=max_needed,
    )

    if args.traj_index >= len(ctx.traj_indices):
        raise IndexError(
            f"traj_index={args.traj_index} but only {len(ctx.traj_indices)} trajectories exist in split '{args.split}'"
        )

    if args.model == "regression_dmd" and "rollout_mode" in ctx.extras:
        print(f"Regression_DMD rollout mode: {ctx.extras['rollout_mode']}")

    summary, rollout_cache = compute_core_metrics_bundle(
        ctx,
        horizons=horizons,
        rollout_horizons=rollout_horizons,
    )

    one_step_metrics = {k: v for k, v in summary.items() if k.startswith("one_step_")}
    horizon_metrics = {k: v for k, v in summary.items() if k.startswith("horizon_") or k == "horizons"}
    rollout_metrics = {k: v for k, v in summary.items() if k.startswith("rollout_")}
    composite_score = float(summary["composite_score"])

    print_core_summary(one_step_metrics, horizon_metrics, rollout_metrics, composite_score)

    summary_payload = {
        "model_name": np.array(args.model),
        "system": np.array(ctx.system),
        "run_name": np.array(ctx.run_name),
        "split": np.array(args.split),
        "traj_indices": np.asarray(ctx.traj_indices),
        "scale_std": ctx.scale_std,
        **summary,
    }

    summary_path = os.path.join(ctx.figdir, f"{args.split}_summary.npz")
    save_summary(summary_payload, summary_path)
    print(f"Saved core summary        : {summary_path}")

    save_metadata_json(
        ctx,
        os.path.join(ctx.figdir, "metadata.json"),
        extra={
            "horizons": horizons,
            "rollout_horizons": rollout_horizons,
            "steps": args.steps,
            "traj_index": args.traj_index,
            "metric_cap": args.metric_cap,
            "used_cache": bool(args.use_cache),
        },
    )

    traj_id = ctx.traj_indices[args.traj_index]
    X_true, X_hat = compute_single_rollout(
        X=ctx.X,
        traj_id=traj_id,
        steps=args.steps,
        model_name=args.model,
        model=ctx.model,
        extras=ctx.extras,
    )

    plot_time_series(X_true, X_hat, ctx.figdir, args.traj_index)
    plot_phase_space(X_true, X_hat, ctx.system, ctx.figdir, args.model, args.traj_index)

    if args.save_rollout_arrays:
        rollout_npz_path = os.path.join(ctx.figdir, f"rollout_example_idx{args.traj_index}.npz")
        save_rollout_example_npz(
            rollout_npz_path,
            X_true=X_true,
            X_hat=X_hat,
            traj_index=args.traj_index,
            steps=args.steps,
            model_name=args.model,
            system=ctx.system,
        )
        print(f"Saved rollout arrays      : {rollout_npz_path}")

    if args.save_rollout_cache and rollout_cache is not None:
        cache_path = os.path.join(ctx.figdir, f"{args.split}_rollout_cache.npz")
        save_optional_cache_npz(cache_path, rollout_cache)
        print(f"Saved rollout cache       : {cache_path}")


if __name__ == "__main__":
    main()

