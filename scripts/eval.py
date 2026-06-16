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
        ml_dmd,
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
from src.eval.diagnostics import format_model_label
from src.eval.rollout_eval import compute_single_rollout
from src.eval.plot_rollout import (
    plot_combined_rollout,
    plot_combined_rollout_with_reference,
    plot_time_series,
    plot_phase_space,
    plot_time_series_with_reference,
    plot_phase_space_with_reference,
)

def print_core_summary(one_step_metrics, horizon_metrics, rollout_metrics, composite_score):
    print("\n--- Test metric summary ---")
    print(f"One-step MSE              : {float(one_step_metrics['one_step_mse']):.6e}")
    print(f"One-step RMSE             : {float(one_step_metrics['one_step_rmse']):.6e}")

    print(f"Mean horizon RMSE         : {float(np.mean(horizon_metrics['horizon_rmse'])):.6e}")
    for h, rmse in zip(
        horizon_metrics["horizons"],
        horizon_metrics["horizon_rmse"],
    ):
        print(f"  Horizon h={int(h):>3d}        : RMSE={float(rmse):.6e}")

    print(f"Mean rollout RMSE         : {float(np.mean(rollout_metrics['rollout_rmse'])):.6e}")
    for h, rmse in zip(
        rollout_metrics["rollout_horizons"],
        rollout_metrics["rollout_rmse"],
    ):
        print(f"  Rollout h={int(h):>3d}        : RMSE={float(rmse):.6e}")

    print(f"Composite test score      : {composite_score:.6e}")


def save_rollout_example_npz(
    out_path,
    *,
    X_true,
    X_hat,
    traj_index,
    steps,
    model_name,
    system,
    X_reference=None,
    reference_label=None,
):
    payload = dict(
        X_true=X_true,
        X_hat=X_hat,
        traj_index=np.array(traj_index),
        steps=np.array(steps),
        model_name=np.array(model_name),
        system=np.array(system),
    )

    if X_reference is not None:
        payload["X_reference"] = X_reference
        payload["reference_label"] = np.array(reference_label or "reference")

    np.savez(out_path, **payload)

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

def load_reference_segment(reference_data_path, split, traj_id, n_points):
    """
    Load matching clean/reference trajectory for overlay plots.
    Assumes same seed, dt, split, and trajectory ordering.
    """
    ref_path = os.path.join(reference_data_path, f"{split}.npz")

    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference split not found: {ref_path}")

    data = np.load(ref_path, allow_pickle=True)
    X_ref = data["X"]

    if X_ref.ndim == 2:
        X_ref = X_ref[:, None, :]
    elif X_ref.ndim != 3:
        raise ValueError(f"Expected reference X to be 2D or 3D, got shape {X_ref.shape}")

    if traj_id >= X_ref.shape[1]:
        raise IndexError(
            f"traj_id={traj_id}, but reference data only has {X_ref.shape[1]} trajectories."
        )

    if n_points > X_ref.shape[0]:
        raise ValueError(
            f"Reference trajectory too short: need {n_points}, has {X_ref.shape[0]}"
        )

    return X_ref[:n_points, traj_id, :]

def main():
    parser = argparse.ArgumentParser(description="Core evaluation for trained models.")

    parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES, help="Model type to evaluate.")
    parser.add_argument("--data_path", type=str, required=True, help="Base path to the dataset (without split suffix).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved trained model checkpoint.")
    parser.add_argument("--name", type=str, default=None, help="Optional run name override for output folders.")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Dataset split to evaluate on.")
    parser.add_argument("--horizons", type=str, default="1,10,100", help="Comma-separated terminal prediction horizons.")
    parser.add_argument("--rollout_horizons", type=str, default="10,100", help="Comma-separated rollout horizons.")
    parser.add_argument("--steps", type=int, default=100, help="Steps for the standard rollout plot.")
    parser.add_argument("--traj_index", type=int, default=0, help="Which trajectory to plot.")
    parser.add_argument("--reference_data_path",type=str,default=None,help="Optional clean/reference dataset path for overlaying rollout plots.")

    parser.add_argument("--metric_cap",type=int,default=64,help="Cap on sampled start points per trajectory for metrics. Use 0 for all.")
    parser.add_argument("--use_cache",action="store_true",help="Reuse rollout cache across metric computations in memory.",)
    parser.add_argument("--save_rollout_cache",action="store_true",help="Optionally save rollout cache to disk for debugging or reuse.")
    parser.add_argument("--save_rollout_arrays",action="store_true",help="Save example rollout arrays as NPZ in addition to the plots.")
    parser.add_argument("--outdir", type=str, default=None, help="Force a custom output directory for all plots and logs.")
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

    if args.outdir:
        ctx.figdir = args.outdir
        os.makedirs(ctx.figdir, exist_ok=True)

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
    model_label = format_model_label(args.model, ctx.model, ctx.extras, system=ctx.system)
    X_true, X_hat = compute_single_rollout(
        X=ctx.X,
        traj_id=traj_id,
        steps=args.steps,
        model_name=args.model,
        model=ctx.model,
        extras=ctx.extras,
    )

    # Save rollout plots into a dedicated sibling 'rollout' folder when the
    # evaluation outputs are rooted under 'data'. Otherwise keep the local subfolder.
    if os.path.basename(os.path.normpath(ctx.figdir)) == "data":
        rollout_figdir = os.path.join(os.path.dirname(os.path.normpath(ctx.figdir)), "rollout")
    else:
        rollout_figdir = os.path.join(ctx.figdir, "rollout")
    os.makedirs(rollout_figdir, exist_ok=True)

    X_ref = None
    if args.reference_data_path is not None:
        X_ref = load_reference_segment(
            reference_data_path=args.reference_data_path,
            split=args.split,
            traj_id=traj_id,
            n_points=X_true.shape[0],
        )

    if X_true.shape[1] == 2:
        if args.reference_data_path is None:
            plot_combined_rollout(
                X_true,
                X_hat,
                rollout_figdir,
                args.traj_index,
                model_label=model_label,
                system=ctx.system,
            )
        else:
            plot_combined_rollout_with_reference(
                X_true,
                X_hat,
                X_ref,
                rollout_figdir,
                args.traj_index,
                true_label="Noisy observed",
                ref_label="Clean true",
                model_label=model_label,
                system=ctx.system,
            )

    elif args.reference_data_path is None:
        plot_time_series(X_true, X_hat, rollout_figdir, args.traj_index, model_label=model_label, system=ctx.system)
        plot_phase_space(X_true, X_hat, ctx.system, rollout_figdir, args.model, args.traj_index, model_label=model_label)

    else:
        plot_time_series_with_reference(
            X_true,
            X_hat,
            X_ref,
            rollout_figdir,
            args.traj_index,
            true_label="Noisy observed",
            ref_label="Clean true",
            model_label=model_label,
            system=ctx.system,
        )

        plot_phase_space_with_reference(
            X_true,
            X_hat,
            X_ref,
            ctx.system,
            rollout_figdir,
            args.model,
            args.traj_index,
            true_label="Noisy observed",
            ref_label="Clean true",
            model_label=model_label,
        )

    if args.save_rollout_arrays:
        rollout_npz_path = os.path.join(ctx.figdir, f"rollout_example_idx{args.traj_index}.npz")
        save_rollout_example_npz(
            rollout_npz_path,
            X_true=X_true,
            X_hat=X_hat,
            X_reference=X_ref,
            reference_label="Clean true" if X_ref is not None else None,
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

