import argparse
import os
import numpy as np

from src.eval.eval_runner import (
    MODEL_CHOICES,
    parse_int_list,
    prepare_eval_context,
    save_metadata_json,
    maybe_load_core_summary,
    compute_core_metrics_bundle,
)

from src.eval.diagnostics import run_diagnostics
"""
Behavior diagnostics for a trained model.

Generates heavier local-analysis plots such as dense true-grid heatmaps,
optional phase-space error maps, and rollout-behavior summaries.
Use this when you want to inspect where in state space the model performs well or poorly.
"""
'''
python -m scripts.eval_behavior \
  --model regression_dmd \
  --data_path data/trajectories/nonlinear/vanderpol \
  --model_path data/models/regression_dmd/vanderpol/default/model.npz \
  --split test \
  --metric_horizons 1,2,5,10 \
  --rollout_metric_horizons 5,10 \
  --phase_map_horizons 1,10,50 \
  --sampled_heatmap_horizon 10 \
  --true_grid_horizons 1 \
  --use_cache \
  --run_true_grid_heatmap
'''
def parse_metric_horizon_spec(text: str):
    """
    Parse metric horizon argument.

    Examples
    --------
    "50"      -> [1, 2, ..., 50]
    "1,10,50" -> [1, 10, 50]
    """
    text = text.strip()
    if "," in text:
        return parse_int_list(text)

    max_h = int(text)
    if max_h < 1:
        raise ValueError("metric_horizons must be >= 1")
    return list(range(1, max_h + 1))

def parse_rollout_horizon_spec(text: str):
    """
    Parse rollout horizon argument.

    Examples
    --------
    "50"      -> [1, 2, ..., 50]
    "5,10,50" -> [5, 10, 50]
    """
    text = text.strip()
    if "," in text:
        return parse_int_list(text)

    max_h = int(text)
    if max_h < 1:
        raise ValueError("rollout_metric_horizons must be >= 1")
    return list(range(1, max_h + 1))


def parse_threshold_spec(text: str):
    """Parse contribution thresholds as percentages or fractions.

    Examples
    --------
    "1,50,90"   -> [0.01, 0.5, 0.9]
    "0.01,0.5"  -> [0.01, 0.5]
    "100"       -> [1.0]
    "1"         -> [0.01]
    ""          -> []
    """
    text = text.strip()
    if text == "":
        return []

    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value > 1:
            value = value / 100.0
        values.append(value)
    return values

def main():
    parser = argparse.ArgumentParser(description="Behavior / local analysis diagnostics for trained models.")

    parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--name", type=str, default=None)

    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--traj_index", type=int, default=0)
    
    parser.add_argument("--metric_horizons", type=str, default="100", help="Either a comma-separated list like '1,10,50' or a single max horizon like '50', which expands to 1..50.")    
    parser.add_argument("--rollout_metric_horizons", type=str, default="100", help="Either a comma-separated list like '5,10,50' or a single max horizon like '50', which expands to 1..50.")    
    parser.add_argument("--phase_map_horizons", type=str, default="1,10,50", help="Comma-separated horizons for optional phase-space error maps.")
    parser.add_argument("--sampled_heatmap_horizon", type=int, default=10, help="Horizon used for the sampled-start heatmap.")
    parser.add_argument("--true_grid_horizons", type=str, default="1", help="Comma-separated horizons for dense true-grid heatmaps when enabled.")
    parser.add_argument(
    "--reuse_core_metrics",
    action="store_true",
    help="Reuse the core summary produced by eval.py instead of recomputing one-step, horizon, and rollout metrics.",
)
    parser.add_argument("--heatmap_mode", type=str, default="traj_initials", choices=["traj_initials", "all_valid_starts"])
    parser.add_argument("--linear_error_scale", action="store_true")
    parser.add_argument("--metric_cap", type=int, default=64)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--run_true_grid_heatmap", action="store_true")
    parser.add_argument("--grid_resolution", type=int, default=100)
    parser.add_argument("--skip_phase_maps", action="store_true", help="Skip phase-space error maps.")
    parser.add_argument("--run_sampled_start_heatmap", action="store_true", help="Also generate sampled-start heatmap.")
    parser.add_argument("--no_overlay_true_trajectory_on_grid", action="store_true", help="Disable overlay of the true trajectory on dense true-grid heatmaps.")
    parser.add_argument("--grid_overlay_n_trajs", type=int, default=1, help="Number of trajectories to overlay on true-grid heatmaps.")
    parser.add_argument("--force_linear_true_grid_error_scale", action="store_true", help="Force linear color scaling for the combined true-grid heatmap figure. Default is automatic log/linear selection.")    
    parser.add_argument("--reuse_if_exists", action="store_true", help="If diagnostics outputs already exist, skip recomputation and exit.")
    parser.add_argument("--mode_subset_thresholds", type=str, default="", help="Comma-separated contribution thresholds for mode heatmaps, e.g. '1,50,90' or '0.01,0.5,0.9'.")
    parser.add_argument("--mode_subset_indices", type=str, default="", help="Comma-separated explicit mode indices for manual mode heatmaps.")
    parser.add_argument("--outdir", type=str, default=None, help="Force a custom output directory for all plots and logs.")

    args = parser.parse_args()

    metric_horizons = parse_metric_horizon_spec(args.metric_horizons)
    rollout_metric_horizons = parse_rollout_horizon_spec(args.rollout_metric_horizons)
    phase_map_horizons = parse_int_list(args.phase_map_horizons)
    true_grid_horizons = parse_int_list(args.true_grid_horizons)
    mode_subset_thresholds = parse_threshold_spec(args.mode_subset_thresholds)
    mode_subset_indices = parse_int_list(args.mode_subset_indices) if args.mode_subset_indices.strip() else []

    phase_max = max(phase_map_horizons) if len(phase_map_horizons) > 0 else 0
    grid_max = max(true_grid_horizons) if args.run_true_grid_heatmap else args.sampled_heatmap_horizon
    max_needed = max(max(metric_horizons), max(rollout_metric_horizons), phase_max, args.sampled_heatmap_horizon, grid_max)

    need_cache_initially = args.use_cache and not args.reuse_core_metrics

    ctx = prepare_eval_context(
        args=args,
        split=args.split,
        subdir=f"diagnostics_{args.split}",
        need_cache=need_cache_initially,
        max_horizon_for_cache=max_needed,
        save_run_metadata=False,
    )

    # Force the outputs to route to the custom directory if provided
    if args.outdir:
        ctx.figdir = args.outdir
        os.makedirs(ctx.figdir, exist_ok=True)

    existing_core = maybe_load_core_summary(ctx)
    if existing_core is not None:
        print("[eval_behavior] Found existing core summary in base figure directory.")
    else:
        print("[eval_behavior] No existing core summary found. Proceeding with diagnostics computation.")

    diagnostics_summary_path = os.path.join(ctx.figdir, "diagnostics_summary.npz")

    expected_heatmap_paths = []
    if args.run_true_grid_heatmap:
        expected_heatmap_paths = [os.path.join(ctx.figdir, "true_grid_error_heatmap_grid_full.png")]
        if mode_subset_thresholds:
            expected_heatmap_paths.append(
                os.path.join(ctx.figdir, "true_grid_error_heatmap_grid_with_subsets.png")
            )

    all_expected_exist = os.path.exists(diagnostics_summary_path) and all(
        os.path.exists(p) for p in expected_heatmap_paths
    )

    if args.reuse_if_exists and all_expected_exist:
        print(f"[eval_behavior] Found existing diagnostics summary: {diagnostics_summary_path}")
        if expected_heatmap_paths:
            print("[eval_behavior] All expected true-grid heatmaps also exist.")
        print("[eval_behavior] --reuse_if_exists set, skipping recomputation.")
        return

    if os.path.exists(diagnostics_summary_path):
        print(f"[eval_behavior] Found existing diagnostics summary but recomputing: {diagnostics_summary_path}")
        for p in expected_heatmap_paths:
            if not os.path.exists(p):
                print(f"[eval_behavior] Missing expected heatmap: {p}")
    else:
        print(f"[eval_behavior] No diagnostics summary found at: {diagnostics_summary_path}")

    if args.traj_index >= len(ctx.traj_indices):
        raise IndexError(
            f"traj_index={args.traj_index} but only {len(ctx.traj_indices)} trajectories exist in split '{args.split}'"
        )
    metric_cap = None if args.metric_cap == 0 else args.metric_cap

    def _split_core_summary(summary):
        one_step_metrics = {
            k: v for k, v in summary.items()
            if k.startswith("one_step_")
        }
        horizon_metrics = {
            k: v for k, v in summary.items()
            if k.startswith("horizon_") or k == "horizons"
        }
        rollout_metrics = {
            k: v for k, v in summary.items()
            if k.startswith("rollout_")
        }
        return one_step_metrics, horizon_metrics, rollout_metrics


    def _core_summary_matches_request(summary, metric_horizons, rollout_metric_horizons):
        if "horizons" not in summary or "rollout_horizons" not in summary:
            return False

        saved_horizons = np.asarray(summary["horizons"], dtype=int)
        saved_rollout_horizons = np.asarray(summary["rollout_horizons"], dtype=int)

        return (
            np.array_equal(saved_horizons, np.asarray(metric_horizons, dtype=int))
            and np.array_equal(saved_rollout_horizons, np.asarray(rollout_metric_horizons, dtype=int))
        )


    if args.reuse_core_metrics:
        if existing_core is None:
            raise FileNotFoundError(
                "--reuse_core_metrics was set, but no core summary from eval.py was found. "
                "Run scripts.eval first with matching --split, --horizons, and --rollout_horizons."
            )

        if not _core_summary_matches_request(existing_core, metric_horizons, rollout_metric_horizons):
            raise ValueError(
                "Existing core summary does not match requested metric horizons.\n"
                f"Requested metric_horizons={metric_horizons}, "
                f"rollout_metric_horizons={rollout_metric_horizons}.\n"
                f"Saved horizons={np.asarray(existing_core.get('horizons', []), dtype=int).tolist()}, "
                f"saved rollout_horizons={np.asarray(existing_core.get('rollout_horizons', []), dtype=int).tolist()}.\n"
                "Either rerun scripts.eval with matching horizons or remove --reuse_core_metrics."
            )

        print("[eval_behavior] Reusing core metrics from eval.py summary.")
        summary = dict(existing_core)
        one_step_metrics, horizon_metrics, rollout_metrics = _split_core_summary(summary)

    else:
        if args.reuse_core_metrics:
            print("[eval_behavior] No reusable matching core summary found. Computing core metrics instead.")

        if args.use_cache and ctx.rollout_cache is None:
            ctx = prepare_eval_context(
                args=args,
                split=args.split,
                subdir=f"diagnostics_{args.split}",
                need_cache=True,
                max_horizon_for_cache=max_needed,
            )

            if args.outdir:
                ctx.figdir = args.outdir
                os.makedirs(ctx.figdir, exist_ok=True)

        print("Computing core metrics...")
        summary, _ = compute_core_metrics_bundle(
            ctx,
            horizons=metric_horizons,
            rollout_horizons=rollout_metric_horizons,
        )

        one_step_metrics, horizon_metrics, rollout_metrics = _split_core_summary(summary)

    # Save a local diagnostics copy either way, so future behavior runs can find it.
    np.savez(os.path.join(ctx.figdir, "diagnostics_summary.npz"), **summary)

    run_diagnostics(
        X=ctx.X,
        split_idx=ctx.traj_indices,
        traj_id=ctx.traj_indices[args.traj_index],
        model_name=args.model,
        model=ctx.model,
        extras=ctx.extras,
        system=ctx.system,
        figdir=ctx.figdir,
        horizon_metrics=horizon_metrics,
        rollout_metrics=rollout_metrics,
        phase_horizons=phase_map_horizons,
        heatmap_horizon=args.sampled_heatmap_horizon,
        heatmap_mode=args.heatmap_mode,
        linear_error_scale=args.linear_error_scale,
        rollout_cache=ctx.rollout_cache,
        data_path=ctx.split_data_path,
        run_true_grid_heatmap=args.run_true_grid_heatmap,
        grid_resolution=args.grid_resolution,
        true_grid_heatmap_horizons=true_grid_horizons,
        run_phase_maps=not args.skip_phase_maps,
        run_sampled_start_heatmap=args.run_sampled_start_heatmap,
        overlay_true_trajectory_on_grid=not args.no_overlay_true_trajectory_on_grid,
        grid_overlay_n_trajs=args.grid_overlay_n_trajs,
        mode_subset_thresholds=mode_subset_thresholds,
        mode_subset_indices=mode_subset_indices,
        force_linear_true_grid_error_scale=args.force_linear_true_grid_error_scale,
    )

    save_metadata_json(
        ctx,
        os.path.join(ctx.figdir, "metadata.json"),
        extra={
            "metric_horizons": metric_horizons,
            "rollout_metric_horizons": rollout_metric_horizons,
            "phase_map_horizons": phase_map_horizons,
            "sampled_heatmap_horizon": args.sampled_heatmap_horizon,
            "heatmap_mode": args.heatmap_mode,
            "true_grid_horizons": true_grid_horizons,
            "run_phase_maps": not args.skip_phase_maps,
            "run_sampled_start_heatmap": args.run_sampled_start_heatmap,
            "overlay_true_trajectory_on_grid": not args.no_overlay_true_trajectory_on_grid,
            "grid_overlay_n_trajs": args.grid_overlay_n_trajs,
            "mode_subset_thresholds": mode_subset_thresholds,
            "mode_subset_indices": mode_subset_indices,
        },
    )

    print(f"Saved behavior diagnostics to: {ctx.figdir}")


if __name__ == "__main__":
    main()