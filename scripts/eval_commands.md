# Evaluation workflow

This project uses a small set of evaluation scripts with different purposes:

- `scripts.eval`  
  Core evaluation for one trained model. Computes one-step, horizon, and rollout metrics, saves a lightweight summary, and generates one standard rollout plot.

- `scripts.eval_behavior`  
  Heavier local diagnostics. Generates dense true-grid heatmaps, optional phase-space error maps, and rollout-behavior summaries.

- `scripts.eval_spectral`  
  Spectral and mode analysis. Saves a spectral summary and produces eigenvalue / transition-matrix plots.

- `scripts.eval_training`  
  Post-hoc analysis of saved training curves. Reads `losses.npz` and plots training / validation losses.

- `scripts.eval_compare`  
  Compares saved summary files from previous eval runs without recomputing predictions.

## Recommended quick smoke tests

For most models, start with:
- one linear system: `saddle_point`
- one nonlinear system: `vanderpol`

This keeps the workflow quick while still testing both simple and nonlinear behavior.

## Model guidance

- `linear_baseline`, `dmd_baseline`  
  Best interpreted on linear systems. They can be run on nonlinear systems as weak baselines, but performance may be poor.

- `regression_dmd`, `ml_lineardynamics`, `ml_dmd`, `sindy_baseline`  
  Test on both one linear and one nonlinear system.

## Typical workflow

1. Train a model with `scripts.train`
2. Run `scripts.eval`
3. Run `scripts.eval_spectral`
4. Run `scripts.eval_behavior`
5. Run `scripts.eval_training` for neural models
6. Run `scripts.eval_compare` to compare saved summaries

---

# Smoke test commands

## ML-DMD: train

```bash
python -m scripts.train --model ml_dmd --data_path data/trajectories/linear/saddle_point --name saddle_mldmd_smoketest --epochs 10 --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5
python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/vanderpol --name vdp_mldmd_smoketest --epochs 10 --expansion_type specific --expansion_degree 10 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4
```

# Eval: ML-DMD

```bash
python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/ml_dmd/saddle_point/saddle_mldmd_smoketest/model.pt --split test --horizons 1,2,5,10 --rollout_horizons 5,10 --steps 200 --traj_index 0 --use_cache --save_rollout_arrays
python -m scripts.eval_spectral --model ml_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/ml_dmd/saddle_point/saddle_mldmd_smoketest/model.pt --split test --plot_eigs --plot_matrix
python -m scripts.eval_behavior --model ml_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/ml_dmd/saddle_point/saddle_mldmd_smoketest/model.pt --split test --horizons 1,2,5,10 --rollout_horizons 5,10 --phase_horizons 1,5,10 --heatmap_horizon 10 --true_grid_heatmap_horizons 1,10 --use_cache --run_true_grid_heatmap --skip_phase_maps
python -m scripts.eval_training --model_path data/models/ml_dmd/saddle_point/saddle_mldmd_smoketest/model.pt

python -m scripts.eval --model ml_dmd --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/ml_dmd/vanderpol/vdp_mldmd_smoketest/model.pt --split test --horizons 1,2,5,10 --rollout_horizons 5,10 --steps 200 --traj_index 0 --use_cache --save_rollout_arrays
python -m scripts.eval_spectral --model ml_dmd --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/ml_dmd/vanderpol/vdp_mldmd_smoketest/model.pt --split test --plot_eigs --plot_matrix
python -m scripts.eval_behavior --model ml_dmd --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/ml_dmd/vanderpol/vdp_mldmd_smoketest/model.pt --split test --horizons 1,2,5,10 --rollout_horizons 5,10 --phase_horizons 1,10,50 --heatmap_horizon 10 --true_grid_heatmap_horizons 1,10 --use_cache --run_true_grid_heatmap --skip_phase_maps
python -m scripts.eval_training --model_path data/models/ml_dmd/vanderpol/vdp_mldmd_smoketest/model.pt

python -m scripts.eval_compare --summary_paths data/figures/ml_dmd/saddle_point/saddle_mldmd_smoketest/test_summary.npz data/figures/ml_dmd/vanderpol/vdp_mldmd_smoketest/test_summary.npz --labels saddle_point vanderpol --metric composite_score --title "ml_dmd smoke test: saddle vs vanderpol"
```

# regression DMD vs DMD baseline train

```bash
python -m scripts.train --model dmd_baseline --data_path data/trajectories/linear/saddle_point --name saddle_dmd_new --ridge 1e-8
python -m scripts.train --model regression_dmd --data_path data/trajectories/linear/saddle_point --name saddle_regdmd_new_DMD --bias true --ridge 1e-8 --regression_rollout_mode DMD

python -m scripts.train --model dmd_baseline --data_path data/trajectories/nonlinear/vanderpol --name vdp_dmd_new --ridge 1e-8
python -m scripts.train --model regression_dmd --data_path data/trajectories/nonlinear/vanderpol --name vdp_regdmd_new_projectedDMD --expansion_type specific --expansion_degree 10 --bias true --ridge 1e-8 --regression_rollout_mode projected_DMD

```

# Eval
```bash
python -m scripts.eval --model dmd_baseline --data_path data/trajectories/linear/saddle_point --model_path data/models/dmd_baseline/saddle_point/saddle_dmd_new/model.npz --split test --horizons 1,2,5,10 --rollout_horizons 5,10 --steps 200 --traj_index 0 --use_cache --save_rollout_arrays
python -m scripts.eval_spectral --model dmd_baseline --data_path data/trajectories/linear/saddle_point --model_path data/models/dmd_baseline/saddle_point/saddle_dmd_new/model.npz --split test --plot_eigs --plot_matrix
python -m scripts.eval_behavior --model dmd_baseline --data_path data/trajectories/linear/saddle_point --model_path data/models/dmd_baseline/saddle_point/saddle_dmd_new/model.npz --split test --horizons 1,2,5,10 --rollout_horizons 5,10 --phase_horizons 1,5,10 --heatmap_horizon 10 --true_grid_heatmap_horizons 1,10 --use_cache --run_true_grid_heatmap --skip_phase_maps

python -m scripts.eval --model regression_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/regression_dmd/saddle_point/saddle_regdmd_new_DMD/model.npz --split test --horizons 1,2,5,10 --rollout_horizons 5,10 --steps 200 --traj_index 0 --use_cache --save_rollout_arrays
python -m scripts.eval_spectral --model regression_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/regression_dmd/saddle_point/saddle_regdmd_new_DMD/model.npz --split test --plot_eigs --plot_matrix
python -m scripts.eval_behavior --model regression_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/regression_dmd/saddle_point/saddle_regdmd_new_DMD/model.npz --split test --horizons 1,2,5,10 --rollout_horizons 5,10 --phase_horizons 1,5,10 --heatmap_horizon 10 --true_grid_heatmap_horizons 1,10 --use_cache --run_true_grid_heatmap --skip_phase_maps

python -m scripts.eval_compare --summary_paths data/figures/dmd_baseline/saddle_point/saddle_dmd_new/test_summary.npz data/figures/regression_dmd/saddle_point/saddle_regdmd_new_DMD/test_summary.npz --labels dmd_baseline regression_dmd --metric composite_score --title "Saddle-point: dmd_baseline vs regression_dmd"

# old docstring from document:
'''

---------------------------------------------------------------------------------------------

# Linear baseline
    python -m scripts.eval --model linear_baseline --data_path data/trajectories/linear/saddle_point --model_path data/models/linear_baseline/saddle_point/default/model.npz
    python -m scripts.eval --model linear_baseline --data_path data/trajectories/linear/degenerate_node --model_path data/models/linear_baseline/degenerate_node/default/model.npz
    python -m scripts.eval --model linear_baseline --data_path data/trajectories/linear/inward_spiral --model_path data/models/linear_baseline/inward_spiral/default/model.npz
    python -m scripts.eval --model linear_baseline --data_path data/trajectories/linear/harmonic_oscillator --model_path data/models/linear_baseline/harmonic_oscillator/default/model.npz

# DMD baseline
    python -m scripts.eval --model dmd_baseline --data_path data/trajectories/linear/saddle_point --model_path data/models/dmd_baseline/saddle_point/default/model.npz
    python -m scripts.eval --model dmd_baseline --data_path data/trajectories/linear/degenerate_node --model_path data/models/dmd_baseline/degenerate_node/default/model.npz
    python -m scripts.eval --model dmd_baseline --data_path data/trajectories/linear/inward_spiral --model_path data/models/dmd_baseline/inward_spiral/default/model.npz
    python -m scripts.eval --model dmd_baseline --data_path data/trajectories/linear/harmonic_oscillator --model_path data/models/dmd_baseline/harmonic_oscillator/default/model.npz

---------------------------------------------------------------------------------------------

# Regression DMD
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/regression_dmd/saddle_point/default/model.npz
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/linear/degenerate_node --model_path data/models/regression_dmd/degenerate_node/default/model.npz
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/linear/inward_spiral --model_path data/models/regression_dmd/inward_spiral/default/model.npz
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/linear/harmonic_oscillator --model_path data/models/regression_dmd/harmonic_oscillator/default/model.npz

    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/regression_dmd/vanderpol/default/model.npz
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/lotka_volterra --model_path data/models/regression_dmd/lotka_volterra/default/model.npz
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/pendulum --model_path data/models/regression_dmd/pendulum/default/model.npz
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/duffing --model_path data/models/regression_dmd/duffing/default/model.npz
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/lorenz --model_path data/models/regression_dmd/lorenz/default/model.npz
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/koopman_poly --model_path data/models/regression_dmd/koopman_poly/default/model.npz
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/koopman_poly_large --model_path data/models/regression_dmd/koopman_poly_large/default/model.npz
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/koopman_poly_trig --model_path data/models/regression_dmd/koopman_poly_trig/default/model.npz
    
    # Final test evaluation + also print the saved validation diagnostics summary for the same run
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/regression_dmd/saddle_point/default/model.npz --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10

# ML Linear Dynamics
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/linear/saddle_point --model_path data/models/ml_lineardynamics/saddle_point/default/model.pt
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/linear/degenerate_node --model_path data/models/ml_lineardynamics/degenerate_node/default/model.pt
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/linear/inward_spiral --model_path data/models/ml_lineardynamics/inward_spiral/default/model.pt
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/linear/harmonic_oscillator --model_path data/models/ml_lineardynamics/harmonic_oscillator/default/model.pt

    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/nonlinear/koopman_poly --model_path data/models/ml_lineardynamics/koopman_poly/default/model.pt
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/nonlinear/koopman_poly_large --model_path data/models/ml_lineardynamics/koopman_poly_large/default/model.pt
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/nonlinear/koopman_poly_trig --model_path data/models/ml_lineardynamics/koopman_poly_trig/default/model.pt

    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/ml_lineardynamics/vanderpol/default/model.pt
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/nonlinear/lotka_volterra --model_path data/models/ml_lineardynamics/lotka_volterra/default/model.pt
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/nonlinear/pendulum --model_path data/models/ml_lineardynamics/pendulum/default/model.pt
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/nonlinear/duffing --model_path data/models/ml_lineardynamics/duffing/default/model.pt
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/nonlinear/lorenz --model_path data/models/ml_lineardynamics/lorenz/default/model.pt

# ML DMD
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/ml_dmd/saddle_point/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/degenerate_node --model_path data/models/ml_dmd/degenerate_node/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/inward_spiral --model_path data/models/ml_dmd/inward_spiral/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/harmonic_oscillator --model_path data/models/ml_dmd/harmonic_oscillator/default/model.pt

    python -m scripts.eval --model ml_dmd --data_path data/trajectories/nonlinear/koopman_poly --model_path data/models/ml_dmd/koopman_poly/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/nonlinear/koopman_poly_large --model_path data/models/ml_dmd/koopman_poly_large/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/nonlinear/koopman_poly_trig --model_path data/models/ml_dmd/koopman_poly_trig/default/model.pt

    python -m scripts.eval --model ml_dmd --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/ml_dmd/vanderpol/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/nonlinear/lotka_volterra --model_path data/models/ml_dmd/lotka_volterra/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/nonlinear/pendulum --model_path data/models/ml_dmd/pendulum/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/nonlinear/duffing --model_path data/models/ml_dmd/duffing/default/model.pt
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/nonlinear/lorenz --model_path data/models/ml_dmd/lorenz/default/model.pt

# SINDy baseline
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/linear/saddle_point --model_path data/models/sindy_baseline/saddle_point/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/linear/degenerate_node --model_path data/models/sindy_baseline/degenerate_node/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/linear/inward_spiral --model_path data/models/sindy_baseline/inward_spiral/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/linear/harmonic_oscillator --model_path data/models/sindy_baseline/harmonic_oscillator/default/model.npz

    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/sindy_baseline/vanderpol/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/lotka_volterra --model_path data/models/sindy_baseline/lotka_volterra/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/pendulum --model_path data/models/sindy_baseline/pendulum/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/duffing --model_path data/models/sindy_baseline/duffing/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/lorenz --model_path data/models/sindy_baseline/lorenz/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/koopman_poly --model_path data/models/sindy_baseline/koopman_poly/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/koopman_poly_large --model_path data/models/sindy_baseline/koopman_poly_large/default/model.npz
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/nonlinear/koopman_poly_trig --model_path data/models/sindy_baseline/koopman_poly_trig/default/model.npz
    
# Final test evaluation + print matching validation summary + save test_summary.npz.
# Add --run_diagnostics to also generate the deeper diagnostic plots on the test split.
# Saddle-point example:
    python -m scripts.eval --model linear_baseline --data_path data/trajectories/linear/saddle_point --model_path data/models/linear_baseline/saddle_point/default/model.npz --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10 --run_diagnostics --phase_horizons 1,5 --heatmap_horizon 5
    python -m scripts.eval --model dmd_baseline --data_path data/trajectories/linear/saddle_point --model_path data/models/dmd_baseline/saddle_point/default/model.npz --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10 --run_diagnostics --phase_horizons 1,5 --heatmap_horizon 5
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/regression_dmd/saddle_point/default/model.npz --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10 --run_diagnostics --phase_horizons 1,5 --heatmap_horizon 5
    python -m scripts.eval --model ml_lineardynamics --data_path data/trajectories/linear/saddle_point --model_path data/models/ml_lineardynamics/saddle_point/default/model.pt --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10 --run_diagnostics --phase_horizons 1,5 --heatmap_horizon 5
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/ml_dmd/saddle_point/default/model.pt --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10 --run_diagnostics --phase_horizons 1,5 --heatmap_horizon 5
    python -m scripts.eval --model sindy_baseline --data_path data/trajectories/linear/saddle_point --model_path data/models/sindy_baseline/saddle_point/default/model.npz --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10 --run_diagnostics --phase_horizons 1,5 --heatmap_horizon 5
--------------------------------------------------------------------------------------------
# Additional diagnostics examples
# --------------------------------
# --run_diagnostics generates the standard test-split diagnostics:
#    * error-vs-horizon plot
#    * phase-space error map(s)
#    * rollout error summary
#
# --run_true_grid_heatmap adds a dense error heatmap over a regular grid of initial states.
# This is a "true simulator vs trained model" comparison, so it is more global than the
# sampled-start initial-condition error map. When enabled, it is the main state-space heatmap.
#
# Useful flags:
#   --heatmap_horizon H      terminal prediction horizon used in the heatmap
#   --grid_resolution N      grid size per axis (N=100 -> 100x100 grid)
#   --phase_horizons ...     horizons shown in the phase-space error maps
#
# Van der Pol example: standard diagnostics + dense true-grid heatmap
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/regression_dmd/vanderpol/default/model.npz --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10 --run_diagnostics --phase_horizons 1,10,50 --heatmap_horizon 1 --run_true_grid_heatmap

# Same as above, but with denser grid (slower, prettier figure)
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/regression_dmd/vanderpol/default/model.npz --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10 --run_diagnostics --phase_horizons 1,10,50 --heatmap_horizon 1 --run_true_grid_heatmap --grid_resolution 150

# Saddle-point example with true-grid heatmap
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/linear/saddle_point --model_path data/models/regression_dmd/saddle_point/default/model.npz --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10 --run_diagnostics --phase_horizons 1,5,10 --heatmap_horizon 1 --run_true_grid_heatmap

# The dense true-grid heatmap works for other evaluated models too, as long as they support
# rollout from an initial condition through the normal eval/model_io pipeline.
# Example with ml_dmd:
    python -m scripts.eval --model ml_dmd --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/ml_dmd/vanderpol/default/model.pt --print_validation_summary --horizons 1,2,5,10 --rollout_horizons 5,10 --run_diagnostics --phase_horizons 1,10,50 --heatmap_horizon 1 --run_true_grid_heatmap

# Quick debug version (faster, lower-resolution heatmap)
    python -m scripts.eval --model regression_dmd --data_path data/trajectories/nonlinear/vanderpol --model_path data/models/regression_dmd/vanderpol/default/model.npz --horizons 1,2 --rollout_horizons 5 --run_diagnostics --phase_horizons 1,5 --heatmap_horizon 1 --run_true_grid_heatmap --grid_resolution 50

Output:
    data/figures/{model}/{system}/{run_name}/time_series_idx{traj_index}.png
    data/figures/{model}/{system}/{run_name}/rollout_idx{traj_index}.png
    data/figures/{model}/{system}/{run_name}/test_summary.npz
    data/figures/{model}/{system}/{run_name}/diagnostics_test/* (if --run_diagnostics)'''
