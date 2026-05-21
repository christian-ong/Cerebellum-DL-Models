# LSBATCH: User input
#!/bin/bash
#BSUB -J train_best_models

# Output files
#BSUB -o hpc/runs/%J.out
#BSUB -e hpc/runs/%J.err

# GPU
#BSUB -q gpua40
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -n 4

# Runtime
#BSUB -W 24:00

# ----------------------------------
# ENV
# ----------------------------------

cd /dtu/blackhole/0d/168141/Cerebellum-DL-Models

module load python3/3.11.11
source .venv/bin/activate


# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/vanderpol/dt_0.01 --epochs 100 --expansion_type general --expansion_degree 5 --bias true --sine_cosine_expansion false --lr 1e-3 --weight_decay 0.0 --rollout_horizon 20 --name linear_vanderpol_gen5_0.01
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/vanderpol/dt_0.05 --epochs 100 --expansion_type general --expansion_degree 5 --bias true --sine_cosine_expansion false --lr 1e-3 --weight_decay 0.0 --rollout_horizon 20 --name linear_vanderpol_gen5_0.05

# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/vanderpol/dt_0.05 --epochs 100 --expansion_type delay --delay_depth 5 --bias true --lr 1e-3 --weight_decay 0.0 --rollout_horizon 20 --name delay_vdp_d5
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/vanderpol/dt_0.05 --epochs 100 --expansion_type delay --delay_depth 15 --bias true --lr 1e-3 --weight_decay 0.0 --rollout_horizon 20 --name delay_vdp_d15
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/vanderpol/dt_0.05 --epochs 100 --expansion_type delay --delay_depth 30 --bias true --lr 1e-3 --weight_decay 0.0 --rollout_horizon 20 --name delay_vdp_d30

# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/pendulum/dt_0.05 --epochs 100 --expansion_type delay --delay_depth 5 --bias true --lr 1e-3 --weight_decay 0.0 --rollout_horizon 20 --name delay_pend_d5
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/pendulum/dt_0.05 --epochs 100 --expansion_type delay --delay_depth 15 --bias true --lr 1e-3 --weight_decay 0.0 --rollout_horizon 20 --name delay_pend_d15
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/pendulum/dt_0.05 --epochs 100 --expansion_type delay --delay_depth 30 --bias true --lr 1e-3 --weight_decay 0.0 --rollout_horizon 20 --name delay_pend_d30

# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/lorenz/dt_0.05 --epochs 100 --expansion_type delay --delay_depth 5 --bias true --lr 1e-3 --weight_decay 0.0 --rollout_horizon 20 --name delay_lorenz_d5
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/lorenz/dt_0.05 --epochs 100 --expansion_type delay --delay_depth 15 --bias true --lr 1e-3 --weight_decay 0.0 --rollout_horizon 20 --name delay_lorenz_d15
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/lorenz/dt_0.05 --epochs 100 --expansion_type delay --delay_depth 30 --bias true --lr 1e-3 --weight_decay 0.0 --rollout_horizon 20 --name delay_lorenz_d30


# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/closed_trig_large --epochs 100 --expansion_type specific --expansion_degree 10 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --name linear_spec10
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/closed_large --epochs 100 --expansion_type specific --expansion_degree 5 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --name linear_spec5
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/closed_small --epochs 100 --expansion_type specific --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --name linear_spec3
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/closed_trig_small --epochs 100 --expansion_type specific --expansion_degree 6 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --name linear_spec6
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/closed_trig_medium --epochs 100 --expansion_type specific --expansion_degree 8 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --name linear_spec8

# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/linear/saddle_point --epochs 100 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --name linear_gen3
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/linear/degenerate_node --epochs 100 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --name linear_gen3
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/linear/inward_spiral --epochs 100 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --name linear_gen3
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/linear/harmonic_oscillator --epochs 100 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --name linear_gen3

# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/vanderpol --epochs 100 --expansion_type general --expansion_degree 10 --bias true --sine_cosine_expansion false --lr 1e-3 --weight_decay 0.0 --rollout_horizon 20 --name linear_vanderpol_gen10_fixed
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/vanderpol --epochs 100 --expansion_type general --expansion_degree 5 --bias true --sine_cosine_expansion false --lr 1e-3 --weight_decay 0.0 --rollout_horizon 20 --name linear_vanderpol_gen5_fixed
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/vanderpol --epochs 100 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --lr 1e-3 --weight_decay 0.0 --rollout_horizon 20 --name linear_vanderpol_gen3_fixed
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/duffing --epochs 100 --expansion_type general --expansion_degree 10 --bias true --sine_cosine_expansion false --lr 1e-3 --weight_decay 0.0 --rollout_horizon 20 --name linear_duffing_gen10
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/lotka_volterra --epochs 100 --expansion_type general --expansion_degree 10 --bias true --sine_cosine_expansion false --lr 1e-3 --weight_decay 0.0 --rollout_horizon 20 --name linear_lotka_volterra_gen10
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/pendulum --epochs 100 --expansion_type general --expansion_degree 10 --bias true --sine_cosine_expansion true --lr 1e-3 --weight_decay 0.0 --rollout_horizon 20 --name linear_pendulum_gen10
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/lorenz --epochs 100 --expansion_type general --expansion_degree 10 --bias true --sine_cosine_expansion false --lr 1e-3 --weight_decay 0.0 --rollout_horizon 20 --name linear_lorenz_gen10


# python -m experiments.eval_trajectory_rollout --model_name ml_lineardynamics --custom_name delay_vdp_d5_0.01 --data_path data/trajectories/nonlinear/vanderpol/dt_0.01
# python -m experiments.eval_trajectory_rollout --model_name ml_lineardynamics --custom_name delay_vdp_d15_0.01 --data_path data/trajectories/nonlinear/vanderpol/dt_0.01
# python -m experiments.eval_trajectory_rollout --model_name ml_lineardynamics --custom_name delay_vdp_d30_0.01 --data_path data/trajectories/nonlinear/vanderpol/dt_0.01



python -m scripts.train --model regression_dmd --data_path data/trajectories/nonlinear/vanderpol/dt_0.05 --expansion_type delay --delay_depth 10 --name reg_delay_10
python -m experiments.eval_trajectory_rollout --model_name regression_dmd --custom_name reg_delay_10 --data_path data/trajectories/nonlinear/vanderpol/dt_0.05

python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/vanderpol/dt_0.05 --epochs 100 --expansion_type hankel_svd --delay_depth 150 --hankel_rank 50 --bias true --lr 1e-3 --rollout_horizon 20 --name ml_hankel_vdp_d150_r50
python -m scripts.eval --model_name ml_lineardynamics --custom_name ml_hankel_vdp_d100_r50 --data_path data/trajectories/nonlinear/vanderpol/dt_0.05


python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/lorenz/dt_0.05 --epochs 100 --expansion_type hankel_svd --delay_depth 100 --hankel_rank 50 --bias true --lr 1e-3 --rollout_horizon 20 --name ml_hankel_d100_r50
python -m scripts.eval --model_name ml_lineardynamics --custom_name ml_hankel_d100_r50 --data_path data/trajectories/nonlinear/lorenz/dt_0.05