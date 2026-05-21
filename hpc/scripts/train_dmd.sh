# LSBATCH: User input
#!/bin/bash
#BSUB -J train_best_models

# Output files
#BSUB -o hpc/runs/%J.out
#BSUB -e hpc/runs/%J.err

# GPU
#BSUB -q gpua10
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

python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/closed_trig_large/dt_0.05 --epochs 100 --expansion_type delay --delay_depth 5 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --l1_weight 1e-3 --name delay_5
python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/closed_trig_large/dt_0.05 --epochs 100 --expansion_type delay --delay_depth 10 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --l1_weight 1e-3 --name delay_10
python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/closed_trig_large/dt_0.05 --epochs 100 --expansion_type delay --delay_depth 30 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --l1_weight 1e-3 --name delay_30

python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/vanderpol/dt_0.05 --epochs 100 --expansion_type delay --delay_depth 5 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --l1_weight 1e-3 --name delay_5
python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/vanderpol/dt_0.05 --epochs 100 --expansion_type delay --delay_depth 10 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --l1_weight 1e-3 --name delay_10
python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/vanderpol/dt_0.05 --epochs 100 --expansion_type delay --delay_depth 5 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --l1_weight 1e-3 --name delay_5

# python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/closed_large/dt_0.05 --epochs 100 --expansion_type delay --delay_depth 10 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --l1_weight 1e-3 --name delay_10
# python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/closed_small/dt_0.05 --epochs 100 --expansion_type delay --delay_depth 10 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --l1_weight 1e-3 --name delay_10
# python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/closed_trig_small/dt_0.05 --epochs 100 --expansion_type delay --delay_depth 10 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --l1_weight 1e-3 --name delay_10
# python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/closed_trig_medium/dt_0.05 --epochs 100 --expansion_type delay --delay_depth 10 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --l1_weight 1e-3 --name delay_10

# python -m scripts.train --model ml_dmd --data_path data/trajectories/linear/saddle_point/dt_0.05 --epochs 100 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4 --rollout_horizon 20 --l1_weight 1e-4 --name l1_gen3_4
# python -m scripts.train --model ml_dmd --data_path data/trajectories/linear/degenerate_node/dt_0.05 --epochs 100 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4 --rollout_horizon 20 --l1_weight 1e-4 --name l1_gen3_4
# python -m scripts.train --model ml_dmd --data_path data/trajectories/linear/inward_spiral/dt_0.05 --epochs 100 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4 --rollout_horizon 20 --l1_weight 1e-4 --name l1_gen3_4
# python -m scripts.train --model ml_dmd --data_path data/trajectories/linear/harmonic_oscillator/dt_0.05 --epochs 100 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-4 --rollout_horizon 20 --l1_weight 1e-4 --name l1_gen3_4

# python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/vanderpol/dt_0.05 --epochs 100 --expansion_type general --expansion_degree 10 --bias true --sine_cosine_expansion false --lr 1e-4 --weight_decay 0.0 --rollout_horizon 20 --l1_weight 1e-4 --name l1_vanderpol_gen10_4
# python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/duffing/dt_0.05 --epochs 100 --expansion_type general --expansion_degree 10 --bias true --sine_cosine_expansion false --lr 1e-4 --weight_decay 0.0 --rollout_horizon 20 --l1_weight 1e-4 --name l1_duffing_gen10_4
# python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/lotka_volterra/dt_0.05 --epochs 100 --expansion_type general --expansion_degree 10 --bias true --sine_cosine_expansion false --lr 1e-4 --weight_decay 0.0 --rollout_horizon 20 --l1_weight 1e-4 --name l1_lotka_volterra_gen10_4
# python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/pendulum/dt_0.05 --epochs 100 --expansion_type general --expansion_degree 10 --bias true --sine_cosine_expansion true --lr 1e-4 --weight_decay 0.0 --rollout_horizon 20 --l1_weight 1e-4 --name l1_pendulum_gen10_4
# python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/lorenz/dt_0.05 --epochs 100 --expansion_type general --expansion_degree 10 --bias true --sine_cosine_expansion false --lr 1e-4 --weight_decay 0.0 --rollout_horizon 20 --l1_weight 1e-4 --name l1_lorenz_gen10_4


python -m experiments.visualize_dynamic_modes --model_name ml_dmd --custom_name delay_5 --data_path data/trajectories/nonlinear/vanderpol/dt_0.05
python -m experiments.eval_trajectory_rollout --model_name ml_dmd --custom_name delay_5 --data_path data/trajectories/nonlinear/vanderpol/dt_0.05

python -m experiments.visualize_dynamic_modes --model_name ml_dmd --custom_name delay_10 --data_path data/trajectories/nonlinear/vanderpol/dt_0.05
python -m experiments.eval_trajectory_rollout --model_name ml_dmd --custom_name delay_10 --data_path data/trajectories/nonlinear/vanderpol/dt_0.05

python -m experiments.visualize_dynamic_modes --model_name ml_dmd --custom_name delay_40 --data_path data/trajectories/nonlinear/vanderpol/dt_0.05
python -m experiments.eval_trajectory_rollout --model_name ml_dmd --custom_name delay_40 --data_path data/trajectories/nonlinear/vanderpol/dt_0.05

python -m experiments.visualize_dynamic_modes --model_name ml_dmd --custom_name delay_5 --data_path data/trajectories/nonlinear/closed_trig_large/dt_0.05
python -m experiments.eval_trajectory_rollout --model_name ml_dmd --custom_name delay_5 --data_path data/trajectories/nonlinear/closed_trig_large/dt_0.05

python -m experiments.visualize_dynamic_modes --model_name ml_dmd --custom_name delay_10 --data_path data/trajectories/nonlinear/closed_trig_large/dt_0.05
python -m experiments.eval_trajectory_rollout --model_name ml_dmd --custom_name delay_10 --data_path data/trajectories/nonlinear/closed_trig_large/dt_0.05

python -m experiments.visualize_dynamic_modes --model_name ml_dmd --custom_name delay_40 --data_path data/trajectories/nonlinear/closed_trig_large/dt_0.05
python -m experiments.eval_trajectory_rollout --model_name ml_dmd --custom_name delay_40 --data_path data/trajectories/nonlinear/closed_trig_large/dt_0.05