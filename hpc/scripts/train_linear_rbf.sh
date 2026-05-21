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


python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/closed_trig_large --epochs 100 --expansion_type rbf --rbf_n_centers 1000 --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --name linear_rbf10
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/closed_large --epochs 100 --expansion_type rbf --expansion_degree 5 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --name linear_rbf5
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/closed_small --epochs 100 --expansion_type rbf --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --name linear_rbf3
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/closed_trig_small --epochs 100 --expansion_type rbf --expansion_degree 6 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --name linear_rbf6
# python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/closed_trig_medium --epochs 100 --expansion_type rbf --expansion_degree 8 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-3 --rollout_horizon 20 --name linear_rbf8

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

