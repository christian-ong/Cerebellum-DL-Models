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

# ----------------------------------
python -m scripts.train --model mlp_baseline --data_path data/trajectories/nonlinear/closed_trig_large --epochs 100 --lr 1e-3 --rollout_horizon 20
python -m scripts.train --model mlp_baseline --data_path data/trajectories/nonlinear/closed_large --epochs 100 --lr 1e-3 --rollout_horizon 20
python -m scripts.train --model mlp_baseline --data_path data/trajectories/nonlinear/closed_trig_small --epochs 100 --lr 1e-3 --rollout_horizon 20
python -m scripts.train --model mlp_baseline --data_path data/trajectories/nonlinear/closed_trig_medium --epochs 100 --lr 1e-3 --rollout_horizon 20
python -m scripts.train --model mlp_baseline --data_path data/trajectories/nonlinear/closed_small --epochs 100 --lr 1e-3 --rollout_horizon 20
python -m scripts.train --model mlp_baseline --data_path data/trajectories/linear/saddle_point --epochs 100 --lr 1e-3 --rollout_horizon 20
python -m scripts.train --model mlp_baseline --data_path data/trajectories/linear/degenerate_node --epochs 100 --lr 1e-3 --rollout_horizon 20
python -m scripts.train --model mlp_baseline --data_path data/trajectories/linear/inward_spiral --epochs 100 --lr 1e-3 --rollout_horizon 20
python -m scripts.train --model mlp_baseline --data_path data/trajectories/linear/harmonic_oscillator --epochs 100 --lr 1e-3 --rollout_horizon 20
python -m scripts.train --model mlp_baseline --data_path data/trajectories/nonlinear/vanderpol --epochs 100 --lr 1e-3 --rollout_horizon 20
python -m scripts.train --model mlp_baseline --data_path data/trajectories/nonlinear/duffing --epochs 100  --lr 1e-3 --rollout_horizon 20
python -m scripts.train --model mlp_baseline --data_path data/trajectories/nonlinear/lotka_volterra --epochs 100 --lr 1e-3 --rollout_horizon 20
python -m scripts.train --model mlp_baseline --data_path data/trajectories/nonlinear/pendulum --epochs 100 --lr 1e-3 --rollout_horizon 20
python -m scripts.train --model mlp_baseline --data_path data/trajectories/nonlinear/lorenz --epochs 100 --lr 1e-3 --rollout_horizon 20