# LSBATCH: User input
#!/bin/bash
#BSUB -J train_best_models

# Output files
#BSUB -o hpc/runs/%J.out
#BSUB -e hpc/runs/%J.err

# GPU
#BSUB -q gpua100
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
# Linear Systems (Using example defaults)
python -m scripts.train --model mlp_baseline --data_path data/trajectories/nonlinear/closed_trig_large/long --epochs 100
python -m scripts.train --model mlp_baseline --data_path data/trajectories/nonlinear/closed_large/short --epochs 100
python -m scripts.train --model mlp_baseline --data_path data/trajectories/nonlinear/closed_trig_small/long --epochs 100
python -m scripts.train --model mlp_baseline --data_path data/trajectories/nonlinear/closed_trig_medium/long --epochs 100
python -m scripts.train --model mlp_baseline --data_path data/trajectories/nonlinear/closed_small/short --epochs 100 
python -m scripts.train --model mlp_baseline --data_path data/trajectories/linear/saddle_point/long --epochs 100
python -m scripts.train --model mlp_baseline --data_path data/trajectories/linear/degenerate_node/long --epochs 100
python -m scripts.train --model mlp_baseline --data_path data/trajectories/linear/inward_spiral/short --epochs 100
python -m scripts.train --model mlp_baseline --data_path data/trajectories/linear/harmonic_oscillator/short --epochs 100
python -m scripts.train --model mlp_baseline --data_path data/trajectories/nonlinear/duffing/short --epochs 100