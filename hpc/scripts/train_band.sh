# LSBATCH: User input
#!/bin/bash
#BSUB -J train_best_models

# Output files
#BSUB -o hpc/runs/%J.out
#BSUB -e hpc/runs/%J.err

# GPU
#BSUB -q gpuv100
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
python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/closed_trig_large/long --epochs 200 --expansion_type specific --expansion_degree 10 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-3 --name band_long_spec10_fix4
python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/closed_large/short --epochs 200 --expansion_type specific --expansion_degree 5 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-3 --name band_short_spec5_fix4
python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/closed_trig_small/long --epochs 200 --expansion_type specific --expansion_degree 6 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-3 --name band_long_spec6_fix4
python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/closed_trig_medium/long --epochs 200 --expansion_type specific --expansion_degree 8 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-3 --name band_long_spec8_fix4
python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/closed_small/short --epochs 200 --expansion_type specific --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-3 --name band_short_spec3_fix4
python -m scripts.train --model ml_dmd_band --data_path data/trajectories/linear/saddle_point/long --epochs 200 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-3 --name band_long_gen3_fix4
python -m scripts.train --model ml_dmd_band --data_path data/trajectories/linear/degenerate_node/long --epochs 200 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-3 --name band_long_gen3_fix4
python -m scripts.train --model ml_dmd_band --data_path data/trajectories/linear/inward_spiral/short --epochs 200 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-3 --name band_short_gen3_fix4
python -m scripts.train --model ml_dmd_band --data_path data/trajectories/linear/harmonic_oscillator/short --epochs 200 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-3 --name band_short_gen3_fix4
python -m scripts.train --model ml_dmd_band --data_path data/trajectories/nonlinear/duffing/short --epochs 200 --expansion_type specific --expansion_degree 10 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-3 --name band_short_spec10_fix4