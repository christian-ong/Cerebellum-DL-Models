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
python -m scripts.train --model ml_dmd --data_path data/trajectories/linear/saddle_point --epochs 50 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5 --name general_deg3_new
python -m scripts.train --model ml_dmd --data_path data/trajectories/linear/degenerate_node --epochs 50 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5 --name general_deg3_new
python -m scripts.train --model ml_dmd --data_path data/trajectories/linear/inward_spiral --epochs 50 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-5 --name general_trig_deg3_new
python -m scripts.train --model ml_dmd --data_path data/trajectories/linear/harmonic_oscillator --epochs 50 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-5 --name general_trig_deg3_new

# Koopman Poly Baselines (Using example defaults)
python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/koopman_poly --epochs 50 --expansion_type specific --expansion_degree 3 --bias false --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5 --name specific_deg3_new
python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/koopman_poly_large --epochs 50 --expansion_type specific --expansion_degree 5 --bias false --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5 --name specific_deg5_new
python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/koopman_poly_trig --epochs 50 --expansion_type specific --expansion_degree 10 --bias false --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-5 --name specific_trig_deg10_new

# Nonlinear Systems (Table Parameters from JSON)
python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/duffing --epochs 50 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-05 --name general_deg3_new
python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/duffing --epochs 50 --expansion_type specific --expansion_degree 5 --bias false --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-05 --name specific_deg5_new
python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/lorenz --epochs 50 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 1e-06 --lr 0.0001 --name general_deg3_new
python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/lorenz --epochs 50 --expansion_type specific --expansion_degree 7 --bias false --sine_cosine_expansion false --weight_decay 1e-06 --lr 1e-05 --name specific_deg7_new
python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/lotka_volterra --epochs 50 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-05 --name general_deg3_new
python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/lotka_volterra --epochs 50 --expansion_type specific --expansion_degree 10 --bias false --sine_cosine_expansion false --weight_decay 1e-06 --lr 1e-05 --name specific_deg10_new
python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/pendulum --epochs 50 --expansion_type general --expansion_degree 5 --bias true --sine_cosine_expansion true --weight_decay 1e-06 --lr 1e-05 --name general_trig_deg5_new
python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/pendulum --epochs 50 --expansion_type specific --expansion_degree 5 --bias false --sine_cosine_expansion true --weight_decay 1e-06 --lr 0.0001 --name specific_trig_deg5_new
python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/vanderpol --epochs 50 --expansion_type general --expansion_degree 5 --bias true --sine_cosine_expansion false --weight_decay 1e-06 --lr 1e-05 --name general_deg5_new
python -m scripts.train --model ml_dmd --data_path data/trajectories/nonlinear/vanderpol --epochs 50 --expansion_type specific --expansion_degree 7 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 0.0001 --name specific_deg7_new

# ----------------------------------
# Linear Systems (Using example defaults)
python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/linear/saddle_point --epochs 50 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5 --name general_deg3_new
python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/linear/degenerate_node --epochs 50 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5 --name general_deg3_new
python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/linear/inward_spiral --epochs 50 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-5 --name general_trig_deg3_new
python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/linear/harmonic_oscillator --epochs 50 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-5 --name general_trig_deg3_new

# Koopman Poly Baselines (Using example defaults)
python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/koopman_poly --epochs 50 --expansion_type specific --expansion_degree 3 --bias false --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5 --name specific_deg3_new
python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/koopman_poly_large --epochs 50 --expansion_type specific --expansion_degree 5 --bias false --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-5 --name specific_deg5_new
python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/koopman_poly_trig --epochs 50 --expansion_type specific --expansion_degree 10 --bias false --sine_cosine_expansion true --weight_decay 0.0 --lr 1e-5 --name specific_trig_deg10_new

# Nonlinear Systems (Table Parameters from JSON)
python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/duffing --epochs 50 --expansion_type general --expansion_degree 5 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 0.0001 --name general_deg5_new
python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/duffing --epochs 50 --expansion_type specific --expansion_degree 5 --bias false --sine_cosine_expansion false --weight_decay 0.0 --lr 0.0001 --name specific_deg5_new
python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/lorenz --epochs 50 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 0.0001 --name general_deg3_new
python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/lorenz --epochs 50 --expansion_type specific --expansion_degree 5 --bias false --sine_cosine_expansion false --weight_decay 1e-06 --lr 1e-05 --name specific_deg5_new
python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/lotka_volterra --epochs 50 --expansion_type general --expansion_degree 3 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 0.0001 --name general_deg3_new
python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/lotka_volterra --epochs 50 --expansion_type specific --expansion_degree 10 --bias false --sine_cosine_expansion false --weight_decay 1e-06 --lr 0.0001 --name specific_deg10_new
python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/pendulum --epochs 50 --expansion_type general --expansion_degree 7 --bias true --sine_cosine_expansion true --weight_decay 0.0 --lr 0.001 --name general_trig_deg7_new
python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/pendulum --epochs 50 --expansion_type specific --expansion_degree 5 --bias false --sine_cosine_expansion true --weight_decay 1e-06 --lr 0.001 --name specific_trig_deg5_new
python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/vanderpol --epochs 50 --expansion_type general --expansion_degree 5 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 1e-05 --name general_deg5_new
python -m scripts.train --model ml_lineardynamics --data_path data/trajectories/nonlinear/vanderpol --epochs 50 --expansion_type specific --expansion_degree 7 --bias true --sine_cosine_expansion false --weight_decay 0.0 --lr 0.0001 --name specific_deg7_new