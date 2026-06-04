# LSBATCH: User input
#!/bin/bash
#BSUB -J eval_best_models

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

# Run the evaluation sweep over the best runs selected from the wandb CSVs.
python -m experiments.eval_best_models --csv "experiments/wandb/wandb_runs_sindy_baseline.csv" --skip_existing