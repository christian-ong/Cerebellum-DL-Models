# LSBATCH: User input
#!/bin/bash
#BSUB -J koopman_ml_dmd

# Output files
#BSUB -o runs/%J.out
#BSUB -e runs/%J.err

# GPU
#BSUB -q gpua10
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB] span[hosts=1]"
#BSUB -n 4

# Runtime
#BSUB -W 04:00

# ----------------------------------
# ENV
# ----------------------------------

cd /dtu/blackhole/0d/168141/Cerebellum-DL-Models

module load python3/3.11.11
source .venv/bin/activate

export WANDB_START_METHOD=thread
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ----------------------------------
# SWEEP
# ----------------------------------

SWEEP_ID=DeepLearningP4Destruction/Cerebellum-DL-Models/<SWEEP_ID>

# ----------------------------------
# RUN (ONLY ONE AGENT!)
# ----------------------------------

wandb agent $SWEEP_ID