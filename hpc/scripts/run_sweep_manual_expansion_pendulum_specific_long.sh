# LSBATCH: User input
#!/bin/bash
#BSUB -J koopman_manual_pendulum_specific

# Output files
#BSUB -o hpc/runs/%J.out
#BSUB -e hpc/runs/%J.err

# GPU
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -n 4

# Runtime
#BSUB -W 6:00

# ----------------------------------
# ENV
# ----------------------------------

cd /dtu/blackhole/0d/168141/Cerebellum-DL-Models

module load python3/3.11.11
source .venv/bin/activate

# ----------------------------------
# SWEEP
# ----------------------------------

SWEEP_ID=DeepLearningP4Destruction/Cerebellum-DL-Models/0l7i9v9y

# ----------------------------------
# RUN (ONLY ONE AGENT!)
# ----------------------------------

wandb agent $SWEEP_ID