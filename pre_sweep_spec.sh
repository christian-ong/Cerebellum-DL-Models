#!/bin/bash

# Fixed parameters for Specific Expansion
DATA="data/trajectories/nonlinear/vanderpol/dt_0.01"
MODEL="ml_dmd_drop"
TYPE="specific"
DEG=10
TRIG="false"
L1="0.0"
HORIZON=50

# Sweep parameters
for LR in 1e-3 1e-4; do
  for BS in 512 2048; do
    for WD in 0.0 1e-5; do
      
      echo "===================================================================="
      echo "Running: LR=$LR | Batch=$BS | WD=$WD (Specific)"
      echo "===================================================================="
      
      python -m scripts.train_sweep \
        --model $MODEL \
        --data_path $DATA \
        --expansion_type $TYPE \
        --expansion_degree $DEG \
        --sine_cosine_expansion $TRIG \
        --l1_weight $L1 \
        --rollout_horizon $HORIZON \
        --bias false \
        --lr $LR \
        --batch_size $BS \
        --weight_decay $WD \
        --epochs 100 \
        --eval_every 1 \
        --num_workers 4
        
    done
  done
done