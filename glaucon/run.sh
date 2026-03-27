#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_gpt.py"

# Environment setup
python3 -m pip install -r "$SCRIPT_DIR/requirements.txt"

# Download dataset
python3 data/cached_challenge_fineweb.py --variant sp1024

# Experiment 3: Depth/width sweep (d_head=64 fixed, MHA, no XSA, no VE)
# All configs ~27M params. Varying depth/width to find optimal L/D tradeoff.

# Baseline: 10 layers, 512 dim (8 heads, MHA) ~27.1M params
SEED=3 RUN_ID="exp3_baseline_L10_D512" NUM_LAYERS=10 NUM_HEADS=8 NUM_KV_HEADS=8 XSA_LAST_N=0 VE_ENABLED=0 \
	torchrun --standalone --nproc_per_node=8 "$TRAIN_SCRIPT"

# A: 13 layers, 448 dim (7 heads, MHA) ~26.9M params
SEED=3 RUN_ID="exp3a_L13_D448" NUM_LAYERS=13 MODEL_DIM=448 NUM_HEADS=7 NUM_KV_HEADS=7 XSA_LAST_N=0 VE_ENABLED=0 \
	torchrun --standalone --nproc_per_node=8 "$TRAIN_SCRIPT"

# B: 18 layers, 384 dim (6 heads, MHA) ~27.3M params
SEED=3 RUN_ID="exp3b_L18_D384" NUM_LAYERS=18 MODEL_DIM=384 NUM_HEADS=6 NUM_KV_HEADS=6 XSA_LAST_N=0 VE_ENABLED=0 \
	torchrun --standalone --nproc_per_node=8 "$TRAIN_SCRIPT"

# C: 26 layers, 320 dim (5 heads, MHA) ~27.3M params
SEED=3 RUN_ID="exp3c_L26_D320" NUM_LAYERS=26 MODEL_DIM=320 NUM_HEADS=5 NUM_KV_HEADS=5 XSA_LAST_N=0 VE_ENABLED=0 \
	torchrun --standalone --nproc_per_node=8 "$TRAIN_SCRIPT"

echo "=== All runs complete ==="
