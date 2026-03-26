#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_gpt.py"

# Environment setup
python3 -m pip install -r "$SCRIPT_DIR/requirements.txt"

# Download dataset
python3 data/cached_challenge_fineweb.py --variant sp1024

# Experiment 3: Depth/width sweep
# Baseline (11 layers, 512 dim) for comparison
SEED=3 RUN_ID="exp3_baseline_L11_D512" \
	torchrun --standalone --nproc_per_node=8 "$TRAIN_SCRIPT"

# A: 15 layers, 432 dim (26.1M params, -3.2% vs baseline)
SEED=3 RUN_ID="exp3a_L15_D432" NUM_LAYERS=15 MODEL_DIM=432 VE_LAYERS="13,14" \
	torchrun --standalone --nproc_per_node=8 "$TRAIN_SCRIPT"

# B: 20 layers, 384 dim (27.4M params, +1.6% vs baseline)
SEED=3 RUN_ID="exp3b_L20_D384" NUM_LAYERS=20 MODEL_DIM=384 VE_LAYERS="18,19" \
	torchrun --standalone --nproc_per_node=8 "$TRAIN_SCRIPT"

# C: 26 layers, 336 dim (27.3M params, +1.0% vs baseline)
SEED=3 RUN_ID="exp3c_L26_D336" NUM_LAYERS=26 MODEL_DIM=336 VE_LAYERS="24,25" \
	torchrun --standalone --nproc_per_node=8 "$TRAIN_SCRIPT"

echo "=== All runs complete ==="
