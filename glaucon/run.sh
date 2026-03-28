#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_gpt.py"

# Environment setup
# python3 -m pip install -r "$SCRIPT_DIR/requirements.txt"

# Download dataset
# python3 data/cached_challenge_fineweb.py --variant sp1024

# Proxy: fixed RoPE buffers + eager VE (1000 steps)
SEED=3 RUN_ID="rope_fix_proxy" ITERATIONS=1000 MAX_WALLCLOCK_SECONDS=0 \
	torchrun --standalone --nproc_per_node=8 "$TRAIN_SCRIPT"

# Experiment 4: Re-introduce GQA + XSA + VE on winning config D (D=576, L=8)
# LR scaling: matrix_lr = 0.025 * sqrt(10/8) = 0.028, embed_lr = 0.035 * sqrt(512/576) = 0.033

# Full features: GQA (9 heads, 3 kv), XSA last 3, VE on layers 6,7
# SEED=3 RUN_ID="exp4_D576_L8_full" NUM_LAYERS=8 MODEL_DIM=576 NUM_HEADS=9 NUM_KV_HEADS=3 MLP_DIM=1792 \
# 	XSA_LAST_N=3 VE_ENABLED=1 VE_LAYERS="6,7" MATRIX_LR=0.028 EMBED_LR=0.033 \
# 	torchrun --standalone --nproc_per_node=8 "$TRAIN_SCRIPT"

echo "=== All runs complete ==="
