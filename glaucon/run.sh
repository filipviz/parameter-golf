#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_gpt.py"

# Proxy sweep: paper-faithful Block AttnRes, varying block size
# ATTN_RES_BLOCK_SIZE is in sublayers (attn+mlp = 2 per transformer layer)
# With L=11: BS=4 → ~7 sources, BS=8 → ~4 sources, BS=22 → 2 sources

SEED=3 RUN_ID="attnres_bs4_lr010" ITERATIONS=500 WARMDOWN_ITERS=3500 \
	MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=10 \
	ATTN_RES_BLOCK_SIZE=4 ATTN_RES_LR=0.1 \
	torchrun --standalone --nproc_per_node=1 "$TRAIN_SCRIPT"

SEED=3 RUN_ID="attnres_bs8_lr010" ITERATIONS=500 WARMDOWN_ITERS=3500 \
	MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=10 \
	ATTN_RES_BLOCK_SIZE=8 ATTN_RES_LR=0.1 \
	torchrun --standalone --nproc_per_node=1 "$TRAIN_SCRIPT"

SEED=3 RUN_ID="attnres_bs22_lr010" ITERATIONS=500 WARMDOWN_ITERS=3500 \
	MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=10 \
	ATTN_RES_BLOCK_SIZE=22 ATTN_RES_LR=0.1 \
	torchrun --standalone --nproc_per_node=1 "$TRAIN_SCRIPT"

echo "=== All runs complete ==="
