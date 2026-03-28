#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_gpt.py"

# Full 500-step experiments: block-level AttnRes with internal residual
SEED=3 RUN_ID="attnres_block_static_lr100" ITERATIONS=500 WARMDOWN_ITERS=3500 \
	MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=10 \
	ATTN_RES_MODE=static ATTN_RES_LR=1.0 ATTN_RES_GRANULARITY=block \
	torchrun --standalone --nproc_per_node=1 "$TRAIN_SCRIPT"

SEED=3 RUN_ID="attnres_block_static_lr050" ITERATIONS=500 WARMDOWN_ITERS=3500 \
	MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=10 \
	ATTN_RES_MODE=static ATTN_RES_LR=0.5 ATTN_RES_GRANULARITY=block \
	torchrun --standalone --nproc_per_node=1 "$TRAIN_SCRIPT"

echo "=== All runs complete ==="
