#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_gpt.py"
BASELINE_SCRIPT="$SCRIPT_DIR/../records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py"

# --- Baseline: PR #549 frontier (500-step proxy) ---
# Architectural env vars from the record's README; TTT/EMA/SWA/QAT disabled
# for proxy; warmdown disabled so LR stays at full scale for comparability.
# NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
# 	ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
# 	MUON_WD=0.04 ADAM_WD=0.04 \
# 	MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
# 	MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
# 	MUON_MOMENTUM_WARMUP_STEPS=1500 \
# 	TTT_ENABLED=0 EMA_ENABLED=0 SWA_ENABLED=0 LATE_QAT=0 \
# 	SEED=3 RUN_ID="baseline_pr549" ITERATIONS=500 WARMDOWN_ITERS=0 \
# 	MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=10 \
# 	torchrun --standalone --nproc_per_node=1 "$BASELINE_SCRIPT"

# --- AttnRes proxy sweep (500 steps each, block-runner two-phase routing) ---
# ATTN_RES_BLOCK_SIZE is in sublayers (attn+mlp = 2 per transformer layer)
# BS=2 → 12 sources (1 layer/block), BS=4 → ~7 sources, BS=8 → ~4 sources

SEED=3 RUN_ID="attnres_bs8_lr050" ITERATIONS=500 WARMDOWN_ITERS=0 \
	MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=10 \
	ATTN_RES_BLOCK_SIZE=8 ATTN_RES_LR=0.5 \
	torchrun --standalone --nproc_per_node=1 "$TRAIN_SCRIPT"

SEED=3 RUN_ID="attnres_bs4_lr050" ITERATIONS=500 WARMDOWN_ITERS=0 \
	MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=10 \
	ATTN_RES_BLOCK_SIZE=4 ATTN_RES_LR=0.5 \
	torchrun --standalone --nproc_per_node=1 "$TRAIN_SCRIPT"

SEED=3 RUN_ID="attnres_bs2_lr050" ITERATIONS=500 WARMDOWN_ITERS=0 \
	MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=10 \
	ATTN_RES_BLOCK_SIZE=2 ATTN_RES_LR=0.5 \
	torchrun --standalone --nproc_per_node=1 "$TRAIN_SCRIPT"

echo "=== All runs complete ==="
