#!/usr/bin/env bash
set -euo pipefail

# Train Glaucon with 3 seeds
# Built on PR #315: 11L Partial RoPE + LN Scale + EMA + XSA4
# New: gated attention, symmetric Muon, brotli. Tweak AdamW eps and Muon/AdamW weight decay.

export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_gpt.py"

ENV=(
    DATA_PATH=./data/datasets/fineweb10B_sp1024
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
    VOCAB_SIZE=1024
    MAX_WALLCLOCK_SECONDS=600
    NUM_LAYERS=11
    BIGRAM_VOCAB_SIZE=2048
    XSA_LAST_N=4
    EMA_ENABLED=1
    EMA_DECAY=0.997
    ROPE_DIMS=16
    LN_SCALE=1
    MUON_WD=0.045
    ADAM_WD=0.045
    MATRIX_LR=0.025
    SCALAR_LR=0.025
    TIED_EMBED_LR=0.035
    MUON_MOMENTUM=0.99
    MUON_MOMENTUM_WARMUP_START=0.92
    MUON_MOMENTUM_WARMUP_STEPS=1500
    WARMDOWN_ITERS=3000
    ITERATIONS=9000
    EVAL_STRIDE=64
)

# Environment setup
python3 -m pip install -r "$SCRIPT_DIR/requirements.txt"

# Download dataset
python3 data/cached_challenge_fineweb.py --variant sp1024

for SEED in 3 33 333; do
    echo "=== Seed ${SEED} ==="
    env "${ENV[@]}" SEED="$SEED" RUN_ID="seed_${SEED}" \
        torchrun --standalone --nproc_per_node=8 "$TRAIN_SCRIPT"
done

echo "=== All runs complete ==="
