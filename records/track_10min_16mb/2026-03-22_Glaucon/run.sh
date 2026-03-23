#!/usr/bin/env bash
set -euo pipefail

# Train Glaucon with 3 seeds
# Built on PR #315: 11L Partial RoPE + LN Scale + EMA + XSA4
# New: gated attention, symmetric Muon, brotli. Tweak AdamW eps and Muon/AdamW weight decay.

export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_gpt.py"

# Environment setup
python3 -m pip install -r "$SCRIPT_DIR/requirements.txt"

# Download dataset
python3 data/cached_challenge_fineweb.py --variant sp1024

for SEED in 3 33 333; do
    echo "=== Seed ${SEED} ==="
    SEED="$SEED" RUN_ID="seed_${SEED}" \
        torchrun --standalone --nproc_per_node=8 "$TRAIN_SCRIPT"
done

echo "=== All runs complete ==="
