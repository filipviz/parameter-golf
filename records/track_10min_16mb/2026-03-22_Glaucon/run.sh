#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_gpt.py"

# Environment setup
python3 -m pip install -r "$SCRIPT_DIR/requirements.txt"

# Download dataset
python3 data/cached_challenge_fineweb.py --variant sp1024

SEED=3 RUN_ID="quant" \
	torchrun --standalone --nproc_per_node=8 "$TRAIN_SCRIPT"

# echo "=== All runs complete ==="
