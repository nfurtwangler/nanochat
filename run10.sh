#!/bin/bash
set -euo pipefail

# The ~$10 tier of nanochat.
# Supports `--gpu=a100` (default, 80GB) and `--gpu=5090` (32GB). The a100 config
# stays on the ~260M parameter depth-12 recipe, while the 5090 path now scales up
# model depth to better use VRAM (with fewer iterations to keep the cost in check).
# Budget target: ~12 hours of uninterrupted run time on the chosen GPU (~$10).

GPU_TYPE="a100"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu=*)
            GPU_TYPE="${1#*=}"
            shift
            ;;
        --gpu)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --gpu" >&2
                exit 1
            fi
            GPU_TYPE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [[ "$GPU_TYPE" != "a100" && "$GPU_TYPE" != "5090" ]]; then
    echo "Unsupported --gpu value: $GPU_TYPE (expected 'a100' or '5090')" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
BASE_DATA_DIR="$NANOCHAT_BASE_DIR/base_data"
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
DATASET_DOWNLOAD_PID=""

# -----------------------------------------------------------------------------
# Environment + dependencies

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

python -m nanochat.report reset

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# -----------------------------------------------------------------------------
# Tokenizer + data (moderate scale)

# Grab enough shards for the first few hundred million training tokens and keep downloading extras
DATASET_READY=0
if [[ -d "$BASE_DATA_DIR" ]]; then
    if find "$BASE_DATA_DIR" -maxdepth 1 -name 'shard_*.parquet' -print -quit | grep -q .; then
        DATASET_READY=1
    fi
fi

if [[ $DATASET_READY -eq 1 ]]; then
    echo "Found dataset shards in $BASE_DATA_DIR, skipping download."
else
    python -m nanochat.dataset -n 8
    python -m nanochat.dataset -n 160 &
    DATASET_DOWNLOAD_PID=$!
fi

# Train tokenizer on ~1B characters (still finishes quickly on A100 boxes)
if [[ -f "$TOKENIZER_DIR/tokenizer.pkl" && -f "$TOKENIZER_DIR/token_bytes.pt" ]]; then
    echo "Found tokenizer artifacts in $TOKENIZER_DIR, skipping tokenizer train/eval."
else
    python -m scripts.tok_train --max_chars=1000000000
    python -m scripts.tok_eval
fi

if [[ -n "$DATASET_DOWNLOAD_PID" ]]; then
    echo "Waiting for dataset background download to finish..."
    wait $DATASET_DOWNLOAD_PID
fi

# -----------------------------------------------------------------------------
# Base model (scales depth/iterations per GPU target)

BASE_DEPTH=28
BASE_DEVICE_BATCH=12
BASE_TOTAL_BATCH=196608
BASE_ITERS=2000
BASE_EVAL_TOKENS=393216
BASE_DESC="~1.1B params (depth 28), ~394M tokens"

if [[ "$GPU_TYPE" == "5090" ]]; then
    BASE_DEPTH=28
    BASE_DEVICE_BATCH=3
    BASE_TOTAL_BATCH=129024
    BASE_EVAL_TOKENS=294912
    BASE_DESC="~1.3B params (depth 28), ~65M tokens (~same FLOPs as depth-12 run)"
fi

echo "Config[$GPU_TYPE]: depth=$BASE_DEPTH, device_batch=$BASE_DEVICE_BATCH, total_batch=$BASE_TOTAL_BATCH, iters=$BASE_ITERS"
echo "Base profile: $BASE_DESC"

python -m scripts.base_train \
    --depth=$BASE_DEPTH \
    --device_batch_size=$BASE_DEVICE_BATCH \
    --total_batch_size=$BASE_TOTAL_BATCH \
    --num_iterations=$BASE_ITERS \
    --eval_every=200 \
    --eval_tokens=$BASE_EVAL_TOKENS \
    --core_metric_every=100 \
    --sample_every=100 \
    --run=$WANDB_RUN

python -m scripts.base_loss \
    --device_batch_size=$BASE_DEVICE_BATCH \
    --split_tokens=$BASE_EVAL_TOKENS

python -m scripts.base_eval --max-per-task=96

# -----------------------------------------------------------------------------
# Midtraining (lightweight conversation/tool warm-up)

python -m scripts.mid_train \
    --device_batch_size=$BASE_DEVICE_BATCH \
    --total_batch_size=$BASE_TOTAL_BATCH \
    --num_iterations=1200 \
    --eval_every=300 \
    --eval_tokens=$BASE_EVAL_TOKENS \
    --run=$WANDB_RUN

MID_CKPT_DIR="$NANOCHAT_BASE_DIR/mid_checkpoints/d$BASE_DEPTH"
if [[ -d "$MID_CKPT_DIR" ]]; then
    python -m scripts.chat_eval -i mid -x 96 -m 256 -t 0.0
else
    echo "Skipping mid chat_eval: $MID_CKPT_DIR not found"
fi

# -----------------------------------------------------------------------------
# Supervised finetuning (short pass over curated dialogs)

python -m scripts.chat_sft \
    --device_batch_size=$BASE_DEVICE_BATCH \
    --target_examples_per_step=64 \
    --num_iterations=600 \
    --eval_steps=150 \
    --eval_metrics_max_problems=96 \
    --run=$WANDB_RUN

SFT_CKPT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/d$BASE_DEPTH"
if [[ -d "$SFT_CKPT_DIR" ]]; then
    python -m scripts.chat_eval -i sft -x 96 -m 256 -t 0.0
else
    echo "Skipping sft chat_eval: $SFT_CKPT_DIR not found"
fi

# -----------------------------------------------------------------------------
# Report + optional chat endpoints

python -m nanochat.report generate

# Talk to the model (optional):
# python -m scripts.chat_cli -p "Why is the sky blue?"
# python -m scripts.chat_web
