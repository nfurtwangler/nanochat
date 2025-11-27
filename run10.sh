#!/bin/bash

# The ~$10 tier of nanochat.
# Supports `--gpu=a100` (default, 80GB) and `--gpu=5090` (32GB). Both follow the
# same training recipe (depth 12), but the 5090 path lowers device batch sizes
# so it fits the smaller VRAM footprint while keeping total batch size constant.
# Budget target: ~12 hours of uninterrupted run time on the chosen GPU (~$10).

GPU_TYPE="a100"
for arg in "$@"; do
    case $arg in
        --gpu=*)
            GPU_TYPE="${arg#*=}"
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

# Grab enough shards for ~450M tokens of training and keep downloading extras
python -m nanochat.dataset -n 4
python -m nanochat.dataset -n 80 &
DATASET_DOWNLOAD_PID=$!

# Train tokenizer on ~1B characters (still finishes quickly on A100 boxes)
python -m scripts.tok_train --max_chars=1000000000
python -m scripts.tok_eval

echo "Waiting for dataset background download to finish..."
wait $DATASET_DOWNLOAD_PID

# -----------------------------------------------------------------------------
# Base model (~260M params at depth 12, ~520M tokens of training)

BASE_DEPTH=12
if [[ "$GPU_TYPE" == "5090" ]]; then
    BASE_DEVICE_BATCH=2
else
    BASE_DEVICE_BATCH=8
fi
BASE_TOTAL_BATCH=131072
BASE_ITERS=4000
BASE_EVAL_TOKENS=262144

python -m scripts.base_train \
    --depth=$BASE_DEPTH \
    --device_batch_size=$BASE_DEVICE_BATCH \
    --total_batch_size=$BASE_TOTAL_BATCH \
    --num_iterations=$BASE_ITERS \
    --eval_every=300 \
    --eval_tokens=$BASE_EVAL_TOKENS \
    --core_metric_every=-1 \
    --sample_every=-1 \
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

python -m scripts.chat_eval -i mid -x 96 -m 256 -t 0.0

# -----------------------------------------------------------------------------
# Supervised finetuning (short pass over curated dialogs)

python -m scripts.chat_sft \
    --device_batch_size=$BASE_DEVICE_BATCH \
    --target_examples_per_step=64 \
    --num_iterations=600 \
    --eval_steps=150 \
    --eval_metrics_max_problems=96 \
    --run=$WANDB_RUN

python -m scripts.chat_eval -i sft -x 96 -m 256 -t 0.0

# -----------------------------------------------------------------------------
# Report + optional chat endpoints

python -m nanochat.report generate

# Talk to the model (optional):
# python -m scripts.chat_cli -p "Why is the sky blue?"
# python -m scripts.chat_web
