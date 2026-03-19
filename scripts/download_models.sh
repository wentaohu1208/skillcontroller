#!/bin/bash
# Download Qwen2.5 Instruct models from HuggingFace Mirror
# Usage: bash scripts/download_models.sh

set +e

export HF_ENDPOINT=https://hf-mirror.com
SAVE_DIR="/data/hwt/hf_ckpt"
mkdir -p "$SAVE_DIR"

echo "============================================"
echo "Downloading Qwen2.5 models to $SAVE_DIR"
echo "Using mirror: $HF_ENDPOINT"
echo "============================================"

echo "[1/3] Qwen2.5-1.5B-Instruct"
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir "$SAVE_DIR/Qwen2.5-1.5B-Instruct"

echo "[2/3] Qwen2.5-3B-Instruct"
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir "$SAVE_DIR/Qwen2.5-3B-Instruct"

echo "[3/3] Qwen2.5-7B-Instruct"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir "$SAVE_DIR/Qwen2.5-7B-Instruct"

echo ""
echo "============================================"
echo "All downloads complete."
du -sh "$SAVE_DIR"/*
echo "============================================"
