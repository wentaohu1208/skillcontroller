#!/bin/bash
# Download all directly-available datasets from HuggingFace Mirror
# Usage: bash download_datasets.sh

set -e

export HF_ENDPOINT=https://hf-mirror.com
SAVE_DIR="/data/hwt/hf_data"
mkdir -p "$SAVE_DIR"

echo "============================================"
echo "Downloading datasets to $SAVE_DIR"
echo "Using mirror: $HF_ENDPOINT"
echo "============================================"

# pip install -U huggingface_hub  # 如果没装的话

# === 通用对话 ===
echo "[1/18] WildChat-1M (~50GB, 最大)"
huggingface-cli download allenai/WildChat-1M --repo-type dataset --local-dir "$SAVE_DIR/WildChat-1M"

echo "[2/18] LMSYS-Chat-1M (~30GB)"
huggingface-cli download lmsys/lmsys-chat-1m --repo-type dataset --local-dir "$SAVE_DIR/lmsys-chat-1m"

echo "[3/18] OpenAssistant (~2GB)"
huggingface-cli download OpenAssistant/oasst1 --repo-type dataset --local-dir "$SAVE_DIR/oasst1"

# === 数学 ===
echo "[4/18] GSM8K (~5MB)"
huggingface-cli download openai/gsm8k --repo-type dataset --local-dir "$SAVE_DIR/gsm8k"

echo "[5/18] MATH (~20MB)"
huggingface-cli download hendrycks/competition_math --repo-type dataset --local-dir "$SAVE_DIR/competition_math"

echo "[6/18] DAPO-Math-17k (~50MB)"
huggingface-cli download BytedTsinghua-SIA/DAPO-Math-17k --repo-type dataset --local-dir "$SAVE_DIR/DAPO-Math-17k"

echo "[7/18] NuminaMath-CoT (~3GB)"
huggingface-cli download AI-MO/NuminaMath-CoT --repo-type dataset --local-dir "$SAVE_DIR/NuminaMath-CoT"

echo "[8/18] MathInstruct (~500MB)"
huggingface-cli download TIGER-Lab/MathInstruct --repo-type dataset --local-dir "$SAVE_DIR/MathInstruct"

echo "[9/18] AIME25 (~1MB)"
huggingface-cli download math-ai/aime25 --repo-type dataset --local-dir "$SAVE_DIR/aime25"

# === 物理/科学 ===
echo "[10/18] ScienceQA (~2GB)"
huggingface-cli download derek-thomas/ScienceQA --repo-type dataset --local-dir "$SAVE_DIR/ScienceQA"

echo "[11/18] GPQA (~1MB)"
huggingface-cli download Idavidrein/gpqa --repo-type dataset --local-dir "$SAVE_DIR/gpqa"

echo "[12/18] TheoremQA (~2MB)"
huggingface-cli download TIGER-Lab/TheoremQA --repo-type dataset --local-dir "$SAVE_DIR/TheoremQA"

# === 编程 ===
echo "[13/18] HumanEval (~200KB)"
huggingface-cli download openai/openai_humaneval --repo-type dataset --local-dir "$SAVE_DIR/humaneval"

echo "[14/18] MBPP (~1MB)"
huggingface-cli download google-research-datasets/mbpp --repo-type dataset --local-dir "$SAVE_DIR/mbpp"

# === 搜索/推理 ===
echo "[15/18] HotpotQA (~600MB)"
huggingface-cli download hotpotqa/hotpot_qa --repo-type dataset --local-dir "$SAVE_DIR/hotpot_qa"

# === 写作/指令 ===
echo "[16/18] Alpaca (~50MB)"
huggingface-cli download tatsu-lab/alpaca --repo-type dataset --local-dir "$SAVE_DIR/alpaca"

echo "[17/18] Dolly (~10MB)"
huggingface-cli download databricks/databricks-dolly-15k --repo-type dataset --local-dir "$SAVE_DIR/dolly"

echo "[18/18] UltraFeedback (~300MB)"
huggingface-cli download openbmb/UltraFeedback --repo-type dataset --local-dir "$SAVE_DIR/UltraFeedback"

echo ""
echo "============================================"
echo "All downloads complete."
echo "Data saved to: $SAVE_DIR"
du -sh "$SAVE_DIR"/*
echo "============================================"
