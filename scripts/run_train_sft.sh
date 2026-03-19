#!/bin/bash
eval "$(conda shell.bash hook)" && \
conda activate /data/hwt/envs/skillcontroller && \
cd /data/hwt/skillcontroller && \
python scripts/train_sft.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --data_path /data/hwt/AutoSkill/data/training_data/training_data_lm.jsonl \
    --output_dir outputs/sft_controller_lora \
    --epochs 2 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 2e-4 \
    --lora_r 16 \
    --max_seq_length 4096 \
    --bf16
