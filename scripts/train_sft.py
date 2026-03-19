"""SFT training for Skill Controller.

Trains a small LM to make add/merge/discard decisions for skill bank management.

Usage:
    # LoRA training (recommended for <1500 samples)
    python scripts/train_sft.py \
        --model_name Qwen/Qwen2.5-3B-Instruct \
        --data_path data/training_data/training_data_lm.jsonl \
        --output_dir outputs/sft_controller_lora \
        --epochs 2 \
        --batch_size 4 \
        --lr 2e-4 \
        --lora_r 16

    # Full fine-tune (for >1500 samples)
    python scripts/train_sft.py \
        --model_name Qwen/Qwen2.5-3B-Instruct \
        --data_path data/training_data/training_data_lm.jsonl \
        --output_dir outputs/sft_controller_full \
        --epochs 2 \
        --batch_size 4 \
        --lr 2e-5 \
        --full_finetune
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_and_split_data(data_path: str, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
    """Load JSONL data and split into train/val/test.

    Args:
        data_path: Path to training_data_lm.jsonl.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test.
        seed: Random seed.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    ds = load_dataset("json", data_files=data_path, split="train")
    logger.info(f"Loaded {len(ds)} samples from {data_path}")

    # Log action distribution
    actions = [s["action"] for s in ds]
    from collections import Counter
    dist = Counter(actions)
    logger.info(f"Action distribution: {dict(dist)}")

    # Split: train / val / test
    test_size = int(len(ds) * test_ratio)
    val_size = int(len(ds) * val_ratio)

    ds_shuffled = ds.shuffle(seed=seed)
    test_ds = ds_shuffled.select(range(test_size))
    val_ds = ds_shuffled.select(range(test_size, test_size + val_size))
    train_ds = ds_shuffled.select(range(test_size + val_size, len(ds)))

    logger.info(f"Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_ds, val_ds, test_ds


def format_sample(sample: dict) -> str:
    """Format a single sample as prompt + completion for SFT.

    Input format from data_converter.py:
        {"prompt": "Current Skill Bank...", "completion": '{"operation": "merge"}', "action": "merge"}

    Output: prompt + newline + completion
    """
    return f"{sample['prompt']}\n{sample['completion']}"


def main(args: argparse.Namespace) -> None:
    """Run SFT training."""
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    # 1. Load data
    train_ds, val_ds, test_ds = load_and_split_data(
        args.data_path, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )

    # Save test set for later evaluation
    test_path = Path(args.output_dir) / "test_data.jsonl"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_path, "w") as f:
        for sample in test_ds:
            f.write(json.dumps(sample, ensure_ascii=False, default=str) + "\n")
    logger.info(f"Saved test set ({len(test_ds)} samples) to {test_path}")

    # 2. Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. LoRA config (if not full fine-tune)
    peft_config = None
    if not args.full_finetune:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        logger.info(f"Using LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    else:
        logger.info("Using full fine-tune (no LoRA)")

    # 4. Training config
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        bf16=args.bf16,
        fp16=not args.bf16,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=args.seed,
        report_to="none",
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        formatting_func=format_sample,
    )

    # 6. Train
    logger.info("Starting SFT training...")
    trainer.train()

    # 7. Save
    final_path = Path(args.output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info(f"Model saved to {final_path}")

    # 8. Log final metrics
    metrics = trainer.evaluate()
    logger.info(f"Final eval metrics: {metrics}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT training for Skill Controller")

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Base model name")

    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to training_data_lm.jsonl")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio")

    # Training
    parser.add_argument("--output_dir", type=str, default="outputs/sft_controller", help="Output directory")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bf16 (default for A800)")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # LoRA vs Full
    parser.add_argument("--full_finetune", action="store_true", help="Full fine-tune instead of LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
