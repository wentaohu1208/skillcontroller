"""Evaluate SFT-trained Skill Controller.

Loads the trained model and evaluates on test set:
- Decision Accuracy (overall)
- Per-class F1 (add / merge / discard)
- JSON Format Rate

Usage:
    python scripts/eval_sft.py \
        --model_path outputs/sft_controller/final \
        --test_data outputs/sft_controller/test_data.jsonl \
        --base_model Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VALID_OPERATIONS = {"add", "merge", "discard"}


def load_model(model_path: str, base_model: str = None):
    """Load trained model (LoRA or full fine-tune).

    Args:
        model_path: Path to saved model.
        base_model: Base model name (needed for LoRA).

    Returns:
        (model, tokenizer)
    """
    # Check if it's a LoRA model
    adapter_config = Path(model_path) / "adapter_config.json"
    is_lora = adapter_config.exists()

    if is_lora:
        logger.info(f"Loading LoRA model from {model_path} (base: {base_model})")
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
    else:
        logger.info(f"Loading full model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, tokenizer, device


def extract_operation(text: str) -> str | None:
    """Extract operation from model output.

    Tries to parse JSON, falls back to regex.

    Returns:
        "add", "merge", "discard", or None if unparseable.
    """
    text = text.strip()

    # Try JSON parse
    try:
        obj = json.loads(text)
        op = obj.get("operation", "")
        if op in VALID_OPERATIONS:
            return op
    except json.JSONDecodeError:
        pass

    # Try to find JSON in text
    match = re.search(r'\{[^}]*"operation"\s*:\s*"(\w+)"[^}]*\}', text)
    if match:
        op = match.group(1)
        if op in VALID_OPERATIONS:
            return op

    # Try plain text match
    for op in VALID_OPERATIONS:
        if op in text.lower():
            return op

    return None


def evaluate(
    model,
    tokenizer,
    device: str,
    test_data: List[Dict[str, Any]],
    max_new_tokens: int = 50,
) -> Dict[str, Any]:
    """Run evaluation on test set.

    Returns:
        Dict with accuracy, per-class metrics, and json_format_rate.
    """
    predictions = []
    ground_truths = []
    json_valid = 0
    total = len(test_data)

    for i, sample in enumerate(test_data):
        prompt = sample["prompt"]
        gt_action = sample["action"]

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        # Parse prediction
        pred_op = extract_operation(generated)

        # Check JSON validity
        try:
            json.loads(generated.strip())
            json_valid += 1
        except (json.JSONDecodeError, ValueError):
            pass

        predictions.append(pred_op)
        ground_truths.append(gt_action)

        if (i + 1) % 10 == 0:
            correct_so_far = sum(p == g for p, g in zip(predictions, ground_truths) if p is not None)
            logger.info(f"  {i+1}/{total} evaluated, running accuracy: {correct_so_far}/{i+1}")

    # Compute metrics
    results = compute_metrics(predictions, ground_truths, json_valid, total)
    return results


def compute_metrics(
    predictions: List[str | None],
    ground_truths: List[str],
    json_valid: int,
    total: int,
) -> Dict[str, Any]:
    """Compute accuracy, per-class F1, and JSON format rate."""

    # Overall accuracy
    correct = sum(p == g for p, g in zip(predictions, ground_truths) if p is not None)
    parseable = sum(1 for p in predictions if p is not None)
    accuracy = correct / total if total > 0 else 0

    # JSON format rate
    json_format_rate = json_valid / total if total > 0 else 0

    # Per-class metrics
    classes = ["add", "merge", "discard"]
    per_class = {}
    for cls in classes:
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p != cls and g == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for g in ground_truths if g == cls),
        }

    # Unparseable rate
    unparseable = sum(1 for p in predictions if p is None)

    return {
        "accuracy": round(accuracy, 4),
        "json_format_rate": round(json_format_rate, 4),
        "parseable_rate": round(parseable / total, 4) if total > 0 else 0,
        "unparseable_count": unparseable,
        "total": total,
        "correct": correct,
        "per_class": per_class,
        "prediction_distribution": dict(Counter(p for p in predictions if p is not None)),
        "ground_truth_distribution": dict(Counter(ground_truths)),
    }


def main(args: argparse.Namespace) -> None:
    """Run evaluation."""

    # 1. Load test data
    test_data = []
    with open(args.test_data, "r") as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    logger.info(f"Loaded {len(test_data)} test samples from {args.test_data}")

    # 2. Load model
    model, tokenizer, device = load_model(args.model_path, args.base_model)

    # 3. Evaluate
    results = evaluate(model, tokenizer, device, test_data, max_new_tokens=args.max_new_tokens)

    # 4. Print results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Accuracy:         {results['accuracy']:.2%}")
    logger.info(f"  JSON Format Rate: {results['json_format_rate']:.2%}")
    logger.info(f"  Parseable Rate:   {results['parseable_rate']:.2%}")
    logger.info(f"  Total / Correct:  {results['total']} / {results['correct']}")
    logger.info("")
    logger.info("  Per-class metrics:")
    for cls, metrics in results["per_class"].items():
        logger.info(
            f"    {cls:10s}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
            f"F1={metrics['f1']:.3f} (support={metrics['support']})"
        )
    logger.info("")
    logger.info(f"  Prediction distribution: {results['prediction_distribution']}")
    logger.info(f"  Ground truth distribution: {results['ground_truth_distribution']}")

    # 5. Save results
    output_path = Path(args.model_path).parent / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SFT-trained Skill Controller")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test JSONL")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Base model (for LoRA)")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max new tokens to generate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
