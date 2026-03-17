"""Phase 1.5: Analyze collected transition data for controller instability.

Reads transition records from collect_data.py output and generates
detailed stability analysis with visualizations.

Usage:
    python scripts/analyze_stability.py \
        --data_path data/collected/math_stability_v1_transitions.json \
        --output_dir data/analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.recorder import TransitionRecorder, TransitionRecord
from src.data_collection.stability import StabilityMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


def analyze_operation_patterns(records: list[TransitionRecord]) -> dict:
    """Analyze which operation patterns correlate with reward changes."""
    op_reward_map = {"ADD": [], "UPDATE": [], "DELETE": [], "NONE": []}

    for record in records:
        if record.delta_reward is None:
            continue
        # Count operations in this step
        op_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0}
        for op in record.operations:
            op_type = op.get("operation", "NONE")
            if op_type in op_counts:
                op_counts[op_type] += 1

        # Associate each operation type with the delta_reward
        for op_type, count in op_counts.items():
            if count > 0:
                op_reward_map[op_type].append({
                    "step": record.step,
                    "count": count,
                    "delta_reward": record.delta_reward,
                })

    # Compute statistics per operation type
    results = {}
    for op_type, entries in op_reward_map.items():
        if not entries:
            results[op_type] = {"count": 0, "avg_delta": None, "positive_rate": None}
            continue

        deltas = [e["delta_reward"] for e in entries]
        results[op_type] = {
            "count": len(entries),
            "avg_delta": sum(deltas) / len(deltas),
            "positive_rate": sum(1 for d in deltas if d > 0) / len(deltas),
            "max_negative": min(deltas) if deltas else None,
            "max_positive": max(deltas) if deltas else None,
        }

    return results


def analyze_bank_growth(records: list[TransitionRecord]) -> dict:
    """Analyze how the experience bank grows over time."""
    trajectory = []
    for record in sorted(records, key=lambda r: r.step):
        before_size = len(record.experience_bank_before)
        after_size = len(record.experience_bank_after)

        # Count content changes (not just key changes)
        unchanged = 0
        modified = 0
        for key in set(record.experience_bank_before.keys()) & set(record.experience_bank_after.keys()):
            if record.experience_bank_before[key] == record.experience_bank_after[key]:
                unchanged += 1
            else:
                modified += 1

        added = after_size - before_size + (before_size - unchanged - modified)
        deleted = before_size - after_size + (after_size - unchanged - modified)

        trajectory.append({
            "step": record.step,
            "size_before": before_size,
            "size_after": after_size,
            "net_change": after_size - before_size,
            "added": added,
            "deleted": deleted,
            "modified": modified,
            "unchanged": unchanged,
        })

    return {
        "trajectory": trajectory,
        "final_size": trajectory[-1]["size_after"] if trajectory else 0,
        "total_adds": sum(t["added"] for t in trajectory),
        "total_deletes": sum(t["deleted"] for t in trajectory),
        "total_modifies": sum(t["modified"] for t in trajectory),
    }


def analyze_experience_volatility(records: list[TransitionRecord]) -> dict:
    """Find experiences that are frequently modified or deleted (volatile)."""
    # Track each experience ID's history
    history: dict[str, list[dict]] = {}

    for record in sorted(records, key=lambda r: r.step):
        after = record.experience_bank_after
        before = record.experience_bank_before

        # Track new appearances
        for key in set(after.keys()) - set(before.keys()):
            if key not in history:
                history[key] = []
            history[key].append({"step": record.step, "event": "ADD", "content": after[key]})

        # Track deletions
        for key in set(before.keys()) - set(after.keys()):
            if key not in history:
                history[key] = []
            history[key].append({"step": record.step, "event": "DELETE", "content": before[key]})

        # Track modifications
        for key in set(before.keys()) & set(after.keys()):
            if before[key] != after[key]:
                if key not in history:
                    history[key] = []
                history[key].append({"step": record.step, "event": "UPDATE", "content": after[key]})

    # Find most volatile experiences
    volatile = []
    for exp_id, events in history.items():
        volatile.append({
            "exp_id": exp_id,
            "num_events": len(events),
            "events": events,
        })

    volatile.sort(key=lambda x: x["num_events"], reverse=True)

    return {
        "total_unique_experiences": len(history),
        "most_volatile": volatile[:10],
        "avg_events_per_experience": (
            sum(len(events) for events in history.values()) / len(history) if history else 0
        ),
    }


def main(args: argparse.Namespace) -> None:
    """Run stability analysis."""
    # Load records
    records = TransitionRecorder.load(args.data_path)
    logger.info(f"Loaded {len(records)} transition records from {args.data_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Basic stability metrics
    logger.info("=" * 60)
    logger.info("1. BASIC STABILITY METRICS")
    logger.info("=" * 60)

    metrics = StabilityMetrics(records)
    report = metrics.full_report()

    logger.info(f"  Rollback rate:      {report['rollback_rate']:.2%}")
    logger.info(f"  Avg churn rate:     {report['avg_churn_rate']:.2%}")
    avg_ls = report['avg_skill_lifespan']
    logger.info(f"  Avg skill lifespan: {f'{avg_ls:.1f} steps' if avg_ls else 'N/A'}")

    # 2. Operation-reward correlation
    logger.info("=" * 60)
    logger.info("2. OPERATION-REWARD CORRELATION")
    logger.info("=" * 60)

    op_analysis = analyze_operation_patterns(records)
    for op_type, stats in op_analysis.items():
        if stats["count"] > 0:
            logger.info(
                f"  {op_type:8s}: count={stats['count']:3d}, "
                f"avg_delta={stats['avg_delta']:+.4f}, "
                f"positive_rate={stats['positive_rate']:.2%}"
            )

    # 3. Bank growth analysis
    logger.info("=" * 60)
    logger.info("3. BANK GROWTH ANALYSIS")
    logger.info("=" * 60)

    growth = analyze_bank_growth(records)
    logger.info(f"  Final bank size:  {growth['final_size']}")
    logger.info(f"  Total ADDs:       {growth['total_adds']}")
    logger.info(f"  Total DELETEs:    {growth['total_deletes']}")
    logger.info(f"  Total MODIFYs:    {growth['total_modifies']}")

    # 4. Experience volatility
    logger.info("=" * 60)
    logger.info("4. EXPERIENCE VOLATILITY")
    logger.info("=" * 60)

    volatility = analyze_experience_volatility(records)
    logger.info(f"  Unique experiences seen: {volatility['total_unique_experiences']}")
    logger.info(f"  Avg events/experience:   {volatility['avg_events_per_experience']:.2f}")

    if volatility["most_volatile"]:
        logger.info("  Most volatile:")
        for v in volatility["most_volatile"][:5]:
            events_summary = ", ".join(e["event"] for e in v["events"])
            logger.info(f"    {v['exp_id']}: {v['num_events']} events ({events_summary})")

    # 5. Save full analysis
    full_analysis = {
        "basic_metrics": report,
        "operation_patterns": op_analysis,
        "bank_growth": growth,
        "volatility": volatility,
    }

    analysis_path = output_dir / f"{Path(args.data_path).stem}_analysis.json"
    analysis_path.write_text(json.dumps(full_analysis, indent=2, default=str, ensure_ascii=False))
    logger.info(f"\nFull analysis saved to: {analysis_path}")

    # 6. Key findings summary
    logger.info("=" * 60)
    logger.info("KEY FINDINGS")
    logger.info("=" * 60)

    if report["rollback_rate"] > 0.3:
        logger.info("  [!] HIGH rollback rate (>30%): Controller updates frequently hurt performance")
    elif report["rollback_rate"] > 0.1:
        logger.info("  [~] MODERATE rollback rate (10-30%): Some instability in controller updates")
    else:
        logger.info("  [ok] LOW rollback rate (<10%): Controller appears relatively stable")

    if report["avg_churn_rate"] > 0.5:
        logger.info("  [!] HIGH churn rate (>50%): Experience bank is very volatile")

    # Check if DELETE operations correlate with reward drops
    if op_analysis["DELETE"]["count"] > 0 and op_analysis["DELETE"]["avg_delta"] is not None:
        if op_analysis["DELETE"]["avg_delta"] < 0:
            logger.info("  [!] DELETE operations correlate with reward drops — potential harmful deletions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Training-Free GRPO transition data for stability",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to transitions JSON")
    parser.add_argument("--output_dir", type=str, default="data/analysis", help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
