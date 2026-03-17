"""Phase 1: Collect (state, action, outcome) transitions from Training-Free GRPO.

Runs instrumented Training-Free GRPO on math tasks to record every
controller decision and its downstream effect on agent performance.

Usage:
    # From youtu-agent environment (with dependencies installed):
    python scripts/collect_data.py \
        --config_name math_reasoning \
        --experiment_name math_stability_v1 \
        --save_dir data/collected \
        --held_out_eval \
        --held_out_size 20

    # Without held-out eval (faster, no delta_reward signal):
    python scripts/collect_data.py \
        --config_name math_reasoning \
        --experiment_name math_no_eval_v1 \
        --save_dir data/collected
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.recorder import TransitionRecorder
from src.data_collection.stability import StabilityMetrics
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

YOUTU_AGENT_PATH = "/Users/wentaohu/project/youtu-agent"


async def main(args: argparse.Namespace) -> None:
    """Run data collection pipeline."""
    set_seed(args.seed)

    # 1. Setup recorder
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    recorder = TransitionRecorder(
        save_dir=save_dir,
        experiment_name=args.experiment_name,
    )

    # 2. Build instrumented GRPO
    from src.data_collection.instrumented_grpo import InstrumentedGRPO

    config_overrides = {}
    if args.epochs is not None:
        config_overrides["practice.epochs"] = args.epochs
    if args.batch_size is not None:
        config_overrides["practice.batch_size"] = args.batch_size
    if args.grpo_n is not None:
        config_overrides["practice.grpo_n"] = args.grpo_n
    if args.restart_step is not None:
        config_overrides["practice.restart_step"] = args.restart_step

    instrumented = InstrumentedGRPO(
        youtu_agent_path=YOUTU_AGENT_PATH,
        config_name=args.config_name,
        recorder=recorder,
        held_out_eval=args.held_out_eval,
        held_out_size=args.held_out_size,
        config_overrides=config_overrides or None,
    )

    # 3. Run
    logger.info(f"Starting data collection: {args.experiment_name}")
    logger.info(f"Config: {args.config_name}, held_out_eval={args.held_out_eval}")

    agent_config_path = await instrumented.run()
    logger.info(f"Agent config saved to: {agent_config_path}")

    # 4. Analyze stability
    logger.info("=" * 60)
    logger.info("STABILITY ANALYSIS")
    logger.info("=" * 60)

    metrics = StabilityMetrics(recorder.records)
    report = metrics.full_report()

    # Print summary
    logger.info(f"Total steps recorded: {report['total_steps']}")
    logger.info(f"Rollback rate: {report['rollback_rate']:.2%}")
    logger.info(f"Avg churn rate: {report['avg_churn_rate']:.2%}")
    avg_lifespan = report['avg_skill_lifespan']
    logger.info(f"Avg skill lifespan: {avg_lifespan:.1f} steps" if avg_lifespan else "Avg skill lifespan: N/A")

    # Reward trajectory
    reward_traj = report["reward_trajectory"]
    if reward_traj:
        deltas = [r["delta"] for r in reward_traj if r["delta"] is not None]
        if deltas:
            positive = sum(1 for d in deltas if d > 0)
            negative = sum(1 for d in deltas if d < 0)
            logger.info(f"Reward deltas: {positive} positive, {negative} negative, avg={sum(deltas)/len(deltas):.4f}")

    # Save full report
    report_path = save_dir / f"{args.experiment_name}_stability_report.json"
    # Convert non-serializable values
    serializable_report = json.loads(json.dumps(report, default=str))
    report_path.write_text(json.dumps(serializable_report, indent=2, ensure_ascii=False))
    logger.info(f"Full stability report saved to: {report_path}")

    # Print recorder summary
    summary = recorder.summary()
    logger.info(f"Recorder summary: {json.dumps(summary, indent=2, default=str)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Training-Free GRPO transitions for controller training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config_name", type=str, default="math_reasoning", help="Practice config name")
    parser.add_argument("--experiment_name", type=str, default="math_stability_v1", help="Experiment name")
    parser.add_argument("--save_dir", type=str, default="data/collected", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Held-out eval
    parser.add_argument("--held_out_eval", action="store_true", help="Run held-out eval for delta_reward")
    parser.add_argument("--held_out_size", type=int, default=20, help="Held-out eval sample count")

    # GRPO overrides
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--grpo_n", type=int, default=None, help="Override grpo_n")
    parser.add_argument("--restart_step", type=int, default=None, help="Restart from step N")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
