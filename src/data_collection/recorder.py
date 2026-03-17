"""Transition recorder for collecting (state, action, outcome) tuples.

Wraps around Training-Free GRPO's ExperienceUpdater to intercept
every controller decision and its downstream effect.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TransitionRecord:
    """A single (state, action, outcome) record.

    Args:
        step: Global step index.
        epoch: Epoch index.
        batch: Batch index within epoch.
        experience_bank_before: Snapshot of experience bank before controller action.
        candidate_experiences: A_text from Group Computation.
        operations: Controller's decisions (list of {operation, id, content}).
        experience_bank_after: Snapshot of experience bank after controller action.
        reward_before: Agent performance before update (on held-out data).
        reward_after: Agent performance after update (on held-out data).
        delta_reward: reward_after - reward_before.
        rollout_stats: Statistics from the rollout batch.
        task_name: Which task this came from.
    """

    step: int
    epoch: int
    batch: int
    experience_bank_before: dict[str, str]
    candidate_experiences: list[str]
    operations: list[dict[str, Any]]
    experience_bank_after: dict[str, str]
    reward_before: Optional[float] = None
    reward_after: Optional[float] = None
    delta_reward: Optional[float] = None
    rollout_stats: dict[str, Any] = field(default_factory=dict)
    task_name: str = ""

    @property
    def is_positive(self) -> bool:
        """Whether this transition improved performance."""
        if self.delta_reward is None:
            return False
        return self.delta_reward > 0

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "batch": self.batch,
            "experience_bank_before": self.experience_bank_before,
            "candidate_experiences": self.candidate_experiences,
            "operations": self.operations,
            "experience_bank_after": self.experience_bank_after,
            "reward_before": self.reward_before,
            "reward_after": self.reward_after,
            "delta_reward": self.delta_reward,
            "rollout_stats": self.rollout_stats,
            "task_name": self.task_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TransitionRecord:
        return cls(**data)


class TransitionRecorder:
    """Collects transition records across Training-Free GRPO runs.

    Args:
        save_dir: Directory to save recorded transitions.
        experiment_name: Name of the experiment.
    """

    def __init__(self, save_dir: str | Path, experiment_name: str) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.records: list[TransitionRecord] = []

    def record(self, transition: TransitionRecord) -> None:
        """Add a transition record."""
        self.records.append(transition)
        logger.info(
            f"[{self.experiment_name}] Step {transition.step}: "
            f"delta_reward={transition.delta_reward}, "
            f"ops={len(transition.operations)}, "
            f"bank_size={len(transition.experience_bank_after)}"
        )

    def save(self) -> Path:
        """Save all records to JSON."""
        filepath = self.save_dir / f"{self.experiment_name}_transitions.json"
        data = [r.to_dict() for r in self.records]
        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        logger.info(f"Saved {len(data)} transitions to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: str | Path) -> list[TransitionRecord]:
        """Load records from JSON file."""
        data = json.loads(Path(filepath).read_text())
        return [TransitionRecord.from_dict(d) for d in data]

    def get_positive_records(self) -> list[TransitionRecord]:
        """Get records where delta_reward > 0."""
        return [r for r in self.records if r.is_positive]

    def get_negative_records(self) -> list[TransitionRecord]:
        """Get records where delta_reward <= 0."""
        return [r for r in self.records if r.delta_reward is not None and not r.is_positive]

    def summary(self) -> dict[str, Any]:
        """Summary statistics of collected data."""
        total = len(self.records)
        with_reward = [r for r in self.records if r.delta_reward is not None]
        positive = self.get_positive_records()
        negative = self.get_negative_records()

        return {
            "total_records": total,
            "with_reward_signal": len(with_reward),
            "positive_transitions": len(positive),
            "negative_transitions": len(negative),
            "positive_rate": len(positive) / len(with_reward) if with_reward else 0,
            "avg_delta_reward": (
                sum(r.delta_reward for r in with_reward) / len(with_reward) if with_reward else 0
            ),
            "tasks": list(set(r.task_name for r in self.records)),
        }
