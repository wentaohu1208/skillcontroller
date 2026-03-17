"""Stability metrics for experience/skill bank evolution.

Measures how stable the controller's updates are across steps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .recorder import TransitionRecord

logger = logging.getLogger(__name__)


@dataclass
class SkillLifecycle:
    """Track a single skill's lifecycle.

    Args:
        skill_id: Original ID when first added.
        content_history: List of (step, content) showing how content evolved.
        created_at: Step when first added.
        deleted_at: Step when deleted (None if still alive).
        update_count: Number of times updated.
    """

    skill_id: str
    content_history: list[tuple[int, str]] = field(default_factory=list)
    created_at: int = 0
    deleted_at: int | None = None
    update_count: int = 0

    @property
    def lifespan(self) -> int | None:
        """Number of steps the skill survived. None if still alive."""
        if self.deleted_at is None:
            return None
        return self.deleted_at - self.created_at

    @property
    def is_alive(self) -> bool:
        return self.deleted_at is None


class StabilityMetrics:
    """Compute stability metrics from transition records.

    Metrics:
    1. Rollback rate: fraction of steps where update decreased performance
    2. Skill churn: rate of ADD + DELETE per step
    3. Skill survival: distribution of skill lifespans
    4. Reward trajectory: reward over steps (monotonic increase = stable)
    5. Bank size trajectory: how bank size evolves
    """

    def __init__(self, records: list[TransitionRecord]) -> None:
        self.records = sorted(records, key=lambda r: r.step)

    def rollback_rate(self) -> float:
        """Fraction of steps where controller update decreased performance."""
        with_reward = [r for r in self.records if r.delta_reward is not None]
        if not with_reward:
            return 0.0
        negative = sum(1 for r in with_reward if r.delta_reward < 0)
        return negative / len(with_reward)

    def skill_churn_per_step(self) -> list[dict[str, int]]:
        """Count ADD/DELETE/UPDATE/NONE operations per step."""
        results = []
        for record in self.records:
            counts: dict[str, int] = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0}
            for op in record.operations:
                op_type = op.get("operation", "NONE")
                if op_type in counts:
                    counts[op_type] += 1
            counts["step"] = record.step
            results.append(counts)
        return results

    def avg_churn_rate(self) -> float:
        """Average (ADD + DELETE) / total_operations per step."""
        churn_data = self.skill_churn_per_step()
        if not churn_data:
            return 0.0
        rates = []
        for step_data in churn_data:
            total = step_data["ADD"] + step_data["UPDATE"] + step_data["DELETE"] + step_data["NONE"]
            if total > 0:
                rates.append((step_data["ADD"] + step_data["DELETE"]) / total)
        return sum(rates) / len(rates) if rates else 0.0

    def skill_lifecycles(self) -> list[SkillLifecycle]:
        """Track each skill's lifecycle across steps."""
        lifecycles: dict[str, SkillLifecycle] = {}

        for record in self.records:
            # Diff before and after to detect ADD/DELETE/UPDATE
            before = set(record.experience_bank_before.keys())
            after = set(record.experience_bank_after.keys())

            # New skills (ADD)
            for sid in after - before:
                lifecycles[sid] = SkillLifecycle(
                    skill_id=sid,
                    content_history=[(record.step, record.experience_bank_after[sid])],
                    created_at=record.step,
                )

            # Deleted skills
            for sid in before - after:
                if sid in lifecycles:
                    lifecycles[sid].deleted_at = record.step

            # Updated skills (same key, different content)
            for sid in before & after:
                if record.experience_bank_before[sid] != record.experience_bank_after[sid]:
                    if sid in lifecycles:
                        lifecycles[sid].update_count += 1
                        lifecycles[sid].content_history.append(
                            (record.step, record.experience_bank_after[sid])
                        )

        return list(lifecycles.values())

    def avg_skill_lifespan(self) -> float | None:
        """Average lifespan of deleted skills (in steps)."""
        lifecycles = self.skill_lifecycles()
        dead = [lc for lc in lifecycles if lc.lifespan is not None]
        if not dead:
            return None
        return sum(lc.lifespan for lc in dead) / len(dead)

    def reward_trajectory(self) -> list[dict[str, Any]]:
        """Reward values over steps."""
        return [
            {"step": r.step, "reward_before": r.reward_before, "reward_after": r.reward_after, "delta": r.delta_reward}
            for r in self.records
            if r.delta_reward is not None
        ]

    def bank_size_trajectory(self) -> list[dict[str, int]]:
        """Bank size over steps."""
        return [
            {"step": r.step, "size_before": len(r.experience_bank_before), "size_after": len(r.experience_bank_after)}
            for r in self.records
        ]

    def full_report(self) -> dict[str, Any]:
        """Generate a comprehensive stability report."""
        return {
            "rollback_rate": self.rollback_rate(),
            "avg_churn_rate": self.avg_churn_rate(),
            "avg_skill_lifespan": self.avg_skill_lifespan(),
            "reward_trajectory": self.reward_trajectory(),
            "bank_size_trajectory": self.bank_size_trajectory(),
            "churn_per_step": self.skill_churn_per_step(),
            "total_steps": len(self.records),
        }
