"""Tests for stability metrics."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection.recorder import TransitionRecord
from src.data_collection.stability import StabilityMetrics


def _make_record(
    step: int,
    bank_before: dict,
    bank_after: dict,
    operations: list,
    delta_reward: float | None = None,
) -> TransitionRecord:
    return TransitionRecord(
        step=step,
        epoch=0,
        batch=step,
        experience_bank_before=bank_before,
        candidate_experiences=[],
        operations=operations,
        experience_bank_after=bank_after,
        reward_before=0.5 if delta_reward is not None else None,
        reward_after=(0.5 + delta_reward) if delta_reward is not None else None,
        delta_reward=delta_reward,
        task_name="test",
    )


class TestStabilityMetrics:
    def test_rollback_rate(self) -> None:
        records = [
            _make_record(0, {}, {"G0": "a"}, [{"operation": "ADD"}], delta_reward=0.1),
            _make_record(1, {"G0": "a"}, {"G0": "a", "G1": "b"}, [{"operation": "ADD"}], delta_reward=-0.05),
            _make_record(2, {"G0": "a", "G1": "b"}, {"G0": "a"}, [{"operation": "DELETE"}], delta_reward=0.02),
        ]
        metrics = StabilityMetrics(records)
        assert metrics.rollback_rate() == pytest.approx(1 / 3)

    def test_churn_rate(self) -> None:
        records = [
            _make_record(0, {}, {"G0": "a"}, [{"operation": "ADD"}, {"operation": "NONE"}]),
            _make_record(1, {"G0": "a"}, {}, [{"operation": "DELETE"}]),
        ]
        metrics = StabilityMetrics(records)
        # Step 0: 1 ADD + 1 NONE = 0.5 churn, Step 1: 1 DELETE = 1.0 churn
        assert metrics.avg_churn_rate() == pytest.approx(0.75)

    def test_skill_lifecycles(self) -> None:
        records = [
            _make_record(0, {}, {"G0": "a", "G1": "b"}, []),
            _make_record(1, {"G0": "a", "G1": "b"}, {"G0": "a_updated", "G1": "b"}, []),
            _make_record(2, {"G0": "a_updated", "G1": "b"}, {"G0": "a_updated"}, []),
        ]
        metrics = StabilityMetrics(records)
        lifecycles = metrics.skill_lifecycles()

        g0 = next(lc for lc in lifecycles if lc.skill_id == "G0")
        assert g0.is_alive
        assert g0.update_count == 1

        g1 = next(lc for lc in lifecycles if lc.skill_id == "G1")
        assert not g1.is_alive
        assert g1.lifespan == 2  # created at 0, deleted at 2

    def test_bank_size_trajectory(self) -> None:
        records = [
            _make_record(0, {}, {"G0": "a"}, []),
            _make_record(1, {"G0": "a"}, {"G0": "a", "G1": "b"}, []),
        ]
        metrics = StabilityMetrics(records)
        traj = metrics.bank_size_trajectory()
        assert traj[0]["size_before"] == 0
        assert traj[0]["size_after"] == 1
        assert traj[1]["size_after"] == 2

    def test_full_report(self) -> None:
        records = [
            _make_record(0, {}, {"G0": "a"}, [{"operation": "ADD"}], delta_reward=0.1),
        ]
        metrics = StabilityMetrics(records)
        report = metrics.full_report()
        assert "rollback_rate" in report
        assert "avg_churn_rate" in report
        assert report["total_steps"] == 1
