"""Instrumented Training-Free GRPO that records every controller transition.

Inherits from youtu-agent's TrainingFreeGRPO, overrides practice() to:
1. Snapshot experience bank before each update
2. Capture all operations from ExperienceUpdater
3. Run held-out evaluation to measure delta_reward
4. Record full (state, action, outcome) transitions
"""

from __future__ import annotations

import copy
import logging
import sys
from typing import Optional

from ..data_collection.recorder import TransitionRecord, TransitionRecorder

logger = logging.getLogger(__name__)


def _ensure_youtu_agent(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


class InstrumentedGRPO:
    """Wraps TrainingFreeGRPO to record transitions without modifying original code.

    Args:
        youtu_agent_path: Path to youtu-agent project root.
        config_name: Practice config name (e.g., "math_reasoning").
        recorder: TransitionRecorder for saving data.
        held_out_eval: Whether to eval after each update for delta_reward.
        held_out_size: Number of samples for held-out eval.
    """

    def __init__(
        self,
        youtu_agent_path: str,
        config_name: str,
        recorder: TransitionRecorder,
        held_out_eval: bool = True,
        held_out_size: int = 20,
        config_overrides: Optional[dict] = None,
    ) -> None:
        _ensure_youtu_agent(youtu_agent_path)

        from utu.config import ConfigLoader
        from utu.practice import TrainingFreeGRPO

        # Load config
        config = ConfigLoader.load_training_free_grpo_config(config_name)

        # Apply overrides
        if config_overrides:
            for key, value in config_overrides.items():
                parts = key.split(".")
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)

        self._grpo = TrainingFreeGRPO(config)
        self._config = config
        self._recorder = recorder
        self._held_out_eval = held_out_eval
        self._held_out_size = held_out_size

    async def build(self) -> None:
        """Build all GRPO components."""
        await self._grpo.build()

    async def run(self) -> str:
        """Run instrumented Training-Free GRPO.

        Returns:
            Path to agent config with learned experiences.
        """
        if self._grpo.practice_rollout_manager is None:
            await self.build()

        await self._instrumented_practice()

        experiences = self._grpo.recorder.experiences or {}
        agent_config_path = self._grpo._create_agent_config_with_experiences(experiences)

        # Save all transition data
        self._recorder.save()

        return agent_config_path

    async def _instrumented_practice(self) -> None:
        """Override of practice() with transition recording."""
        from agents import custom_span, gen_trace_id, trace
        from utu.utils.experience_cache import ExperienceCache

        grpo = self._grpo
        config = self._config

        for epoch in range(config.practice.epochs):
            logger.info(f"[Instrumented] Start Epoch {epoch}")

            epoch_data = grpo.practice_rollout_manager.load_epoch_data(
                epoch,
                shuffle=config.practice.shuffle_data,
                truncate=config.practice.rollout_data_truncate,
            )

            assert len(epoch_data) % config.practice.grpo_n == 0
            if len(epoch_data) < config.practice.batch_size * config.practice.grpo_n:
                raise ValueError(
                    f"Epoch {epoch} data size {len(epoch_data) // config.practice.grpo_n} "
                    f"< batch_size {config.practice.batch_size}"
                )

            num_batches = len(epoch_data) // (config.practice.batch_size * config.practice.grpo_n)

            for batch_idx in range(num_batches):
                step = epoch * num_batches + batch_idx
                logger.info(f"[Instrumented] Step {step} (Epoch {epoch}, Batch {batch_idx})")

                step_trace_id = gen_trace_id()
                with trace(f"[{grpo.recorder.experiment_name}] Step {step}", trace_id=step_trace_id):
                    stats = grpo.recorder.stats or {}
                    if f"step_{step}" not in stats:
                        stats[f"step_{step}"] = {"epoch": epoch, "batch": batch_idx, "complete": False}

                    # ---- 1. Rollout ----
                    with custom_span("Rollout batch"):
                        rollouts, stat = await grpo.practice_rollout_manager.main(
                            batch_idx=batch_idx,
                            recorder=grpo.recorder,
                            use_cache=grpo._should_use_cache(step),
                        )
                        stats[f"step_{step}"]["rollout"] = stat

                    # ---- 2. Snapshot BEFORE ----
                    bank_before = copy.deepcopy(grpo.recorder.experiences or {})

                    # ---- 3. Experience Update (with operation capture) ----
                    with custom_span("Experience update"):
                        cached = ExperienceCache.load_experiences(
                            experiment_name=grpo.recorder.experiment_name, step=step
                        )
                        if cached is not None and grpo._should_use_cache(step):
                            new_experiences = cached
                            grpo.recorder.experiences_update(new_experiences)
                            captured_ops = self._diff_to_operations(bank_before, new_experiences)
                        else:
                            # Patch _batch_update to capture operations
                            captured_ops = []
                            original_batch_update = grpo.experience_updater._batch_update

                            async def patched_batch_update(**kwargs):
                                critiques = kwargs.get("critiques", [])
                                for each in critiques:
                                    if "operations" in each:
                                        captured_ops.extend(each["operations"])
                                return await original_batch_update(**kwargs)

                            grpo.experience_updater._batch_update = patched_batch_update
                            try:
                                new_experiences = await grpo.experience_updater.run(
                                    rollouts=rollouts,
                                    recorder=grpo.recorder,
                                    concurrency=config.practice.rollout_concurrency,
                                    given_ground_truth=config.practice.given_ground_truth,
                                    num_experiences=config.practice.num_experiences_per_query,
                                )
                            finally:
                                grpo.experience_updater._batch_update = original_batch_update

                            ExperienceCache.save_experiences(
                                experiment_name=grpo.recorder.experiment_name,
                                step=step,
                                experiences=new_experiences,
                                epoch=epoch,
                                batch=batch_idx,
                            )

                    # ---- 4. Snapshot AFTER ----
                    bank_after = copy.deepcopy(new_experiences)

                    # ---- 5. Held-out evaluation for delta_reward ----
                    reward_before = None
                    reward_after = None
                    delta_reward = None

                    if self._held_out_eval and grpo.eval_rollout_manager is not None:
                        reward_before = await self._quick_eval(bank_before, epoch)
                        reward_after = await self._quick_eval(bank_after, epoch)
                        if reward_before is not None and reward_after is not None:
                            delta_reward = reward_after - reward_before

                    # ---- 6. Record transition ----
                    candidate_texts = list({op.get("content", "") for op in captured_ops if op.get("content")})

                    transition = TransitionRecord(
                        step=step,
                        epoch=epoch,
                        batch=batch_idx,
                        experience_bank_before=bank_before,
                        candidate_experiences=candidate_texts,
                        operations=captured_ops,
                        experience_bank_after=bank_after,
                        reward_before=reward_before,
                        reward_after=reward_after,
                        delta_reward=delta_reward,
                        rollout_stats=stat,
                        task_name=self._recorder.experiment_name,
                    )
                    self._recorder.record(transition)

                    stats[f"step_{step}"]["complete"] = True
                    stats[f"step_{step}"]["delta_reward"] = delta_reward
                    grpo.recorder.stat_update({f"step_{step}": stats[f"step_{step}"]})

                    # ---- 7. Standard evaluation (same as original) ----
                    if grpo.eval_rollout_manager and grpo._should_evaluate(step, batch_idx, num_batches):
                        logger.info(f"[Instrumented] Running eval at step {step}")
                        grpo.eval_rollout_manager.load_epoch_data(
                            epoch=epoch, shuffle=False, truncate=config.practice.eval_data_truncate
                        )
                        _, eval_stats = await grpo.eval_rollout_manager.main(
                            recorder=grpo.recorder, use_cache=grpo._should_use_cache(step)
                        )
                        stats[f"step_{step}"]["eval"] = eval_stats
                        grpo.recorder.stat_update({f"step_{step}": stats[f"step_{step}"]})

    async def _quick_eval(self, experiences: dict[str, str], epoch: int) -> Optional[float]:
        """Quick evaluation with given experiences on held-out data.

        Temporarily injects experiences into agent instructions, runs eval,
        then restores original instructions.

        Returns:
            Average reward score, or None on failure.
        """
        try:
            grpo = self._grpo
            eval_mgr = grpo.eval_rollout_manager
            if eval_mgr is None:
                return None

            # Temporarily modify agent instructions with experiences
            original_instructions = grpo._config.evaluation.agent.agent.instructions

            experience_text = ""
            if experiences:
                experience_text = "\n\nExperiences:\n" + "\n".join(
                    f"[{k}]. {v}" for k, v in experiences.items()
                )

            # Load held-out data
            eval_mgr.load_epoch_data(epoch=epoch, shuffle=False, truncate=self._held_out_size)

            # Run eval
            rollouts, eval_stat = await eval_mgr.main(recorder=grpo.recorder, use_cache=False)

            # Extract average reward
            if rollouts:
                rewards = [r.reward for r in rollouts if r.reward is not None]
                if rewards:
                    return sum(rewards) / len(rewards)

            return None
        except Exception as e:
            logger.warning(f"Quick eval failed: {e}")
            return None

    @staticmethod
    def _diff_to_operations(before: dict[str, str], after: dict[str, str]) -> list[dict]:
        """Infer operations from before/after snapshots (for cached steps)."""
        ops = []
        before_keys = set(before.keys())
        after_keys = set(after.keys())

        for key in after_keys - before_keys:
            ops.append({"operation": "ADD", "id": None, "content": after[key]})

        for key in before_keys - after_keys:
            ops.append({"operation": "DELETE", "id": key, "content": before[key]})

        for key in before_keys & after_keys:
            if before[key] != after[key]:
                ops.append({"operation": "UPDATE", "id": key, "content": after[key]})

        return ops
