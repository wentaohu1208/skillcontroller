"""Configuration definitions for SkillController."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SkillBankConfig:
    """Hierarchical Skill Bank configuration."""

    max_levels: int = 3
    # L0: Meta-Principles, L1: Domain Strategies, L2: Specific Tactics
    level_names: tuple[str, ...] = ("meta_principle", "domain_strategy", "specific_tactic")
    max_skills_per_level: tuple[int, ...] = (10, 30, 100)
    # Retrieval
    retrieval_method: str = "embedding"  # "embedding" | "llm" | "hybrid"
    top_k_retrieval: int = 5


@dataclass(frozen=True)
class ControllerConfig:
    """Trained Controller configuration."""

    # Architecture: "llm_baseline" | "small_lm" | "classifier"
    architecture: str = "llm_baseline"
    # For small_lm
    model_name: Optional[str] = None
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    # Operations
    operations: tuple[str, ...] = ("ADD", "UPDATE", "DELETE", "NONE")


@dataclass(frozen=True)
class DataCollectionConfig:
    """Data collection configuration for Phase 1."""

    # youtu-agent path for importing
    youtu_agent_path: str = "/Users/wentaohu/project/youtu-agent"
    # Tasks to collect data from
    tasks: tuple[str, ...] = ("math",)
    # Training-Free GRPO configs to run
    practice_configs: tuple[str, ...] = ("math_reasoning",)
    # Recording
    record_dir: str = "data/collected"
    # Stability measurement
    eval_after_each_update: bool = True
    held_out_ratio: float = 0.1
    rollback_threshold: float = 0.0  # rollback if delta_reward < threshold


@dataclass(frozen=True)
class SkillControllerConfig:
    """Top-level configuration."""

    exp_id: str = "default"
    skill_bank: SkillBankConfig = field(default_factory=SkillBankConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    data_collection: DataCollectionConfig = field(default_factory=DataCollectionConfig)
    seed: int = 42
