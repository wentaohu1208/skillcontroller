"""Skill node definition for hierarchical skill bank."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class SkillLevel(enum.IntEnum):
    """Skill hierarchy levels."""

    META_PRINCIPLE = 0      # e.g., "多方法交叉验证"
    DOMAIN_STRATEGY = 1     # e.g., "组合问题: 枚举小案例 + 理论推导"
    SPECIFIC_TACTIC = 2     # e.g., "Burnside 引理: 检查 cycle 长度"


@dataclass
class SkillNode:
    """A single node in the hierarchical skill bank.

    Args:
        node_id: Unique identifier.
        content: Skill description text.
        level: Hierarchy level.
        parent_id: Parent node ID (None for root nodes).
        children_ids: List of child node IDs.
        metadata: Additional metadata (source_task, creation_step, etc.).
    """

    node_id: str
    content: str
    level: SkillLevel
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    @property
    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "node_id": self.node_id,
            "content": self.content,
            "level": self.level.value,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SkillNode:
        """Deserialize from dictionary."""
        return cls(
            node_id=data["node_id"],
            content=data["content"],
            level=SkillLevel(data["level"]),
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            metadata=data.get("metadata", {}),
        )
