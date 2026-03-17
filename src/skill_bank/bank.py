"""Hierarchical skill bank implementation."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Optional

from .node import SkillLevel, SkillNode

logger = logging.getLogger(__name__)


class HierarchicalSkillBank:
    """A tree-structured skill bank with ADD/UPDATE/DELETE/MOVE operations.

    Args:
        max_skills_per_level: Maximum number of skills allowed at each level.
    """

    def __init__(self, max_skills_per_level: tuple[int, ...] = (10, 30, 100)) -> None:
        self._nodes: dict[str, SkillNode] = {}
        self._next_id: int = 0
        self._max_skills_per_level = max_skills_per_level

    @property
    def size(self) -> int:
        return len(self._nodes)

    def get_node(self, node_id: str) -> Optional[SkillNode]:
        return self._nodes.get(node_id)

    def get_roots(self) -> list[SkillNode]:
        return [n for n in self._nodes.values() if n.is_root]

    def get_children(self, node_id: str) -> list[SkillNode]:
        node = self._nodes.get(node_id)
        if node is None:
            return []
        return [self._nodes[cid] for cid in node.children_ids if cid in self._nodes]

    def get_level_nodes(self, level: SkillLevel) -> list[SkillNode]:
        return [n for n in self._nodes.values() if n.level == level]

    def get_subtree(self, node_id: str) -> list[SkillNode]:
        """Get all nodes in the subtree rooted at node_id (inclusive)."""
        node = self._nodes.get(node_id)
        if node is None:
            return []
        result = [node]
        for child_id in node.children_ids:
            result.extend(self.get_subtree(child_id))
        return result

    # ---- Operations ----

    def add(
        self,
        content: str,
        level: SkillLevel,
        parent_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        """Add a new skill node.

        Args:
            content: Skill description.
            level: Hierarchy level.
            parent_id: Parent node ID (None for L0 roots).
            metadata: Optional metadata.

        Returns:
            New node ID, or None if level capacity exceeded.
        """
        level_count = len(self.get_level_nodes(level))
        if level.value < len(self._max_skills_per_level) and level_count >= self._max_skills_per_level[level.value]:
            logger.warning(f"Level {level.name} capacity ({self._max_skills_per_level[level.value]}) reached, skip ADD.")
            return None

        if parent_id is not None and parent_id not in self._nodes:
            logger.warning(f"Parent {parent_id} not found, skip ADD.")
            return None

        node_id = f"S{self._next_id}"
        self._next_id += 1

        node = SkillNode(
            node_id=node_id,
            content=content,
            level=level,
            parent_id=parent_id,
            metadata=metadata or {},
        )
        self._nodes[node_id] = node

        if parent_id is not None:
            self._nodes[parent_id].children_ids.append(node_id)

        logger.debug(f"ADD {node_id} at L{level.value}: {content[:50]}...")
        return node_id

    def update(self, node_id: str, content: str) -> bool:
        """Update a skill node's content.

        Args:
            node_id: Target node ID.
            content: New content.

        Returns:
            True if updated, False if node not found.
        """
        node = self._nodes.get(node_id)
        if node is None:
            logger.warning(f"Node {node_id} not found, skip UPDATE.")
            return False
        node.content = content
        logger.debug(f"UPDATE {node_id}: {content[:50]}...")
        return True

    def delete(self, node_id: str, cascade: bool = True) -> bool:
        """Delete a skill node.

        Args:
            node_id: Target node ID.
            cascade: If True, delete all descendants. If False, reparent children.

        Returns:
            True if deleted, False if node not found.
        """
        node = self._nodes.get(node_id)
        if node is None:
            logger.warning(f"Node {node_id} not found, skip DELETE.")
            return False

        if cascade:
            subtree = self.get_subtree(node_id)
            for n in subtree:
                del self._nodes[n.node_id]
                if n.parent_id and n.parent_id in self._nodes:
                    parent = self._nodes[n.parent_id]
                    if n.node_id in parent.children_ids:
                        parent.children_ids.remove(n.node_id)
        else:
            # Reparent children to deleted node's parent
            for child_id in node.children_ids:
                if child_id in self._nodes:
                    self._nodes[child_id].parent_id = node.parent_id
                    if node.parent_id and node.parent_id in self._nodes:
                        self._nodes[node.parent_id].children_ids.append(child_id)

            if node.parent_id and node.parent_id in self._nodes:
                self._nodes[node.parent_id].children_ids.remove(node_id)
            del self._nodes[node_id]

        logger.debug(f"DELETE {node_id} (cascade={cascade})")
        return True

    def move(self, node_id: str, new_parent_id: Optional[str]) -> bool:
        """Move a node to a new parent.

        Args:
            node_id: Node to move.
            new_parent_id: New parent (None to make root).

        Returns:
            True if moved, False on error.
        """
        node = self._nodes.get(node_id)
        if node is None:
            return False
        if new_parent_id is not None and new_parent_id not in self._nodes:
            return False

        # Remove from old parent
        if node.parent_id and node.parent_id in self._nodes:
            self._nodes[node.parent_id].children_ids.remove(node_id)

        # Attach to new parent
        node.parent_id = new_parent_id
        if new_parent_id is not None:
            self._nodes[new_parent_id].children_ids.append(node_id)

        return True

    # ---- Snapshot & Restore ----

    def snapshot(self) -> dict[str, dict]:
        """Create a deep copy snapshot of current state."""
        return {nid: copy.deepcopy(n.to_dict()) for nid, n in self._nodes.items()}

    def restore(self, snapshot: dict[str, dict]) -> None:
        """Restore from a snapshot."""
        self._nodes = {nid: SkillNode.from_dict(data) for nid, data in snapshot.items()}
        max_id = 0
        for nid in self._nodes:
            num = int(nid[1:]) if nid[1:].isdigit() else 0
            max_id = max(max_id, num + 1)
        self._next_id = max_id

    # ---- Serialization ----

    def to_flat_dict(self) -> dict[str, str]:
        """Convert to flat dict format (compatible with Training-Free GRPO)."""
        result = {}
        for i, node in enumerate(self._nodes.values()):
            result[f"G{i}"] = node.content
        return result

    @classmethod
    def from_flat_dict(cls, flat: dict[str, str]) -> HierarchicalSkillBank:
        """Create from flat dict (all at L0, no hierarchy)."""
        bank = cls()
        for content in flat.values():
            bank.add(content=content, level=SkillLevel.META_PRINCIPLE)
        return bank

    def to_prompt(self, nodes: Optional[list[SkillNode]] = None) -> str:
        """Format skills for prompt injection.

        Args:
            nodes: Specific nodes to include. If None, include all.

        Returns:
            Formatted string for injection into agent instructions.
        """
        if nodes is None:
            nodes = list(self._nodes.values())
        if not nodes:
            return ""

        lines = []
        for node in sorted(nodes, key=lambda n: (n.level, n.node_id)):
            indent = "  " * node.level.value
            lines.append(f"{indent}[{node.node_id}] {node.content}")
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save to JSON file."""
        data = {
            "next_id": self._next_id,
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> HierarchicalSkillBank:
        """Load from JSON file."""
        data = json.loads(Path(path).read_text())
        bank = cls()
        bank._next_id = data["next_id"]
        bank._nodes = {nid: SkillNode.from_dict(nd) for nid, nd in data["nodes"].items()}
        return bank
