"""Tests for hierarchical skill bank."""

import json
import tempfile
from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.skill_bank.bank import HierarchicalSkillBank
from src.skill_bank.node import SkillLevel, SkillNode


class TestSkillNode:
    def test_create_node(self) -> None:
        node = SkillNode(node_id="S0", content="test skill", level=SkillLevel.META_PRINCIPLE)
        assert node.node_id == "S0"
        assert node.is_root
        assert node.is_leaf

    def test_serialization(self) -> None:
        node = SkillNode(
            node_id="S1",
            content="domain skill",
            level=SkillLevel.DOMAIN_STRATEGY,
            parent_id="S0",
            children_ids=["S2"],
            metadata={"source": "math"},
        )
        data = node.to_dict()
        restored = SkillNode.from_dict(data)
        assert restored.node_id == "S1"
        assert restored.parent_id == "S0"
        assert restored.children_ids == ["S2"]


class TestHierarchicalSkillBank:
    def test_add_and_get(self) -> None:
        bank = HierarchicalSkillBank()
        nid = bank.add("Meta principle A", SkillLevel.META_PRINCIPLE)
        assert nid is not None
        node = bank.get_node(nid)
        assert node.content == "Meta principle A"
        assert node.is_root
        assert bank.size == 1

    def test_hierarchy(self) -> None:
        bank = HierarchicalSkillBank()
        root = bank.add("Root", SkillLevel.META_PRINCIPLE)
        child = bank.add("Child", SkillLevel.DOMAIN_STRATEGY, parent_id=root)
        leaf = bank.add("Leaf", SkillLevel.SPECIFIC_TACTIC, parent_id=child)

        assert bank.size == 3
        assert bank.get_children(root) == [bank.get_node(child)]
        subtree = bank.get_subtree(root)
        assert len(subtree) == 3

    def test_update(self) -> None:
        bank = HierarchicalSkillBank()
        nid = bank.add("Old content", SkillLevel.META_PRINCIPLE)
        assert bank.update(nid, "New content")
        assert bank.get_node(nid).content == "New content"
        assert not bank.update("nonexistent", "fail")

    def test_delete_cascade(self) -> None:
        bank = HierarchicalSkillBank()
        root = bank.add("Root", SkillLevel.META_PRINCIPLE)
        child = bank.add("Child", SkillLevel.DOMAIN_STRATEGY, parent_id=root)
        bank.add("Leaf", SkillLevel.SPECIFIC_TACTIC, parent_id=child)

        bank.delete(root, cascade=True)
        assert bank.size == 0

    def test_delete_reparent(self) -> None:
        bank = HierarchicalSkillBank()
        root = bank.add("Root", SkillLevel.META_PRINCIPLE)
        mid = bank.add("Mid", SkillLevel.DOMAIN_STRATEGY, parent_id=root)
        leaf = bank.add("Leaf", SkillLevel.SPECIFIC_TACTIC, parent_id=mid)

        bank.delete(mid, cascade=False)
        assert bank.size == 2
        assert bank.get_node(leaf).parent_id == root

    def test_move(self) -> None:
        bank = HierarchicalSkillBank()
        r1 = bank.add("Root 1", SkillLevel.META_PRINCIPLE)
        r2 = bank.add("Root 2", SkillLevel.META_PRINCIPLE)
        child = bank.add("Child", SkillLevel.DOMAIN_STRATEGY, parent_id=r1)

        bank.move(child, r2)
        assert bank.get_node(child).parent_id == r2
        assert child not in bank.get_node(r1).children_ids
        assert child in bank.get_node(r2).children_ids

    def test_snapshot_restore(self) -> None:
        bank = HierarchicalSkillBank()
        bank.add("Skill A", SkillLevel.META_PRINCIPLE)
        bank.add("Skill B", SkillLevel.META_PRINCIPLE)

        snap = bank.snapshot()
        bank.add("Skill C", SkillLevel.META_PRINCIPLE)
        assert bank.size == 3

        bank.restore(snap)
        assert bank.size == 2

    def test_flat_dict_conversion(self) -> None:
        bank = HierarchicalSkillBank()
        bank.add("Skill A", SkillLevel.META_PRINCIPLE)
        bank.add("Skill B", SkillLevel.META_PRINCIPLE)

        flat = bank.to_flat_dict()
        assert len(flat) == 2

        restored = HierarchicalSkillBank.from_flat_dict(flat)
        assert restored.size == 2

    def test_save_load(self) -> None:
        bank = HierarchicalSkillBank()
        root = bank.add("Root", SkillLevel.META_PRINCIPLE)
        bank.add("Child", SkillLevel.DOMAIN_STRATEGY, parent_id=root)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            bank.save(f.name)
            loaded = HierarchicalSkillBank.load(f.name)
            assert loaded.size == 2
            assert loaded.get_node(root).content == "Root"

    def test_level_capacity(self) -> None:
        bank = HierarchicalSkillBank(max_skills_per_level=(2, 5, 10))
        bank.add("A", SkillLevel.META_PRINCIPLE)
        bank.add("B", SkillLevel.META_PRINCIPLE)
        result = bank.add("C", SkillLevel.META_PRINCIPLE)  # should fail
        assert result is None
        assert bank.size == 2

    def test_to_prompt(self) -> None:
        bank = HierarchicalSkillBank()
        root = bank.add("Meta: verify results", SkillLevel.META_PRINCIPLE)
        bank.add("Domain: enumerate cases", SkillLevel.DOMAIN_STRATEGY, parent_id=root)

        prompt = bank.to_prompt()
        assert "Meta: verify results" in prompt
        assert "  " in prompt  # indentation for L1
