"""Base controller interface for skill bank management."""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..skill_bank.node import SkillLevel


class OperationType(enum.Enum):
    """Controller operation types."""

    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NONE = "NONE"
    MOVE = "MOVE"


@dataclass
class ControllerAction:
    """A single action produced by the controller.

    Args:
        operation: Type of operation.
        content: Skill content (for ADD/UPDATE).
        target_node_id: Target node ID (for UPDATE/DELETE/MOVE).
        level: Target level (for ADD).
        parent_id: Parent node ID (for ADD/MOVE).
        confidence: Controller's confidence in this action.
        reasoning: Optional explanation (for analysis).
    """

    operation: OperationType
    content: str = ""
    target_node_id: Optional[str] = None
    level: SkillLevel = SkillLevel.META_PRINCIPLE
    parent_id: Optional[str] = None
    confidence: float = 1.0
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "operation": self.operation.value,
            "content": self.content,
            "target_node_id": self.target_node_id,
            "level": self.level.value,
            "parent_id": self.parent_id,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


@dataclass
class ControllerInput:
    """Input to the controller.

    Args:
        skill_bank_snapshot: Current state of the skill bank.
        candidate_experiences: New experiences extracted from Group Computation (A_text).
        agent_objective: Description of the agent's task.
        learning_objective: Description of the learning goal.
    """

    skill_bank_snapshot: dict[str, dict]
    candidate_experiences: list[str]
    agent_objective: str = ""
    learning_objective: str = ""


@dataclass
class ControllerOutput:
    """Output from the controller.

    Args:
        actions: List of actions to apply to the skill bank.
        metadata: Additional info (LLM tokens used, latency, etc.).
    """

    actions: list[ControllerAction] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BaseController(ABC):
    """Abstract base class for skill bank controllers.

    All controllers (LLM-based, trained, rule-based) implement this interface.
    """

    @abstractmethod
    async def decide(self, controller_input: ControllerInput) -> ControllerOutput:
        """Decide what operations to perform on the skill bank.

        Args:
            controller_input: Current skill bank state and candidate experiences.

        Returns:
            Actions to apply.
        """

    def apply(
        self,
        actions: list[ControllerAction],
        skill_bank: "HierarchicalSkillBank",
    ) -> dict[str, bool]:
        """Apply actions to the skill bank.

        Args:
            actions: Actions from decide().
            skill_bank: The skill bank to modify.

        Returns:
            Mapping from action index to success/failure.
        """
        results = {}
        for i, action in enumerate(actions):
            if action.operation == OperationType.ADD:
                node_id = skill_bank.add(
                    content=action.content,
                    level=action.level,
                    parent_id=action.parent_id,
                    metadata={"source": "controller"},
                )
                results[str(i)] = node_id is not None

            elif action.operation == OperationType.UPDATE:
                results[str(i)] = skill_bank.update(
                    node_id=action.target_node_id,
                    content=action.content,
                )

            elif action.operation == OperationType.DELETE:
                results[str(i)] = skill_bank.delete(
                    node_id=action.target_node_id,
                )

            elif action.operation == OperationType.MOVE:
                results[str(i)] = skill_bank.move(
                    node_id=action.target_node_id,
                    new_parent_id=action.parent_id,
                )

            elif action.operation == OperationType.NONE:
                results[str(i)] = True

        return results
