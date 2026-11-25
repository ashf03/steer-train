"""Toy intervene rule — docs/protocol.md."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TOY_RULE_ID = "toy-loss-streak-cut-lr"

TRIGGER_LOSS = 0.05
TRIGGER_STREAK = 3
TRIGGER_MIN_STEP = 60
STEER_LR = 0.02
DEFAULT_LR = 0.05


@dataclass
class ToyRuleDecision:
    fire: bool
    rule_id: str = TOY_RULE_ID
    decision_step: int | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToyLossStreakRule:
    """One-shot: loss < 0.05 × 3 steps after step ≥ 60 → lr 0.05 → 0.02."""

    trigger_loss: float = TRIGGER_LOSS
    streak_needed: int = TRIGGER_STREAK
    min_step: int = TRIGGER_MIN_STEP
    steer_lr: float = STEER_LR
    default_lr: float = DEFAULT_LR
    streak: int = 0
    trigger_steps: list[int] = field(default_factory=list)
    fired: bool = False

    def observe(self, *, step: int, loss: float, lr: float) -> ToyRuleDecision:  # noqa: ARG002
        if self.fired:
            return ToyRuleDecision(fire=False)

        if step >= self.min_step and loss < self.trigger_loss:
            self.streak += 1
            self.trigger_steps.append(step)
        else:
            self.streak = 0
            self.trigger_steps.clear()

        if self.streak < self.streak_needed:
            return ToyRuleDecision(fire=False)

        self.fired = True
        return ToyRuleDecision(
            fire=True,
            decision_step=step,
            evidence={
                "steps": list(self.trigger_steps),
                "threshold": self.trigger_loss,
                "streak": self.streak_needed,
            },
            action={
                "type": "lr_set",
                "lr_from": self.default_lr,
                "lr_to": self.steer_lr,
            },
        )
