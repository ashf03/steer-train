"""S1 OURS-RULE — docs/protocol.scale.md (frozen)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

S1_RULE_ID = "S1-plateau-half-lr"

# Frozen constants
MIN_STEP = 500
MIN_DELTA = 1e-3
PATIENCE_EVALS = 3
MAX_INTERVENTIONS = 2
LR_FACTOR = 0.5
MIN_LR = 1e-6
BASE_LR = 2e-4


@dataclass
class S1RuleDecision:
    fire: bool
    rule_id: str = S1_RULE_ID
    decision_step: int | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)


@dataclass
class S1OursRule:
    """Plateau on val_loss → half LR (max 2 cuts, step ≥ 500).

    Observe **eval** rows only (val_loss present). Use ``note_lr`` on later
    metric rows to record decision→effect lag.
    """

    min_step: int = MIN_STEP
    min_delta: float = MIN_DELTA
    patience: int = PATIENCE_EVALS
    max_interventions: int = MAX_INTERVENTIONS
    lr_factor: float = LR_FACTOR
    min_lr: float = MIN_LR

    best_val: float | None = None
    plateau_streak: int = 0
    interventions: int = 0
    last_decision_step: int | None = None
    pending_effect: bool = False
    effect_step: int | None = None
    effect_delay_steps: int | None = None
    _target_lr: float | None = None

    def _maybe_mark_effect(self, *, step: int, lr: float) -> None:
        if not self.pending_effect or self._target_lr is None:
            return
        if abs(float(lr) - self._target_lr) < 1e-12:
            self.effect_step = step
            if self.last_decision_step is not None:
                self.effect_delay_steps = step - self.last_decision_step
            self.pending_effect = False

    def observe_eval(
        self,
        *,
        step: int,
        val_loss: float,
        lr: float,
    ) -> S1RuleDecision:
        """Call on every eval row that has a real ``val_loss``."""
        self._maybe_mark_effect(step=step, lr=lr)

        if self.best_val is None:
            self.best_val = float(val_loss)
            self.plateau_streak = 0
            return S1RuleDecision(fire=False)

        improved = val_loss < (self.best_val - self.min_delta)
        if improved:
            self.best_val = float(val_loss)
            self.plateau_streak = 0
            return S1RuleDecision(fire=False)

        self.plateau_streak += 1

        if step < self.min_step:
            return S1RuleDecision(fire=False)
        if self.interventions >= self.max_interventions:
            return S1RuleDecision(fire=False)
        if self.plateau_streak < self.patience:
            return S1RuleDecision(fire=False)

        new_lr = max(float(lr) * self.lr_factor, self.min_lr)
        if new_lr >= float(lr) - 1e-15:
            return S1RuleDecision(fire=False)

        self.interventions += 1
        self.last_decision_step = step
        self.pending_effect = True
        self._target_lr = new_lr
        self.plateau_streak = 0

        return S1RuleDecision(
            fire=True,
            decision_step=step,
            evidence={
                "val_loss": float(val_loss),
                "best_val": float(self.best_val),
                "plateau_evals": self.patience,
                "min_delta": self.min_delta,
                "intervention_index": self.interventions,
            },
            action={
                "type": "lr_mul",
                "factor": self.lr_factor,
                "lr_from": float(lr),
                "lr_to": new_lr,
            },
        )

    def note_lr(self, *, step: int, lr: float) -> None:
        """Mark effect when a metric row first shows the new lr."""
        self._maybe_mark_effect(step=step, lr=lr)
