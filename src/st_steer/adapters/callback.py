"""Callback adapter — in-process metrics + LR set (S1 default).

Framework-agnostic ``SteerHook`` drives logging and optional rules.
``as_transformers_callback`` wraps it for HuggingFace ``Trainer`` when
``transformers`` is installed (``pip install st-steer[s1]``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from st_steer.adapters.base import AdapterInfo, register_adapter
from st_steer.schema import (
    append_jsonl,
    config_snapshot_record,
    intervention_record,
    metric_record,
    patch_intervention_effect,
    write_config_snapshot,
    write_intervention,
)

GetLrFn = Callable[[], float]
SetLrFn = Callable[[float], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class OptimizerLrController:
    """Set LR on all param groups of a torch optimizer (duck-typed)."""

    optimizer: Any

    def get_lr(self) -> float:
        groups = getattr(self.optimizer, "param_groups", None)
        if not groups:
            raise RuntimeError("optimizer has no param_groups")
        return float(groups[0]["lr"])

    def set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)


@dataclass
class SteerHook:
    """In-process observe / intervene surface (lag target ≤ 1 step).

    Call ``on_step`` every training step and ``on_eval`` on eval rows.
    Attach an optional rule object with ``observe_eval`` / ``observe`` /
    ``note_lr`` as appropriate.
    """

    metrics_path: Path
    get_lr: GetLrFn
    set_lr: SetLrFn
    rule: Any | None = None
    setting_id: str = "S1-qwen15b-alpaca-lora"
    condition: str = "OURS-RULE"
    seed: int = 7
    steps: int = 2000
    interventions_path: Path | None = None
    write_snapshot: bool = True
    _t0: float = field(default_factory=time.perf_counter, repr=False)
    _snapshot_done: bool = False
    _last_decision_step: int | None = None

    def __post_init__(self) -> None:
        self.metrics_path = Path(self.metrics_path)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if self.interventions_path is None:
            self.interventions_path = self.metrics_path.parent / "interventions.jsonl"

    def ensure_snapshot(self, *, lr: float | None = None) -> Path | None:
        if not self.write_snapshot or self._snapshot_done:
            return None
        path = self.metrics_path.parent / "config.snapshot.json"
        write_config_snapshot(
            path,
            config_snapshot_record(
                seed=self.seed,
                steps=self.steps,
                lr=float(lr if lr is not None else self.get_lr()),
                started_at=_utc_now(),
                setting_id=self.setting_id,
                condition=self.condition,
            ),
        )
        self._snapshot_done = True
        return path

    def reset_metrics(self) -> None:
        self.metrics_path.write_text("", encoding="utf-8")
        self._t0 = time.perf_counter()
        self._snapshot_done = False

    def wall_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def on_step(
        self,
        *,
        step: int,
        train_loss: float,
        lr: float | None = None,
        val_loss: float | None = None,
        epoch: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        lr_v = float(self.get_lr() if lr is None else lr)
        self.ensure_snapshot(lr=lr_v)
        row = metric_record(
            step=step,
            lr=lr_v,
            wall_ms=round(self.wall_ms(), 3),
            train_loss=train_loss,
            val_loss=val_loss,
            epoch=epoch,
            extra=extra,
        )
        append_jsonl(self.metrics_path, row)

        if self.rule is not None and hasattr(self.rule, "note_lr"):
            self.rule.note_lr(step=step, lr=lr_v)
            self._maybe_patch_effect()

        # Toy-style rules that observe every step
        if self.rule is not None and hasattr(self.rule, "observe"):
            decision = self.rule.observe(step=step, loss=float(train_loss), lr=lr_v)
            if getattr(decision, "fire", False):
                self._apply_decision(decision)

        return row

    def on_eval(
        self,
        *,
        step: int,
        val_loss: float,
        train_loss: float | None = None,
        lr: float | None = None,
        epoch: float | None = None,
    ) -> dict[str, Any]:
        lr_v = float(self.get_lr() if lr is None else lr)
        self.ensure_snapshot(lr=lr_v)
        tl = float(train_loss) if train_loss is not None else float("nan")
        row = metric_record(
            step=step,
            lr=lr_v,
            wall_ms=round(self.wall_ms(), 3),
            train_loss=tl if train_loss is not None else 0.0,
            val_loss=float(val_loss),
            epoch=epoch,
        )
        if train_loss is None:
            # keep train_loss key but mark absent via extra? schema wants float —
            # use last known; callers should pass train_loss when available.
            row["train_loss"] = row.get("train_loss", 0.0)
        append_jsonl(self.metrics_path, row)

        if self.rule is not None and hasattr(self.rule, "observe_eval"):
            decision = self.rule.observe_eval(step=step, val_loss=float(val_loss), lr=lr_v)
            if getattr(decision, "fire", False):
                self._apply_decision(decision)
        elif self.rule is not None and hasattr(self.rule, "note_lr"):
            self.rule.note_lr(step=step, lr=lr_v)
            self._maybe_patch_effect()

        return row

    def _apply_decision(self, decision: Any) -> None:
        action = getattr(decision, "action", {}) or {}
        lr_to = action.get("lr_to")
        if lr_to is None:
            return
        self.set_lr(float(lr_to))
        self._last_decision_step = getattr(decision, "decision_step", None)
        if self.interventions_path is not None:
            write_intervention(
                self.interventions_path,
                intervention_record(
                    timestamp=_utc_now(),
                    rule_id=getattr(decision, "rule_id", "unknown"),
                    decision_step=int(getattr(decision, "decision_step", 0) or 0),
                    evidence=getattr(decision, "evidence", {}) or {},
                    action=action,
                    effect_step=None,
                    effect_delay_steps=None,
                    extra={"adapter": "callback"},
                ),
            )

    def _maybe_patch_effect(self) -> None:
        if self.rule is None or self.interventions_path is None:
            return
        effect_step = getattr(self.rule, "effect_step", None)
        delay = getattr(self.rule, "effect_delay_steps", None)
        decision_step = getattr(self.rule, "last_decision_step", self._last_decision_step)
        if effect_step is None or delay is None or decision_step is None:
            return
        if not self.interventions_path.exists():
            return
        patch_intervention_effect(
            self.interventions_path,
            decision_step=int(decision_step),
            effect_step=int(effect_step),
            effect_delay_steps=int(delay),
        )


class CallbackAdapter:
    """S1-oriented adapter: ``SteerHook`` + optional HF callback factory."""

    name = "callback"

    def __init__(
        self,
        *,
        metrics_path: Path,
        get_lr: GetLrFn | None = None,
        set_lr: SetLrFn | None = None,
        optimizer: Any | None = None,
        rule: Any | None = None,
        setting_id: str = "S1-qwen15b-alpaca-lora",
        condition: str = "OURS-RULE",
        seed: int = 7,
        steps: int = 2000,
    ) -> None:
        if optimizer is not None:
            ctrl = OptimizerLrController(optimizer)
            get_lr = ctrl.get_lr
            set_lr = ctrl.set_lr
        if get_lr is None or set_lr is None:
            raise ValueError("CallbackAdapter needs get_lr/set_lr or optimizer")

        self.hook = SteerHook(
            metrics_path=Path(metrics_path),
            get_lr=get_lr,
            set_lr=set_lr,
            rule=rule,
            setting_id=setting_id,
            condition=condition,
            seed=seed,
            steps=steps,
        )
        self.controller = OptimizerLrController(optimizer) if optimizer is not None else None

    @property
    def metrics_path(self) -> Path:
        return self.hook.metrics_path

    def as_transformers_callback(self) -> Any:
        """Return an HF ``TrainerCallback`` or raise with install hint."""
        return as_transformers_callback(self.hook)


def as_transformers_callback(hook: SteerHook) -> Any:
    """Wrap ``SteerHook`` as ``transformers.TrainerCallback``."""
    try:
        from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
    except ImportError as err:
        raise ImportError(
            "callback→HF requires transformers — pip install 'st-steer[s1]'"
        ) from err

    class SteerTrainerCallback(TrainerCallback):
        """Writes schema-1 metrics; applies hook.rule on evaluate."""

        def __init__(self, steer_hook: SteerHook) -> None:
            self.hook = steer_hook

        def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
            opt = kwargs.get("optimizer")
            if opt is not None:
                ctrl = OptimizerLrController(opt)
                self.hook.get_lr = ctrl.get_lr
                self.hook.set_lr = ctrl.set_lr
            self.hook.ensure_snapshot()

        def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: dict[str, float] | None = None, **kwargs: Any) -> None:
            logs = logs or {}
            step = int(state.global_step)
            train_loss = logs.get("loss", logs.get("train_loss"))
            if train_loss is None:
                return
            lr = logs.get("learning_rate")
            self.hook.on_step(
                step=step,
                train_loss=float(train_loss),
                lr=float(lr) if lr is not None else None,
                epoch=float(state.epoch) if state.epoch is not None else None,
            )

        def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict[str, float] | None = None, **kwargs: Any) -> None:
            metrics = metrics or {}
            # HF uses eval_loss
            val = metrics.get("eval_loss", metrics.get("val_loss"))
            if val is None:
                return
            step = int(state.global_step)
            self.hook.on_eval(
                step=step,
                val_loss=float(val),
                train_loss=metrics.get("loss"),
                epoch=float(state.epoch) if state.epoch is not None else None,
            )

    return SteerTrainerCallback(hook)


def _factory(
    *,
    metrics_path: Path | str,
    get_lr: GetLrFn | None = None,
    set_lr: SetLrFn | None = None,
    optimizer: Any | None = None,
    rule: Any | None = "s1",
    setting_id: str = "S1-qwen15b-alpaca-lora",
    condition: str = "OURS-RULE",
    seed: int = 7,
    steps: int = 2000,
    **_kwargs: Any,
) -> CallbackAdapter:
    resolved_rule: Any | None
    if rule == "s1":
        from st_steer.rules.s1 import S1OursRule

        resolved_rule = S1OursRule()
    elif rule == "toy":
        from st_steer.rules.toy import ToyLossStreakRule

        resolved_rule = ToyLossStreakRule()
    else:
        resolved_rule = rule

    # Allow constructing without optimizer for dry wiring — use no-op LR until bound
    if get_lr is None and set_lr is None and optimizer is None:
        _lr = {"v": 2e-4}

        def get_lr() -> float:
            return float(_lr["v"])

        def set_lr(lr: float) -> None:
            _lr["v"] = float(lr)

    return CallbackAdapter(
        metrics_path=Path(metrics_path),
        get_lr=get_lr,
        set_lr=set_lr,
        optimizer=optimizer,
        rule=resolved_rule,
        setting_id=setting_id,
        condition=condition,
        seed=seed,
        steps=steps,
    )


register_adapter(
    AdapterInfo(
        name="callback",
        summary="In-process HF TrainerCallback / SteerHook (S1 default)",
        status="ready",
        lag_note="≤1 step when LR set on optimizer param groups",
        requires=("torch", "transformers"),  # for HF wrapper; SteerHook is pure
    ),
    _factory,
)

# Deferred aq entry so ``st adapters`` lists it.
register_adapter(
    AdapterInfo(
        name="aq",
        summary="aq mission/daemon bridge (deferred)",
        status="deferred",
        lag_note="n/a until implemented",
        requires=("aq",),
    ),
    None,
)
