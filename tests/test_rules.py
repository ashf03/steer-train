"""Frozen intervene rules."""

from __future__ import annotations

from st_steer.rules.s1 import BASE_LR, S1OursRule
from st_steer.rules.toy import ToyLossStreakRule


def test_toy_rule_fires_after_streak() -> None:
    rule = ToyLossStreakRule()
    for step in range(60, 63):
        d = rule.observe(step=step, loss=0.04, lr=0.05)
        if step < 62:
            assert not d.fire
    assert d.fire
    assert d.action["lr_to"] == 0.02
    assert not rule.observe(step=63, loss=0.01, lr=0.02).fire  # one-shot


def test_s1_plateau_half_lr_and_cap() -> None:
    rule = S1OursRule()
    box = {"lr": BASE_LR}
    fires = 0
    # establish best, then flat plateau
    for step, val in [
        (100, 1.0),
        (500, 0.8),
        (550, 0.8),
        (600, 0.8),
        (650, 0.8),  # fire 1
        (700, 0.8),
        (750, 0.8),
        (800, 0.8),  # fire 2
        (850, 0.8),
        (900, 0.8),
        (950, 0.8),  # no fire — cap
    ]:
        d = rule.observe_eval(step=step, val_loss=val, lr=box["lr"])
        if d.fire:
            fires += 1
            box["lr"] = d.action["lr_to"]
    assert fires == 2
    assert abs(box["lr"] - BASE_LR * 0.25) < 1e-12
