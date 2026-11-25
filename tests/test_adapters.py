"""Trainer adapters (file-tail + callback; aq deferred)."""

from __future__ import annotations

from pathlib import Path

import pytest

from st_steer.adapters import FileTailAdapter, get_adapter, list_adapters
from st_steer.rules.s1 import BASE_LR, S1OursRule
from st_steer.schema import validate_interventions_file, validate_metrics_file


def test_adapter_registry() -> None:
    names = {a.name for a in list_adapters()}
    assert names == {"file-tail", "callback", "aq"}
    aq = next(a for a in list_adapters() if a.name == "aq")
    assert aq.status == "deferred"
    with pytest.raises(NotImplementedError):
        get_adapter("aq")


def test_file_tail_set_lr_and_metrics(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        '{"lr": 0.05, "steps": 10, "seed": 7, "sleep_seconds": 0}\n',
        encoding="utf-8",
    )
    adapter = FileTailAdapter(
        config_path=cfg,
        metrics_path=tmp_path / "metrics.jsonl",
    )
    adapter.controller.set_lr(0.02)
    assert adapter.controller.get_lr() == 0.02
    adapter.metrics.write({"step": 0, "lr": 0.02, "wall_ms": 1.0, "loss": 1.0})
    assert validate_metrics_file(adapter.metrics_path, strict=True).ok


def test_callback_s1_rule_writes_artifacts(tmp_path: Path) -> None:
    box = {"lr": BASE_LR}
    adapter = get_adapter(
        "callback",
        metrics_path=tmp_path / "metrics.jsonl",
        rule=S1OursRule(),
        get_lr=lambda: box["lr"],
        set_lr=lambda v: box.__setitem__("lr", float(v)),
        seed=7,
        steps=2000,
    )
    hook = adapter.hook
    for step, val in [
        (100, 1.0),
        (500, 0.8),
        (550, 0.8),
        (600, 0.8),
        (650, 0.8),
    ]:
        hook.on_eval(step=step, val_loss=val, train_loss=val - 0.05)
    assert box["lr"] == BASE_LR * 0.5
    hook.on_step(step=651, train_loss=0.7)
    assert validate_metrics_file(tmp_path / "metrics.jsonl", strict=True).ok
    assert validate_interventions_file(tmp_path / "interventions.jsonl", strict=True).ok
    assert (tmp_path / "config.snapshot.json").exists()
