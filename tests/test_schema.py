"""Schema v1 writers / validators."""

from __future__ import annotations

import json
from pathlib import Path

from st_steer.schema import (
    SCHEMA_VERSION,
    config_snapshot_record,
    intervention_record,
    metric_record,
    validate_config_snapshot,
    validate_intervention_row,
    validate_metric_row,
    validate_metrics_file,
    write_config_snapshot,
    write_intervention,
    write_metric,
)


def test_metric_record_has_schema_and_aliases() -> None:
    row = metric_record(step=0, lr=0.05, wall_ms=1.0, loss=1.2)
    assert row["schema"] == SCHEMA_VERSION
    assert row["train_loss"] == row["loss"] == 1.2
    assert validate_metric_row(row, strict=True).ok


def test_legacy_metric_compat(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps({"step": 0, "loss": 1.0, "lr": 0.05, "wall_ms": 0.1}) + "\n",
        encoding="utf-8",
    )
    soft = validate_metrics_file(path, strict=False)
    hard = validate_metrics_file(path, strict=True)
    assert soft.ok
    assert not hard.ok


def test_snapshot_and_intervention_roundtrip(tmp_path: Path) -> None:
    snap = config_snapshot_record(
        seed=7,
        steps=100,
        lr=0.05,
        started_at="2026-01-01T00:00:00Z",
        setting_id="toy-synthetic",
        condition="baseline",
    )
    sp = tmp_path / "config.snapshot.json"
    write_config_snapshot(sp, snap)
    assert validate_config_snapshot(json.loads(sp.read_text()), strict=True).ok

    ip = tmp_path / "interventions.jsonl"
    rec = intervention_record(
        timestamp="2026-01-01T00:00:00Z",
        rule_id="toy-loss-streak-cut-lr",
        decision_step=69,
        evidence={"steps": [67, 68, 69]},
        action={"type": "lr_set", "lr_from": 0.05, "lr_to": 0.02},
        effect_step=70,
        effect_delay_steps=1,
    )
    write_intervention(ip, rec)
    row = json.loads(ip.read_text().strip())
    assert validate_intervention_row(row, strict=True).ok

    mp = tmp_path / "metrics.jsonl"
    write_metric(mp, step=0, lr=0.05, wall_ms=0.1, loss=1.0)
    assert validate_metrics_file(mp, strict=True).ok
