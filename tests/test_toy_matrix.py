"""Toy matrix harness — CI must stay green even when research verdict is FAIL."""

from __future__ import annotations

import json
from pathlib import Path

from st_steer.run_matrix import aggregate, main as matrix_main
from st_steer.schema import validate_run_dir
from st_steer.summarize import load_metrics, summarize


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {"steps": 100, "lr": 0.05, "sleep_seconds": 0.0, "seed": 7},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_aggregate_protocol() -> None:
    rows = [{"verdict": v} for v in ("fail", "fail", "fail", "pass", "inconclusive")]
    assert aggregate(rows)["matrix_call"] == "fail"
    rows = [{"verdict": v} for v in ("pass", "pass", "pass", "fail", "fail")]
    assert aggregate(rows)["matrix_call"] == "pass"


def test_single_seed_harness(tmp_path: Path) -> None:
    """One seed end-to-end: artifacts schema-ok; intervene may or may not help."""
    cfg = tmp_path / "config.json"
    _write_config(cfg)
    results = tmp_path / "results"
    summary = tmp_path / "summary.json"
    rc = matrix_main(
        [
            "--seeds",
            "7",
            "--sleep",
            "0",
            "--results-root",
            str(results),
            "--config",
            str(cfg),
            "--summary",
            str(summary),
        ]
    )
    assert rc == 0
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["aggregate"]["n"] == 1
    assert payload["aggregate"]["matrix_call"] in {"pass", "fail", "inconclusive"}

    base = results / "seed_7" / "baseline"
    steer = results / "seed_7" / "steered"
    assert validate_run_dir(base, strict=True).ok
    # steered may lack interventions if rule never fired (seed-dependent);
    # metrics + snapshot must still validate.
    assert validate_run_dir(steer, strict=True).ok
    assert summarize(load_metrics(base / "metrics.jsonl"))["n"] == 100
    assert (steer / "compare.json").exists()


def test_full_toy_matrix_harness(tmp_path: Path) -> None:
    """Frozen seed set completes; research FAIL is allowed (harness still green)."""
    cfg = tmp_path / "config.json"
    _write_config(cfg)
    results = tmp_path / "results"
    summary = tmp_path / "matrix_summary.json"
    rc = matrix_main(
        [
            "--seeds",
            "7",
            "11",
            "13",
            "17",
            "19",
            "--sleep",
            "0",
            "--results-root",
            str(results),
            "--config",
            str(cfg),
            "--summary",
            str(summary),
        ]
    )
    assert rc == 0
    payload = json.loads(summary.read_text(encoding="utf-8"))
    agg = payload["aggregate"]
    assert agg["n"] == 5
    assert agg["pass"] + agg["fail"] + agg["inconclusive"] == 5
    assert agg["matrix_call"] in {"pass", "fail", "inconclusive"}

    for seed in (7, 11, 13, 17, 19):
        base = results / f"seed_{seed}" / "baseline"
        steer = results / f"seed_{seed}" / "steered"
        assert validate_run_dir(base, strict=True).ok, seed
        assert validate_run_dir(steer, strict=True).ok, seed
        assert (steer / "compare.json").exists(), seed
