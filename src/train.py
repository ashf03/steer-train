#!/usr/bin/env python3
"""Synthetic training loop with live-reloadable hyperparameters.

Reads ``configs/default.json`` (or ``--config``) before every step so an
external process (or the TUI) can change ``lr`` without restarting.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.json"
DEFAULT_CONFIG: dict[str, Any] = {
    "steps": 100,
    "lr": 0.05,
    "sleep_seconds": 0.1,
    "seed": 7,
}

StepCallback = Callable[[dict[str, Any]], None]


def resolve_path(path: Path, root: Path = ROOT) -> Path:
    return path if path.is_absolute() else root / path


def load_config(path: Path) -> dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            config.update(json.load(handle))
    return config


def load_config_safe(
    path: Path,
    last_good: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Load config; on failure keep ``last_good`` (or defaults) and return a warning."""
    fallback = dict(last_good) if last_good is not None else DEFAULT_CONFIG.copy()
    try:
        if not path.exists():
            return fallback, f"config missing ({path}); using last good"
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            return fallback, f"config not an object ({path}); using last good"
        config = DEFAULT_CONFIG.copy()
        config.update(raw)
        return config, None
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as err:
        return fallback, f"config reload failed ({err}); using last good"


def append_metric(path: Path, metric: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric, sort_keys=True) + "\n")
        handle.flush()


def write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def snapshot_path_for(metrics_path: Path) -> Path:
    """``results/<run>/metrics.jsonl`` → ``results/<run>/config.snapshot.json``."""
    return metrics_path.parent / "config.snapshot.json"


def write_config_snapshot(
    metrics_path: Path,
    config: dict[str, Any],
    *,
    sleep_override: float | None = None,
) -> Path:
    """Freeze starting seed/steps/lr next to the metrics file."""
    path = snapshot_path_for(metrics_path)
    snapshot = {
        "seed": int(config["seed"]),
        "steps": int(config["steps"]),
        "lr": float(config["lr"]),
        "sleep_seconds": float(config.get("sleep_seconds", DEFAULT_CONFIG["sleep_seconds"])),
        "sleep_override": sleep_override,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_config(path, snapshot)
    return path


def set_lr(config_path: Path, lr: float) -> dict[str, Any]:
    config, _ = load_config_safe(config_path)
    config["lr"] = float(lr)
    write_config(config_path, config)
    return config


def iter_training(
    *,
    config_path: Path,
    metrics_path: Path,
    reset: bool = False,
    sleep: float | None = None,
    stop_flag: Callable[[], bool] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield one metric dict per step. ``sleep`` overrides config sleep_seconds."""
    config_path = resolve_path(config_path)
    metrics_path = resolve_path(metrics_path)

    initial, warn = load_config_safe(config_path)
    if warn:
        warnings.warn(warn, stacklevel=2)

    random.seed(int(initial["seed"]))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if reset:
        metrics_path.write_text("", encoding="utf-8")
    write_config_snapshot(metrics_path, initial, sleep_override=sleep)

    loss = 2.0
    max_steps = int(initial["steps"])
    last_good = dict(initial)
    t0 = time.perf_counter()

    for step in range(max_steps):
        if stop_flag and stop_flag():
            break

        config, warn = load_config_safe(config_path, last_good)
        if warn:
            warnings.warn(warn, stacklevel=2)
        else:
            last_good = dict(config)

        lr = float(config["lr"])
        noise = random.gauss(0.0, 0.01)
        loss = max(0.0, loss * math.exp(-lr) + noise)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        metric = {
            "step": step,
            "loss": loss,
            "lr": lr,
            "wall_ms": round(wall_ms, 3),
        }
        append_metric(metrics_path, metric)
        yield metric

        delay = float(config["sleep_seconds"]) if sleep is None else float(sleep)
        if delay > 0:
            time.sleep(delay)


def run_training(
    *,
    config_path: Path,
    metrics_path: Path,
    reset: bool = False,
    sleep: float | None = None,
    on_step: StepCallback | None = None,
    stop_flag: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in iter_training(
        config_path=config_path,
        metrics_path=metrics_path,
        reset=reset,
        sleep=sleep,
        stop_flag=stop_flag,
    ):
        rows.append(metric)
        if on_step:
            on_step(metric)
        else:
            print(
                f"step={metric['step']:04d} loss={metric['loss']:.6f} "
                f"lr={metric['lr']:.6g} wall_ms={metric['wall_ms']:.1f}",
                flush=True,
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="JSON hyperparameter file (re-read every step)",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="JSONL path for step/loss/lr/wall_ms records",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="truncate metrics file before training",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        metavar="SEC",
        help="override config sleep_seconds (0 = fast batch; omit = use JSON)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_training(
        config_path=args.config,
        metrics_path=args.metrics,
        reset=args.reset,
        sleep=args.sleep,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
