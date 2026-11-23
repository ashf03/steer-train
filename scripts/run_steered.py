#!/usr/bin/env python3
"""One command: train + frozen-rule steer together (no second terminal).

Applies the intervene rule on each emitted step in-process, so even
``--sleep 0`` still cuts ``lr`` before later steps are written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = Path(__file__).resolve().parent
for p in (str(SRC), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from steer_watch import DEFAULT_LR, SteerWatch, resolve  # noqa: E402
from train import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_config_safe,
    run_training,
    write_config,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/steered/metrics.jsonl"),
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument(
        "--sleep",
        type=float,
        default=0.08,
        help="trainer step delay (CLI override; default 0.08)",
    )
    p.add_argument("--seed", type=int, default=None, help="override config seed")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="apply the rule in memory only; do not edit config",
    )
    p.add_argument(
        "--no-restore",
        action="store_true",
        help="leave config lr as-is after the run",
    )
    p.add_argument(
        "--quiet-steps",
        action="store_true",
        help="do not print every training step line",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = resolve(args.metrics)
    config = args.config if args.config.is_absolute() else (
        DEFAULT_CONFIG_PATH if args.config == DEFAULT_CONFIG_PATH else ROOT / args.config
    )
    if not config.is_absolute():
        config = ROOT / config

    cfg, warn = load_config_safe(config)
    if warn:
        print(f"warn: {warn}", flush=True)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
        cfg["lr"] = DEFAULT_LR
        write_config(config, cfg)
        print(f"seed → {args.seed}  lr reset → {DEFAULT_LR}", flush=True)
    else:
        # ensure we start from the protocol default lr
        if float(cfg.get("lr", DEFAULT_LR)) != DEFAULT_LR:
            cfg["lr"] = DEFAULT_LR
            write_config(config, cfg)
            print(f"lr reset → {DEFAULT_LR}", flush=True)

    metrics.parent.mkdir(parents=True, exist_ok=True)

    watch = SteerWatch(
        metrics_path=metrics,
        config_path=config,
        dry_run=args.dry_run,
        poll_s=0.05,
        restore=not args.no_restore,
        log_md=ROOT / "docs" / "interventions.md",
        log_jsonl=metrics.parent / "interventions.jsonl",
    )
    watch.prepare()

    print(
        f"training → {metrics.relative_to(ROOT)}  sleep={args.sleep}  "
        f"dry_run={args.dry_run}",
        flush=True,
    )

    def on_step(metric: dict) -> None:
        if not args.quiet_steps:
            print(
                f"step={metric['step']:04d} loss={metric['loss']:.6f} "
                f"lr={metric['lr']:.6g} wall_ms={metric['wall_ms']:.1f}",
                flush=True,
            )
        watch.on_metric(metric)

    try:
        run_training(
            config_path=config,
            metrics_path=metrics,
            reset=True,
            sleep=args.sleep,
            on_step=on_step,
        )
    finally:
        watch._finish()

    print("run_steered complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
