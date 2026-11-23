#!/usr/bin/env python3
"""Watch steered metrics and apply the frozen live-lr intervene rule.

Protocol (docs/protocol.md):
  trigger: loss < 0.05 for 3 consecutive steps AND step >= 60
  action:  one-shot lr 0.05 → 0.02 in configs/default.json
  then:    restore lr to 0.05 on exit
"""

from __future__ import annotations

import argparse
import atexit
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_config_safe,
    set_lr,
)

# Frozen rule constants — keep in sync with docs/protocol.md / src/tui.py
TRIGGER_LOSS = 0.05
TRIGGER_STREAK = 3
TRIGGER_MIN_STEP = 60
STEER_LR = 0.02
DEFAULT_LR = 0.05


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_metric_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    row = json.loads(line)
    if not isinstance(row, dict):
        raise ValueError("metric row must be an object")
    return row


class SteerWatch:
    def __init__(
        self,
        *,
        metrics_path: Path,
        config_path: Path,
        dry_run: bool,
        poll_s: float,
        restore: bool,
        log_md: Path,
        log_jsonl: Path | None,
    ) -> None:
        self.metrics_path = metrics_path
        self.config_path = config_path
        self.dry_run = dry_run
        self.poll_s = poll_s
        self.restore = restore
        self.log_md = log_md
        self.log_jsonl = log_jsonl

        self.streak = 0
        self.trigger_steps: list[int] = []
        self.intervened = False
        self.decision_step: int | None = None
        self.decision_at = None
        self.effect_step: int | None = None
        self.effect_delay: int | None = None
        self._orig_lr: float | None = None
        self._did_edit = False
        self._stop = False

    def prepare(self) -> None:
        cfg, warn = load_config_safe(self.config_path)
        if warn:
            print(f"warn: {warn}", flush=True)
        self._orig_lr = float(cfg.get("lr", DEFAULT_LR))
        if self.restore and not self.dry_run:
            atexit.register(self._restore_lr)
            signal.signal(signal.SIGINT, self._signal_stop)
            signal.signal(signal.SIGTERM, self._signal_stop)
        print(
            f"watching {self.metrics_path}  config={self.config_path}  "
            f"dry_run={self.dry_run}  rule=loss<{TRIGGER_LOSS}×{TRIGGER_STREAK} "
            f"step≥{TRIGGER_MIN_STEP} → lr {STEER_LR}",
            flush=True,
        )

    def _signal_stop(self, signum, frame) -> None:  # noqa: ANN001
        self._stop = True
        print("\nstopping…", flush=True)

    def _restore_lr(self) -> None:
        if self.dry_run or not self._did_edit:
            return
        set_lr(self.config_path, DEFAULT_LR)
        print(f"restored lr → {DEFAULT_LR}", flush=True)

    def on_metric(self, row: dict[str, Any]) -> None:
        step = int(row["step"])
        loss = float(row["loss"])
        lr = float(row["lr"])

        if self.intervened and self.effect_step is None and lr == STEER_LR:
            self.effect_step = step
            if self.decision_step is not None:
                self.effect_delay = step - self.decision_step
            print(
                f"effect  step={step} lr={lr}  "
                f"decision→effect delay={self.effect_delay} step(s)",
                flush=True,
            )
            self._patch_logs_with_delay()

        if self.intervened:
            return

        if step >= TRIGGER_MIN_STEP and loss < TRIGGER_LOSS:
            self.streak += 1
            self.trigger_steps.append(step)
        else:
            self.streak = 0
            self.trigger_steps.clear()

        if self.streak < TRIGGER_STREAK:
            return

        self._fire(evidence=list(self.trigger_steps), decision_step=step)

    def _fire(self, *, evidence: list[int], decision_step: int) -> None:
        self.intervened = True
        self.decision_step = decision_step
        self.decision_at = utc_now()
        action = f"lr {DEFAULT_LR} → {STEER_LR}"
        if self.dry_run:
            print(
                f"DRY-RUN intervene at step {decision_step}  "
                f"evidence={evidence}  would {action}",
                flush=True,
            )
        else:
            set_lr(self.config_path, STEER_LR)
            self._did_edit = True
            print(
                f"INTERVENE at step {decision_step}  evidence={evidence}  {action}",
                flush=True,
            )
        self._write_logs(evidence=evidence)

    def _write_logs(self, *, evidence: list[int]) -> None:
        delay_note = (
            f"{self.effect_delay}"
            if self.effect_delay is not None
            else "_(pending — waiting for lr in metrics)_"
        )
        md = (
            f"\n## Intervention (steer_watch{' dry-run' if self.dry_run else ''})\n\n"
            f"- **Timestamp:** {self.decision_at}\n"
            f"- **Decision step:** {self.decision_step}\n"
            f"- **Evidence steps:** {evidence}\n"
            f"- **Action:** `lr` {DEFAULT_LR} → {STEER_LR}"
            f"{' (dry-run — config not written)' if self.dry_run else ''}\n"
            f"- **Config:** `{self.config_path}`\n"
            f"- **Metrics:** `{self.metrics_path}`\n"
            f"- **Decision→effect delay (steps):** {delay_note}\n"
            f"- **Outcome:** _(fill after run / compare_runs)_\n"
        )
        self.log_md.parent.mkdir(parents=True, exist_ok=True)
        with self.log_md.open("a", encoding="utf-8") as handle:
            handle.write(md)

        if self.log_jsonl is not None:
            record = {
                "timestamp": self.decision_at,
                "dry_run": self.dry_run,
                "decision_step": self.decision_step,
                "evidence_steps": evidence,
                "lr_from": DEFAULT_LR,
                "lr_to": STEER_LR,
                "effect_step": self.effect_step,
                "effect_delay_steps": self.effect_delay,
                "metrics": str(self.metrics_path),
                "config": str(self.config_path),
            }
            self.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with self.log_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _patch_logs_with_delay(self) -> None:
        """Append a follow-up JSONL row once delay is known (md already has pending)."""
        if self.log_jsonl is None or self.effect_delay is None:
            return
        record = {
            "timestamp": utc_now(),
            "event": "effect_observed",
            "decision_step": self.decision_step,
            "effect_step": self.effect_step,
            "effect_delay_steps": self.effect_delay,
            "metrics": str(self.metrics_path),
        }
        with self.log_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def replay(self) -> None:
        if not self.metrics_path.exists():
            raise SystemExit(f"not found: {self.metrics_path}")
        with self.metrics_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = parse_metric_line(line)
                if row is None:
                    continue
                self.on_metric(row)
        self._finish()

    def tail(self, *, from_start: bool = False) -> None:
        if not self.metrics_path.exists():
            print("waiting for metrics file to appear…", flush=True)
        while not self.metrics_path.exists() and not self._stop:
            time.sleep(self.poll_s)

        offset = 0
        # Live runs: start at EOF so we only see new steps.
        # Launcher uses from_start=True after truncating the file.
        # For an already-finished file, use --replay instead.
        if self.metrics_path.exists():
            size = self.metrics_path.stat().st_size
            if from_start:
                offset = 0
                print("tailing from start — waiting for steps…", flush=True)
            elif size > 0:
                offset = size
                print(
                    f"tailing from end ({offset} bytes already there) — "
                    f"start training, use scripts/run_steered.py, or --replay",
                    flush=True,
                )
            else:
                print("metrics file empty — waiting for first steps…", flush=True)

        expected_steps: int | None = None
        cfg, _ = load_config_safe(self.config_path)
        expected_steps = int(cfg.get("steps", 100))

        last_step = -1
        while not self._stop:
            if not self.metrics_path.exists():
                time.sleep(self.poll_s)
                continue
            size = self.metrics_path.stat().st_size
            if size < offset:
                # train --reset truncated the file
                offset = 0
            with self.metrics_path.open(encoding="utf-8") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()
            if chunk:
                for line in chunk.splitlines():
                    try:
                        row = parse_metric_line(line)
                    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as err:
                        print(f"skip bad line: {err}", flush=True)
                        continue
                    if row is None:
                        continue
                    last_step = int(row["step"])
                    self.on_metric(row)
            if expected_steps is not None and last_step >= expected_steps - 1:
                break
            time.sleep(self.poll_s)
        self._finish()

    def _finish(self) -> None:
        if not self.intervened:
            print("done · no intervene (trigger never fired)", flush=True)
        elif self.effect_delay is None:
            print(
                "done · intervened but effect not seen in metrics yet "
                "(trainer may still be on prior lr for 1 step)",
                flush=True,
            )
        else:
            print(
                f"done · decision→effect delay = {self.effect_delay} step(s) "
                f"(target ≤ 2)",
                flush=True,
            )
        if self.restore and not self.dry_run:
            self._restore_lr()
            # avoid double-restore via atexit
            try:
                atexit.unregister(self._restore_lr)
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/steered/metrics.jsonl"),
        help="steered metrics to watch (never baseline)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print would-be edits; do not touch config",
    )
    p.add_argument(
        "--replay",
        action="store_true",
        help="process existing metrics from the start (no live tail)",
    )
    p.add_argument(
        "--poll",
        type=float,
        default=0.05,
        help="tail poll interval seconds (default 0.05)",
    )
    p.add_argument(
        "--no-restore",
        action="store_true",
        help="do not restore lr=0.05 on exit",
    )
    p.add_argument(
        "--log-md",
        type=Path,
        default=ROOT / "docs" / "interventions.md",
    )
    p.add_argument(
        "--log-jsonl",
        type=Path,
        default=None,
        help="default: <metrics_dir>/interventions.jsonl",
    )
    return p


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = resolve(args.metrics)
    config = args.config if args.config.is_absolute() else (
        DEFAULT_CONFIG_PATH if args.config == DEFAULT_CONFIG_PATH else ROOT / args.config
    )
    # argparse default is already absolute DEFAULT_CONFIG_PATH
    if not config.is_absolute():
        config = ROOT / config

    log_jsonl = (
        metrics.parent / "interventions.jsonl"
        if args.log_jsonl is None
        else resolve(args.log_jsonl)
    )

    watch = SteerWatch(
        metrics_path=metrics,
        config_path=config,
        dry_run=args.dry_run,
        poll_s=args.poll,
        restore=not args.no_restore,
        log_md=resolve(args.log_md),
        log_jsonl=log_jsonl,
    )
    watch.prepare()
    if args.replay:
        watch.replay()
    else:
        watch.tail()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
