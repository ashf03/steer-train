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

from st_steer.adapters.file_tail import FileTailLrController
from st_steer.paths import repo_root, resolve
from st_steer.rules.toy import (
    DEFAULT_LR,
    STEER_LR,
    TOY_RULE_ID,
    TRIGGER_LOSS,
    TRIGGER_MIN_STEP,
    TRIGGER_STREAK,
    ToyLossStreakRule,
)
from st_steer.schema import (
    intervention_record,
    patch_intervention_effect,
    train_loss,
    write_intervention,
)
from st_steer.train import (
    DEFAULT_CONFIG_PATH,
    load_config_safe,
    set_lr,
)

ROOT = repo_root()


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

        self.rule = ToyLossStreakRule()
        self.intervened = False
        self.decision_step: int | None = None
        self.decision_at = None
        self.effect_step: int | None = None
        self.effect_delay: int | None = None
        self._orig_lr: float | None = None
        self._did_edit = False
        self._stop = False
        self._lr_ctrl = FileTailLrController(config_path)

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
            f"adapter=file-tail  dry_run={self.dry_run}  "
            f"rule=loss<{TRIGGER_LOSS}×{TRIGGER_STREAK} "
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
        loss = train_loss(row)
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

        decision = self.rule.observe(step=step, loss=loss, lr=lr)
        if not decision.fire:
            return

        evidence = list(decision.evidence.get("steps", []))
        self._fire(evidence=evidence, decision_step=int(decision.decision_step or step))

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
            self._lr_ctrl.set_lr(STEER_LR)
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
            f"- **Rule:** `{TOY_RULE_ID}`\n"
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
            record = intervention_record(
                timestamp=self.decision_at or utc_now(),
                rule_id=TOY_RULE_ID,
                decision_step=int(self.decision_step or 0),
                evidence={"steps": evidence, "threshold": TRIGGER_LOSS, "streak": TRIGGER_STREAK},
                action={
                    "type": "lr_set",
                    "lr_from": DEFAULT_LR,
                    "lr_to": STEER_LR,
                },
                effect_step=self.effect_step,
                effect_delay_steps=self.effect_delay,
                extra={
                    "dry_run": self.dry_run,
                    "metrics": str(self.metrics_path),
                    "config": str(self.config_path),
                    "adapter": "file-tail",
                },
            )
            write_intervention(self.log_jsonl, record)

    def _patch_logs_with_delay(self) -> None:
        """Fill effect_step / effect_delay_steps on the decision row once known."""
        if (
            self.log_jsonl is None
            or self.effect_delay is None
            or self.effect_step is None
            or self.decision_step is None
        ):
            return
        patched = patch_intervention_effect(
            self.log_jsonl,
            decision_step=self.decision_step,
            effect_step=self.effect_step,
            effect_delay_steps=self.effect_delay,
        )
        if patched == 0:
            # Fallback if file was empty / legacy — append a complete v1 row
            write_intervention(
                self.log_jsonl,
                intervention_record(
                    timestamp=utc_now(),
                    rule_id=TOY_RULE_ID,
                    decision_step=self.decision_step,
                    evidence={"steps": list(self.rule.trigger_steps)},
                    action={
                        "type": "lr_set",
                        "lr_from": DEFAULT_LR,
                        "lr_to": STEER_LR,
                    },
                    effect_step=self.effect_step,
                    effect_delay_steps=self.effect_delay,
                    extra={
                        "metrics": str(self.metrics_path),
                        "adapter": "file-tail",
                    },
                ),
            )

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
