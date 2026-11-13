#!/usr/bin/env python3
"""Summarize one metrics JSONL: n, final loss, first-zero step, lr change step."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as err:
                raise SystemExit(f"{path}:{line_no}: invalid JSON ({err})") from err
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_no}: expected object, got {type(row).__name__}")
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "final_loss": None,
            "final_lr": None,
            "first_zero_step": None,
            "lr_start": None,
            "lr_change_step": None,
            "lr_after_change": None,
        }

    lr_start = float(rows[0]["lr"])
    first_zero: int | None = None
    lr_change_step: int | None = None
    lr_after: float | None = None

    for row in rows:
        step = int(row["step"])
        loss = float(row["loss"])
        lr = float(row["lr"])
        if first_zero is None and loss == 0.0:
            first_zero = step
        if lr_change_step is None and lr != lr_start:
            lr_change_step = step
            lr_after = lr

    last = rows[-1]
    return {
        "n": len(rows),
        "final_loss": float(last["loss"]),
        "final_lr": float(last["lr"]),
        "first_zero_step": first_zero,
        "lr_start": lr_start,
        "lr_change_step": lr_change_step,
        "lr_after_change": lr_after,
    }


def format_text(path: Path, summary: dict[str, Any]) -> str:
    zero = summary["first_zero_step"]
    change = summary["lr_change_step"]
    lines = [
        f"file             {path}",
        f"n                {summary['n']}",
        f"final_loss       {summary['final_loss']:.6f}"
        if summary["final_loss"] is not None
        else "final_loss       —",
        f"final_lr         {summary['final_lr']}"
        if summary["final_lr"] is not None
        else "final_lr         —",
        f"first_zero_step  {zero if zero is not None else '—'}",
        f"lr_start         {summary['lr_start']}"
        if summary["lr_start"] is not None
        else "lr_start         —",
        (
            f"lr_change_step   {change}  ({summary['lr_start']} → {summary['lr_after_change']})"
            if change is not None
            else "lr_change_step   —  (no change)"
        ),
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "metrics",
        type=Path,
        nargs="+",
        help="one or more metrics.jsonl paths",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="emit JSON (list if multiple files)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payloads: list[dict[str, Any]] = []
    texts: list[str] = []

    for path in args.metrics:
        if not path.exists():
            raise SystemExit(f"not found: {path}")
        summary = summarize(load_metrics(path))
        payloads.append({"path": str(path), **summary})
        texts.append(format_text(path, summary))

    if args.json:
        out = payloads[0] if len(payloads) == 1 else payloads
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print("\n\n".join(texts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
