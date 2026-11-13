#!/usr/bin/env python3
"""Compare baseline vs steered metrics against the frozen protocol."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from summarize_run import load_metrics, summarize

# Frozen in docs/protocol.md — do not drift.
POST_WINDOW_START = 87
POST_WINDOW_END = 99  # inclusive
FINAL_LOSS_BUDGET = 1.10  # steered may be at most 10% worse
LATE_INTERVENE_AFTER = 88  # first lr change after this → inconclusive


def losses_in_window(
    rows: list[dict[str, Any]],
    start: int,
    end: int,
) -> list[float]:
    by_step = {int(r["step"]): float(r["loss"]) for r in rows}
    return [by_step[s] for s in range(start, end + 1) if s in by_step]


def identical_prefix_len(baseline: list[dict[str, Any]], steered: list[dict[str, Any]]) -> int:
    n = 0
    for a, b in zip(baseline, steered):
        if int(a["step"]) != int(b["step"]):
            break
        if float(a["loss"]) != float(b["loss"]) or float(a["lr"]) != float(b["lr"]):
            break
        n += 1
    return n


def window_stats(losses: list[float]) -> dict[str, float | None]:
    if not losses:
        return {"n": 0, "min": None, "max": None, "std": None, "range": None}
    std = statistics.stdev(losses) if len(losses) > 1 else 0.0
    lo, hi = min(losses), max(losses)
    return {
        "n": len(losses),
        "min": lo,
        "max": hi,
        "std": std,
        "range": hi - lo,
    }


def verdict(
    *,
    baseline_n: int,
    steered_n: int,
    expected_n: int,
    lr_change_step: int | None,
    final_base: float,
    final_steer: float,
    vol_base: float | None,
    vol_steer: float | None,
) -> tuple[str, str]:
    if baseline_n < expected_n or steered_n < expected_n:
        return "inconclusive", "incomplete runs"
    if lr_change_step is None:
        return "inconclusive", "no lr change in steered run"
    if lr_change_step > LATE_INTERVENE_AFTER:
        return "inconclusive", f"intervene late (first change at step {lr_change_step} > {LATE_INTERVENE_AFTER})"
    if vol_base is None or vol_steer is None:
        return "inconclusive", "post-change window empty"

    final_ok = final_steer <= FINAL_LOSS_BUDGET * final_base
    vol_ok = vol_steer < vol_base

    if vol_ok and final_ok:
        return "pass", "lower post-window σ and final loss within 10%"
    reasons = []
    if not vol_ok:
        reasons.append(f"vol_steered ({vol_steer:.6f}) ≥ vol_baseline ({vol_base:.6f})")
    if not final_ok:
        reasons.append(
            f"final_steered ({final_steer:.6f}) > {FINAL_LOSS_BUDGET:.0%} × baseline "
            f"({FINAL_LOSS_BUDGET * final_base:.6f})"
        )
    return "fail", "; ".join(reasons)


def compare(
    baseline: list[dict[str, Any]],
    steered: list[dict[str, Any]],
    *,
    window_start: int = POST_WINDOW_START,
    window_end: int = POST_WINDOW_END,
) -> dict[str, Any]:
    base_sum = summarize(baseline)
    steer_sum = summarize(steered)
    prefix = identical_prefix_len(baseline, steered)

    base_win = window_stats(losses_in_window(baseline, window_start, window_end))
    steer_win = window_stats(losses_in_window(steered, window_start, window_end))

    final_base = base_sum["final_loss"]
    final_steer = steer_sum["final_loss"]
    final_delta = None
    final_rel = None
    if final_base is not None and final_steer is not None:
        final_delta = final_steer - final_base
        final_rel = (final_steer / final_base - 1.0) if final_base != 0 else None

    expected_n = window_end + 1  # steps 0..end
    label, reason = verdict(
        baseline_n=base_sum["n"],
        steered_n=steer_sum["n"],
        expected_n=expected_n,
        lr_change_step=steer_sum["lr_change_step"],
        final_base=final_base or 0.0,
        final_steer=final_steer or 0.0,
        vol_base=base_win["std"],
        vol_steer=steer_win["std"],
    )

    return {
        "identical_prefix_length": prefix,
        "first_diverge_step": prefix if prefix < min(len(baseline), len(steered)) else None,
        "baseline": base_sum,
        "steered": steer_sum,
        "final_delta": final_delta,
        "final_rel": final_rel,
        "post_window": {"start": window_start, "end": window_end},
        "post_baseline": base_win,
        "post_steered": steer_win,
        "verdict": label,
        "verdict_reason": reason,
    }


def _fmt_float(v: float | None, digits: int = 6) -> str:
    return f"{v:.{digits}f}" if v is not None else "—"


def format_text(baseline_path: Path, steered_path: Path, result: dict[str, Any]) -> str:
    b, s = result["baseline"], result["steered"]
    pb, ps = result["post_baseline"], result["post_steered"]
    win = result["post_window"]
    diverge = result["first_diverge_step"]
    rel = result["final_rel"]
    rel_s = f"{rel:+.2%}" if rel is not None else "—"

    lines = [
        f"baseline         {baseline_path}",
        f"steered          {steered_path}",
        f"identical_prefix {result['identical_prefix_length']}"
        + (f"  (diverge at step {diverge})" if diverge is not None else "  (no diverge)"),
        f"final_loss       base {_fmt_float(b['final_loss'])}  "
        f"steer {_fmt_float(s['final_loss'])}  "
        f"Δ {_fmt_float(result['final_delta'])}  ({rel_s})",
        f"lr_change_step   {s['lr_change_step'] if s['lr_change_step'] is not None else '—'} "
        f"({s['lr_start']} → {s['lr_after_change']})"
        if s["lr_change_step"] is not None
        else "lr_change_step   —",
        f"post_window      steps {win['start']}–{win['end']}",
        f"post_σ           base {_fmt_float(pb['std'])}  steer {_fmt_float(ps['std'])}",
        f"post_range       base {_fmt_float(pb['range'])}  steer {_fmt_float(ps['range'])}",
        f"verdict          {result['verdict'].upper()}  — {result['verdict_reason']}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--baseline",
        type=Path,
        default=Path("results/baseline/metrics.jsonl"),
    )
    p.add_argument(
        "--steered",
        type=Path,
        default=Path("results/steered/metrics.jsonl"),
    )
    p.add_argument(
        "--window-start",
        type=int,
        default=POST_WINDOW_START,
        help=f"post-change window start (default {POST_WINDOW_START}, frozen protocol)",
    )
    p.add_argument(
        "--window-end",
        type=int,
        default=POST_WINDOW_END,
        help=f"post-change window end inclusive (default {POST_WINDOW_END})",
    )
    p.add_argument("--json", action="store_true", help="emit JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.baseline, args.steered):
        if not path.exists():
            raise SystemExit(f"not found: {path}")

    result = compare(
        load_metrics(args.baseline),
        load_metrics(args.steered),
        window_start=args.window_start,
        window_end=args.window_end,
    )

    if args.json:
        json.dump(
            {
                "baseline_path": str(args.baseline),
                "steered_path": str(args.steered),
                **result,
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        print(format_text(args.baseline, args.steered, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
