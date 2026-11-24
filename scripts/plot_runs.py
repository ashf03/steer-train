#!/usr/bin/env python3
"""Plot baseline vs steered loss (+ optional lr) → results/figures/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from summarize_run import load_metrics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "figures" / "loss_curves.png"


def series(rows: list[dict]) -> tuple[list[int], list[float], list[float]]:
    steps = [int(r["step"]) for r in rows]
    losses = [float(r["loss"]) for r in rows]
    lrs = [float(r["lr"]) for r in rows]
    return steps, losses, lrs


def first_lr_change(rows: list[dict]) -> int | None:
    if not rows:
        return None
    start = float(rows[0]["lr"])
    for r in rows:
        if float(r["lr"]) != start:
            return int(r["step"])
    return None


def plot(
    *,
    baseline_path: Path,
    steered_path: Path,
    out: Path,
    show_lr: bool,
) -> Path:
    base = load_metrics(baseline_path)
    steer = load_metrics(steered_path)
    b_steps, b_loss, b_lr = series(base)
    s_steps, s_loss, s_lr = series(steer)
    cut = first_lr_change(steer)

    if show_lr:
        fig, (ax_loss, ax_lr) = plt.subplots(
            2,
            1,
            figsize=(9, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1]},
        )
    else:
        fig, ax_loss = plt.subplots(figsize=(9, 4.5))
        ax_lr = None

    ax_loss.plot(b_steps, b_loss, color="#3d5a80", linewidth=1.6, label="baseline")
    ax_loss.plot(s_steps, s_loss, color="#ee6c4d", linewidth=1.6, label="steered")
    if cut is not None:
        ax_loss.axvline(cut, color="#293241", linestyle="--", linewidth=1, alpha=0.7)
        ax_loss.annotate(
            f"lr cut @ {cut}",
            xy=(cut, ax_loss.get_ylim()[1]),
            xytext=(6, -12),
            textcoords="offset points",
            fontsize=9,
            color="#293241",
        )
    ax_loss.axvspan(87, 99, color="#98c1d9", alpha=0.15, label="post window 87–99")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("baseline vs steered")
    ax_loss.legend(loc="upper right", frameon=False)
    ax_loss.grid(True, alpha=0.25)

    if ax_lr is not None:
        ax_lr.step(b_steps, b_lr, where="post", color="#3d5a80", linewidth=1.4, label="baseline lr")
        ax_lr.step(s_steps, s_lr, where="post", color="#ee6c4d", linewidth=1.4, label="steered lr")
        if cut is not None:
            ax_lr.axvline(cut, color="#293241", linestyle="--", linewidth=1, alpha=0.7)
        ax_lr.set_ylabel("lr")
        ax_lr.set_xlabel("step")
        ax_lr.legend(loc="upper right", frameon=False)
        ax_lr.grid(True, alpha=0.25)
    else:
        ax_loss.set_xlabel("step")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--baseline",
        type=Path,
        default=Path("results/pilot/baseline/metrics.jsonl"),
    )
    p.add_argument(
        "--steered",
        type=Path,
        default=Path("results/pilot/steered/metrics.jsonl"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="output image path (default results/figures/loss_curves.png)",
    )
    p.add_argument(
        "--lr",
        action="store_true",
        help="also plot learning-rate panel",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.baseline, args.steered):
        if not path.exists():
            raise SystemExit(f"not found: {path}")

    out = args.out if args.out.is_absolute() else ROOT / args.out
    written = plot(
        baseline_path=args.baseline,
        steered_path=args.steered,
        out=out,
        show_lr=args.lr,
    )
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
