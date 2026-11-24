#!/usr/bin/env python3
"""Run the frozen-protocol seed matrix → results/seed_<k>/{baseline,steered}/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = Path(__file__).resolve().parent
for p in (str(SRC), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from compare_runs import compare  # noqa: E402
from run_steered import main as run_steered_main  # noqa: E402
from steer_watch import DEFAULT_LR  # noqa: E402
from summarize_run import load_metrics  # noqa: E402
from train import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_config_safe,
    run_training,
    write_config,
)

DEFAULT_SEEDS = (7, 11, 13, 17, 19)


def run_baseline(*, seed: int, metrics: Path, sleep: float, config: Path) -> None:
    cfg, _ = load_config_safe(config)
    cfg["seed"] = int(seed)
    cfg["lr"] = DEFAULT_LR
    write_config(config, cfg)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== seed {seed} baseline → {metrics.relative_to(ROOT)} ===", flush=True)
    run_training(
        config_path=config,
        metrics_path=metrics,
        reset=True,
        sleep=sleep,
        on_step=lambda m: None,
    )


def run_one_seed(seed: int, *, sleep: float, dry_run: bool) -> dict:
    base = ROOT / "results" / f"seed_{seed}" / "baseline" / "metrics.jsonl"
    steer = ROOT / "results" / f"seed_{seed}" / "steered" / "metrics.jsonl"

    run_baseline(
        seed=seed,
        metrics=base,
        sleep=sleep,
        config=DEFAULT_CONFIG_PATH,
    )
    rc = run_steered_main(
        [
            "--seed",
            str(seed),
            "--sleep",
            str(sleep),
            "--metrics",
            str(steer.relative_to(ROOT)),
            "--quiet-steps",
            *(["--dry-run"] if dry_run else []),
        ]
    )
    if rc != 0:
        raise SystemExit(f"run_steered failed for seed {seed} (exit {rc})")

    result = compare(load_metrics(base), load_metrics(steer))
    row = {
        "seed": seed,
        "verdict": result["verdict"],
        "verdict_reason": result["verdict_reason"],
        "identical_prefix_length": result["identical_prefix_length"],
        "lr_change_step": result["steered"]["lr_change_step"],
        "final_baseline": result["baseline"]["final_loss"],
        "final_steered": result["steered"]["final_loss"],
        "final_rel": result["final_rel"],
        "vol_baseline": result["post_baseline"]["std"],
        "vol_steered": result["post_steered"]["std"],
        "baseline_path": str(base.relative_to(ROOT)),
        "steered_path": str(steer.relative_to(ROOT)),
    }
    print(
        f"seed {seed}: {row['verdict'].upper()}  "
        f"final {row['final_baseline']:.6f}→{row['final_steered']:.6f}  "
        f"σ {row['vol_baseline']:.6f}→{row['vol_steered']:.6f}",
        flush=True,
    )
    return row


def aggregate(rows: list[dict]) -> dict:
    passes = sum(1 for r in rows if r["verdict"] == "pass")
    fails = sum(1 for r in rows if r["verdict"] == "fail")
    incon = sum(1 for r in rows if r["verdict"] == "inconclusive")
    # protocol: ≥3/5 pass → pass; ≥3/5 fail → fail; else inconclusive
    if passes >= 3:
        call = "pass"
    elif fails >= 3:
        call = "fail"
    else:
        call = "inconclusive"
    return {
        "n": len(rows),
        "pass": passes,
        "fail": fails,
        "inconclusive": incon,
        "matrix_call": call,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help=f"seeds to run (default {list(DEFAULT_SEEDS)})",
    )
    p.add_argument("--sleep", type=float, default=0.0, help="step delay (default 0)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--summary",
        type=Path,
        default=Path("results/matrix_summary.json"),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = [
        run_one_seed(seed, sleep=args.sleep, dry_run=args.dry_run) for seed in args.seeds
    ]
    summary = {"seeds": rows, "aggregate": aggregate(rows)}
    out = args.summary if args.summary.is_absolute() else ROOT / args.summary
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    agg = summary["aggregate"]
    print("\n=== matrix ===", flush=True)
    print(
        f"pass={agg['pass']} fail={agg['fail']} inconclusive={agg['inconclusive']}  "
        f"→ {agg['matrix_call'].upper()}",
        flush=True,
    )
    print(f"wrote {out.relative_to(ROOT)}", flush=True)

    # leave config at protocol default seed 7
    cfg, _ = load_config_safe(DEFAULT_CONFIG_PATH)
    cfg["seed"] = 7
    cfg["lr"] = DEFAULT_LR
    write_config(DEFAULT_CONFIG_PATH, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
