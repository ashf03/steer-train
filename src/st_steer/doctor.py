"""Environment and install self-check."""

from __future__ import annotations

import importlib
import sys


def main(argv: list[str] | None = None) -> int:
    _ = argv  # reserved
    from st_steer import __version__
    from st_steer.paths import repo_root
    from st_steer.schema import SCHEMA_VERSION, validate_run_dir

    print(f"st  {__version__}")
    print(f"schema  {SCHEMA_VERSION}")
    print(f"python  {sys.version.split()[0]}  {sys.executable}")

    root = repo_root()
    print(f"root  {root}")

    ok = True
    for rel in ("configs/default.json", "docs/protocol.md", "docs/protocol.scale.md"):
        path = root / rel
        status = "ok" if path.exists() else "MISSING"
        if not path.exists():
            ok = False
        print(f"  {status:7}  {rel}")

    for mod in (
        "st_steer.train",
        "st_steer.summarize",
        "st_steer.compare",
        "st_steer.plot",
        "st_steer.schema",
        "st_steer.adapters",
        "st_steer.rules.s1",
    ):
        try:
            importlib.import_module(mod)
            print(f"  ok       import {mod}")
        except Exception as err:  # noqa: BLE001
            ok = False
            print(f"  FAIL     import {mod}: {err}")

    try:
        from st_steer.adapters import list_adapters

        for info in list_adapters():
            print(f"  {'ok' if info.status != 'deferred' else 'skip':7}  adapter {info.name} ({info.status})")
    except Exception as err:  # noqa: BLE001
        ok = False
        print(f"  FAIL     adapters: {err}")

    for mod, label in (("textual", "tui"), ("matplotlib", "plot")):
        try:
            importlib.import_module(mod)
            print(f"  ok       optional {label} ({mod})")
        except Exception as err:  # noqa: BLE001
            print(f"  warn     optional {label} ({mod}): {err}")

    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"  {'ok' if cuda else 'warn':7}  torch {torch.__version__}  cuda={cuda}")
    except Exception:
        print("  warn     torch not installed (needed for S1 scale only)")

    # Schema sample: prefer a seed run with snapshot; else metrics-only on pilot.
    from st_steer.schema import validate_metrics_file

    sample = root / "results" / "seed_7" / "baseline"
    if not sample.is_dir():
        sample = root / "results" / "pilot" / "baseline"
    if (sample / "metrics.jsonl").exists():
        if (sample / "config.snapshot.json").exists():
            r = validate_run_dir(sample, strict=False)
            label = str(sample.relative_to(root))
        else:
            r = validate_metrics_file(sample / "metrics.jsonl", strict=False)
            label = str((sample / "metrics.jsonl").relative_to(root))
        status = "ok" if r.ok else "FAIL"
        if not r.ok:
            ok = False
        warns = sum(1 for i in r.issues if i.level == "warn")
        print(f"  {status:7}  schema {label} (compat; {warns} warn)")
        for issue in r.issues:
            if issue.level == "error":
                print(f"           {issue.format()}")

    print("doctor  " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1
