"""``st`` console entrypoint — small memorized surface."""

from __future__ import annotations

import argparse
import sys


def _cmd_train(argv: list[str]) -> int:
    from st_steer.train import main

    return main(argv)


def _cmd_steer(argv: list[str]) -> int:
    from st_steer.run_steered import main

    return main(argv)


def _cmd_watch(argv: list[str]) -> int:
    from st_steer.steer_watch import main

    return main(argv)


def _cmd_summarize(argv: list[str]) -> int:
    from st_steer.summarize import main

    return main(argv)


def _cmd_compare(argv: list[str]) -> int:
    # Accept --a/--b aliases for --baseline/--steered
    mapped: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--a" and i + 1 < len(argv):
            mapped.extend(["--baseline", argv[i + 1]])
            i += 2
        elif argv[i] == "--b" and i + 1 < len(argv):
            mapped.extend(["--steered", argv[i + 1]])
            i += 2
        else:
            mapped.append(argv[i])
            i += 1
    from st_steer.compare import main

    return main(mapped)


def _cmd_plot(argv: list[str]) -> int:
    from st_steer.plot import main

    return main(argv)


def _cmd_matrix(argv: list[str]) -> int:
    from st_steer.run_matrix import main

    return main(argv)


def _cmd_tui(argv: list[str]) -> int:
    from st_steer.tui import main

    return main(argv)


def _cmd_doctor(argv: list[str]) -> int:
    from st_steer.doctor import main

    return main(argv)


def _cmd_validate(argv: list[str]) -> int:
    from st_steer.schema import main

    return main(argv)


def _cmd_adapters(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="st adapters", description="List trainer adapters")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    # Import package to populate registry
    import st_steer.adapters  # noqa: F401
    from st_steer.adapters import list_adapters

    infos = list_adapters()
    if args.json:
        import json

        json.dump(
            [
                {
                    "name": a.name,
                    "summary": a.summary,
                    "status": a.status,
                    "lag_note": a.lag_note,
                    "requires": list(a.requires),
                }
                for a in infos
            ],
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 0

    print(f"{'name':12} {'status':10} summary")
    for a in infos:
        print(f"{a.name:12} {a.status:10} {a.summary}")
        print(f"{'':12} {'':10} lag: {a.lag_note}")
        if a.requires:
            print(f"{'':12} {'':10} requires: {', '.join(a.requires)}")
    print("\naq is deferred — use file-tail (toy) or callback (S1).")
    return 0


def _cmd_demo(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="st demo")
    parser.add_argument("--toy", action="store_true", help="toy pilot compare + plot")
    parser.add_argument(
        "--s1-dry-run",
        action="store_true",
        help="print frozen S1 plan without training",
    )
    args = parser.parse_args(argv)

    if args.s1_dry_run:
        from st_steer.paths import repo_root

        path = repo_root() / "docs" / "protocol.scale.md"
        print(f"S1 protocol: {path}")
        if path.exists():
            print(path.read_text(encoding="utf-8")[:1200])
            print("\n… (see docs/protocol.scale.md for full freeze)")
        print("dry-run only — no GPU training started")
        return 0

    if args.toy or not argv:
        from st_steer.compare import main as compare_main
        from st_steer.plot import main as plot_main

        print("demo --toy: compare + plot pilot artifacts")
        rc = compare_main([])
        if rc != 0:
            return rc
        return plot_main(["--lr"])

    parser.error("pass --toy or --s1-dry-run")
    return 2


SUBCOMMANDS = {
    "train": (_cmd_train, "run synthetic / adapter trainer"),
    "steer": (_cmd_steer, "train + frozen-rule intervene (one command)"),
    "watch": (_cmd_watch, "attach intervene loop to metrics stream"),
    "summarize": (_cmd_summarize, "summarize metrics.jsonl"),
    "compare": (_cmd_compare, "baseline vs steered verdict"),
    "plot": (_cmd_plot, "loss (+lr) curves"),
    "matrix": (_cmd_matrix, "multi-seed toy/scale driver"),
    "tui": (_cmd_tui, "live steer dashboard"),
    "doctor": (_cmd_doctor, "environment self-check"),
    "validate": (_cmd_validate, "validate artifact schema v1"),
    "adapters": (_cmd_adapters, "list trainer adapters"),
    "demo": (_cmd_demo, "quickstart demos"),
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="st",
        description="st — live outer-loop training intervention",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="print version and exit",
    )
    sub = parser.add_subparsers(dest="command")
    for name, (_fn, help_) in SUBCOMMANDS.items():
        sub.add_parser(name, help=help_, add_help=False)

    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        print("\ncommands:", ", ".join(SUBCOMMANDS))
        return 0

    if argv[0] in ("-V", "--version"):
        from st_steer import __version__

        print(__version__)
        return 0

    cmd = argv[0]
    if cmd not in SUBCOMMANDS:
        parser.error(f"unknown command {cmd!r}. choose from: {', '.join(SUBCOMMANDS)}")

    fn, _ = SUBCOMMANDS[cmd]
    return fn(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
