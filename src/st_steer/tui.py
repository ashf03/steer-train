#!/usr/bin/env python3
"""Full-screen TUI for watching and live-steering a synthetic training run."""

from __future__ import annotations

import argparse
import statistics
import threading
from datetime import datetime, timezone
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Footer,
    Header,
    Label,
    ProgressBar,
    RichLog,
    Sparkline,
    Static,
)

from st_steer.rules.toy import (
    DEFAULT_LR,
    STEER_LR,
    TOY_RULE_ID,
    TRIGGER_LOSS,
    TRIGGER_MIN_STEP,
    TRIGGER_STREAK,
)
from st_steer.schema import intervention_record, train_loss, write_intervention
from st_steer.train import (
    DEFAULT_CONFIG_PATH,
    ROOT,
    iter_training,
    load_config_safe,
    resolve_path,
    set_lr,
    write_config,
)

POST_WINDOW = range(87, 100)


class Stat(Static):
    """One labeled metric tile."""

    DEFAULT_CSS = """
    Stat {
        width: 1fr;
        height: 5;
        padding: 0 1;
        border: tall $surface;
        background: $boost;
    }
    Stat .label {
        color: $text-muted;
        text-style: bold;
    }
    Stat .value {
        text-style: bold;
        color: $accent;
    }
    """

    def __init__(self, label: str, value: str = "—", **kwargs) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._value = value

    def compose(self) -> ComposeResult:
        yield Label(self._label, classes="label")
        yield Label(self._value, classes="value")

    def set_value(self, value: str) -> None:
        self.query_one("Label.value", Label).update(value)


class SteerApp(App[None]):
    """Live training dashboard with protocol auto-steer + manual intervene."""

    TITLE = "st · live LR steer"
    SUB_TITLE = "synthetic trainer"
    CSS = """
    Screen {
        layout: vertical;
    }
    #top {
        height: 7;
        padding: 0 1;
    }
    #stats {
        height: 5;
    }
    #mid {
        height: 1fr;
        padding: 0 1 1 1;
    }
    #chart-pane {
        width: 3fr;
        border: tall $primary;
        padding: 0 1;
    }
    #side {
        width: 2fr;
        border: tall $accent;
        padding: 0 1 1 1;
        background: $boost;
    }
    #chart-title, #side-title, #log-title {
        text-style: bold;
        color: $text-muted;
        padding: 1 0 0 0;
    }
    #controls {
        padding: 1 0 0 0;
    }
    #controls .key {
        text-style: bold;
        color: $accent;
        width: 5;
    }
    #controls .desc {
        color: $text;
    }
    #controls .row {
        height: 1;
        layout: horizontal;
    }
    #auto-state {
        margin-top: 1;
        text-style: bold;
    }
    #spark {
        height: 8;
        margin: 1 0;
    }
    #progress {
        padding: 0 1 1 1;
    }
    #log-pane {
        height: 8;
        border: tall $surface;
        margin: 0 1 1 1;
        padding: 0 1;
    }
    #status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("i", "intervene", "Slow down (cut lr)", show=True, priority=True),
        Binding("r", "restore_lr", "Speed up (restore lr)", show=True),
        Binding("a", "toggle_auto", "Auto on/off", show=True),
        Binding("p", "toggle_pause", "Pause", show=True),
        Binding("q", "quit", "Quit", show=True, priority=True),
    ]

    def __init__(
        self,
        *,
        mode: str,
        config_path: Path,
        metrics_path: Path,
        reset: bool,
        sleep: float,
        seed: int | None,
        auto_steer: bool,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.config_path = resolve_path(config_path)
        self.metrics_path = resolve_path(metrics_path)
        self.reset = reset
        self.sleep = sleep
        self.seed_override = seed
        self.auto_steer = auto_steer and mode == "steered"
        self.paused = False
        self.stop = False
        self.intervened = False
        self.losses: list[float] = []
        self.metrics: list[dict] = []
        self.streak = 0
        self.trigger_steps: list[int] = []
        self._worker: threading.Thread | None = None
        self._orig_config: dict | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="top"):
            yield Static(id="status")
            with Horizontal(id="stats"):
                yield Stat("step", id="stat-step")
                yield Stat("loss", id="stat-loss")
                yield Stat("lr", id="stat-lr")
                yield Stat("wall", id="stat-wall")
                yield Stat("mode", id="stat-mode")
        with Horizontal(id="mid"):
            with Vertical(id="chart-pane"):
                yield Label("loss", id="chart-title")
                yield Sparkline(id="spark", min_color="#3d5a80", max_color="#ee6c4d")
                yield Static(id="summary")
            with Vertical(id="side"):
                yield Label("controls — press a key", id="side-title")
                with Vertical(id="controls"):
                    yield Static("[b]i[/]   slow down  (cut lr to 0.02)", classes="row")
                    yield Static("[b]r[/]   speed up   (restore lr to 0.05)", classes="row")
                    yield Static("[b]a[/]   auto       (let protocol cut lr)", classes="row")
                    yield Static("[b]p[/]   pause      (freeze the run)", classes="row")
                    yield Static("[b]q[/]   quit", classes="row")
                yield Static(id="auto-state")
        yield ProgressBar(id="progress", total=100, show_eta=False)
        with Vertical(id="log-pane"):
            yield Label("what just happened", id="log-title")
            yield RichLog(id="log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        cfg, warn = load_config_safe(self.config_path)
        self._orig_config = dict(cfg)
        if self.seed_override is not None:
            cfg["seed"] = int(self.seed_override)
            write_config(self.config_path, cfg)
        if warn:
            self.log_line(f"[yellow]config:[/] {warn}")

        steps = int(cfg.get("steps", 100))
        self.query_one("#progress", ProgressBar).update(total=steps, progress=0)
        self.query_one("#stat-mode", Stat).set_value(self.mode)
        self._render_auto_state()
        self._set_status(
            f"seed {cfg.get('seed')} · {self.mode} · "
            f"file {self.metrics_path.relative_to(ROOT)}"
        )
        self.log_line(f"started · press [b]i[/] to cut lr, [b]a[/] for auto")
        self.run_train()

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def _render_auto_state(self) -> None:
        if self.mode != "steered":
            text = "auto: off (baseline — no cuts)"
        elif self.intervened:
            text = "auto: done (already cut once)"
        elif self.auto_steer:
            text = (
                f"auto: ON · cuts when loss < {TRIGGER_LOSS} "
                f"for {TRIGGER_STREAK} steps after step {TRIGGER_MIN_STEP}"
            )
        else:
            text = "auto: OFF · press a to turn on"
        self.query_one("#auto-state", Static).update(text)

    def log_line(self, msg: str) -> None:
        self.query_one("#log", RichLog).write(msg)

    def action_toggle_auto(self) -> None:
        if self.mode != "steered":
            self.log_line("auto only works in steered mode")
            return
        self.auto_steer = not self.auto_steer
        self._render_auto_state()
        self.log_line(f"auto → {'ON' if self.auto_steer else 'OFF'}")

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused
        self.log_line("paused" if self.paused else "resumed")

    def action_intervene(self) -> None:
        if self.intervened:
            self.log_line("already cut once this run")
            return
        self._apply_intervene(manual=True)

    def action_restore_lr(self) -> None:
        set_lr(self.config_path, DEFAULT_LR)
        self.log_line(f"sped up · lr back to {DEFAULT_LR}")
        self.query_one("#stat-lr", Stat).set_value(f"{DEFAULT_LR:.3g}")

    def action_quit(self) -> None:
        self.stop = True
        self.paused = False
        self.exit()

    def _apply_intervene(self, *, manual: bool, evidence: list[int] | None = None) -> None:
        set_lr(self.config_path, STEER_LR)
        self.intervened = True
        why = "manual" if manual else "auto protocol"
        steps = evidence or ([self.metrics[-1]["step"]] if self.metrics else [])
        self.log_line(
            f"[bold green]slowed down[/] ({why})  lr {DEFAULT_LR}→{STEER_LR}  "
            f"at steps {steps}"
        )
        self._append_intervention_md(manual=manual, evidence=steps)
        self.query_one("#stat-lr", Stat).set_value(f"{STEER_LR:.3g}")
        self._render_auto_state()

    def _append_intervention_md(self, *, manual: bool, evidence: list[int]) -> None:
        path = ROOT / "docs" / "interventions.md"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        kind = "manual TUI" if manual else "auto protocol"
        block = (
            f"\n## Intervention ({kind})\n\n"
            f"- **Timestamp:** {ts}\n"
            f"- **Rule:** `{TOY_RULE_ID}`\n"
            f"- **Mode:** {self.mode}\n"
            f"- **Evidence steps:** {evidence}\n"
            f"- **Action:** `lr` {DEFAULT_LR} → {STEER_LR} via TUI "
            f"(`{self.config_path.relative_to(ROOT)}`)\n"
            f"- **Metrics:** `{self.metrics_path.relative_to(ROOT)}`\n"
            f"- **Outcome:** _(fill after run)_\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(block)

        decision_step = evidence[-1] if evidence else 0
        write_intervention(
            self.metrics_path.parent / "interventions.jsonl",
            intervention_record(
                timestamp=ts,
                rule_id=TOY_RULE_ID if not manual else f"{TOY_RULE_ID}+manual",
                decision_step=int(decision_step),
                evidence={"steps": evidence, "manual": manual},
                action={
                    "type": "lr_set",
                    "lr_from": DEFAULT_LR,
                    "lr_to": STEER_LR,
                },
                extra={"source": "tui", "mode": self.mode},
            ),
        )

    def _maybe_auto_steer(self, metric: dict) -> None:
        if not self.auto_steer or self.intervened:
            return
        step = int(metric["step"])
        loss = train_loss(metric)
        if step >= TRIGGER_MIN_STEP and loss < TRIGGER_LOSS:
            self.streak += 1
            self.trigger_steps.append(step)
        else:
            self.streak = 0
            self.trigger_steps.clear()
        if self.streak >= TRIGGER_STREAK:
            self._apply_intervene(manual=False, evidence=list(self.trigger_steps))

    def _update_ui(self, metric: dict) -> None:
        step = int(metric["step"])
        loss = train_loss(metric)
        lr = float(metric["lr"])
        wall = float(metric["wall_ms"])
        self.losses.append(loss)
        self.metrics.append(metric)

        self.query_one("#stat-step", Stat).set_value(f"{step}")
        self.query_one("#stat-loss", Stat).set_value(f"{loss:.6f}")
        self.query_one("#stat-lr", Stat).set_value(f"{lr:.3g}")
        self.query_one("#stat-wall", Stat).set_value(f"{wall:.0f} ms")
        self.query_one("#progress", ProgressBar).update(progress=step + 1)

        spark = self.query_one("#spark", Sparkline)
        spark.data = self.losses[-80:]

        summary = f"points={len(self.losses)}  min={min(self.losses):.4f}  max={max(self.losses):.4f}"
        if len(self.losses) >= 2:
            summary += f"  std={statistics.pstdev(self.losses):.4f}"
        post = [m["loss"] for m in self.metrics if int(m["step"]) in POST_WINDOW]
        if post:
            summary += f"  post87-99 σ={statistics.stdev(post) if len(post) > 1 else 0:.4f}"
        if self.intervened:
            summary += "  [intervened]"
        self.query_one("#summary", Static).update(summary)

    @work(thread=True)
    def run_train(self) -> None:
        try:
            for metric in iter_training(
                config_path=self.config_path,
                metrics_path=self.metrics_path,
                reset=self.reset,
                sleep=self.sleep,
                stop_flag=lambda: self.stop,
            ):
                while self.paused and not self.stop:
                    threading.Event().wait(0.05)
                if self.stop:
                    break
                self.call_from_thread(self._update_ui, metric)
                self.call_from_thread(self._maybe_auto_steer, metric)
        finally:
            self.call_from_thread(self._on_finished)

    def _on_finished(self) -> None:
        if self._orig_config is not None:
            # restore default lr after steered runs; keep user's seed if overridden mid-prep
            restore = dict(self._orig_config)
            restore["lr"] = DEFAULT_LR
            if self.seed_override is not None:
                restore["seed"] = int(self.seed_override)
            write_config(self.config_path, restore)

        if not self.metrics:
            self.log_line("[red]no metrics written[/]")
            return

        final = self.metrics[-1]
        self.log_line(
            f"[bold]done[/] steps={len(self.metrics)} "
            f"final_loss={final['loss']:.6f} lr={final['lr']:.3g} "
            f"intervened={self.intervened}"
        )
        self._set_status(
            f"finished · final loss {final['loss']:.6f} · "
            f"{'intervened' if self.intervened else 'no intervene'}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("baseline", "steered"),
        default="steered",
        help="baseline disables auto-steer; steered enables protocol rule",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="default: results/<mode>/metrics.jsonl",
    )
    p.add_argument("--reset", action="store_true", default=True)
    p.add_argument("--no-reset", action="store_false", dest="reset")
    p.add_argument(
        "--sleep",
        type=float,
        default=0.08,
        help="seconds between steps (CLI override; default 0.08 for live feel)",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--no-auto",
        action="store_true",
        help="disable protocol auto-steer even in steered mode",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = args.metrics or Path(f"results/{args.mode}/metrics.jsonl")
    app = SteerApp(
        mode=args.mode,
        config_path=args.config,
        metrics_path=metrics,
        reset=args.reset,
        sleep=args.sleep,
        seed=args.seed,
        auto_steer=not args.no_auto,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
