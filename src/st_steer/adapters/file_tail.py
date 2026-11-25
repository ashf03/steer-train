"""File-tail adapter — live-reloadable JSON config + metrics.jsonl.

This is today's toy path: an outer loop (``st watch`` / ``st steer``) edits
``configs/*.json``; the trainer re-reads ``lr`` each step. Target lag is
typically 0–1 step when in-process (``run_steered``), or poll-interval when
tailing from another process.
"""

from __future__ import annotations

import atexit
from pathlib import Path
from typing import Any

from st_steer.adapters.base import AdapterInfo, register_adapter
from st_steer.schema import append_jsonl, metric_record


class FileTailLrController:
    """Mutate ``lr`` inside a JSON hyperparameter file."""

    def __init__(self, config_path: Path, *, restore_on_exit: bool = False) -> None:
        from st_steer.train import load_config_safe, write_config

        self.config_path = Path(config_path)
        self._load = load_config_safe
        self._write = write_config
        self._restore_on_exit = restore_on_exit
        self._orig_lr: float | None = None
        self._did_edit = False
        cfg, _ = self._load(self.config_path)
        self._orig_lr = float(cfg.get("lr", 0.05))
        if restore_on_exit:
            atexit.register(self.restore)

    def get_lr(self) -> float:
        cfg, _ = self._load(self.config_path)
        return float(cfg.get("lr", self._orig_lr if self._orig_lr is not None else 0.05))

    def set_lr(self, lr: float) -> None:
        cfg, _ = self._load(self.config_path)
        cfg["lr"] = float(lr)
        self._write(self.config_path, cfg)
        self._did_edit = True

    def restore(self) -> None:
        if not self._restore_on_exit or not self._did_edit or self._orig_lr is None:
            return
        self.set_lr(self._orig_lr)
        self._did_edit = False


class FileTailMetricsSink:
    """Append schema-1 rows to a metrics.jsonl path."""

    def __init__(self, metrics_path: Path) -> None:
        self._path = Path(metrics_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def write(self, row: dict[str, Any]) -> dict[str, Any]:
        if "schema" not in row:
            row = metric_record(
                step=int(row["step"]),
                lr=float(row["lr"]),
                wall_ms=float(row["wall_ms"]),
                train_loss=float(row.get("train_loss", row.get("loss"))),
                val_loss=row.get("val_loss"),
                epoch=row.get("epoch"),
            )
        append_jsonl(self._path, row)
        return row

    def reset(self) -> None:
        self._path.write_text("", encoding="utf-8")


class FileTailAdapter:
    """Bundle controller + metrics sink for the file-tail pattern."""

    name = "file-tail"

    def __init__(
        self,
        *,
        config_path: Path,
        metrics_path: Path,
        restore_on_exit: bool = False,
    ) -> None:
        self.controller = FileTailLrController(
            config_path, restore_on_exit=restore_on_exit
        )
        self.metrics = FileTailMetricsSink(metrics_path)

    @property
    def config_path(self) -> Path:
        return self.controller.config_path

    @property
    def metrics_path(self) -> Path:
        return self.metrics.path


def _factory(
    *,
    config_path: Path | str,
    metrics_path: Path | str,
    restore_on_exit: bool = False,
    **_kwargs: Any,
) -> FileTailAdapter:
    return FileTailAdapter(
        config_path=Path(config_path),
        metrics_path=Path(metrics_path),
        restore_on_exit=restore_on_exit,
    )


register_adapter(
    AdapterInfo(
        name="file-tail",
        summary="Reloadable JSON config + metrics.jsonl (toy default)",
        status="ready",
        lag_note="0–1 step in-process; poll-bound when cross-process",
        requires=(),
    ),
    _factory,
)
