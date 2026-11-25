"""Adapter contracts + registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class LrController(Protocol):
    """Read / write optimizer (or live-config) learning rate."""

    def get_lr(self) -> float: ...

    def set_lr(self, lr: float) -> None: ...


@runtime_checkable
class MetricsSink(Protocol):
    """Append schema-1 metric rows."""

    @property
    def path(self) -> Path: ...

    def write(self, row: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class AdapterInfo:
    name: str
    summary: str
    status: str  # ready | optional | deferred
    lag_note: str
    requires: tuple[str, ...] = ()


_REGISTRY: dict[str, AdapterInfo] = {}
_FACTORIES: dict[str, Callable[..., Any]] = {}


def register_adapter(
    info: AdapterInfo,
    factory: Callable[..., Any] | None = None,
) -> None:
    _REGISTRY[info.name] = info
    if factory is not None:
        _FACTORIES[info.name] = factory


def list_adapters() -> list[AdapterInfo]:
    order = ("file-tail", "callback", "aq")
    out: list[AdapterInfo] = []
    for name in order:
        if name in _REGISTRY:
            out.append(_REGISTRY[name])
    for name, info in _REGISTRY.items():
        if name not in order:
            out.append(info)
    return out


def get_adapter(name: str, **kwargs: Any) -> Any:
    if name == "aq":
        raise NotImplementedError(
            "aq adapter is deferred — use file-tail (toy) or callback (S1). "
            "See docs/tool-mvp.md."
        )
    if name not in _FACTORIES:
        known = ", ".join(list_adapters_names())
        raise KeyError(f"unknown adapter {name!r}; known: {known}")
    return _FACTORIES[name](**kwargs)


def list_adapters_names() -> list[str]:
    return [a.name for a in list_adapters()]
