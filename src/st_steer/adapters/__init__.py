"""Trainer adapters for live intervene.

Order (MVP): toy uses **file-tail**; S1 default is **callback**; **aq** deferred.
"""

from __future__ import annotations

from st_steer.adapters.base import (
    AdapterInfo,
    LrController,
    MetricsSink,
    list_adapters,
    get_adapter,
)
from st_steer.adapters.callback import CallbackAdapter, SteerHook
from st_steer.adapters.file_tail import FileTailAdapter, FileTailLrController

__all__ = [
    "AdapterInfo",
    "CallbackAdapter",
    "FileTailAdapter",
    "FileTailLrController",
    "LrController",
    "MetricsSink",
    "SteerHook",
    "get_adapter",
    "list_adapters",
]
