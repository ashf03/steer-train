"""Repo / package path helpers."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Project root (configs/, results/, docs/).

    Order: ``ST_ROOT`` env → walk up from this file looking for markers → cwd.
    """
    env = os.environ.get("ST_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve().parent
    for cand in [here, *here.parents]:
        if (cand / "configs" / "default.json").exists() and (cand / "docs").is_dir():
            return cand
        if (cand / "pyproject.toml").exists() and (cand / "configs").is_dir():
            return cand

    return Path.cwd().resolve()


def resolve(path: Path, root: Path | None = None) -> Path:
    root = root or repo_root()
    return path if path.is_absolute() else root / path
