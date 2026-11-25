#!/usr/bin/env python3
"""Shim — prefer ``st train`` or ``python -m st_steer.train``."""

from st_steer.train import main

if __name__ == "__main__":
    raise SystemExit(main())
