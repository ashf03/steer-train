#!/usr/bin/env python3
"""Shim — prefer ``st tui``."""

from st_steer.tui import main

if __name__ == "__main__":
    raise SystemExit(main())
