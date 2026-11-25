#!/usr/bin/env python3
"""Shim — prefer ``st plot``."""

from st_steer.plot import main

if __name__ == "__main__":
    raise SystemExit(main())
