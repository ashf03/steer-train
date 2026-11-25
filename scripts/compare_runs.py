#!/usr/bin/env python3
"""Shim — prefer ``st compare``."""

from st_steer.compare import main

if __name__ == "__main__":
    raise SystemExit(main())
