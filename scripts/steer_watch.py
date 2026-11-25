#!/usr/bin/env python3
"""Shim — prefer ``st watch``."""

from st_steer.steer_watch import main

if __name__ == "__main__":
    raise SystemExit(main())
