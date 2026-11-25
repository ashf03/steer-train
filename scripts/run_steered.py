#!/usr/bin/env python3
"""Shim — prefer ``st steer``."""

from st_steer.run_steered import main

if __name__ == "__main__":
    raise SystemExit(main())
