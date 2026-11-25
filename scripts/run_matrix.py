#!/usr/bin/env python3
"""Shim — prefer ``st matrix``."""

from st_steer.run_matrix import main

if __name__ == "__main__":
    raise SystemExit(main())
