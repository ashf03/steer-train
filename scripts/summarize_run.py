#!/usr/bin/env python3
"""Shim — prefer ``st summarize``."""

from st_steer.summarize import main

if __name__ == "__main__":
    raise SystemExit(main())
