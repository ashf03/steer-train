"""CLI smoke — install surface used by CI."""

from __future__ import annotations

from st_steer.cli import main
from st_steer.doctor import main as doctor_main


def test_st_help() -> None:
    assert main(["--help"]) == 0


def test_st_version() -> None:
    assert main(["-V"]) == 0


def test_st_adapters() -> None:
    assert main(["adapters"]) == 0


def test_doctor_ok() -> None:
    assert doctor_main([]) == 0
