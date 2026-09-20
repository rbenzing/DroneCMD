"""Regression: the CLI capture handler must read a real CaptureMetadata field.

`dronecmd capture` captured samples fine but then crashed with
``'CaptureMetadata' object has no attribute 'signal_power_dbfs'`` — the metadata
field is ``signal_level_dbfs``. These pins keep the CLI aligned with the
dataclass contract.
"""
from __future__ import annotations

from pathlib import Path

from core.capture import CaptureMetadata


def test_capture_metadata_exposes_signal_level_not_power() -> None:
    fields = CaptureMetadata.__dataclass_fields__
    assert "signal_level_dbfs" in fields
    assert "signal_power_dbfs" not in fields  # never existed on this dataclass


def test_cli_does_not_read_nonexistent_metadata_attr() -> None:
    src = (
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("cli.py")
        .read_text(encoding="utf-8")
    )
    # The CLI must not access metadata.signal_power_dbfs (AttributeError at
    # runtime); it should use metadata.signal_level_dbfs.
    assert "metadata.signal_power_dbfs" not in src
    assert "metadata.signal_level_dbfs" in src
