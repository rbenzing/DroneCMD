"""Tests for the `dronecmd validate` CLI command (synth/ingest/run)."""
from __future__ import annotations

import pytest

from cli import create_parser


def test_parser_has_validate_synth():
    parser = create_parser()
    args = parser.parse_args(
        ["validate", "synth", "--protocols", "mavlink", "--snr", "0:20:10",
         "--n", "2", "--out", "ds"]
    )
    assert args.command == "validate"
    assert args.validate_action == "synth"
    assert args.protocols == "mavlink"


@pytest.mark.integration
def test_validate_synth_writes_dataset(tmp_path):
    from cli import cmd_validate, ConfigManager, CLIOutput

    parser = create_parser()
    out = tmp_path / "ds"
    args = parser.parse_args(
        ["validate", "synth", "--protocols", "mavlink,dji", "--snr", "0:20:20",
         "--n", "1", "--out", str(out)]
    )
    cmd_validate(args, ConfigManager(), CLIOutput(json_output=False))
    assert any(out.glob("*.sigmf-data")) or any(out.glob("*.json"))
