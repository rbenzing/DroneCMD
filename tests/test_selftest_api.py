"""Non-hardware tests for the self-test module + its CLI wiring.

The full self-test needs a HackRF; these pin the reusable API and the
``dronecmd selftest`` subcommand registration without touching hardware.
"""
from __future__ import annotations

from core.selftest import Result, print_summary, run_selftest, summarize


def test_public_api_exists() -> None:
    assert callable(run_selftest)
    assert callable(summarize)
    assert callable(print_summary)


def test_summarize_counts_and_pass_flag() -> None:
    results = [
        Result("a", "OK", "ok"),
        Result("b", "FAIL", "boom"),
        Result("c", "SKIP", "n/a"),
        Result("d", "OK", "ok"),
    ]
    s = summarize(results)
    assert s["ok"] == 2 and s["fail"] == 1 and s["skip"] == 1 and s["total"] == 4
    assert s["passed"] is False
    assert {c["name"] for c in s["checks"]} == {"a", "b", "c", "d"}


def test_summarize_all_ok_passes() -> None:
    s = summarize([Result("a", "OK"), Result("b", "SKIP")])
    assert s["passed"] is True and s["fail"] == 0


def test_print_summary_exit_code(capsys) -> None:
    assert print_summary([Result("a", "OK")]) == 0
    assert print_summary([Result("a", "FAIL", "x")]) == 1


def test_cli_registers_selftest_subcommand() -> None:
    from cli import create_parser

    args = create_parser().parse_args(["selftest"])
    assert args.command == "selftest"
