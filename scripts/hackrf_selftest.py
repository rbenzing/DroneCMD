#!/usr/bin/env python3
"""HackRF RX self-test — thin CLI wrapper around ``core.selftest``.

Drives a real HackRF (receive-only) across DroneCMD's capture parameters and
code paths and prints a pass/fail table. Never transmits. The reusable logic
lives in :mod:`core.selftest` (also exposed as ``dronecmd selftest``).

Run:  python scripts/hackrf_selftest.py
Exit: 0 if no checks FAILED (SKIPs are allowed), 1 otherwise.
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from core.selftest import print_summary, run_selftest  # noqa: E402

if __name__ == "__main__":
    sys.exit(print_summary(run_selftest(verbose=True)))
