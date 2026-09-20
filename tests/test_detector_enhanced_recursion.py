"""Regression: the enhanced SignalDetector path must not infinitely recurse.

``capture.detector`` imports ``core.signal_processing.detect_packets`` but also
defines a module-level ``detect_packets`` backward-compat wrapper that builds a
SignalDetector and calls ``detect_signals``. The wrapper used to shadow the
import, so ``_enhanced_detection`` called the wrapper and recursed
(detect_signals -> _enhanced_detection -> detect_packets -> detect_signals ...)
until RecursionError, silently falling back to threshold detection. These tests
pin the fix: the enhanced path completes quickly and actually detects a burst.
"""
from __future__ import annotations

import time

import numpy as np

from capture.detector import SignalDetector, detect_packets


def _signal_with_burst() -> np.ndarray:
    rng = np.random.default_rng(0)
    noise = 0.01 * (rng.standard_normal(30000) + 1j * rng.standard_normal(30000))
    sig = noise.astype(np.complex64)
    sig[10000:14000] += 0.8 + 0.0j  # a clear 4000-sample burst (> min_length)
    return sig


def test_enhanced_detection_does_not_recurse_and_finds_burst() -> None:
    det = SignalDetector(sample_rate=8e6)
    assert det.enable_enhanced, "enhanced mode should be active for this test"
    t0 = time.perf_counter()
    signals = det.detect_signals(
        _signal_with_burst(), threshold=0.05, method="enhanced"
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"enhanced detection too slow ({elapsed:.1f}s)"
    # It must actually detect the burst via the enhanced path (not silently
    # fall back to 0 regions).
    assert len(signals) >= 1
    s = signals[0]
    assert s["detection_method"] == "enhanced"
    assert s["start_sample"] <= 10050 and s["end_sample"] >= 13950


def test_module_level_detect_packets_wrapper_still_works() -> None:
    # The public backward-compat wrapper must return regions without recursing.
    regions = detect_packets(_signal_with_burst(), threshold=0.05, min_gap=1000)
    assert isinstance(regions, list) and len(regions) >= 1
    start, end = regions[0]
    assert start <= 10050 and end >= 13950
