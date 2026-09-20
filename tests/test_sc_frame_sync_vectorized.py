"""Correctness pin for the vectorized ``sc_frame_sync``.

``sc_frame_sync`` was a per-offset Python loop; it is being replaced by an
FFT-convolution + sliding-energy vectorization for speed. This test pins the
new implementation to the *exact* math of the original loop (reimplemented
here as ``_oracle``) across random signals, signals with an embedded preamble,
and the edge cases — so the optimization cannot silently change any
blind-resolution decision.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.single_carrier import sc_frame_sync


def _oracle(rx, ref_wave, search_span):
    """The original O(n_positions x L) loop — the behavioral reference."""
    ref = np.asarray(ref_wave, dtype=np.complex128)
    signal = np.asarray(rx, dtype=np.complex128)
    length = ref.size
    ref_norm = float(np.linalg.norm(ref))
    n_positions = min(signal.size - length + 1, search_span)
    if n_positions <= 0:
        return 0, complex(0.0, 0.0)
    corr = np.empty(n_positions, dtype=np.complex128)
    for d in range(n_positions):
        window = signal[d : d + length]
        wn = float(np.linalg.norm(window))
        corr[d] = np.sum(window * np.conj(ref)) / (wn * ref_norm + 1e-12)
    best = int(np.argmax(np.abs(corr)))
    return best, complex(corr[best])


def _ref(length: int, seed: int = 99) -> np.ndarray:
    """A deterministic unit-modulus reference waveform of the given length."""
    rng = np.random.default_rng(seed)
    phases = rng.uniform(-np.pi, np.pi, size=length)
    return np.exp(1j * phases).astype(np.complex128)


def _assert_matches(rx, ref, span):
    o_best, o_peak = _oracle(rx, ref, span)
    n_best, n_peak = sc_frame_sync(rx, ref, span)
    assert n_best == o_best
    assert n_peak == pytest.approx(o_peak, abs=1e-8, rel=1e-6)


def test_matches_oracle_on_random_signals():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(50, 4000))
        length = int(rng.integers(4, min(n, 300)))
        rx = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        ref = rng.standard_normal(length) + 1j * rng.standard_normal(length)
        span = int(rng.integers(1, n + 50))
        _assert_matches(rx.astype(np.complex128), ref.astype(np.complex128), span)


def test_matches_oracle_with_embedded_preamble():
    # A clean peak: the reference appears verbatim inside the signal.
    rng = np.random.default_rng(1)
    ref = _ref(208)  # ~ a 26-symbol preamble at sps=8
    for start in (0, 37, 200):
        rx = (0.01 * (rng.standard_normal(600) + 1j * rng.standard_normal(600))).astype(
            np.complex128
        )
        rx[start : start + ref.size] += ref
        best, _ = sc_frame_sync(rx, ref, search_span=8 * 40)
        assert best == start  # locks exactly where the preamble sits
        _assert_matches(rx, ref, 8 * 40)


def test_matches_oracle_high_sps_preamble():
    # The slow case that motivated the vectorization (long ref, wide search
    # span). Must still match the oracle exactly.
    rng = np.random.default_rng(2)
    ref = _ref(1664)  # ~ a 26-symbol preamble at sps=64
    rx = (rng.standard_normal(6000) + 1j * rng.standard_normal(6000)).astype(
        np.complex128
    )
    rx[1000 : 1000 + ref.size] += ref
    _assert_matches(rx, ref, 64 * 40)


def test_signal_shorter_than_ref_returns_zero():
    ref = _ref(208)
    rx = ref[: ref.size // 2].copy()
    assert sc_frame_sync(rx, ref, 100) == (0, complex(0.0, 0.0))


def test_nonpositive_search_span_returns_zero():
    ref = _ref(208)
    rx = np.zeros(500, dtype=np.complex128)
    assert sc_frame_sync(rx, ref, 0) == (0, complex(0.0, 0.0))


def test_vectorized_is_fast_at_high_sps():
    # A single high-sps sync that dominated blind-resolve runtime in the loop
    # must now be well under a second. Guards against a perf regression.
    import time

    rng = np.random.default_rng(3)
    ref = _ref(13312)  # ~ 26-symbol preamble at sps=512
    rx = (rng.standard_normal(100000) + 1j * rng.standard_normal(100000)).astype(
        np.complex128
    )
    rx[40000 : 40000 + ref.size] += ref
    t0 = time.perf_counter()
    sc_frame_sync(rx, ref, 512 * 40)
    assert time.perf_counter() - t0 < 1.0
