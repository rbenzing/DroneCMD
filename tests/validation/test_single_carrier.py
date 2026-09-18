"""Tests for the shared single-carrier PHY (core.single_carrier)."""
from __future__ import annotations

import numpy as np

from core.single_carrier import (
    BARKER13,
    DEFAULT_SC_PROFILE,
    PREAMBLE_SYMBOLS,
    preamble_wave_psk,
    sc_frame_sync,
    sc_lock_confidence,
)


def test_preamble_structure() -> None:
    assert len(BARKER13) == 13
    assert PREAMBLE_SYMBOLS.shape == (26,)
    # two identical halves
    assert np.array_equal(PREAMBLE_SYMBOLS[:13], PREAMBLE_SYMBOLS[13:])
    assert set(np.unique(PREAMBLE_SYMBOLS)).issubset({-1.0, 1.0})


def test_frame_sync_finds_known_offset() -> None:
    p = DEFAULT_SC_PROFILE
    ref = preamble_wave_psk(p)
    offset = 37
    rx = np.concatenate(
        [np.zeros(offset, dtype=np.complex128), ref, np.zeros(50, dtype=np.complex128)]
    )
    start, peak = sc_frame_sync(rx, ref, search_span=200)
    assert abs(start - offset) <= 1
    assert abs(peak) > 0.9  # strong, aligned correlation


def test_frame_sync_phase_estimate() -> None:
    p = DEFAULT_SC_PROFILE
    ref = preamble_wave_psk(p)
    rot = np.exp(1j * 0.7)
    rx = np.concatenate([np.zeros(10, dtype=np.complex128), ref * rot])
    _, peak = sc_frame_sync(rx, ref, search_span=100)
    # peak phase recovers the applied rotation
    assert abs(np.angle(peak) - 0.7) < 0.1


def test_lock_confidence_low_on_noise() -> None:
    p = DEFAULT_SC_PROFILE
    ref = preamble_wave_psk(p)
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(400) + 1j * rng.standard_normal(400)).astype(
        np.complex128
    )
    assert sc_lock_confidence(noise, ref, search_span=200) < 0.5
