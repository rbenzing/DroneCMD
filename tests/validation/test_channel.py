"""SNR-calibration and channel-impairment tests for validation.synth.channel.

The AWGN calibration is the correctness landmine for the whole T&E spine
(spec Sec8): if signal power is measured as magnitude instead of power, or
the noise variance is split incorrectly across I/Q, the achieved SNR will be
silently wrong while everything still "runs". These tests measure the
output SNR independently (from the returned noisy signal, not from internal
implementation values) across an SNR grid to catch that class of bug.
"""
from __future__ import annotations

import numpy as np

from validation.repro import rng
from validation.synth.channel import add_awgn_at_snr, apply_channel
from validation.types import ChannelParams


def _measure_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    noise = noisy - clean
    s = np.mean(np.abs(clean) ** 2)
    n = np.mean(np.abs(noise) ** 2)
    return 10 * np.log10(s / n)


def test_awgn_hits_target_snr():
    g = rng(0)
    clean = (np.exp(1j * 2 * np.pi * 0.03 * np.arange(20000))).astype(np.complex64)
    clean /= np.sqrt(np.mean(np.abs(clean) ** 2))
    for target in (-5.0, 0.0, 10.0, 20.0):
        noisy, _std, achieved = add_awgn_at_snr(clean, target, g)
        assert abs(achieved - target) < 0.5, (target, achieved)
        assert abs(_measure_snr_db(clean, noisy) - target) < 0.7


def test_awgn_is_deterministic_by_seed():
    clean = np.ones(1000, dtype=np.complex64)
    a, _, _ = add_awgn_at_snr(clean, 10.0, rng(7))
    b, _, _ = add_awgn_at_snr(clean, 10.0, rng(7))
    assert np.allclose(a, b)


def test_apply_channel_timing_offset_shifts():
    clean = np.ones(100, dtype=np.complex64)
    p = ChannelParams(snr_db=40.0, timing_offset=10)
    out, _ = apply_channel(clean, p, rng(1))
    assert len(out) == 110
    assert np.mean(np.abs(out[:10])) < np.mean(np.abs(out[10:]))
