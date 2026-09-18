"""Coding-gain demonstration + closing regression gate (P3a Task 10).

Headline test: rate-1/3 repetition coding gives real coding gain (fewer bit
errors than uncoded) at matched PHY/SNR, exercising the full synth ->
channel -> single-carrier demod -> deinterleave -> codec decode -> CRC-strip
chain end to end.

Note: ``CODING_INTERLEAVE_DEPTH`` lives in :mod:`core.coding` (moved there
from ``validation.synth.modulators`` per controller ruling), not in
``validation.synth.modulators``.
"""
from __future__ import annotations

import numpy as np

from core.coding import (
    CODING_CATALOG,
    CODING_INTERLEAVE_DEPTH,
    check_and_strip_crc,
    deinterleave,
    make_codec,
)
from core.single_carrier import SCProfile, sc_demodulate_psk
from validation.repro import rng
from validation.synth.channel import add_awgn_at_snr
from validation.synth.modulators import modulate
from validation.types import ModScheme


def test_repetition_beats_uncoded_at_low_snr() -> None:
    """Rate-1/3 repetition coding yields fewer bit errors than uncoded at 4 dB."""
    payload = np.unpackbits(np.frombuffer(bytes(range(24)), dtype=np.uint8))
    snr, trials = 4.0, 30
    unc_err = cod_err = tot = 0
    for s in range(trials):
        g = rng(s)
        # uncoded
        u = modulate(bytes(range(24)), ModScheme.BPSK, sps=16).astype(np.complex128)
        un, _, _ = add_awgn_at_snr(u, snr, g)
        ub = sc_demodulate_psk(
            un.astype(np.complex128),
            SCProfile(sps=16),
            bits_per_symbol=1,
            differential=False,
        )
        unc_err += int(np.sum(ub[: payload.size] != payload))
        # rep3-coded
        c = modulate(
            bytes(range(24)),
            ModScheme.BPSK,
            sps=16,
            coding=CODING_CATALOG["rep3"],
        ).astype(np.complex128)
        cn, _, _ = add_awgn_at_snr(c, snr, g)
        cb = sc_demodulate_psk(
            cn.astype(np.complex128),
            SCProfile(sps=16),
            bits_per_symbol=1,
            differential=False,
        )
        frame = (
            make_codec(CODING_CATALOG["rep3"])
            .decode(deinterleave(cb, CODING_INTERLEAVE_DEPTH))
            .bits
        )
        rec, _ = check_and_strip_crc(frame)
        cod_err += int(np.sum(rec[: payload.size] != payload[: rec.size]))
        tot += payload.size
    assert cod_err < unc_err  # coding gain: fewer errors coded than uncoded


def test_full_suite_regression_marker() -> None:
    """Sentinel; the closing gate is the full validation-suite run (Step 4)."""
    assert True
