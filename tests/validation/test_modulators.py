"""Round-trip correctness tests for validation.synth.modulators.

These tests are the correctness anchor for the modulator implementations:
each scheme is modulated and then demodulated with an independent, inline
reference demodulator (not the production code) to prove exact-bit recovery
at high SNR (noiseless channel, 0 BER).
"""
from __future__ import annotations

import numpy as np

from validation.synth.modulators import modulate
from validation.types import ModScheme

DATA = bytes([0b10110010, 0b01011101, 0xA5, 0x3C])


def _bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _ref_demod_fsk(iq: np.ndarray, sps: int) -> np.ndarray:
    # instantaneous frequency = d(phase)/dt; sign at symbol centre -> bit
    phase = np.unwrap(np.angle(iq))
    inst_freq = np.diff(phase, prepend=phase[0])
    n_sym = len(iq) // sps
    bits = np.empty(n_sym, dtype=np.uint8)
    for k in range(n_sym):
        seg = inst_freq[k * sps + sps // 4 : k * sps + 3 * sps // 4]
        bits[k] = 1 if np.mean(seg) > 0 else 0
    return bits


def _ref_demod_qpsk(iq: np.ndarray, sps: int) -> np.ndarray:
    n_sym = len(iq) // sps
    out = []
    for k in range(n_sym):
        c = iq[k * sps + sps // 2]
        i_bit = 0 if c.real >= 0 else 1
        q_bit = 0 if c.imag >= 0 else 1
        out.extend([i_bit, q_bit])
    return np.array(out, dtype=np.uint8)


def test_fsk_roundtrip_recovers_bits() -> None:
    iq = modulate(DATA, ModScheme.FSK, sps=8)
    assert iq.dtype == np.complex64
    assert len(iq) == 8 * len(DATA) * 8  # sps * n_bits
    rec = _ref_demod_fsk(iq, sps=8)
    assert np.array_equal(rec, _bits(DATA))


def test_qpsk_roundtrip_recovers_bits() -> None:
    iq = modulate(DATA, ModScheme.QPSK, sps=8)
    assert len(iq) == 8 * (len(DATA) * 8 // 2)
    rec = _ref_demod_qpsk(iq, sps=8)
    assert np.array_equal(rec, _bits(DATA))


def test_gfsk_roundtrip_recovers_bits() -> None:
    iq = modulate(DATA, ModScheme.GFSK, sps=8, bt=0.5)
    rec = _ref_demod_fsk(iq, sps=8)
    assert np.array_equal(rec, _bits(DATA))


def test_output_average_power_normalized() -> None:
    iq = modulate(DATA, ModScheme.FSK, sps=8)
    p = float(np.mean(np.abs(iq) ** 2))
    assert abs(p - 1.0) < 0.05
