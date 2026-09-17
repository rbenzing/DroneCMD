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


def _ref_demod_ofdm(iq: np.ndarray) -> np.ndarray:
    # Independent minimal OFDM reference: assumes the burst starts at sample 0
    # (noiseless synth output), strips STF+LTF, and does CP-strip + FFT + QPSK
    # sign-demap on the 48 data subcarriers of each data symbol.
    from core.ofdm import DEFAULT_OFDM_PROFILE

    p = DEFAULT_OFDM_PROFILE
    n, cp, slen = p.fft_size, p.cp_len, p.symbol_len
    data_bins = np.array([k % n for k in p.data_carriers], dtype=np.intp)
    bits = []
    pos = 2 * slen  # skip STF + LTF
    while pos + slen <= len(iq):
        body = iq[pos + cp : pos + slen]
        y = np.fft.fft(body, n)[data_bins]
        out = np.empty(2 * y.size, dtype=np.uint8)
        out[0::2] = (y.real < 0).astype(np.uint8)
        out[1::2] = (y.imag < 0).astype(np.uint8)
        bits.append(out)
        pos += slen
    return np.concatenate(bits) if bits else np.zeros(0, dtype=np.uint8)


def test_ofdm_roundtrip_recovers_bits() -> None:
    payload = bytes(range(24))  # 192 bits = exactly 2 OFDM symbols
    iq = modulate(payload, ModScheme.OFDM)
    assert iq.dtype == np.complex64
    rec = _ref_demod_ofdm(iq)
    assert np.array_equal(rec[: len(_bits(payload))], _bits(payload))
