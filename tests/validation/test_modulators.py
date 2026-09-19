"""Round-trip correctness tests for validation.synth.modulators.

These tests are the correctness anchor for the modulator implementations:
each scheme is modulated and then demodulated with an independent, inline
reference demodulator (not the production code) to prove exact-bit recovery
at high SNR (noiseless channel, 0 BER).

Single-carrier schemes (FSK/GFSK/BPSK/QPSK) now prepend the shared
``core.single_carrier`` preamble waveform to the payload, so the reference
demods below locate the payload the same way a receiver would: skip the
first ``len(preamble_wave_*)`` samples (using the shared preamble-geometry
helpers to compute that offset), then demodulate independently.
"""
from __future__ import annotations

import numpy as np

from core.single_carrier import DEFAULT_SC_PROFILE, preamble_wave_fsk, preamble_wave_psk
from validation.synth.modulators import modulate
from validation.types import ModScheme

DATA = bytes([0b10110010, 0b01011101, 0xA5, 0x3C])


def _bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _ref_demod_fsk(payload: np.ndarray, sps: int) -> np.ndarray:
    # instantaneous frequency = d(phase)/dt; sign at symbol centre -> bit
    phase = np.unwrap(np.angle(payload))
    inst_freq = np.diff(phase, prepend=phase[0])
    n_sym = len(payload) // sps
    bits = np.empty(n_sym, dtype=np.uint8)
    for k in range(n_sym):
        seg = inst_freq[k * sps + sps // 4 : k * sps + 3 * sps // 4]
        bits[k] = 1 if np.mean(seg) > 0 else 0
    return bits


def _ref_demod_psk(payload: np.ndarray, sps: int, bits_per_symbol: int) -> np.ndarray:
    n_sym = len(payload) // sps
    centers = np.array([payload[k * sps + sps // 2] for k in range(n_sym)])
    if bits_per_symbol == 1:
        return np.where(centers.real >= 0, 0, 1).astype(np.uint8)
    out = np.empty(n_sym * 2, dtype=np.uint8)
    out[0::2] = np.where(centers.real >= 0, 0, 1)
    out[1::2] = np.where(centers.imag >= 0, 0, 1)
    return out


def _ref_diff_decode(symbols: np.ndarray) -> np.ndarray:
    # d[k] = symbols[k] * conj(symbols[k-1]), symbols[-1] := 1 -- independent
    # restatement of the differential-decode recurrence (mirrors, but does
    # not call, core.single_carrier.sc_diff_decode).
    prev = np.concatenate([[complex(1.0, 0.0)], symbols[:-1]])
    return symbols * np.conj(prev)


def _psk_symbol_centers(payload: np.ndarray, sps: int) -> np.ndarray:
    n_sym = len(payload) // sps
    return np.array([payload[k * sps + sps // 2] for k in range(n_sym)])


def test_fsk_roundtrip_recovers_bits() -> None:
    iq = modulate(DATA, ModScheme.FSK, sps=8)
    assert iq.dtype == np.complex64
    off = len(preamble_wave_fsk(DEFAULT_SC_PROFILE, gfsk=False))
    assert len(iq) == off + 8 * len(DATA) * 8  # preamble + sps * n_bits
    rec = _ref_demod_fsk(iq[off:], sps=8)
    assert np.array_equal(rec, _bits(DATA))


def test_gfsk_roundtrip_recovers_bits() -> None:
    iq = modulate(DATA, ModScheme.GFSK, sps=8, bt=0.5)
    off = len(preamble_wave_fsk(DEFAULT_SC_PROFILE, gfsk=True))
    rec = _ref_demod_fsk(iq[off:], sps=8)
    assert np.array_equal(rec, _bits(DATA))


def test_bpsk_roundtrip_recovers_bits() -> None:
    iq = modulate(DATA, ModScheme.BPSK, sps=8)
    off = len(preamble_wave_psk(DEFAULT_SC_PROFILE))
    assert len(iq) == off + 8 * (len(DATA) * 8)  # sps * n_bits (1 bit/symbol)
    rec = _ref_demod_psk(iq[off:], sps=8, bits_per_symbol=1)
    assert np.array_equal(rec, _bits(DATA))


def test_qpsk_roundtrip_recovers_bits() -> None:
    iq = modulate(DATA, ModScheme.QPSK, sps=8)
    off = len(preamble_wave_psk(DEFAULT_SC_PROFILE))
    assert len(iq) == off + 8 * (len(DATA) * 8 // 2)
    rec = _ref_demod_psk(iq[off:], sps=8, bits_per_symbol=2)
    assert np.array_equal(rec, _bits(DATA))


def test_bpsk_differential_roundtrip_recovers_bits() -> None:
    iq = modulate(DATA, ModScheme.BPSK, sps=8, differential=True)
    off = len(preamble_wave_psk(DEFAULT_SC_PROFILE))
    centers = _psk_symbol_centers(iq[off:], sps=8)
    decoded = _ref_diff_decode(centers)
    rec = np.where(decoded.real >= 0, 0, 1).astype(np.uint8)
    assert np.array_equal(rec, _bits(DATA))


def test_qpsk_differential_roundtrip_recovers_bits() -> None:
    iq = modulate(DATA, ModScheme.QPSK, sps=8, differential=True)
    off = len(preamble_wave_psk(DEFAULT_SC_PROFILE))
    centers = _psk_symbol_centers(iq[off:], sps=8)
    decoded = _ref_diff_decode(centers)
    rec = np.empty(len(decoded) * 2, dtype=np.uint8)
    rec[0::2] = np.where(decoded.real >= 0, 0, 1)
    rec[1::2] = np.where(decoded.imag >= 0, 0, 1)
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


def test_modulate_pilot_spacing_zero_matches_pilotless() -> None:
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    data = bytes(range(16))
    for scheme in (ModScheme.BPSK, ModScheme.QPSK):
        base = modulate(data, scheme, sps=8)
        with_zero = modulate(data, scheme, sps=8, pilot_spacing=0)
        assert np.array_equal(base, with_zero)


def test_modulate_pilots_lengthen_psk_frame() -> None:
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    data = bytes(range(16))
    base = modulate(data, ModScheme.QPSK, sps=8)
    piloted = modulate(data, ModScheme.QPSK, sps=8, pilot_spacing=8)
    assert len(piloted) > len(base)  # pilots add symbols


def test_modulate_pilots_ignored_for_differential() -> None:
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    data = bytes(range(16))
    diff_plain = modulate(data, ModScheme.QPSK, sps=8, differential=True)
    diff_pilots = modulate(
        data, ModScheme.QPSK, sps=8, differential=True, pilot_spacing=8
    )
    assert np.array_equal(diff_plain, diff_pilots)


def test_modulate_ofdm_profile_roundtrips() -> None:
    import numpy as np

    from core.ofdm import demodulate_ofdm
    from core.profiles import OFDM_CATALOG
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    payload = bytes(range(20))
    for name in ("wifi_40", "ofdm_nb"):
        iq = modulate(payload, ModScheme.OFDM, ofdm_profile=OFDM_CATALOG[name])
        assert iq.dtype == np.complex64
        bits = demodulate_ofdm(iq.astype(np.complex128), OFDM_CATALOG[name])
        expect = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        assert np.array_equal(bits[: len(expect)], expect)


def test_modulate_ofdm_default_profile_backcompat() -> None:
    import numpy as np

    from core.ofdm import DEFAULT_OFDM_PROFILE
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    payload = bytes(range(24))
    a = modulate(payload, ModScheme.OFDM)
    b = modulate(payload, ModScheme.OFDM, ofdm_profile=DEFAULT_OFDM_PROFILE)
    assert np.array_equal(a, b)


def test_modulate_coding_expands_and_backcompat() -> None:
    import numpy as np

    from core.coding import CODING_CATALOG
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    payload = bytes(range(12))
    unc = modulate(payload, ModScheme.BPSK, sps=16)
    cod = modulate(payload, ModScheme.BPSK, sps=16, coding=CODING_CATALOG["rep3"])
    assert cod.size > 2 * unc.size  # ~3x payload symbols (rep3) + CRC
    # back-compat: default None == today's output exactly
    assert np.array_equal(unc, modulate(payload, ModScheme.BPSK, sps=16, coding=None))
