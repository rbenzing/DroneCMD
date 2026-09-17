"""Tests for the shared OFDM PHY (core.ofdm) and its full-chain integration."""
from __future__ import annotations

import numpy as np

from core.ofdm import (
    DEFAULT_OFDM_PROFILE,
    demodulate_ofdm,
    modulate_ofdm,
    qpsk_demap,
    qpsk_map,
)
from validation.repro import rng
from validation.synth.channel import add_awgn_at_snr, apply_channel
from validation.types import ChannelParams


def _bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


# 24 bytes = 192 bits = exactly 2 OFDM data symbols (96 bits each).
DATA_2SYM = bytes(range(24))
# 48 bytes = 384 bits = exactly 4 OFDM data symbols.
DATA_4SYM = bytes(range(48))


def test_profile_structure() -> None:
    p = DEFAULT_OFDM_PROFILE
    assert p.fft_size == 64 and p.cp_len == 16
    assert p.symbol_len == 80
    assert len(p.data_carriers) == 48
    assert len(p.pilot_carriers) == 4
    assert len(p.occupied_carriers) == 52
    assert 0 not in p.occupied_carriers  # DC null
    assert p.n_data_bits_per_symbol == 96


def test_qpsk_map_demap_roundtrip() -> None:
    b = _bits(bytes([0b10110010, 0x3C, 0xA5, 0x00]))
    assert np.array_equal(qpsk_demap(qpsk_map(b)), b)


def test_ofdm_frame_length_two_symbols() -> None:
    tx = modulate_ofdm(_bits(DATA_2SYM))
    # STF + LTF + 2 data symbols, 80 samples each.
    assert len(tx) == (2 + 2) * 80
    assert tx.dtype == np.complex128


def test_ofdm_noiseless_roundtrip_exact_bits() -> None:
    b = _bits(DATA_2SYM)
    rec = demodulate_ofdm(modulate_ofdm(b))
    assert np.array_equal(rec[: len(b)], b)  # 0 BER noiseless


def test_ofdm_ber_decreases_with_snr() -> None:
    b = _bits(DATA_4SYM)
    tx = modulate_ofdm(b).astype(np.complex64)
    g = rng(123)
    lo, _, _ = add_awgn_at_snr(tx, -5.0, g)
    hi, _, _ = add_awgn_at_snr(tx, 25.0, g)
    ber_lo = float(np.mean(demodulate_ofdm(lo.astype(np.complex128))[: len(b)] != b))
    ber_hi = float(np.mean(demodulate_ofdm(hi.astype(np.complex128))[: len(b)] != b))
    assert ber_hi <= ber_lo
    assert ber_hi < 0.01


def test_ofdm_survives_cfo_and_timing_offset() -> None:
    b = _bits(DATA_4SYM)
    tx = modulate_ofdm(b).astype(np.complex64)
    g = rng(7)
    # 0.3 subcarrier of CFO (well within the ±1 S&C range) + 20-sample delay.
    params = ChannelParams(snr_db=30.0, cfo_hz=0.3 / 64.0, timing_offset=20)
    rx, _ = apply_channel(tx, params, g)
    rec = demodulate_ofdm(rx.astype(np.complex128))
    assert float(np.mean(rec[: len(b)] != b)) < 0.01


def test_ofdm_survives_multipath_within_cp() -> None:
    b = _bits(DATA_4SYM)
    tx = modulate_ofdm(b).astype(np.complex64)
    g = rng(9)
    params = ChannelParams(snr_db=30.0, multipath_taps=(1.0 + 0j, 0.3 + 0.1j, 0.1 + 0j))
    rx, _ = apply_channel(tx, params, g)
    rec = demodulate_ofdm(rx.astype(np.complex128))
    assert float(np.mean(rec[: len(b)] != b)) < 0.05


def test_demodulate_ofdm_too_short_returns_empty() -> None:
    assert demodulate_ofdm(np.zeros(10, dtype=np.complex128)).size == 0


def test_ofdm_demodulator_recovers_bits() -> None:
    from core.demodulation import DemodConfig, ModulationScheme, OFDMDemodulator
    from core.ofdm import modulate_ofdm

    b = _bits(DATA_2SYM)
    tx = modulate_ofdm(b).astype(np.complex64)
    cfg = DemodConfig(scheme=ModulationScheme.OFDM)
    res = OFDMDemodulator(cfg).demodulate(tx)
    assert res.is_valid is True
    assert np.array_equal(res.bits[: len(b)].astype(np.uint8), b)


def test_ofdm_demodulator_short_input_invalid() -> None:
    from core.demodulation import DemodConfig, ModulationScheme, OFDMDemodulator

    cfg = DemodConfig(scheme=ModulationScheme.OFDM)
    res = OFDMDemodulator(cfg).demodulate(np.zeros(10, dtype=np.complex64))
    assert res.is_valid is False
    assert res.error_message


def test_demodconfig_ofdm_properties() -> None:
    from core.demodulation import DemodConfig, ModulationScheme

    cfg = DemodConfig(scheme=ModulationScheme.OFDM, sample_rate_hz=1_000_000.0)
    assert cfg.samples_per_symbol == 80  # N + CP
    assert cfg.symbol_rate_hz == 1_000_000.0 / 80
    assert ModulationScheme.OFDM.bits_per_symbol == 96


def test_engine_demodulates_ofdm_via_override() -> None:
    from core.demodulation import DemodConfig, DemodulationEngine, ModulationScheme
    from core.ofdm import modulate_ofdm

    b = _bits(DATA_2SYM)
    tx = modulate_ofdm(b).astype(np.complex64)
    engine = DemodulationEngine(DemodConfig(scheme=ModulationScheme.OFDM))
    res = engine.demodulate(tx, scheme_override=ModulationScheme.OFDM)
    assert res.is_valid and np.array_equal(res.bits[: len(b)].astype(np.uint8), b)


def test_engine_ofdm_override_from_other_scheme() -> None:
    from core.demodulation import DemodConfig, DemodulationEngine, ModulationScheme
    from core.ofdm import modulate_ofdm

    b = _bits(DATA_2SYM)
    tx = modulate_ofdm(b).astype(np.complex64)
    engine = DemodulationEngine(DemodConfig(scheme=ModulationScheme.FSK))
    res = engine.demodulate(tx, scheme_override=ModulationScheme.OFDM)
    assert res.is_valid and np.array_equal(res.bits[: len(b)].astype(np.uint8), b)


def test_ofdm_synth_output_power_normalized() -> None:
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    iq = modulate(bytes(range(24)), ModScheme.OFDM)
    assert iq.dtype == np.complex64
    p = float(np.mean(np.abs(iq) ** 2))
    assert abs(p - 1.0) < 0.1
