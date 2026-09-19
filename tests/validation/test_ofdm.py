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


def test_ofdm_demodulator_rejects_unsynced_region() -> None:
    from core.demodulation import DemodConfig, ModulationScheme, OFDMDemodulator

    b = _bits(DATA_2SYM)
    tx = modulate_ofdm(b)
    g = rng(0)
    # Scale noise to the burst's own amplitude so this is a realistic
    # mis-lock (comparable-power interference/lead-in), not silence.
    noise_scale = float(np.std(np.abs(tx)))
    noise = noise_scale * (g.standard_normal(200) + 1j * g.standard_normal(200))
    # STF starts at sample 200 -- well beyond the ~80-sample (symbol_len)
    # coarse-timing search window, so the receiver never finds the true STF.
    rx = np.concatenate([noise, tx]).astype(np.complex64)
    cfg = DemodConfig(scheme=ModulationScheme.OFDM)
    res = OFDMDemodulator(cfg).demodulate(rx)
    assert res.is_valid is False
    assert res.error_message


def test_ofdm_demodulator_rejects_pure_noise() -> None:
    from core.demodulation import DemodConfig, ModulationScheme, OFDMDemodulator

    g = rng(1)
    noise = (g.standard_normal(3 * 80) + 1j * g.standard_normal(3 * 80)).astype(
        np.complex64
    )
    cfg = DemodConfig(scheme=ModulationScheme.OFDM)
    res = OFDMDemodulator(cfg).demodulate(noise)
    assert res.is_valid is False
    assert res.error_message


def test_create_synth_dataset_ofdm_protocol() -> None:
    from validation import create_synth_dataset
    from validation.types import ModScheme

    ds = create_synth_dataset(
        protocols=["ocusync"],
        snr_grid_db=[20.0, 30.0],
        n_per_cell=2,
        scheme_by_protocol={"ocusync": ModScheme.OFDM},
        sample_rate=1e6,
        seed=1,
    )
    assert len(ds) == 4
    for cap in ds:
        assert cap.provenance["scheme"] == "ofdm"
        assert cap.truth_regions is not None and len(cap.truth_regions) == 1


def test_ofdm_equalized_symbols_matches_demod() -> None:
    """qpsk_demap(ofdm_equalized_symbols(rx)) is byte-identical to demodulate_ofdm(rx)."""
    from core.ofdm import (
        DEFAULT_OFDM_PROFILE,
        demodulate_ofdm,
        modulate_ofdm,
        ofdm_equalized_symbols,
        qpsk_demap,
    )

    b = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 40, dtype=np.uint8)
    rx = modulate_ofdm(b, DEFAULT_OFDM_PROFILE)
    syms = ofdm_equalized_symbols(rx, DEFAULT_OFDM_PROFILE)
    assert syms.dtype == np.complex128
    assert np.array_equal(qpsk_demap(syms), demodulate_ofdm(rx, DEFAULT_OFDM_PROFILE))


def test_ofdm_equalized_symbols_roundtrip_and_empty() -> None:
    from core.ofdm import (
        DEFAULT_OFDM_PROFILE,
        demodulate_ofdm,
        modulate_ofdm,
        ofdm_equalized_symbols,
    )

    b = np.array([0, 1, 1, 0] * 30, dtype=np.uint8)
    rx = modulate_ofdm(b, DEFAULT_OFDM_PROFILE)
    out = demodulate_ofdm(rx, DEFAULT_OFDM_PROFILE)
    assert np.array_equal(out[: len(b)], b)  # noiseless round-trip preserved
    # Too short for STF+LTF -> empty (guards the short-input (len(x) < 2*slen) early-return path).
    empty = ofdm_equalized_symbols(
        np.zeros(4, dtype=np.complex128), DEFAULT_OFDM_PROFILE
    )
    assert empty.size == 0 and empty.dtype == np.complex128


def test_ofdm_soft_bits_sign_matches_hard_and_scales_with_snr() -> None:
    from core.ofdm import (
        DEFAULT_OFDM_PROFILE,
        demodulate_ofdm,
        modulate_ofdm,
        ofdm_soft_bits,
    )

    b = np.array([1, 0, 0, 1, 1, 1, 0, 0] * 12, dtype=np.uint8)
    clean = modulate_ofdm(b, DEFAULT_OFDM_PROFILE).astype(np.complex128)
    hi, _, _ = add_awgn_at_snr(clean, 30.0, rng(1))
    hard = demodulate_ofdm(hi.astype(np.complex128), DEFAULT_OFDM_PROFILE)
    llr = ofdm_soft_bits(hi.astype(np.complex128), DEFAULT_OFDM_PROFILE)
    assert llr.size == hard.size
    assert np.array_equal((llr < 0).astype(np.uint8), hard)  # sign == hard bit
    lo, _, _ = add_awgn_at_snr(clean, 8.0, rng(1))
    llr_lo = ofdm_soft_bits(lo.astype(np.complex128), DEFAULT_OFDM_PROFILE)
    assert np.mean(np.abs(llr)) > np.mean(np.abs(llr_lo))  # |LLR| grows with SNR


def test_ofdm_soft_bits_empty_on_short() -> None:
    from core.ofdm import DEFAULT_OFDM_PROFILE, ofdm_soft_bits

    assert (
        ofdm_soft_bits(np.zeros(4, dtype=np.complex128), DEFAULT_OFDM_PROFILE).size == 0
    )
