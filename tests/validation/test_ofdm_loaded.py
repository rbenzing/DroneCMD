import numpy as np


def test_pack_unpack_allocation() -> None:
    from core.ofdm import DEFAULT_OFDM_PROFILE, pack_allocation, unpack_allocation

    nd = len(DEFAULT_OFDM_PROFILE.data_carriers)  # 48
    rng = np.random.default_rng(0)
    alloc = rng.choice([0, 2, 4, 6], size=nd).astype(np.intp)
    bits = pack_allocation(alloc)
    assert bits.size == 2 * nd  # 96 bits = one QPSK symbol
    np.testing.assert_array_equal(unpack_allocation(bits), alloc)


def test_modulate_loaded_structure() -> None:
    from core.ofdm import DEFAULT_OFDM_PROFILE as P
    from core.ofdm import modulate_ofdm_loaded

    nd = len(P.data_carriers)
    alloc = np.full(nd, 2, dtype=np.intp)  # all QPSK
    payload = np.random.default_rng(1).integers(0, 2, size=2 * nd * 3).astype(np.uint8)
    iq = modulate_ofdm_loaded(payload, alloc, P)
    # STF + LTF + header(1) + data symbols; length multiple of symbol_len
    assert iq.dtype == np.complex128 and iq.size % P.symbol_len == 0
    # at least preamble(2) + header(1) + ceil(payload_bits / sum(alloc)) data symbols
    assert iq.size // P.symbol_len >= 2 + 1 + 3


def test_loaded_roundtrip_noiseless() -> None:
    from core.ofdm import DEFAULT_OFDM_PROFILE as P
    from core.ofdm import demodulate_ofdm_loaded, modulate_ofdm_loaded

    rng = np.random.default_rng(4)
    nd = len(P.data_carriers)
    alloc = rng.choice([2, 4, 6], size=nd).astype(np.intp)  # no nulls: exact bit count
    nbits = int(alloc.sum()) * 4  # 4 data symbols
    payload = rng.integers(0, 2, size=nbits).astype(np.uint8)
    iq = modulate_ofdm_loaded(payload, alloc, P)
    out = demodulate_ofdm_loaded(iq, P)
    np.testing.assert_array_equal(out[: payload.size], payload)


def test_loaded_recovers_allocation_and_nulls() -> None:
    from core.ofdm import DEFAULT_OFDM_PROFILE as P
    from core.ofdm import demodulate_ofdm_loaded, modulate_ofdm_loaded

    rng = np.random.default_rng(5)
    nd = len(P.data_carriers)
    alloc = rng.choice([0, 2, 4, 6], size=nd).astype(np.intp)
    nbits = int(alloc.sum()) * 2
    payload = rng.integers(0, 2, size=nbits).astype(np.uint8)
    iq = modulate_ofdm_loaded(payload, alloc, P)
    out = demodulate_ofdm_loaded(iq, P)
    np.testing.assert_array_equal(
        out[: payload.size], payload
    )  # nulls skipped correctly


def test_data_channel_response() -> None:
    from core.ofdm import DEFAULT_OFDM_PROFILE as P
    from core.ofdm import data_channel_response

    h = data_channel_response((1.0 + 0j,), P)  # flat channel -> |H|==1
    assert h.size == len(P.data_carriers)
    np.testing.assert_allclose(np.abs(h), 1.0, atol=1e-9)


def test_bitloading_beats_fixed_qpsk_goodput_selective_channel() -> None:
    """On a frequency-selective channel with perfect CSI at TX, bit-loaded OFDM
    delivers more correct payload bits per burst (goodput) than fixed-QPSK
    OFDM. FER/goodput metric (not per-bit BER).

    Operating point (see ``task-5-report.md`` for the tap/SNR sweep that
    backs this): a 4-tap FIR with ~4.5 dB of frequency-selective ripple
    (``|H|`` in ``[1.21, 1.71]`` across the data subcarriers) at 16 dB SNR
    gives bit-loading 16/64-QAM headroom on the strong subcarriers while
    fixed QPSK pays the same per-carrier error rate everywhere it is used.
    Over 20 trials of 20 data symbols each, bit-loaded delivers
    68082/68800 correct payload bits vs. fixed-QPSK's 32640/38400 -- more
    than double, well outside the margin of a single unlucky trial.
    """
    from core.bitloading import chow_load, subcarrier_snr
    from core.ofdm import DEFAULT_OFDM_PROFILE as P
    from core.ofdm import (
        data_channel_response,
        demodulate_ofdm,
        demodulate_ofdm_loaded,
        modulate_ofdm,
        modulate_ofdm_loaded,
    )
    from validation.repro import rng as rng_fn
    from validation.synth.channel import apply_channel
    from validation.types import ChannelParams

    taps = (1.0 + 0j, 0.6 + 0j, -0.6 + 0j, 0.3 + 0j)
    snr_db, trials, n_data_syms = 16.0, 20, 20
    h = data_channel_response(taps, P)
    # perfect-CSI noise_var from the requested SNR at unit signal power
    noise_var = 10 ** (-snr_db / 10.0)
    alloc = chow_load(subcarrier_snr(h, noise_var), target_ber=1e-3)
    fixed_good = loaded_good = 0
    for t in range(trials):
        g = rng_fn(t)
        # fixed QPSK: 96 bits/symbol * n_data_syms
        pf = g.integers(0, 2, size=P.n_data_bits_per_symbol * n_data_syms).astype(
            np.uint8
        )
        cp = ChannelParams(
            snr_db=snr_db,
            cfo_hz=0.0,
            doppler_hz=0.0,
            multipath_taps=taps,
            timing_offset=0,
        )
        yf, _ = apply_channel(modulate_ofdm(pf, P).astype(np.complex64), cp, g)
        rf = demodulate_ofdm(yf.astype(np.complex128), P)
        fixed_good += int(np.sum(rf[: pf.size] == pf) if rf.size >= pf.size else 0)
        # bit-loaded: sum(alloc) bits/symbol * n_data_syms
        pl = g.integers(0, 2, size=int(alloc.sum()) * n_data_syms).astype(np.uint8)
        yl, _ = apply_channel(
            modulate_ofdm_loaded(pl, alloc, P).astype(np.complex64), cp, g
        )
        rl = demodulate_ofdm_loaded(yl.astype(np.complex128), P)
        loaded_good += int(np.sum(rl[: pl.size] == pl) if rl.size >= pl.size else 0)
    assert loaded_good > fixed_good  # more correctly-delivered payload bits
