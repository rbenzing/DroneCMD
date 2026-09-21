"""Coded bit-loaded OFDM beats its uncoded and fixed-QPSK baselines.

On a frequency-selective channel, coded bit-loaded OFDM (BICM) should deliver
BOTH (1) a coding gain over *uncoded* bit-loaded OFDM at the same allocation,
and (2) the bit-loading gain over *coded fixed-QPSK* -- carrying the same payload
in a shorter burst (higher spectral efficiency) while still decoding reliably.
"""
import numpy as np

from core.bitloading import chow_load, subcarrier_snr
from core.coding import CODING_CATALOG
from core.ofdm import DEFAULT_OFDM_PROFILE as P
from core.ofdm import (
    data_channel_response,
    decode_coded_ofdm_loaded,
    demodulate_ofdm_loaded,
    modulate_coded_ofdm_loaded,
    modulate_ofdm_loaded,
)
from validation.repro import rng as rng_fn
from validation.synth.channel import apply_channel
from validation.types import ChannelParams

# A 4-tap frequency-selective channel: strong and weak subcarriers coexist, so
# adaptive loading has something to exploit and fixed QPSK leaves headroom. The
# allocation is *designed* at DESIGN_SNR but the channel actually delivers the
# lower ACTUAL_SNR -- so uncoded loading runs at a deficit (high FER) that the
# code's ~4-5 dB gain recovers. This creates a clean coding-gain window.
TAPS = (1.0 + 0j, 0.6 + 0j, -0.45 + 0j, 0.3 + 0j)
DESIGN_SNR_DB = 16.0
ACTUAL_SNR_DB = 14.0
TRIALS = 30
DATA = bytes(range(24))


def _alloc(target_ber):
    h = data_channel_response(TAPS, P)
    n0 = 10 ** (-DESIGN_SNR_DB / 10.0)
    return chow_load(subcarrier_snr(h, n0), target_ber=target_ber)


def _channel(iq, seed):
    cp = ChannelParams(
        snr_db=ACTUAL_SNR_DB,
        cfo_hz=0.0,
        doppler_hz=0.0,
        multipath_taps=TAPS,
        timing_offset=0,
    )
    y, _ = apply_channel(iq.astype(np.complex64), cp, rng_fn(seed))
    return y.astype(np.complex128)


def test_coded_bitloaded_beats_uncoded_and_fixed_qpsk():
    """Measured (deterministic, seeded) at DESIGN_SNR=16 dB / ACTUAL_SNR=14 dB
    over 30 trials with conv_k7_r12: coded bit-loaded 23/30 vs **uncoded
    bit-loaded 0/30** (the coding gain absorbs the 2 dB CSI deficit that leaves
    uncoded loading broken); coded fixed-QPSK 25/30. The adaptive allocation
    carries the payload at sum(alloc)=176 vs fixed-QPSK's 96 bits/symbol -- ~1.8x
    spectral efficiency, so the coded adaptive burst is strictly shorter at
    comparable reliability. Thresholds have margin around the measured values.
    """
    spec = CODING_CATALOG["conv_k7_r12"]
    nd = len(P.data_carriers)
    alloc = _alloc(target_ber=1e-3)  # designed at DESIGN_SNR; run at ACTUAL_SNR
    fixed = np.full(nd, 2, dtype=np.intp)  # coded fixed-QPSK baseline
    payload_bits = np.unpackbits(np.frombuffer(DATA, dtype=np.uint8))

    coded_ok = uncoded_ok = fixed_ok = 0
    for t in range(TRIALS):
        # (1) coded adaptive bit-loaded
        y = _channel(modulate_coded_ofdm_loaded(DATA, alloc, spec, P), t)
        _, ok = decode_coded_ofdm_loaded(y, spec, P)
        coded_ok += int(ok)
        # (2) uncoded adaptive bit-loaded (same allocation)
        yu = _channel(modulate_ofdm_loaded(payload_bits, alloc, P), 1000 + t)
        rb = demodulate_ofdm_loaded(yu, P)
        uncoded_ok += int(
            rb.size >= payload_bits.size
            and np.array_equal(rb[: payload_bits.size], payload_bits)
        )
        # (3) coded fixed-QPSK (same codec, uniform QPSK allocation)
        yf = _channel(modulate_coded_ofdm_loaded(DATA, fixed, spec, P), 2000 + t)
        _, okf = decode_coded_ofdm_loaded(yf, spec, P)
        fixed_ok += int(okf)

    # (1) coding gain: coded recovers the CSI deficit where uncoded almost always
    # fails -- a large, unambiguous margin.
    assert uncoded_ok <= int(0.2 * TRIALS), f"uncoded too good: {uncoded_ok}/{TRIALS}"
    assert coded_ok >= int(0.6 * TRIALS), f"coded unreliable: {coded_ok}/{TRIALS}"
    assert coded_ok >= uncoded_ok + int(
        0.4 * TRIALS
    ), f"coding gain too small: coded={coded_ok} uncoded={uncoded_ok}"

    # (2) bit-loading = spectral efficiency: same payload+codec, adaptive loads
    # the strong carriers higher -> a strictly shorter burst, at reliability
    # comparable to coded fixed-QPSK.
    burst_adaptive = modulate_coded_ofdm_loaded(DATA, alloc, spec, P)
    burst_fixed = modulate_coded_ofdm_loaded(DATA, fixed, spec, P)
    assert int(alloc.sum()) > int(fixed.sum())  # higher spectral efficiency
    assert burst_adaptive.size < burst_fixed.size  # -> shorter burst
    assert fixed_ok >= int(0.6 * TRIALS)  # fixed-QPSK also reliable
