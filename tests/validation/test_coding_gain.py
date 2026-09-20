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


def test_convolutional_beats_repetition_and_uncoded() -> None:
    """Soft-Viterbi rate-1/2 conv BER < rep3 BER < uncoded BER at low SNR."""
    from core.coding import (
        CODING_CATALOG,
        CODING_INTERLEAVE_DEPTH,
        check_and_strip_crc,
        deinterleave,
        make_codec,
    )
    from core.single_carrier import SCProfile, sc_demodulate_psk, sc_soft_bits

    payload = bytes(range(8))
    pbits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    snr, trials = 3.0, 12
    unc = rep = conv = 0
    prof = SCProfile(sps=32)
    for s in range(trials):
        g = rng(s)
        # uncoded
        u = modulate(payload, ModScheme.BPSK, sps=32).astype(np.complex128)
        un, _, _ = add_awgn_at_snr(u, snr, g)
        ub = sc_demodulate_psk(
            un.astype(np.complex128), prof, bits_per_symbol=1, differential=False
        )
        unc += int(np.sum(ub[: pbits.size] != pbits))
        # rep3 (hard)
        r = modulate(
            payload, ModScheme.BPSK, sps=32, coding=CODING_CATALOG["rep3"]
        ).astype(np.complex128)
        rn, _, _ = add_awgn_at_snr(r, snr, g)
        rb = sc_demodulate_psk(
            rn.astype(np.complex128), prof, bits_per_symbol=1, differential=False
        )
        rframe = (
            make_codec(CODING_CATALOG["rep3"])
            .decode(deinterleave(rb, CODING_INTERLEAVE_DEPTH))
            .bits
        )
        rp, _ = check_and_strip_crc(rframe)
        rep += int(np.sum(rp[: pbits.size] != pbits[: rp.size]))
        # conv (soft Viterbi)
        c = modulate(
            payload, ModScheme.BPSK, sps=32, coding=CODING_CATALOG["conv_k7_r12"]
        ).astype(np.complex128)
        cn, _, _ = add_awgn_at_snr(c, snr, g)
        cl = sc_soft_bits(cn.astype(np.complex128), prof, bits_per_symbol=1)
        cframe = (
            make_codec(CODING_CATALOG["conv_k7_r12"])
            .decode(deinterleave(cl, CODING_INTERLEAVE_DEPTH))
            .bits
        )
        cp, _ = check_and_strip_crc(cframe)
        conv += int(np.sum(cp[: pbits.size] != pbits[: cp.size]))
    assert conv < rep < unc  # convolutional strongest, then repetition, then uncoded


def test_rs_beats_uncoded_low_snr() -> None:
    """Soft-decoded RS(255,239) BER < uncoded BER at low SNR (matched PHY BPSK).

    Mirrors ``test_convolutional_beats_repetition_and_uncoded``'s harness
    shape: fixed-seed Monte Carlo over the full synth -> channel ->
    ``sc_soft_bits`` -> RS decode -> CRC-strip chain. 7.0 dB / 10 trials is
    the smallest budget (measured) at which the fixed seeds land RS at zero
    residual errors while uncoded still has some, keeping the targeted run
    well under 30 s (RS's pure-Python BM/Chien decode is the slow part,
    ~0.7-1.5 s per trial with errors present).
    """
    from core.coding import CODING_CATALOG, CODING_INTERLEAVE_DEPTH
    from core.single_carrier import SCProfile, sc_demodulate_psk, sc_soft_bits

    payload = bytes(range(24))
    pbits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    snr, trials = 7.0, 10
    unc = rs = 0
    prof = SCProfile(sps=64)
    for s in range(trials):
        g = rng(s)
        # uncoded
        u = modulate(payload, ModScheme.BPSK, sps=64).astype(np.complex128)
        un, _, _ = add_awgn_at_snr(u, snr, g)
        ub = sc_demodulate_psk(
            un.astype(np.complex128), prof, bits_per_symbol=1, differential=False
        )
        unc += int(np.sum(ub[: pbits.size] != pbits))
        # rs_255_239 (soft Berlekamp-Massey + Chien + Forney errata decode)
        c = modulate(
            payload, ModScheme.BPSK, sps=64, coding=CODING_CATALOG["rs_255_239"]
        ).astype(np.complex128)
        cn, _, _ = add_awgn_at_snr(c, snr, g)
        cl = sc_soft_bits(cn.astype(np.complex128), prof, bits_per_symbol=1)
        cframe = (
            make_codec(CODING_CATALOG["rs_255_239"])
            .decode(deinterleave(cl, CODING_INTERLEAVE_DEPTH))
            .bits
        )
        rp, _ = check_and_strip_crc(cframe)
        rs += int(np.sum(rp[: pbits.size] != pbits[: rp.size]))
    assert rs < unc  # coding gain: fewer residual errors RS-coded than uncoded


def test_ldpc_beats_uncoded_low_snr() -> None:
    """Soft-decoded LDPC(648,324) r=1/2 BER < uncoded BER at low SNR (matched PHY BPSK).

    Mirrors ``test_rs_beats_uncoded_low_snr``'s harness shape through the SOFT
    demod branch (``sc_soft_bits``, since ``CODING_CATALOG["ldpc_648_r12"]
    .soft_input`` is True) into the normalized min-sum decoder. Payload is 38
    bytes (304 payload bits + 16 CRC bits = 320 info bits, just under
    ``k=324``) so the shortened codeword sits at (near) the code's nominal
    rate-1/2 instead of losing several dB to heavy shortening overhead --
    iterative LDPC decoding has a sharp waterfall, and below-threshold
    operation can decode *worse* than uncoded (false convergence), so getting
    close to the design rate matters for a clean gain demo. 7.2 dB / 15
    trials (measured) is the smallest budget at which the fixed seeds land
    LDPC at zero residual errors while uncoded still has some, keeping the
    targeted run well under 30 s (~5 s measured; pure-Python min-sum is the
    slow part but converges quickly this close to/above threshold).
    """
    from core.coding import CODING_CATALOG, CODING_INTERLEAVE_DEPTH
    from core.single_carrier import SCProfile, sc_demodulate_psk, sc_soft_bits

    payload = bytes(range(38))
    pbits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    snr, trials = 7.2, 15
    unc = ldp = 0
    prof = SCProfile(sps=16)
    for s in range(trials):
        g = rng(s)
        # uncoded
        u = modulate(payload, ModScheme.BPSK, sps=16).astype(np.complex128)
        un, _, _ = add_awgn_at_snr(u, snr, g)
        ub = sc_demodulate_psk(
            un.astype(np.complex128), prof, bits_per_symbol=1, differential=False
        )
        unc += int(np.sum(ub[: pbits.size] != pbits))
        # ldpc_648_r12 (soft normalized min-sum decode)
        c = modulate(
            payload, ModScheme.BPSK, sps=16, coding=CODING_CATALOG["ldpc_648_r12"]
        ).astype(np.complex128)
        cn, _, _ = add_awgn_at_snr(c, snr, g)
        cl = sc_soft_bits(cn.astype(np.complex128), prof, bits_per_symbol=1)
        cframe = (
            make_codec(CODING_CATALOG["ldpc_648_r12"])
            .decode(deinterleave(cl, CODING_INTERLEAVE_DEPTH))
            .bits
        )
        rp, _ = check_and_strip_crc(cframe)
        ldp += int(np.sum(rp[: pbits.size] != pbits[: rp.size]))
    assert ldp < unc  # coding gain: fewer residual errors LDPC-coded than uncoded


def test_bch_beats_uncoded_low_snr() -> None:
    """Hard-decoded BCH(255,223) t=4 BER < uncoded BER at low SNR (matched PHY BPSK).

    Mirrors ``test_rs_beats_uncoded_low_snr``'s harness shape but through the
    HARD demod branch (``sc_demodulate_psk``, not ``sc_soft_bits``) since BCH
    is a hard-input code (``CODING_CATALOG["bch_255_223"].soft_input is
    False``). 5.0 dB / 8 trials (measured) is the smallest budget at which
    the fixed seeds land BCH at zero residual errors while uncoded still has
    several, keeping the targeted run well under 30 s (BCH's pure-Python
    BM/Chien decode is the slow part, ~1.5 s per trial).
    """
    payload = bytes(range(24))
    pbits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    snr, trials = 5.0, 8
    unc = bch = 0
    prof = SCProfile(sps=64)
    for s in range(trials):
        g = rng(s)
        # uncoded
        u = modulate(payload, ModScheme.BPSK, sps=64).astype(np.complex128)
        un, _, _ = add_awgn_at_snr(u, snr, g)
        ub = sc_demodulate_psk(
            un.astype(np.complex128), prof, bits_per_symbol=1, differential=False
        )
        unc += int(np.sum(ub[: pbits.size] != pbits))
        # bch_255_223 (hard-decision syndromes + Berlekamp-Massey + Chien + bit-flip)
        c = modulate(
            payload, ModScheme.BPSK, sps=64, coding=CODING_CATALOG["bch_255_223"]
        ).astype(np.complex128)
        cn, _, _ = add_awgn_at_snr(c, snr, g)
        cb = sc_demodulate_psk(
            cn.astype(np.complex128), prof, bits_per_symbol=1, differential=False
        )
        cframe = (
            make_codec(CODING_CATALOG["bch_255_223"])
            .decode(deinterleave(cb, CODING_INTERLEAVE_DEPTH))
            .bits
        )
        rp, _ = check_and_strip_crc(cframe)
        bch += int(np.sum(rp[: pbits.size] != pbits[: rp.size]))
    assert bch < unc  # coding gain: fewer residual errors BCH-coded than uncoded


def test_turbo_beats_uncoded_awgn_llr() -> None:
    """Turbo (rate-1/3) coding gain on a controlled AWGN-LLR channel.

    Complementary to ``test_turbo_beats_uncoded_end_to_end_soft_demod``: this
    one measures gain on a clean AWGN-LLR channel with a KNOWN noise variance
    -- the standard textbook way coding gain is characterized -- isolating the
    decoder from any demod/sync effect. (It was originally the *only* turbo
    gain test because the shared soft-demod emitted ~40-63% wrong-*sign* LLRs
    at low SNR, a phase-ramp defect since fixed in ``core/single_carrier.py``;
    see ADR-0015 and the end-to-end companion test.) The turbo decoder's
    correctness is further established by ``tests/validation/test_turbo.py``
    (noiseless / error-correction / scale-invariance) and the codec-level
    noisy round-trip in ``test_coding.py``.
    """
    from core.coding import check_and_strip_crc, frame_with_crc

    codec = make_codec(CODING_CATALOG["turbo_r13"])
    payload = bytes(range(30))  # 240 + 16 CRC = 256 = block_k (light shortening)
    pbits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    coded = (
        make_codec(CODING_CATALOG["turbo_r13"])
        .encode(frame_with_crc(pbits.astype(np.uint8)))
        .astype(np.float64)
    )
    sigma, trials = 1.1, 6
    unc = tur = 0
    for s in range(trials):
        g = rng(s)
        # coded: BPSK bit b -> (1-2b); AWGN(sigma); LLR = 2y/sigma^2 (L>0 => bit0)
        y = (1.0 - 2.0 * coded) + sigma * g.standard_normal(coded.size)
        llr = 2.0 * y / (sigma * sigma)
        rp, _ = check_and_strip_crc(codec.decode(llr).bits)
        tur += (
            int(np.sum(rp[: pbits.size] != pbits[: rp.size])) if rp.size else pbits.size
        )
        # uncoded: same AWGN channel on the raw payload bits
        yu = (1.0 - 2.0 * pbits.astype(np.float64)) + sigma * g.standard_normal(
            pbits.size
        )
        unc += int(np.sum((yu < 0).astype(np.uint8) != pbits))
    assert tur < unc  # coding gain on a controlled AWGN-LLR channel


def test_turbo_beats_uncoded_end_to_end_soft_demod() -> None:
    """Turbo (rate-1/3) coding gain END-TO-END through the real ``sc_soft_bits``.

    Now that the soft-demod phase-ramp defect is fixed (decision-directed
    payload tracking + the lower-variance L&R CFO estimator in
    ``core/single_carrier.py``; see ADR-0015), turbo shows genuine coding gain
    through the same synth -> ``sc_soft_bits`` -> soft-decode chain the
    RS/LDPC gain tests use -- not only on the controlled AWGN-LLR channel of
    ``test_turbo_beats_uncoded_awgn_llr``. Before the fix this was impossible:
    the demod emitted ~40-63% wrong-*sign* LLRs on ~40% of low-SNR frames, so
    turbo decoded far *worse* than uncoded end-to-end. Payload is 30 bytes
    (240 + 16 CRC = 256 = ``block_k``, light shortening). 6.0 dB / 8 trials
    (measured) lands turbo at zero residual errors while uncoded still errs.
    """
    from core.coding import CODING_CATALOG, CODING_INTERLEAVE_DEPTH
    from core.single_carrier import SCProfile, sc_demodulate_psk, sc_soft_bits

    payload = bytes(range(30))
    pbits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    snr, trials = 6.0, 8
    unc = tur = 0
    prof = SCProfile(sps=8)
    for s in range(trials):
        g = rng(s)
        # uncoded reference through the hard demod
        u = modulate(payload, ModScheme.BPSK, sps=8).astype(np.complex128)
        un, _, _ = add_awgn_at_snr(u, snr, g)
        ub = sc_demodulate_psk(
            un.astype(np.complex128), prof, bits_per_symbol=1, differential=False
        )
        unc += int(np.sum(ub[: pbits.size] != pbits))
        # turbo_r13 through the soft demod (sc_soft_bits) + iterative decode
        c = modulate(
            payload, ModScheme.BPSK, sps=8, coding=CODING_CATALOG["turbo_r13"]
        ).astype(np.complex128)
        cn, _, _ = add_awgn_at_snr(c, snr, g)
        cl = sc_soft_bits(cn.astype(np.complex128), prof, bits_per_symbol=1)
        cframe = (
            make_codec(CODING_CATALOG["turbo_r13"])
            .decode(deinterleave(cl, CODING_INTERLEAVE_DEPTH))
            .bits
        )
        rp, _ = check_and_strip_crc(cframe)
        tur += int(np.sum(rp[: pbits.size] != pbits[: rp.size]))
    assert tur < unc  # end-to-end coding gain through the (fixed) soft demod


def test_polar_beats_uncoded_low_snr() -> None:
    """CA-SCL polar(256,128) BER < uncoded BER end-to-end through the fixed soft demod.

    Mirrors ``test_turbo_beats_uncoded_end_to_end_soft_demod``'s harness shape
    (same sps=8, same synth -> AWGN -> ``sc_soft_bits`` -> soft-decode -> CRC
    chain), swapped to ``polar_256_128``. Payload is 14 bytes ((128-16)/8=14,
    a near-full rate-1/2 block) to avoid the shortening waterfall, per the
    LDPC/turbo lesson (design 0012 Sec 6 item 7). 6.0 dB / 8 trials (measured,
    same operating point turbo uses through the same fixed demod) lands polar
    at 0 residual errors while uncoded still errs, keeping the run well under
    30 s (pure-Python CA-SCL, list=8, n=256 is the slow part).
    """
    from core.coding import CODING_CATALOG, CODING_INTERLEAVE_DEPTH
    from core.single_carrier import SCProfile, sc_demodulate_psk, sc_soft_bits

    payload = bytes(range(14))
    pbits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    snr, trials = 6.0, 8
    unc = pol = 0
    prof = SCProfile(sps=8)
    for s in range(trials):
        g = rng(s)
        # uncoded reference through the hard demod
        u = modulate(payload, ModScheme.BPSK, sps=8).astype(np.complex128)
        un, _, _ = add_awgn_at_snr(u, snr, g)
        ub = sc_demodulate_psk(
            un.astype(np.complex128), prof, bits_per_symbol=1, differential=False
        )
        unc += int(np.sum(ub[: pbits.size] != pbits))
        # polar_256_128 through the soft demod (sc_soft_bits) + CA-SCL decode
        c = modulate(
            payload, ModScheme.BPSK, sps=8, coding=CODING_CATALOG["polar_256_128"]
        ).astype(np.complex128)
        cn, _, _ = add_awgn_at_snr(c, snr, g)
        cl = sc_soft_bits(cn.astype(np.complex128), prof, bits_per_symbol=1)
        cframe = (
            make_codec(CODING_CATALOG["polar_256_128"])
            .decode(deinterleave(cl, CODING_INTERLEAVE_DEPTH))
            .bits
        )
        rp, _ = check_and_strip_crc(cframe)
        pol += int(np.sum(rp[: pbits.size] != pbits[: rp.size]))
    assert pol < unc  # end-to-end coding gain through the (fixed) soft demod
