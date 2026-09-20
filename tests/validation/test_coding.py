import numpy as np


def test_catalog_has_all_seven_families() -> None:
    from core.coding import CODING_CATALOG, CodeFamily

    fams = {s.family for s in CODING_CATALOG.values()}
    assert {
        CodeFamily.CONVOLUTIONAL,
        CodeFamily.REED_SOLOMON,
        CodeFamily.BCH,
        CodeFamily.LDPC,
        CodeFamily.TURBO,
        CodeFamily.POLAR,
        CodeFamily.FOUNTAIN,
    } <= fams
    assert "uncoded" in CODING_CATALOG and "rep3" in CODING_CATALOG


def test_make_codec_unimplemented_families_raise() -> None:
    # All nine families are now implemented (fountain lands last, P3h) -- every
    # catalog spec must build without raising. Name kept for history/grep
    # continuity even though nothing raises anymore.
    from core.coding import CODING_CATALOG, make_codec

    for name, spec in CODING_CATALOG.items():
        make_codec(spec)  # must not raise


def test_uncoded_and_repetition_roundtrip() -> None:
    from core.coding import CODING_CATALOG, make_codec

    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 4, dtype=np.uint8)
    for name in ("uncoded", "rep3"):
        codec = make_codec(CODING_CATALOG[name])
        coded = codec.encode(bits)
        out = codec.decode(coded).bits
        assert np.array_equal(out[: len(bits)], bits)
    # rep3 expands 3x
    assert make_codec(CODING_CATALOG["rep3"]).encode(bits).size == 3 * bits.size


def test_repetition_corrects_minority_errors() -> None:
    from core.coding import CODING_CATALOG, make_codec

    codec = make_codec(CODING_CATALOG["rep3"])
    bits = np.array([1, 0, 1, 0], dtype=np.uint8)
    coded = codec.encode(bits).copy()
    coded[0] ^= 1  # flip one of the 3 copies of bit0 -> majority still correct
    assert np.array_equal(codec.decode(coded).bits, bits)


def test_crc_frame_roundtrip_and_detects_error() -> None:
    from core.coding import check_and_strip_crc, frame_with_crc

    payload = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=np.uint8)
    frame = frame_with_crc(payload)
    assert frame.size == payload.size + 16
    got, ok = check_and_strip_crc(frame)
    assert ok and np.array_equal(got, payload)
    bad = frame.copy()
    bad[3] ^= 1
    _, ok2 = check_and_strip_crc(bad)
    assert ok2 is False  # loud: corruption detected


def test_interleaver_roundtrip_bits_and_llrs() -> None:
    from core.coding import deinterleave, interleave

    for depth in (0, 1, 4, 7):
        b = np.arange(20, dtype=np.uint8) % 2
        assert np.array_equal(deinterleave(interleave(b, depth), depth), b)
        llr = np.linspace(-3, 3, 20).astype(np.float64)
        assert np.allclose(deinterleave(interleave(llr, depth), depth), llr)


def test_conv_encoder_structure() -> None:
    from core.coding import CODING_CATALOG, make_codec

    codec = make_codec(CODING_CATALOG["conv_k7_r12"])
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    coded = codec.encode(info)
    # rate 1/2 with K-1=6 zero-tail bits -> 2*(len+6) coded bits
    assert coded.size == 2 * (info.size + 6)
    assert coded.dtype == np.uint8
    # all-zero input -> all-zero output (encoder stays in state 0)
    z = make_codec(CODING_CATALOG["conv_k7_r12"]).encode(np.zeros(8, dtype=np.uint8))
    assert not z.any()
    # deterministic
    assert np.array_equal(coded, make_codec(CODING_CATALOG["conv_k7_r12"]).encode(info))
    # a leading 1 (state 0) emits both generator MSB taps = (1,1) for 133/171
    lead = make_codec(CODING_CATALOG["conv_k7_r12"]).encode(
        np.array([1], dtype=np.uint8)
    )
    assert lead[0] == 1 and lead[1] == 1


def test_make_codec_builds_convolutional() -> None:
    from core.coding import CODING_CATALOG, make_codec

    codec = make_codec(CODING_CATALOG["conv_k7_r12"])
    assert codec.spec.family.value == "convolutional"


def test_puncture_lengths_and_depuncture_roundtrip() -> None:
    import numpy as np

    from core.coding import _depuncture, _puncture

    x = np.arange(24, dtype=np.uint8) % 2
    p23 = _puncture(x, (1, 1, 1, 0))  # keep 3 of every 4
    assert p23.size == 24 * 3 // 4
    p34 = _puncture(x, (1, 1, 1, 0, 0, 1))  # keep 4 of every 6
    assert p34.size == 24 * 4 // 6
    # de-puncture reinserts erasures (0) at punctured positions, restoring length
    d = _depuncture(p23.astype(np.float64), (1, 1, 1, 0))
    assert d.size == 24 and np.all(d[3::4] == 0.0)  # punctured slots are 0


def test_convolutional_punctured_encode_lengths() -> None:
    import numpy as np

    from core.coding import CODING_CATALOG, _puncture, make_codec

    info = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    b12 = make_codec(CODING_CATALOG["conv_k7_r12"]).encode(info)
    assert (
        make_codec(CODING_CATALOG["conv_k7_r23"]).encode(info).size
        == _puncture(b12, (1, 1, 1, 0)).size
    )
    assert (
        make_codec(CODING_CATALOG["conv_k7_r34"]).encode(info).size
        == _puncture(b12, (1, 1, 1, 0, 0, 1)).size
    )


def test_conv_roundtrip_all_rates_noiseless() -> None:
    import numpy as np

    from core.coding import CODING_CATALOG, make_codec

    info = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=np.uint8)
    for name in ("conv_k7_r12", "conv_k7_r23", "conv_k7_r34"):
        codec = make_codec(CODING_CATALOG[name])
        coded = codec.encode(info)
        # clean channel: map bits -> LLR (+/-6), L>0 => bit 0
        llr = np.where(coded == 0, 6.0, -6.0).astype(np.float64)
        out = codec.decode(llr).bits
        assert np.array_equal(out, info), f"{name} noiseless round-trip failed"


def test_conv_soft_corrects_errors() -> None:
    import numpy as np

    from core.coding import CODING_CATALOG, make_codec

    codec = make_codec(CODING_CATALOG["conv_k7_r12"])
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=np.uint8)
    llr = np.where(codec.encode(info) == 0, 4.0, -4.0).astype(np.float64)
    llr[2] = -llr[2]
    llr[9] = -llr[9]
    llr[10] = -llr[10]  # a few flipped soft bits
    assert np.array_equal(codec.decode(llr).bits, info)  # Viterbi corrects them


def test_rs_bit_symbol_roundtrip() -> None:
    import numpy as np

    from core.coding import _bits_to_symbols, _symbols_to_bits

    syms = [0x00, 0xFF, 0x2A, 0x71, 0x80, 0x01]
    bits = _symbols_to_bits(syms)
    assert bits.dtype == np.uint8 and bits.size == 8 * len(syms)
    assert _bits_to_symbols(bits) == syms


def test_rs_encode_systematic_parity_len_and_clean_syndromes() -> None:
    import numpy as np

    from core.coding import CODING_CATALOG, _bits_to_symbols, make_codec
    from core.galois import GF256

    spec = CODING_CATALOG["rs_255_239"]  # t=8 -> 16 parity symbols
    codec = make_codec(spec)
    payload = bytes(range(20))
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    coded = codec.encode(bits.astype(np.uint8))
    syms = _bits_to_symbols(coded)
    # message (20) + 2t parity (16) = 36 symbols
    assert len(syms) == 20 + 16
    # a clean codeword evaluates to 0 at alpha^(fcr..fcr+2t-1)
    for j in range(16):
        assert GF256.poly_eval(syms, GF256.pow(2, j + 1)) == 0


def test_rs_catalog_params() -> None:
    from core.coding import CODING_CATALOG, CodeFamily

    rs16 = CODING_CATALOG["rs_255_223"]
    rs8 = CODING_CATALOG["rs_255_239"]
    assert rs16.family == CodeFamily.REED_SOLOMON and rs16.params["t"] == 16
    assert rs8.family == CodeFamily.REED_SOLOMON and rs8.params["t"] == 8
    assert rs16.soft_input is True and rs8.soft_input is True
    assert rs16.params["prim_poly"] == 0x11D and rs16.params["fcr"] == 1


def _rs_codec(name: str):
    from core.coding import CODING_CATALOG, make_codec

    return make_codec(CODING_CATALOG[name])


def test_rs_noiseless_roundtrip_both_codes() -> None:
    import numpy as np

    from core.coding import check_and_strip_crc, frame_with_crc

    for name in ("rs_255_223", "rs_255_239"):
        codec = _rs_codec(name)
        payload = np.unpackbits(np.frombuffer(bytes(range(24)), dtype=np.uint8))
        frame = frame_with_crc(payload.astype(np.uint8))
        coded = codec.encode(frame)
        out = codec.decode(coded)  # hard bits in
        recovered, ok = check_and_strip_crc(out.bits)
        assert ok and np.array_equal(recovered, payload)


def test_rs_corrects_up_to_t_symbol_errors() -> None:
    import numpy as np

    from core.coding import (
        _bits_to_symbols,
        _symbols_to_bits,
        check_and_strip_crc,
        frame_with_crc,
    )

    codec = _rs_codec("rs_255_239")  # t=8
    payload = np.unpackbits(np.frombuffer(bytes(range(24)), dtype=np.uint8))
    frame = frame_with_crc(payload.astype(np.uint8))
    coded = codec.encode(frame)
    syms = _bits_to_symbols(coded)
    for p in range(8):  # flip 8 = t symbols
        syms[p] ^= 0x5A
    corrupted = _symbols_to_bits(syms)
    out = codec.decode(corrupted)
    recovered, ok = check_and_strip_crc(out.bits)
    assert ok and np.array_equal(recovered, payload)


def test_rs_fails_loudly_beyond_t() -> None:
    import numpy as np

    from core.coding import (
        _bits_to_symbols,
        _symbols_to_bits,
        check_and_strip_crc,
        frame_with_crc,
    )

    codec = _rs_codec("rs_255_239")  # t=8
    payload = np.unpackbits(np.frombuffer(bytes(range(24)), dtype=np.uint8))
    frame = frame_with_crc(payload.astype(np.uint8))
    coded = codec.encode(frame)
    syms = _bits_to_symbols(coded)
    for p in range(12):  # 12 > t=8 symbol errors: uncorrectable
        syms[p] ^= 0x5A
    corrupted = _symbols_to_bits(syms)
    out = codec.decode(corrupted)
    _, ok = check_and_strip_crc(out.bits)
    assert ok is False  # loud failure, not a silent wrong payload


def test_rs_errors_and_erasures_explicit_mask() -> None:
    import numpy as np

    from core.coding import (
        _bits_to_symbols,
        _symbols_to_bits,
        check_and_strip_crc,
        frame_with_crc,
    )

    codec = _rs_codec("rs_255_239")  # t=8, nsym=16 -> 2e+f<=16
    payload = np.unpackbits(np.frombuffer(bytes(range(24)), dtype=np.uint8))
    frame = frame_with_crc(payload.astype(np.uint8))
    syms = _bits_to_symbols(codec.encode(frame))
    # 10 erasures + 3 errors: 2*3 + 10 = 16 == 2t -> correctable
    erase_pos = list(range(10))
    for p in erase_pos:
        syms[p] ^= 0x33
    for p in (15, 16, 17):
        syms[p] ^= 0x9C
    msg, ok = codec._decode_symbols(syms, erase_pos)
    recovered, crc_ok = check_and_strip_crc(_symbols_to_bits(msg))
    assert ok and crc_ok and np.array_equal(recovered, payload)
    # beyond the bound: 10 erasures + 5 errors -> 2*5+10=20 > 16 -> fail
    syms2 = _bits_to_symbols(codec.encode(frame))
    for p in erase_pos:
        syms2[p] ^= 0x33
    for p in (15, 16, 17, 18, 19):
        syms2[p] ^= 0x9C
    _, ok2 = codec._decode_symbols(syms2, erase_pos)
    assert ok2 is False


def test_rs_reliability_flagged_erasures_soft_and_scale_invariant() -> None:
    import numpy as np

    from core.coding import (
        CODING_CATALOG,
        check_and_strip_crc,
        frame_with_crc,
        make_codec,
    )

    codec = make_codec(CODING_CATALOG["rs_255_239"])  # t=8
    payload = np.unpackbits(np.frombuffer(bytes(range(24)), dtype=np.uint8))
    frame = frame_with_crc(payload.astype(np.uint8))
    coded = codec.encode(frame)
    # Build LLRs: correct sign, high magnitude, EXCEPT a burst of 12 symbols
    # that are (a) sign-flipped (wrong hard bit) and (b) very low magnitude.
    # 12 errors alone > t=8, but flagged as erasures 12 <= 2t=16 -> correctable.
    llr = np.where(coded > 0, -6.0, 6.0)  # bit=1 -> negative LLR
    burst = range(0, 12)
    for s in burst:
        for b in range(8):
            idx = s * 8 + b
            llr[idx] = -np.sign(llr[idx]) * 0.05  # flip sign, tiny magnitude
    out = codec.decode(llr.astype(np.float64))
    recovered, ok = check_and_strip_crc(out.bits)
    assert ok and np.array_equal(recovered, payload)
    assert out.meta["n_erasures"] >= 12
    # scale invariance: multiply all LLRs by 10 -> identical decode
    out2 = codec.decode((llr * 10.0).astype(np.float64))
    assert np.array_equal(out2.bits, out.bits)


def test_bch_min_poly_and_generator_degrees() -> None:
    from core.coding import _bch_generator_poly, _bch_min_poly
    from core.galois import GF256, GF2m

    # minimal polynomial of alpha^1 over GF(2^8): degree 8, binary coeffs
    m1 = _bch_min_poly(GF256, 1)
    assert len(m1) - 1 == 8
    assert all(c in (0, 1) for c in m1)

    # generator degrees: GF(2^8) t=2 -> 16, t=4 -> 32; GF(2^6) t=2 -> 12
    assert len(_bch_generator_poly(GF256, 2)) - 1 == 16
    assert len(_bch_generator_poly(GF256, 4)) - 1 == 32
    f6 = GF2m(6, 0x43)
    assert len(_bch_generator_poly(f6, 2)) - 1 == 12


def test_bch_generator_roots_are_consecutive_powers() -> None:
    from core.coding import _bch_generator_poly
    from core.galois import GF256

    for t in (2, 4):
        g = _bch_generator_poly(GF256, t)
        # g(alpha^i) == 0 for i = 1 .. 2t (the designed consecutive roots)
        for i in range(1, 2 * t + 1):
            assert GF256.poly_eval(g, GF256.pow(2, i)) == 0
        # binary coefficients
        assert all(c in (0, 1) for c in g)


def test_bch_encode_systematic_parity_and_clean_syndromes() -> None:
    import numpy as np

    from core.coding import CODING_CATALOG, make_codec
    from core.galois import GF256

    codec = make_codec(CODING_CATALOG["bch_255_223"])  # t=4 -> deg g = 32
    info = np.unpackbits(np.frombuffer(bytes(range(20)), dtype=np.uint8))  # 160 bits
    coded = codec.encode(info.astype(np.uint8))
    # systematic: info prefix preserved, 32 parity bits appended
    assert coded.size == info.size + 32
    assert np.array_equal(coded[: info.size], info)
    # clean codeword evaluates to 0 at alpha^1..alpha^8
    poly = [int(b) for b in coded]
    for i in range(1, 9):
        assert GF256.poly_eval(poly, GF256.pow(2, i)) == 0


def test_bch_catalog_params() -> None:
    from core.coding import CODING_CATALOG, CodeFamily

    for name, gf_m, t in (
        ("bch_63_51", 6, 2),
        ("bch_255_239", 8, 2),
        ("bch_255_223", 8, 4),
    ):
        spec = CODING_CATALOG[name]
        assert spec.family == CodeFamily.BCH
        assert spec.params["gf_m"] == gf_m and spec.params["t"] == t
        assert spec.soft_input is False  # BCH is hard-input
    assert CODING_CATALOG["bch_63_51"].params["prim_poly"] == 0x43
    assert CODING_CATALOG["bch_255_223"].params["prim_poly"] == 0x11D


def _bch(name: str):
    from core.coding import CODING_CATALOG, make_codec

    return make_codec(CODING_CATALOG[name])


def test_bch_noiseless_roundtrip_all_codes() -> None:
    import numpy as np

    from core.coding import check_and_strip_crc, frame_with_crc

    # bch_63_51 (k=51) needs a small payload; 255-codes take the full frame.
    cases = [
        ("bch_255_239", bytes(range(24))),
        ("bch_255_223", bytes(range(24))),
        ("bch_63_51", bytes([0xA5, 0x3C, 0x81])),  # 24 bits + 16 CRC = 40 <= 51
    ]
    for name, pl in cases:
        codec = _bch(name)
        payload = np.unpackbits(np.frombuffer(pl, dtype=np.uint8))
        frame = frame_with_crc(payload.astype(np.uint8))
        coded = codec.encode(frame)
        out = codec.decode(coded)  # hard bits, no errors
        recovered, ok = check_and_strip_crc(out.bits)
        assert ok and np.array_equal(recovered, payload)


def test_bch_corrects_up_to_t_bit_errors() -> None:
    import numpy as np

    from core.coding import check_and_strip_crc, frame_with_crc

    codec = _bch("bch_255_223")  # t=4
    payload = np.unpackbits(np.frombuffer(bytes(range(24)), dtype=np.uint8))
    frame = frame_with_crc(payload.astype(np.uint8))
    coded = codec.encode(frame).copy()
    for p in (3, 40, 100, 175):  # flip 4 = t bits
        coded[p] ^= 1
    out = codec.decode(coded)
    recovered, ok = check_and_strip_crc(out.bits)
    assert ok and np.array_equal(recovered, payload)


def test_bch_fails_loudly_beyond_t() -> None:
    import numpy as np

    from core.coding import check_and_strip_crc, frame_with_crc

    codec = _bch("bch_255_239")  # t=2
    payload = np.unpackbits(np.frombuffer(bytes(range(24)), dtype=np.uint8))
    frame = frame_with_crc(payload.astype(np.uint8))
    coded = codec.encode(frame).copy()
    for p in (3, 40, 100, 175, 200):  # 5 > t=2 bit errors: uncorrectable
        coded[p] ^= 1
    out = codec.decode(coded)
    _, ok = check_and_strip_crc(out.bits)
    assert ok is False  # loud failure, not silent wrong payload


def _ldpc(name):
    from core.coding import CODING_CATALOG, make_codec

    return make_codec(CODING_CATALOG[name])


def test_ldpc_catalog_params() -> None:
    from core.coding import CODING_CATALOG, CodeFamily

    for name, rate in (
        ("ldpc_648_r12", "1/2"),
        ("ldpc_648_r23", "2/3"),
        ("ldpc_648_r34", "3/4"),
    ):
        spec = CODING_CATALOG[name]
        assert spec.family == CodeFamily.LDPC
        assert spec.soft_input is True
        assert spec.params["rate"] == rate


def test_ldpc_shortened_roundtrip_all_rates() -> None:
    import numpy as np

    from core.coding import check_and_strip_crc, frame_with_crc

    for name in ("ldpc_648_r12", "ldpc_648_r23", "ldpc_648_r34"):
        codec = _ldpc(name)
        payload = np.unpackbits(np.frombuffer(bytes(range(24)), dtype=np.uint8))
        frame = frame_with_crc(payload.astype(np.uint8))
        coded = codec.encode(frame)
        llr = np.where(coded == 0, 8.0, -8.0).astype(np.float64)  # clean channel
        out = codec.decode(llr)
        recovered, ok = check_and_strip_crc(out.bits)
        assert ok and np.array_equal(recovered, payload)


def test_make_codec_unimplemented_families_raise_ldpc() -> None:
    # LDPC now builds; ensure the enumerating test's builds-set includes it.
    from core.coding import CODING_CATALOG, make_codec

    make_codec(CODING_CATALOG["ldpc_648_r12"])  # must not raise


def _turbo(name):
    from core.coding import CODING_CATALOG, make_codec

    return make_codec(CODING_CATALOG[name])


def test_turbo_catalog_params() -> None:
    from core.coding import CODING_CATALOG, CodeFamily

    for name in ("turbo_r13", "turbo_r12"):
        spec = CODING_CATALOG[name]
        assert spec.family == CodeFamily.TURBO
        assert spec.soft_input is True
        assert spec.params["block_k"] == 256
    assert "puncture" not in CODING_CATALOG["turbo_r13"].params
    assert CODING_CATALOG["turbo_r12"].params["puncture"] is True


def test_turbo_shortened_roundtrip() -> None:
    from core.coding import check_and_strip_crc, frame_with_crc

    for name in ("turbo_r13", "turbo_r12"):
        codec = _turbo(name)
        payload = np.unpackbits(np.frombuffer(bytes(range(24)), dtype=np.uint8))
        frame = frame_with_crc(payload.astype(np.uint8))
        coded = codec.encode(frame)
        llr = np.where(coded == 0, 8.0, -8.0).astype(np.float64)  # clean channel
        out = codec.decode(llr)
        recovered, ok = check_and_strip_crc(out.bits)
        assert ok and np.array_equal(recovered, payload)


def test_make_codec_unimplemented_families_raise_turbo() -> None:
    # TURBO now builds; ensure the enumerating test's builds-set includes it.
    from core.coding import CODING_CATALOG, make_codec

    make_codec(CODING_CATALOG["turbo_r13"])  # must not raise
    make_codec(CODING_CATALOG["turbo_r12"])  # must not raise


def test_make_codec_unimplemented_families_raise_polar() -> None:
    # POLAR now builds; ensure the enumerating test's builds-set includes it.
    from core.coding import CODING_CATALOG, CodeFamily, make_codec

    for spec in CODING_CATALOG.values():
        if spec.family == CodeFamily.POLAR:
            make_codec(spec)  # must not raise


def test_polar_roundtrip_all_rates_with_shortening() -> None:
    import numpy as np

    from core.coding import (
        CODING_CATALOG,
        check_and_strip_crc,
        frame_with_crc,
        make_codec,
    )

    for name in ("polar_256_128", "polar_256_85", "polar_256_170"):
        spec = CODING_CATALOG[name]
        codec = make_codec(spec)
        cap = (spec.k - 16) // 8
        for nbytes in (1, cap // 2 or 1, cap):
            payload = bytes(range(nbytes))
            pbits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
            frame = frame_with_crc(pbits.astype(np.uint8))
            coded = codec.encode(frame)
            llr = (1.0 - 2.0 * coded.astype(np.float64)) * 8.0
            out = codec.decode(llr)
            rp, ok = check_and_strip_crc(out.bits)
            assert ok and np.array_equal(rp[: pbits.size], pbits)


def test_polar_full_rate_l_equals_k_all_rates() -> None:
    """L=K (no shortening) round-trips for each rate: 85/128/170.

    ``build_shortened_mask(code, K)`` should reduce to exactly ``code.frozen_mask``
    (the shortening tail is empty at L=K), and ``polar_encode``/``scl_decode``
    should round-trip a K-bit info vector directly at that mask. Since payloads
    are byte-aligned (frame = 8*nbytes + 16 CRC bits), an exact L=K frame is not
    generally byte-aligned, so this exercises ``core.polar`` directly with a
    K-bit info array rather than going through the byte/CRC-framed codec API
    (whose own CRC-aided path selection expects a CRC-framed payload). The
    ``crc_check`` callback is stubbed to always pass, isolating this test to the
    encode/decode/mask correctness at L=K.
    """
    import numpy as np

    from core.coding import CODING_CATALOG
    from core.polar import build_code, build_shortened_mask, polar_encode, scl_decode

    for name, k in (
        ("polar_256_85", 85),
        ("polar_256_128", 128),
        ("polar_256_170", 170),
    ):
        spec = CODING_CATALOG[name]
        assert spec.k == k
        dsnr = float(spec.params["design_snr_db"])  # type: ignore[arg-type]
        code = build_code(spec.n, k, dsnr)
        mask = build_shortened_mask(code, k)
        assert np.array_equal(mask, code.frozen_mask)  # no shortening tail at L=K

        rng = np.random.default_rng(0)
        info = rng.integers(0, 2, size=k).astype(np.uint8)
        cw = polar_encode(info, mask)
        assert cw.size == spec.n
        llr = (1.0 - 2.0 * cw.astype(np.float64)) * 8.0  # clean channel
        decoded, passed = scl_decode(
            llr, mask, int(spec.params["list_size"]), lambda bits: True  # type: ignore[arg-type]
        )
        assert passed
        assert np.array_equal(decoded, info)


def test_polar_loud_fail_on_garbage() -> None:
    import numpy as np

    from core.coding import CODING_CATALOG, check_and_strip_crc, make_codec

    codec = make_codec(CODING_CATALOG["polar_256_128"])
    rng = np.random.default_rng(9)
    llr = rng.standard_normal(256) * 0.1  # near-zero info -> should not pass CRC
    out = codec.decode(llr)
    _, ok = check_and_strip_crc(out.bits)
    assert not ok and out.meta["decode_ok"] is False


def test_polar_scale_invariance() -> None:
    import numpy as np

    from core.coding import CODING_CATALOG, frame_with_crc, make_codec

    codec = make_codec(CODING_CATALOG["polar_256_128"])
    payload = bytes(range(12))
    pbits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    coded = codec.encode(frame_with_crc(pbits.astype(np.uint8)))
    base = (1.0 - 2.0 * coded.astype(np.float64)) + 0.3 * np.random.default_rng(
        1
    ).standard_normal(coded.size)
    ref = codec.decode(base).bits
    for k in (1e-3, 1e-1, 1e1, 1e3):
        np.testing.assert_array_equal(codec.decode(k * base).bits, ref)


def test_make_codec_builds_fountain() -> None:
    from core.coding import CODING_CATALOG, CodeFamily, make_codec

    for spec in CODING_CATALOG.values():
        if spec.family == CodeFamily.FOUNTAIN:
            make_codec(spec)  # must not raise


def test_fountain_roundtrip_and_loud_fail() -> None:
    import numpy as np

    from core.coding import (
        CODING_CATALOG,
        check_and_strip_crc,
        frame_with_crc,
        make_codec,
    )

    for name in ("fountain_r05", "fountain_r10", "fountain_r15"):
        codec = make_codec(CODING_CATALOG[name])
        payload = np.unpackbits(np.frombuffer(bytes(range(24)), dtype=np.uint8))
        frame = frame_with_crc(payload.astype(np.uint8))
        coded = codec.encode(frame)
        out = codec.decode(coded)  # noiseless -> recovers
        rp, ok = check_and_strip_crc(out.bits)
        assert ok and np.array_equal(rp[: payload.size], payload)
        # corrupt beyond capability -> loud fail (never silent wrong payload)
        bad = coded.copy()
        bad[:] ^= 1
        rp2, ok2 = check_and_strip_crc(
            make_codec(CODING_CATALOG[name]).decode(bad).bits
        )
        assert not ok2
