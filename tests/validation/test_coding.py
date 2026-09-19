import numpy as np
import pytest


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
    from core.coding import CODING_CATALOG, CodeFamily, make_codec

    for name, spec in CODING_CATALOG.items():
        if spec.family in (
            CodeFamily.UNCODED,
            CodeFamily.REPETITION,
            CodeFamily.CONVOLUTIONAL,
        ):
            make_codec(spec)  # builds
        else:
            with pytest.raises(NotImplementedError):
                make_codec(spec)


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
