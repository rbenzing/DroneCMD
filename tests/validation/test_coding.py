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
