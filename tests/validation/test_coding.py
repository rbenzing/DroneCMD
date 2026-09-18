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
        if spec.family in (CodeFamily.UNCODED, CodeFamily.REPETITION):
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
