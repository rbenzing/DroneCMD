import numpy as np


def _dense_gn(m: int) -> np.ndarray:
    F = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    G = np.array([[1]], dtype=np.uint8)
    for _ in range(m):
        G = np.kron(G, F)  # F^{⊗m}, non-bit-reversed
    return G % 2


def test_polar_transform_matches_dense_gn() -> None:
    from core.polar import polar_transform

    rng = np.random.default_rng(0)
    for m in (1, 2, 3, 8):
        n = 2**m
        G = _dense_gn(m)
        for _ in range(5):
            u = rng.integers(0, 2, size=n).astype(np.uint8)
            expected = (u @ G) % 2
            np.testing.assert_array_equal(polar_transform(u), expected.astype(np.uint8))


def test_polar_transform_lower_triangular_tail() -> None:
    # Freezing the top s inputs zeros the last s codeword bits (shorten-from-end).
    from core.polar import polar_transform

    n = 256
    rng = np.random.default_rng(1)
    for s in (1, 8, 40):
        u = rng.integers(0, 2, size=n).astype(np.uint8)
        u[n - s :] = 0
        x = polar_transform(u)
        assert np.all(x[n - s :] == 0)


def test_ga_frozen_set_deterministic_and_sized() -> None:
    from core.polar import build_code

    c1 = build_code(256, 128, 2.0)
    c2 = build_code(256, 128, 2.0)
    np.testing.assert_array_equal(c1.frozen_mask, c2.frozen_mask)
    assert c1.info_positions.size == 128
    assert int(c1.frozen_mask.sum()) == 128
    # info positions are exactly the unfrozen ones
    assert np.array_equal(np.where(~c1.frozen_mask)[0], c1.info_positions)


def test_ga_monotone_with_rate() -> None:
    # Lower rate (fewer info) must be a subset of higher-rate info positions
    # (nested/monotone reliability order).
    from core.polar import build_code

    lo = build_code(256, 85, 2.0)
    hi = build_code(256, 170, 2.0)
    assert set(lo.info_positions.tolist()) <= set(hi.info_positions.tolist())


def test_build_shortened_mask_freezes_top_s() -> None:
    from core.polar import build_code, build_shortened_mask

    code = build_code(256, 128, 2.0)
    mask = build_shortened_mask(code, info_len=100)  # s = 28
    assert np.all(mask[256 - 28 :])  # top s force-frozen
    assert int((~mask).sum()) == 100  # exactly info_len info positions


def test_build_shortened_mask_at_full_k_matches_nominal_frozen_mask() -> None:
    # Invariant relied on by full-rate encode/decode: shortening to info_len=k
    # (s=0, no forced tail) must reproduce the nominal frozen mask exactly.
    from core.polar import build_code, build_shortened_mask

    code = build_code(256, 128, 2.0)
    mask = build_shortened_mask(code, info_len=code.k)
    np.testing.assert_array_equal(mask, code.frozen_mask)


def test_polar_encode_scatters_and_transforms() -> None:
    from core.polar import build_code, polar_encode, polar_transform

    code = build_code(256, 128, 2.0)
    rng = np.random.default_rng(2)
    info = rng.integers(0, 2, size=128).astype(np.uint8)
    x = polar_encode(info, code.frozen_mask)
    # reconstruct u and compare to a direct transform
    u = np.zeros(256, dtype=np.uint8)
    u[code.info_positions] = info
    np.testing.assert_array_equal(x, polar_transform(u))
    assert x.size == 256


def test_polar_encode_all_frozen_is_zero() -> None:
    from core.polar import polar_encode

    all_frozen = np.ones(256, dtype=np.bool_)
    x = polar_encode(np.zeros(0, dtype=np.uint8), all_frozen)
    assert np.all(x == 0)


def test_scl_noiseless_roundtrip() -> None:
    from core.polar import build_code, polar_encode, scl_decode

    code = build_code(256, 128, 2.0)
    rng = np.random.default_rng(3)
    info = rng.integers(0, 2, size=128).astype(np.uint8)
    x = polar_encode(info, code.frozen_mask)
    llr = (1.0 - 2.0 * x.astype(np.float64)) * 8.0  # L>0 => bit 0
    out, _ = scl_decode(llr, code.frozen_mask, 8, crc_check=lambda b: True)
    np.testing.assert_array_equal(out, info)


def test_scl_corrects_a_few_errors() -> None:
    from core.polar import build_code, polar_encode, scl_decode

    code = build_code(256, 128, 2.0)
    rng = np.random.default_rng(4)
    info = rng.integers(0, 2, size=128).astype(np.uint8)
    x = polar_encode(info, code.frozen_mask)
    llr = (1.0 - 2.0 * x.astype(np.float64)) * 4.0
    llr[:6] *= -0.5  # weaken/flip a few
    out, _ = scl_decode(llr, code.frozen_mask, 8, crc_check=lambda b: True)
    assert int(np.sum(out != info)) <= 4  # list decoding recovers most/all


def test_scl_crc_aided_selection() -> None:
    # The CRC callback selects a valid path even if the ML path is wrong.
    from core.polar import build_code, polar_encode, scl_decode

    code = build_code(256, 128, 2.0)
    rng = np.random.default_rng(5)
    info = rng.integers(0, 2, size=128).astype(np.uint8)
    x = polar_encode(info, code.frozen_mask)
    llr = (1.0 - 2.0 * x.astype(np.float64)) * 3.0
    llr[10:18] *= -1.0  # inject a burst that misleads the ML path
    good = info.tobytes()
    out, passed = scl_decode(
        llr, code.frozen_mask, 8, crc_check=lambda b: b.tobytes() == good
    )
    assert passed and np.array_equal(out, info)


def test_scl_decode_is_scale_invariant() -> None:
    # Every metric (f-node, g-node, path metric) is linearly homogeneous in a
    # common LLR scale, so decoding must be invariant to a positive rescale.
    from core.polar import build_code, polar_encode, scl_decode

    code = build_code(256, 128, 2.0)
    rng = np.random.default_rng(6)
    info = rng.integers(0, 2, size=128).astype(np.uint8)
    x = polar_encode(info, code.frozen_mask)
    llr = (1.0 - 2.0 * x.astype(np.float64)) * 4.0
    llr[:6] *= -0.5  # weaken/flip a few, same perturbation as the errors test

    out1, passed1 = scl_decode(llr, code.frozen_mask, 8, crc_check=lambda b: True)
    out2, passed2 = scl_decode(
        llr * 17.0, code.frozen_mask, 8, crc_check=lambda b: True
    )
    np.testing.assert_array_equal(out1, out2)
    assert passed1 == passed2
