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
