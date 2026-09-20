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
        n = 2 ** m
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
        u[n - s:] = 0
        x = polar_transform(u)
        assert np.all(x[n - s:] == 0)
