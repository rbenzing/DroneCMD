import numpy as np


def test_crc8_detects_single_bit_flip() -> None:
    from core.fountain import crc8

    rng = np.random.default_rng(0)
    data = rng.integers(0, 2, size=32).astype(np.uint8)
    c = crc8(data)
    assert c.shape == (8,)
    bad = data.copy()
    bad[5] ^= 1
    assert not np.array_equal(crc8(bad), c)  # single-bit error changes CRC


def test_robust_soliton_is_pmf() -> None:
    from core.fountain import robust_soliton

    pmf = robust_soliton(40, 0.03, 0.5)
    assert pmf.shape == (41,)  # index 0 unused
    assert pmf[0] == 0.0
    assert abs(pmf.sum() - 1.0) < 1e-9
    assert np.all(pmf >= 0.0)


def test_sample_degree_and_neighbors_deterministic() -> None:
    from core.fountain import robust_soliton, sample_degree, symbol_neighbors

    pmf = robust_soliton(40, 0.03, 0.5)
    d1 = sample_degree(pmf, np.random.default_rng(7))
    d2 = sample_degree(pmf, np.random.default_rng(7))
    assert d1 == d2 and 1 <= d1 <= 40
    nb1 = symbol_neighbors(np.random.default_rng(7), 5, 40)
    nb2 = symbol_neighbors(np.random.default_rng(7), 5, 40)
    assert np.array_equal(nb1, nb2)
    assert nb1.size == 5 and len(set(nb1.tolist())) == 5 and nb1.max() < 40
