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


def test_build_precode_shape_and_determinism() -> None:
    from core.fountain import build_precode

    p1 = build_precode(40, 12345, 0.9, 4)  # rate 0.9 -> R = round(40*(1/0.9-1)) = 4
    p2 = build_precode(40, 12345, 0.9, 4)
    assert len(p1) == len(p2) == 4
    for a, b in zip(p1, p2):
        assert np.array_equal(a, b)
    for row in p1:
        assert row.size >= 1 and row.max() < 40 and len(set(row.tolist())) == row.size


def test_build_precode_rate_one_is_empty() -> None:
    from core.fountain import build_precode

    assert build_precode(40, 1, 1.0, 4) == []  # rate 1.0 -> no parity


def test_fountain_encode_structure() -> None:
    from core.fountain import crc8, fountain_encode

    rng = np.random.default_rng(3)
    S, eps = 16, 1.0
    payload = rng.integers(0, 2, size=200).astype(np.uint8)  # arbitrary length
    K = int(np.ceil(payload.size / S))
    N = int(np.ceil((1 + eps) * K))
    coded = fountain_encode(
        payload,
        symbol_bits=S,
        seed=42,
        c=0.03,
        delta=0.5,
        precode_rate=0.9,
        precode_degree=4,
        overhead=eps,
    )
    assert coded.size == N * (S + 8)  # N (S + crc8) symbols
    # each encoded symbol's per-symbol CRC is valid on a clean encode
    sym = coded.reshape(N, S + 8)
    for row in sym:
        assert np.array_equal(crc8(row[:S]), row[S:])
    # deterministic
    coded2 = fountain_encode(
        payload,
        symbol_bits=S,
        seed=42,
        c=0.03,
        delta=0.5,
        precode_rate=0.9,
        precode_degree=4,
        overhead=eps,
    )
    assert np.array_equal(coded, coded2)
