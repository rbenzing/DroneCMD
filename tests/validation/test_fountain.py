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


def test_fountain_noiseless_roundtrip_all_survive() -> None:
    from core.fountain import fountain_decode, fountain_encode

    rng = np.random.default_rng(5)
    S, eps = 16, 1.0
    payload = rng.integers(0, 2, size=200).astype(np.uint8)
    kw = dict(
        symbol_bits=S,
        seed=42,
        c=0.03,
        delta=0.5,
        precode_rate=0.9,
        precode_degree=4,
        overhead=eps,
    )
    coded = fountain_encode(payload, **kw)
    rec, ok, n_erased = fountain_decode(coded, **kw)
    assert ok and n_erased == 0
    assert np.array_equal(rec[: payload.size], payload)


def test_fountain_recovers_with_erasures() -> None:
    from core.fountain import fountain_decode, fountain_encode

    rng = np.random.default_rng(6)
    S, eps = 16, 1.0
    payload = rng.integers(0, 2, size=320).astype(np.uint8)  # K=20, N=40
    kw = dict(
        symbol_bits=S,
        seed=42,
        c=0.03,
        delta=0.5,
        precode_rate=0.9,
        precode_degree=4,
        overhead=eps,
    )
    coded = fountain_encode(payload, **kw).reshape(-1, S + 8)
    # erase ~25% of symbols by corrupting their data so per-symbol CRC fails
    n = coded.shape[0]
    for i in rng.choice(n, size=n // 4, replace=False):
        coded[i, 0] ^= 1  # flips data, CRC no longer matches -> erased
    rec, ok, n_erased = fountain_decode(coded.reshape(-1), **kw)
    assert ok and n_erased == n // 4
    assert np.array_equal(rec[: payload.size], payload)


def test_fountain_loud_fail_too_many_erasures() -> None:
    from core.fountain import fountain_decode, fountain_encode

    rng = np.random.default_rng(9)
    S, eps = 16, 1.0
    payload = rng.integers(0, 2, size=320).astype(np.uint8)
    kw = dict(
        symbol_bits=S,
        seed=42,
        c=0.03,
        delta=0.5,
        precode_rate=0.9,
        precode_degree=4,
        overhead=eps,
    )
    coded = fountain_encode(payload, **kw).reshape(-1, S + 8)
    coded[:, 0] ^= 1  # erase (almost) everything
    rec, ok, _ = fountain_decode(coded.reshape(-1), **kw)
    assert not ok  # loud fail, no silent wrong payload


def test_fountain_decode_non_aligned_length_graceful() -> None:
    """Regression test: fountain_decode guards against non-symbol-aligned input."""
    from core.fountain import fountain_decode

    S = 16
    kw = dict(
        symbol_bits=S,
        seed=42,
        c=0.03,
        delta=0.5,
        precode_rate=0.9,
        precode_degree=4,
        overhead=1.0,
    )
    # S + 8 = 24; feed a length that is NOT a multiple of 24
    misaligned_length = 25  # 25 % 24 != 0
    bad_coded = np.zeros(misaligned_length, dtype=np.uint8)
    rec, ok, n_erased = fountain_decode(bad_coded, **kw)
    assert ok is False
    assert rec.size == 0
    assert n_erased == 0

    # Also test with a larger non-multiple length
    bad_coded2 = np.zeros(100, dtype=np.uint8)  # 100 % 24 != 0
    rec2, ok2, n_erased2 = fountain_decode(bad_coded2, **kw)
    assert ok2 is False
    assert rec2.size == 0
    assert n_erased2 == 0


def test_fountain_codec_path_non_aligned_graceful() -> None:
    """Regression test: codec path gracefully handles non-aligned input."""
    from core.coding import CODING_CATALOG, make_codec

    codec = make_codec(CODING_CATALOG["fountain_r10"])
    misaligned_input = np.zeros(25, dtype=np.uint8)  # 25 % 24 != 0
    result = codec.decode(misaligned_input)
    assert result.meta["decode_ok"] is False
    assert result.bits.size == 0
