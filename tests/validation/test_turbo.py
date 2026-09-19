import numpy as np

from core.turbo import (
    NXT,
    deinterleave_qpp,
    interleave_qpp,
    qpp_perm,
    rsc_encode,
    turbo_decode,
    turbo_encode,
)


def test_rsc_systematic_and_returns_to_zero() -> None:
    rng = np.random.default_rng(0)
    info = rng.integers(0, 2, size=40).astype(np.uint8)
    sys, par = rsc_encode(info)
    assert sys.size == info.size + 3 and par.size == info.size + 3
    assert np.array_equal(sys[: info.size], info)  # systematic
    # replay through the trellis: final state must be 0 (tail terminated)
    state = 0
    for u in sys:
        state = NXT[state][int(u)]
    assert state == 0


def test_qpp_is_a_bijection() -> None:
    K, f1, f2 = 256, 31, 64
    p = qpp_perm(K, f1, f2)
    assert sorted(p.tolist()) == list(range(K))
    # f1 odd, f2 even (validity for K = power of two)
    assert f1 % 2 == 1 and f2 % 2 == 0
    x = np.arange(K)
    assert np.array_equal(deinterleave_qpp(interleave_qpp(x, p), p), x)


def _valid_small_perm(K: int) -> np.ndarray:
    # any K: search small (f1 odd, f2 even) that yields a permutation
    for f2 in range(2, K, 2):
        for f1 in range(1, K, 2):
            i = np.arange(K)
            p = (f1 * i + f2 * i * i) % K
            if sorted(p.tolist()) == list(range(K)):
                return p.astype(np.intp)
    raise AssertionError("no perm")


def test_turbo_encode_layout_and_recover_systematic() -> None:
    K = 32
    info = (np.arange(K) % 2).astype(np.uint8)
    # use a valid small perm for the test
    perm = _valid_small_perm(K)
    cw = turbo_encode(info, perm)
    assert cw.size == 3 * K + 12
    assert np.array_equal(cw[:K], info)  # systematic prefix
    # parity1 block equals rsc_encode(info) parity
    _, par1 = rsc_encode(info)
    assert np.array_equal(cw[K + 6 : K + 6 + (K + 3)], par1)


def _split(cw: np.ndarray, K: int) -> tuple:
    info = cw[:K]
    t1s = cw[K : K + 3]
    t2s = cw[K + 3 : K + 6]
    par1 = cw[K + 6 : K + 6 + (K + 3)]
    par2 = cw[K + 6 + (K + 3) :]
    return info, t1s, t2s, par1, par2


def _perm256() -> np.ndarray:
    return qpp_perm(256, 31, 64)


def test_turbo_noiseless_roundtrip_rate13() -> None:
    K = 256
    perm = _perm256()
    rng = np.random.default_rng(1)
    info = rng.integers(0, 2, size=K).astype(np.uint8)
    cw = turbo_encode(info, perm)
    llr = np.where(cw == 0, 6.0, -6.0).astype(np.float64)
    i, t1, t2, p1, p2 = _split(llr, K)
    out = turbo_decode(i, t1, t2, p1, p2, perm)
    assert np.array_equal(out, info)


def test_turbo_corrects_errors() -> None:
    K = 256
    perm = _perm256()
    rng = np.random.default_rng(2)
    info = rng.integers(0, 2, size=K).astype(np.uint8)
    cw = turbo_encode(info, perm)
    llr = np.where(cw == 0, 3.0, -3.0).astype(np.float64)
    flip = rng.choice(llr.size, size=llr.size // 12, replace=False)  # ~8% flips
    llr[flip] = -llr[flip]
    i, t1, t2, p1, p2 = _split(llr, K)
    out = turbo_decode(i, t1, t2, p1, p2, perm)
    assert np.array_equal(out, info)


def test_turbo_scale_invariance() -> None:
    K = 256
    perm = _perm256()
    rng = np.random.default_rng(3)
    info = rng.integers(0, 2, size=K).astype(np.uint8)
    cw = turbo_encode(info, perm)
    llr = np.where(cw == 0, 2.0, -2.0).astype(np.float64)
    flip = rng.choice(llr.size, size=20, replace=False)
    llr[flip] = -llr[flip]
    i, t1, t2, p1, p2 = _split(llr, K)
    base = turbo_decode(i, t1, t2, p1, p2, perm)
    for k in (1e-3, 5.0, 1e3):
        out = turbo_decode(k * i, k * t1, k * t2, k * p1, k * p2, perm)
        assert np.array_equal(out, base)
