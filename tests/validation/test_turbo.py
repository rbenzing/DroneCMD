import numpy as np

from core.turbo import (
    NXT,
    deinterleave_qpp,
    interleave_qpp,
    qpp_perm,
    rsc_encode,
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
