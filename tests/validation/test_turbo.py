import numpy as np

from core.turbo import NXT, deinterleave_qpp, interleave_qpp, qpp_perm, rsc_encode


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
