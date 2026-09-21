import numpy as np
import pytest

from core.bitloading import qam_demap, qam_map, qam_soft_demap


@pytest.mark.parametrize("order", [2, 4, 6])
def test_soft_hard_consistency_noiseless(order):
    rng = np.random.default_rng(order)
    bits = rng.integers(0, 2, size=order * 200).astype(np.uint8)
    syms = qam_map(bits, order)
    llr = qam_soft_demap(syms, order, weight=1.0)
    assert llr.shape == (order * 200,)
    np.testing.assert_array_equal((llr < 0).astype(np.uint8), qam_demap(syms, order))
    # noiseless: the hard slice must also recover the transmitted bits exactly
    np.testing.assert_array_equal((llr < 0).astype(np.uint8), bits)


@pytest.mark.parametrize("order", [2, 4, 6])
def test_soft_sign_and_weight_scaling(order):
    bits = np.zeros(order, dtype=np.uint8)  # all-zero group -> most-positive rails
    sym = qam_map(bits, order)
    l1 = qam_soft_demap(sym, order, weight=1.0)
    l3 = qam_soft_demap(sym, order, weight=3.0)
    assert np.all(l1 > 0)  # every bit is 0 -> all LLR>0
    np.testing.assert_allclose(l3, 3.0 * l1, rtol=1e-9)  # weight scales linearly


def test_per_symbol_weight_broadcast():
    order = 4
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=order * 3).astype(np.uint8)
    syms = qam_map(bits, order)
    w = np.array([1.0, 2.0, 5.0])
    llr = qam_soft_demap(syms, order, weight=w)
    base = qam_soft_demap(syms, order, weight=1.0).reshape(3, order)
    np.testing.assert_allclose(llr.reshape(3, order), base * w[:, None], rtol=1e-9)


def test_bad_order_raises():
    with pytest.raises(ValueError):
        qam_soft_demap(np.zeros(4, dtype=np.complex128), 3)
