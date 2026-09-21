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


def _oracle_soft(sym, order, weight):
    """Independent unit-energy max-log LLR over the full 2-D constellation."""
    labels = np.array(
        [list(map(int, format(v, f"0{order}b"))) for v in range(1 << order)],
        dtype=np.uint8,
    )
    pts = np.array([qam_map(labels[v], order)[0] for v in range(1 << order)])
    d = np.abs(sym - pts) ** 2  # unit-average-energy distances
    out = np.empty(order, dtype=np.float64)
    for j in range(order):
        b1 = labels[:, j] == 1
        out[j] = weight * (d[b1].min() - d[~b1].min())
    return out


@pytest.mark.parametrize("order", [2, 4, 6])
def test_soft_matches_unit_energy_oracle_cross_order(order):
    # Pins the *relative* per-order weighting: the LLR must be weight x
    # unit-energy max-log distances, with NO order-dependent norm^2 factor. A
    # regression here (e.g. 5x/21x inflation for 16/64-QAM) breaks BICM.
    rng = np.random.default_rng(7)
    for _ in range(50):
        sym = (rng.standard_normal() + 1j * rng.standard_normal()) * 0.5
        w = float(rng.uniform(0.3, 5.0))
        got = qam_soft_demap(np.array([sym], dtype=np.complex128), order, w)
        exp = _oracle_soft(sym, order, w)
        np.testing.assert_allclose(got, exp, rtol=1e-9, atol=1e-9)


def test_bad_order_raises():
    with pytest.raises(ValueError):
        qam_soft_demap(np.zeros(4, dtype=np.complex128), 3)
