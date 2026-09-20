import numpy as np


def test_qam_order2_equals_qpsk() -> None:
    from core.bitloading import qam_demap, qam_map
    from core.ofdm import qpsk_demap, qpsk_map

    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=48).astype(np.uint8)  # even length
    np.testing.assert_allclose(qam_map(bits, 2), qpsk_map(bits))
    np.testing.assert_array_equal(
        qam_demap(qpsk_map(bits), 2), qpsk_demap(qpsk_map(bits))
    )


def test_qam_roundtrip_and_unit_energy() -> None:
    from core.bitloading import qam_demap, qam_map

    rng = np.random.default_rng(1)
    for order in (2, 4, 6):
        bits = rng.integers(0, 2, size=order * 200).astype(np.uint8)
        syms = qam_map(bits, order)
        assert syms.size == bits.size // order
        np.testing.assert_array_equal(
            qam_demap(syms, order), bits
        )  # noiseless round-trip
        assert (
            abs(float(np.mean(np.abs(syms) ** 2)) - 1.0) < 0.05
        )  # unit average energy


def test_qam_gray_single_bit_neighbor() -> None:
    # A nearest-neighbor symbol perturbation flips at most 1 bit per rail (Gray).
    from core.bitloading import qam_demap, qam_map

    order = 4
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, size=order * 500).astype(np.uint8)
    syms = qam_map(bits, order)
    step = 2.0 / np.sqrt((2.0 / 3.0) * (2**order - 1))  # one PAM step, normalized
    noisy = syms + 0.2 * step  # small real nudge toward the I neighbor
    out = qam_demap(noisy, order)
    per_sym_errs = (out.reshape(-1, order) != bits.reshape(-1, order)).sum(axis=1)
    assert per_sym_errs.max() <= 1
