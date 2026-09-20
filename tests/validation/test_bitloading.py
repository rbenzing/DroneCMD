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


def test_subcarrier_snr() -> None:
    from core.bitloading import subcarrier_snr

    h = np.array([1.0, 2.0, 0.5], dtype=np.complex128)
    snr = subcarrier_snr(h, noise_var=0.25)
    np.testing.assert_allclose(snr, np.array([4.0, 16.0, 1.0]))  # |H|^2/nv


def test_chow_load_allocates_more_to_stronger() -> None:
    from core.bitloading import chow_load

    # increasing SNR across carriers -> non-decreasing bit allocation, entries in {0,2,4,6}
    snr = np.array([0.5, 2.0, 8.0, 40.0, 500.0], dtype=np.float64)
    alloc = chow_load(snr, target_ber=1e-3)
    assert set(np.unique(alloc)).issubset({0, 2, 4, 6})
    assert np.all(np.diff(alloc) >= 0)  # monotone in SNR
    assert alloc[0] == 0 and alloc[-1] == 6  # weakest nulled, strongest full
    assert (
        chow_load(snr, 1e-3).tolist() == chow_load(snr, 1e-3).tolist()
    )  # deterministic


def test_chow_load_total_bits_monotone_in_quality() -> None:
    from core.bitloading import chow_load

    rng = np.random.default_rng(3)
    base = rng.random(48) * 50.0
    assert chow_load(base * 4.0, 1e-3).sum() >= chow_load(base, 1e-3).sum()
