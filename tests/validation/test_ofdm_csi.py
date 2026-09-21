import numpy as np

from core.ofdm import DEFAULT_OFDM_PROFILE as P
from core.ofdm import (
    modulate_ofdm,
    ofdm_equalized_symbols,
    ofdm_equalized_symbols_csi,
)


def test_csi_matches_symbols_and_flat_channel():
    bits = (
        np.random.default_rng(0)
        .integers(0, 2, size=P.n_data_bits_per_symbol * 5)
        .astype(np.uint8)
    )
    iq = modulate_ofdm(bits, P).astype(np.complex128)
    syms_ref = ofdm_equalized_symbols(iq, P)
    syms, gain_sq, n0 = ofdm_equalized_symbols_csi(iq, P)
    np.testing.assert_allclose(syms, syms_ref, atol=1e-9)  # same equalized symbols
    assert gain_sq.shape == syms.shape and np.all(gain_sq > 0)
    np.testing.assert_allclose(gain_sq, 1.0, atol=0.05)  # flat channel -> |h|^2 ~ 1
    assert np.isfinite(n0) and n0 >= 0.0  # tiny (noiseless)


def test_csi_empty_on_short_input():
    syms, gain_sq, n0 = ofdm_equalized_symbols_csi(np.zeros(4, dtype=np.complex128), P)
    assert syms.size == 0 and gain_sq.size == 0 and np.isnan(n0)
