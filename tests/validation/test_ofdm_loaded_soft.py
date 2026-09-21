import numpy as np

from core.ofdm import DEFAULT_OFDM_PROFILE as P
from core.ofdm import (
    demodulate_ofdm_loaded,
    demodulate_ofdm_loaded_soft,
    modulate_ofdm_loaded,
)


def test_soft_loaded_hardslice_matches_hard_demod_noiseless():
    rng = np.random.default_rng(3)
    nd = len(P.data_carriers)
    alloc = rng.choice([2, 4, 6], size=nd).astype(np.intp)  # no nulls: exact bit count
    nbits = int(alloc.sum()) * 4
    payload = rng.integers(0, 2, size=nbits).astype(np.uint8)
    iq = modulate_ofdm_loaded(payload, alloc, P)
    hard = demodulate_ofdm_loaded(iq, P)
    llr = demodulate_ofdm_loaded_soft(iq, P)
    assert llr.shape[0] >= payload.size
    np.testing.assert_array_equal(
        (llr[: payload.size] < 0).astype(np.uint8), hard[: payload.size]
    )
    np.testing.assert_array_equal(hard[: payload.size], payload)  # sanity


def test_soft_loaded_recovers_nulls():
    rng = np.random.default_rng(5)
    nd = len(P.data_carriers)
    alloc = rng.choice([0, 2, 4, 6], size=nd).astype(np.intp)
    nbits = int(alloc.sum()) * 2
    payload = rng.integers(0, 2, size=nbits).astype(np.uint8)
    iq = modulate_ofdm_loaded(payload, alloc, P)
    llr = demodulate_ofdm_loaded_soft(iq, P)
    np.testing.assert_array_equal((llr[: payload.size] < 0).astype(np.uint8), payload)


def test_soft_loaded_empty_on_short_input():
    assert demodulate_ofdm_loaded_soft(np.zeros(4, dtype=np.complex128), P).size == 0
