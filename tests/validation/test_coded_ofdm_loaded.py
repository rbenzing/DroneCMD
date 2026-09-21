import numpy as np

from core.bitloading import chow_load, subcarrier_snr
from core.coding import CODING_CATALOG
from core.ofdm import DEFAULT_OFDM_PROFILE as P
from core.ofdm import (
    data_channel_response,
    decode_coded_ofdm_loaded,
    modulate_coded_ofdm_loaded,
)


def _alloc(taps, snr_db):
    h = data_channel_response(taps, P)
    n0 = 10 ** (-snr_db / 10.0)
    return chow_load(subcarrier_snr(h, n0), target_ber=1e-3)


def test_coded_loaded_roundtrip_noiseless():
    taps = (1.0 + 0j, 0.5 + 0j, -0.3 + 0j)
    alloc = _alloc(taps, 25.0)
    data = bytes(range(24))
    for name in ("conv_k7_r12", "ldpc_648_r12"):
        spec = CODING_CATALOG[name]
        iq = modulate_coded_ofdm_loaded(data, alloc, spec, P)
        payload, ok = decode_coded_ofdm_loaded(iq, spec, P)
        assert ok, name
        assert payload[: len(data)] == data, name


def test_coded_loaded_hard_only_codec_roundtrip():
    # A hard-input codec (rep3) must also round-trip via the hard demod path.
    taps = (1.0 + 0j, 0.4 + 0j)
    alloc = _alloc(taps, 25.0)
    data = bytes(range(12))
    spec = CODING_CATALOG["rep3"]
    iq = modulate_coded_ofdm_loaded(data, alloc, spec, P)
    payload, ok = decode_coded_ofdm_loaded(iq, spec, P)
    assert ok and payload[: len(data)] == data


def test_coded_loaded_empty_on_garbage():
    spec = CODING_CATALOG["conv_k7_r12"]
    payload, ok = decode_coded_ofdm_loaded(np.zeros(4, dtype=np.complex128), spec, P)
    assert payload == b"" and ok is False
