import numpy as np


def test_pack_unpack_allocation() -> None:
    from core.ofdm import DEFAULT_OFDM_PROFILE, pack_allocation, unpack_allocation

    nd = len(DEFAULT_OFDM_PROFILE.data_carriers)  # 48
    rng = np.random.default_rng(0)
    alloc = rng.choice([0, 2, 4, 6], size=nd).astype(np.intp)
    bits = pack_allocation(alloc)
    assert bits.size == 2 * nd  # 96 bits = one QPSK symbol
    np.testing.assert_array_equal(unpack_allocation(bits), alloc)


def test_modulate_loaded_structure() -> None:
    from core.ofdm import DEFAULT_OFDM_PROFILE as P
    from core.ofdm import modulate_ofdm_loaded

    nd = len(P.data_carriers)
    alloc = np.full(nd, 2, dtype=np.intp)  # all QPSK
    payload = np.random.default_rng(1).integers(0, 2, size=2 * nd * 3).astype(np.uint8)
    iq = modulate_ofdm_loaded(payload, alloc, P)
    # STF + LTF + header(1) + data symbols; length multiple of symbol_len
    assert iq.dtype == np.complex128 and iq.size % P.symbol_len == 0
    # at least preamble(2) + header(1) + ceil(payload_bits / sum(alloc)) data symbols
    assert iq.size // P.symbol_len >= 2 + 1 + 3
