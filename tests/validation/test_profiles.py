from __future__ import annotations

from core.profiles import (
    DEFAULT_SC_PROFILE_NAME,
    OFDM_CATALOG,
    SC_CATALOG,
    Family,
    SCMod,
    family_of,
)
from core.single_carrier import DEFAULT_SC_PROFILE


def test_catalog_has_expected_entries() -> None:
    assert set(SC_CATALOG) == {
        "sik_gfsk",
        "ble_1m",
        "ble_2m",
        "fsk_basic",
        "psk_c2",
        "qpsk_link",
        "rep_bpsk",
        "conv_bpsk",
        "rs_bpsk",
        "bch_bpsk",
    }
    assert "wifi_20" in OFDM_CATALOG


def test_entries_are_phy_distinct() -> None:
    # No two SC entries share (mod, sps, mod_index, bt) -- blind resolution
    # needs each to be distinguishable.
    sigs = set()
    for spec in SC_CATALOG.values():
        sig = (spec.mod, spec.profile.sps, spec.profile.mod_index, spec.profile.bt)
        assert sig not in sigs, f"duplicate PHY signature for {spec.name}"
        sigs.add(sig)


def test_default_matches_framework_default() -> None:
    # sik_gfsk is the current GFSK default (DEFAULT_SC_PROFILE params).
    spec = SC_CATALOG[DEFAULT_SC_PROFILE_NAME]
    assert spec.mod == SCMod.GFSK
    assert spec.profile == DEFAULT_SC_PROFILE  # sps=8, mod_index=0.7, bt=0.5


def test_ble_2m_uses_non_default_sps() -> None:
    assert SC_CATALOG["ble_2m"].profile.sps == 4  # exercises the old sps==8 guard


def test_bits_per_symbol_and_family() -> None:
    assert SC_CATALOG["qpsk_link"].bits_per_symbol == 2
    assert SC_CATALOG["psk_c2"].bits_per_symbol == 1
    assert SC_CATALOG["sik_gfsk"].is_fsk and SC_CATALOG["sik_gfsk"].gfsk
    assert not SC_CATALOG["psk_c2"].is_fsk
    assert family_of("sik_gfsk") == Family.SINGLE_CARRIER
    assert family_of("wifi_20") == Family.OFDM


def test_spec_is_frozen() -> None:
    import pytest

    spec = SC_CATALOG["psk_c2"]
    with pytest.raises(Exception):
        spec.name = "x"  # type: ignore[misc]


def test_ofdm_catalog_has_broader_profiles() -> None:
    from core.ofdm import OFDMProfile
    from core.profiles import OFDM_CATALOG, Family, all_profile_names, family_of

    expected = {"wifi_20", "wifi_40", "ofdm_nb", "wifi_20_longcp"}
    assert expected <= set(OFDM_CATALOG)
    for name in expected:
        assert isinstance(OFDM_CATALOG[name], OFDMProfile)
        assert family_of(name) == Family.OFDM
        assert name in all_profile_names()


def test_ofdm_catalog_distinct_shapes() -> None:
    from core.profiles import OFDM_CATALOG

    c = OFDM_CATALOG
    assert (c["wifi_20"].fft_size, c["wifi_40"].fft_size, c["ofdm_nb"].fft_size) == (
        64,
        128,
        32,
    )
    # same-N CP variant differs only in cp_len
    assert c["wifi_20_longcp"].fft_size == 64 and c["wifi_20_longcp"].cp_len == 32
    assert c["wifi_20"].cp_len == 16


def test_ofdm_catalog_profiles_roundtrip() -> None:
    import numpy as np

    from core.ofdm import demodulate_ofdm, modulate_ofdm
    from core.profiles import OFDM_CATALOG

    b = np.array([1, 0, 0, 1, 1, 1, 0, 0] * 12, dtype=np.uint8)
    for name, profile in OFDM_CATALOG.items():
        rx = modulate_ofdm(b, profile)
        out = demodulate_ofdm(rx, profile)
        assert np.array_equal(out[: len(b)], b), f"round-trip failed for {name}"


def test_coding_of_and_coded_profile() -> None:
    from core.coding import CODING_CATALOG
    from core.profiles import SC_CATALOG, coding_of

    assert coding_of("sik_gfsk") is None  # existing profiles uncoded
    assert coding_of("rep_bpsk") == "rep3"
    assert SC_CATALOG["rep_bpsk"].coding in CODING_CATALOG
    assert SC_CATALOG["rep_bpsk"].profile.sps == 16  # PHY-distinct


def test_rep_bpsk_blind_resolves() -> None:
    import numpy as np

    from core.blind import resolve_sc_profile
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    iq = modulate(bytes(range(24)), ModScheme.BPSK, sps=16).astype(np.complex128)
    spec, _ = resolve_sc_profile(iq)
    assert spec is not None and spec.name == "rep_bpsk"


def test_conv_bpsk_profile() -> None:
    from core.coding import CODING_CATALOG
    from core.profiles import SC_CATALOG, coding_of

    assert coding_of("conv_bpsk") == "conv_k7_r12"
    assert SC_CATALOG["conv_bpsk"].coding in CODING_CATALOG
    assert SC_CATALOG["conv_bpsk"].profile.sps == 32


def test_conv_bpsk_blind_resolves() -> None:
    import numpy as np

    from core.blind import resolve_sc_profile
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    iq = modulate(bytes(range(24)), ModScheme.BPSK, sps=32).astype(np.complex128)
    spec, _ = resolve_sc_profile(iq)
    assert spec is not None and spec.name == "conv_bpsk"


def test_rs_bpsk_profile() -> None:
    from core.coding import CODING_CATALOG
    from core.profiles import SC_CATALOG, coding_of

    assert coding_of("rs_bpsk") == "rs_255_239"
    assert SC_CATALOG["rs_bpsk"].coding in CODING_CATALOG
    assert SC_CATALOG["rs_bpsk"].profile.sps == 64
    # sps=64 must be unique across SC_CATALOG (blind resolution key)
    sps_vals = [p.profile.sps for p in SC_CATALOG.values() if p.profile.sps == 64]
    assert sps_vals == [64]


def test_rs_bpsk_blind_resolves() -> None:
    import numpy as np

    from core.blind import resolve_sc_profile
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    iq = modulate(bytes(range(24)), ModScheme.BPSK, sps=64).astype(np.complex128)
    spec, _ = resolve_sc_profile(iq)
    assert spec is not None and spec.name == "rs_bpsk"


def test_bch_bpsk_profile() -> None:
    from core.coding import CODING_CATALOG
    from core.profiles import SC_CATALOG, coding_of

    assert coding_of("bch_bpsk") == "bch_255_223"
    assert SC_CATALOG["bch_bpsk"].coding in CODING_CATALOG
    assert SC_CATALOG["bch_bpsk"].profile.sps == 128
    # sps=128 must be unique in SC_CATALOG (blind resolution key)
    assert [p.profile.sps for p in SC_CATALOG.values() if p.profile.sps == 128] == [128]
