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
