"""Named PHY profile registry for DroneCMD (single source of truth).

A profile bundles a modulation with its single-carrier PHY parameters
(:class:`core.single_carrier.SCProfile`) under a stable name, so datasets,
the blind resolver, and the CLI all refer to the same catalog. Split by
:class:`Family`; the OFDM catalog holds only the fixed ``wifi_20`` profile
for now (parametric OFDM profiles are a later phase). This module depends
only on :mod:`core.single_carrier` and :mod:`core.ofdm` -- no heavy imports.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from core.ofdm import DEFAULT_OFDM_PROFILE
from core.single_carrier import SCProfile


class SCMod(Enum):
    """Single-carrier modulation of a profile (mirrors the SC members of
    :class:`validation.types.ModScheme`; kept local so ``core`` does not
    depend on ``validation``)."""

    FSK = "fsk"
    GFSK = "gfsk"
    BPSK = "bpsk"
    QPSK = "qpsk"


class Family(Enum):
    """Waveform family a profile belongs to."""

    SINGLE_CARRIER = "single_carrier"
    OFDM = "ofdm"


@dataclass(frozen=True)
class SCProfileSpec:
    """A named single-carrier profile: a modulation plus its PHY params.

    Attributes:
        name: Stable catalog key (e.g. ``"ble_2m"``).
        mod: The single-carrier modulation.
        profile: The PHY parameters (sps/mod_index/bt).
    """

    name: str
    mod: SCMod
    profile: SCProfile

    @property
    def bits_per_symbol(self) -> int:
        """2 for QPSK, else 1 (BPSK/FSK/GFSK)."""
        return 2 if self.mod == SCMod.QPSK else 1

    @property
    def is_fsk(self) -> bool:
        """True for FSK/GFSK (frequency modulations)."""
        return self.mod in (SCMod.FSK, SCMod.GFSK)

    @property
    def gfsk(self) -> bool:
        """True only for GFSK (Gaussian-shaped)."""
        return self.mod == SCMod.GFSK


# Curated, PHY-distinct single-carrier catalog (see the design spec). Each
# entry is distinguishable by blind resolution -- by preamble waveform (sps
# and FSK/GFSK/PSK shape) for all but the BPSK<->QPSK pair, which the
# payload-order discriminator (`core.blind`) separates. `sik_gfsk` equals the
# framework's current GFSK default (`DEFAULT_SC_PROFILE`).
SC_CATALOG: Dict[str, SCProfileSpec] = {
    "sik_gfsk": SCProfileSpec(
        "sik_gfsk", SCMod.GFSK, SCProfile(sps=8, mod_index=0.7, bt=0.5)
    ),
    "ble_1m": SCProfileSpec(
        "ble_1m", SCMod.GFSK, SCProfile(sps=8, mod_index=0.5, bt=0.5)
    ),
    "ble_2m": SCProfileSpec(
        "ble_2m", SCMod.GFSK, SCProfile(sps=4, mod_index=0.5, bt=0.5)
    ),
    "fsk_basic": SCProfileSpec(
        "fsk_basic", SCMod.FSK, SCProfile(sps=8, mod_index=0.7, bt=0.5)
    ),
    "psk_c2": SCProfileSpec("psk_c2", SCMod.BPSK, SCProfile(sps=8)),
    "qpsk_link": SCProfileSpec("qpsk_link", SCMod.QPSK, SCProfile(sps=8)),
}

# OFDM catalog: only the fixed 802.11a-style profile for now (P2-OFDM adds
# parametric OFDM profiles). Value is the `core.ofdm` profile object.
OFDM_CATALOG: Dict[str, object] = {"wifi_20": DEFAULT_OFDM_PROFILE}

# The framework default single-carrier profile name (the GFSK default).
DEFAULT_SC_PROFILE_NAME = "sik_gfsk"


def family_of(name: str) -> Family:
    """Return the :class:`Family` of a catalog profile name.

    Raises:
        KeyError: If ``name`` is in neither catalog.
    """
    if name in SC_CATALOG:
        return Family.SINGLE_CARRIER
    if name in OFDM_CATALOG:
        return Family.OFDM
    raise KeyError(f"unknown profile: {name}")


def all_profile_names() -> List[str]:
    """All known profile names (single-carrier then OFDM)."""
    return list(SC_CATALOG) + list(OFDM_CATALOG)
