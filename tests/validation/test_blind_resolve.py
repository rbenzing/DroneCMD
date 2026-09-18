from __future__ import annotations

import numpy as np

from core.blind import resolve_sc_profile
from core.profiles import SC_CATALOG
from core.single_carrier import (  # noqa: F401  (kept for parity; not used directly)
    preamble_wave_fsk,
    preamble_wave_psk,
    sc_diff_encode,
    sc_map_psk,
)
from validation.synth.modulators import modulate
from validation.types import ModScheme

_SCMOD_TO_SCHEME = {
    "sik_gfsk": ModScheme.GFSK,
    "ble_1m": ModScheme.GFSK,
    "ble_2m": ModScheme.GFSK,
    "fsk_basic": ModScheme.FSK,
    "psk_c2": ModScheme.BPSK,
    "qpsk_link": ModScheme.QPSK,
}


def _burst(name: str) -> np.ndarray:
    spec = SC_CATALOG[name]
    data = bytes(range(24))
    iq = modulate(
        data,
        _SCMOD_TO_SCHEME[name],
        sps=spec.profile.sps,
        mod_index=spec.profile.mod_index,
        bt=spec.profile.bt,
    )
    return iq.astype(np.complex128)


def test_each_profile_resolves_to_itself() -> None:
    for name in SC_CATALOG:
        spec, conf = resolve_sc_profile(_burst(name))
        assert spec is not None, f"{name} failed to lock (conf={conf})"
        assert spec.name == name, f"{name} resolved to {spec.name}"


def test_bpsk_qpsk_disambiguated() -> None:
    # Same sps + identical BPSK preamble -> only the payload-order
    # discriminator separates these.
    assert resolve_sc_profile(_burst("psk_c2"))[0].name == "psk_c2"
    assert resolve_sc_profile(_burst("qpsk_link"))[0].name == "qpsk_link"


def test_ble_2m_sps4_resolves() -> None:
    assert resolve_sc_profile(_burst("ble_2m"))[0].name == "ble_2m"


def test_noise_returns_no_lock() -> None:
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(600) + 1j * rng.standard_normal(600)).astype(
        np.complex128
    )
    spec, conf = resolve_sc_profile(noise)
    assert spec is None
    assert conf < 0.5


def test_aligned_centers_separate_bpsk_qpsk() -> None:
    from core.single_carrier import sc_aligned_payload_centers

    bpsk = sc_aligned_payload_centers(_burst("psk_c2"), SC_CATALOG["psk_c2"].profile)
    qpsk = sc_aligned_payload_centers(
        _burst("qpsk_link"), SC_CATALOG["qpsk_link"].profile
    )
    bpsk_ratio = float(np.mean(np.abs(bpsk.imag))) / (
        float(np.mean(np.abs(bpsk.real))) + 1e-9
    )
    qpsk_ratio = float(np.mean(np.abs(qpsk.imag))) / (
        float(np.mean(np.abs(qpsk.real))) + 1e-9
    )
    assert bpsk_ratio < 0.3
    assert qpsk_ratio > 0.7
