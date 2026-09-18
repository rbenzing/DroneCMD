from __future__ import annotations

import numpy as np

from core.blind import classify_family
from core.profiles import SC_CATALOG, Family
from core.single_carrier import preamble_wave_fsk, preamble_wave_psk


def _sc_burst(spec) -> np.ndarray:
    # preamble + a short constant-envelope payload for the profile.
    rng = np.random.default_rng(1)
    if spec.is_fsk:
        pre = preamble_wave_fsk(spec.profile, gfsk=spec.gfsk)
    else:
        pre = preamble_wave_psk(spec.profile)
    pay = np.exp(1j * rng.uniform(0, 2 * np.pi, 300)).astype(np.complex128)
    return np.concatenate([pre, pay])


def _ofdm_burst() -> np.ndarray:
    from core.ofdm import modulate_ofdm

    bits = np.unpackbits(np.frombuffer(bytes(range(48)), dtype=np.uint8))
    return modulate_ofdm(bits.astype(np.uint8)).astype(np.complex128)


def test_ofdm_region_classifies_ofdm() -> None:
    fam, score = classify_family(_ofdm_burst())
    assert fam == Family.OFDM
    assert score > 0.5


def test_every_sc_profile_classifies_single_carrier() -> None:
    for spec in SC_CATALOG.values():
        fam, score = classify_family(_sc_burst(spec))
        assert (
            fam == Family.SINGLE_CARRIER
        ), f"{spec.name} misclassified (score={score})"


def test_noise_is_single_carrier_default() -> None:
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(500) + 1j * rng.standard_normal(500)).astype(
        np.complex128
    )
    fam, _ = classify_family(noise)
    assert fam == Family.SINGLE_CARRIER  # conservative default; SC resolver then gates
