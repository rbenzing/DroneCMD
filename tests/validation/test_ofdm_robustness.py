"""Robustness pins for OFDM equalization + blind OFDM resolution.

On noise / no-signal the LS channel estimate ``h`` has near-zero occupied bins,
so the one-tap equalizer's ``y / h`` produced nan/inf, ``_ofdm_data_evm``
returned ``nan``, and ``resolve_ofdm_profile``'s ``evm > OFDM_EVM_MAX`` ceiling
was silently bypassed (``nan > x`` is ``False``) -> a false confident lock.
These pin the guarded division + the finite-EVM ceiling.
"""
from __future__ import annotations

import warnings

import numpy as np

from core.blind import OFDM_EVM_MAX, _ofdm_data_evm, resolve_ofdm_profile
from core.ofdm import modulate_ofdm, ofdm_equalized_symbols
from core.profiles import OFDM_CATALOG


def test_equalized_symbols_finite_on_degenerate_input() -> None:
    # Long zero buffer: sync proceeds but every channel bin is ~0, so the
    # equalizer used to emit nan/inf. It must now be all-finite, with no
    # divide-by-zero RuntimeWarning.
    z = np.zeros(8192, dtype=np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        syms = ofdm_equalized_symbols(z, OFDM_CATALOG["wifi_20"])
    assert np.isfinite(syms).all()


def test_ofdm_data_evm_finite_and_rejected_on_zeros() -> None:
    evm = _ofdm_data_evm(np.zeros(8192, dtype=np.complex128), OFDM_CATALOG["wifi_20"])
    assert np.isfinite(evm), "degenerate EVM must be finite, not nan"
    assert evm > OFDM_EVM_MAX, "degenerate input must score above the reject ceiling"


def test_resolve_ofdm_rejects_nonfinite_evm(monkeypatch) -> None:
    # Defense in depth: even if the EVM ever came back non-finite, the ceiling
    # must reject rather than silently lock. Use a real burst so sync locks,
    # then force a nan EVM.
    import core.blind as blind

    rx = modulate_ofdm(
        np.array([1, 0, 1, 1, 0, 0, 1, 0] * 40, dtype=np.uint8),
        OFDM_CATALOG["wifi_20"],
    ).astype(np.complex128)
    monkeypatch.setattr(blind, "_ofdm_data_evm", lambda iq, profile: float("nan"))
    name, _conf = resolve_ofdm_profile(rx)
    assert name is None, "non-finite EVM must not produce a lock"


def test_resolve_ofdm_still_locks_real_burst() -> None:
    # The guards must not cost a legitimate lock.
    rx = modulate_ofdm(
        np.array([1, 0, 1, 1, 0, 0, 1, 0] * 40, dtype=np.uint8),
        OFDM_CATALOG["wifi_20"],
    ).astype(np.complex128)
    name, conf = resolve_ofdm_profile(rx)
    assert name == "wifi_20" and conf >= 0.6
