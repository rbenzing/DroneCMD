"""Blind waveform-family and single-carrier-profile resolution.

Given a detected region with no side information, decide OFDM vs
single-carrier (:func:`classify_family`) and, for single-carrier, infer the
profile from lock confidence (:func:`resolve_sc_profile`, Task 3). Used by the
validation pipeline so decode does not depend on provenance -- provenance is
kept only as ground truth for the profile-ID metric.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, TypeVar

import numpy as np
import numpy.typing as npt

from core.ofdm import (
    OFDM_SYNC_THRESHOLD,
    OFDMProfile,
    ofdm_equalized_symbols,
    ofdm_sync_confidence,
)
from core.profiles import OFDM_CATALOG, SC_CATALOG, Family, SCMod, SCProfileSpec
from core.single_carrier import (
    SC_SYNC_THRESHOLD,
    preamble_wave_fsk,
    preamble_wave_psk,
    sc_acquire,
    sc_aligned_payload_centers,
)

Complex = npt.NDArray[np.complex128]
_A = TypeVar("_A", bound=np.generic)

# Cyclic-prefix autocorrelation score above which a region MAY be OFDM. An
# OFDM symbol repeats its last CP samples one FFT-size later, giving a
# normalized CP-metric ~1 at symbol boundaries. This alone is insufficient:
# the short deterministic single-carrier Barker preamble spuriously self-matches
# at lag N too, so the OFDM decision ALSO requires high PAPR (below).
OFDM_FAMILY_THRESHOLD = 0.5

# Peak-to-average power ratio (linear) above which a region MAY be OFDM.
# Single-carrier here is near-constant-envelope (FSK/GFSK: |y|=1; rect PSK:
# |y|=1 after normalization) -> PAPR ~1-2; OFDM is high-PAPR -> ~8-13. This is
# the co-requirement that rejects the single-carrier preamble's spurious CP
# self-match, which PAPR does not share.
PAPR_OFDM_THRESHOLD = 4.0

# Mean data-subcarrier EVM (distance to the nearest ideal QPSK point) above
# which a region's best-fitting OFDM profile is NOT credible OFDM -- a loud-
# failure ceiling for `resolve_ofdm_profile`. Genuine correct-profile OFDM
# equalizes onto the unit-magnitude QPSK constellation (EVM ~0 at high SNR,
# rising with noise); a wrong CP mis-windows every FFT and a wrong layout
# mis-equalizes every bin, and a misrouted single-carrier region has no
# OFDM structure at all -> EVM near/above the ~1.4 max constellation spacing.
#
# Tuned from 0.9 to 0.65 (P2-OFDM plan T3 measurement): the naive 0.9 sat
# ABOVE the misrouted-single-carrier floor, not below it. Measured, wifi_20
# correct-profile EVM (30 finite-region seeds per SNR): mean/max 0.38/0.45 @
# 10 dB, 0.12/0.15 @ 20 dB, 0.04/0.05 @ 30 dB. Measured misrouted-region EVM
# (best OFDM profile fit, GFSK sps=8 mod_index=0.7 bt=0.5 over payload
# lengths 24-200 bytes): 0.726-0.753; FSK: ~0.74. 0.65 sits in the resulting
# gap (above genuine OFDM down to the ~10 dB normal-SNR floor, below every
# measured misrouted-single-carrier fit). As with OFDM_SYNC_THRESHOLD, the
# distributions overlap below the normal-SNR band (correct-profile EVM
# reaches ~0.86 at 5 dB, above this ceiling; no scalar is airtight there);
# the residual is measured by the profile-ID/BER metrics, not claimed as a
# hard guarantee. The loud-failure guarantee holds at normal operating SNR.
OFDM_EVM_MAX = 0.65

_QPSK_SCALE = 1.0 / np.sqrt(2.0)


def _ofdm_data_evm(iq: Complex, profile: OFDMProfile) -> float:
    """Mean distance of equalized data subcarriers to the nearest ideal QPSK
    point over a trial demod with ``profile`` (``+inf`` if no data symbol).

    Low for the true profile; high when a wrong CP mis-windows the per-symbol
    FFT or a wrong pilot/data layout mis-equalizes. This is the CP/layout-
    selective signal the FFT-size-selective sync metric cannot provide.
    """
    syms = ofdm_equalized_symbols(np.asarray(iq, dtype=np.complex128), profile)
    if syms.size == 0:
        return float("inf")
    ideal = (
        np.where(syms.real >= 0.0, 1.0, -1.0)
        + 1j * np.where(syms.imag >= 0.0, 1.0, -1.0)
    ) * _QPSK_SCALE
    return float(np.mean(np.abs(syms - ideal)))


def _cp_autocorr_peak(iq: Complex, n_fft: int, cp: int) -> float:
    """Max normalized cyclic-prefix autocorrelation metric over the region.

    ``M(d) = |sum_{k<cp} conj(y[d+k]) y[d+k+n_fft]|^2 /
             (sum_{k<cp}|y[d+k]|^2 * sum_{k<cp}|y[d+k+n_fft]|^2)`` in [0, 1].
    """
    y = np.asarray(iq, dtype=np.complex128)
    if y.size < n_fft + cp:
        return 0.0
    prod = np.conj(y[:-n_fft]) * y[n_fft:]
    a = (np.abs(y[:-n_fft]) ** 2).astype(np.float64)
    b = (np.abs(y[n_fft:]) ** 2).astype(np.float64)

    def _slide(x: npt.NDArray[_A]) -> npt.NDArray[_A]:
        zero = np.zeros(1, dtype=x.dtype)
        c = np.cumsum(np.concatenate([zero, x]))
        result: npt.NDArray[_A] = (c[cp:] - c[:-cp]).astype(x.dtype)
        return result

    p = _slide(prod)
    ra = _slide(a)
    rb = _slide(b)
    denom = ra * rb + 1e-12
    metric = (np.abs(p) ** 2) / denom
    return float(np.max(metric)) if metric.size else 0.0


def _papr(iq: Complex) -> float:
    """Linear peak-to-average power ratio (~1-2 constant-envelope SC, ~8-13
    OFDM)."""
    y = np.asarray(iq, dtype=np.complex128)
    if y.size == 0:
        return 0.0
    power = (np.abs(y) ** 2).astype(np.float64)
    mean = float(np.mean(power))
    if mean <= 0.0:
        return 0.0
    return float(np.max(power)) / mean


def classify_family(
    iq: Complex,
    *,
    fft_sizes: Tuple[int, ...] = (64,),
    cp_ratio: float = 0.25,
    ofdm_threshold: float = OFDM_FAMILY_THRESHOLD,
    papr_threshold: float = PAPR_OFDM_THRESHOLD,
) -> Tuple[Family, float]:
    """Classify a region as OFDM vs single-carrier (blind).

    A region is judged OFDM only if it has BOTH a cyclic-prefix
    autocorrelation peak (``>= ofdm_threshold`` at some candidate FFT size --
    just ``64`` = ``wifi_20`` for now) AND high PAPR (``>= papr_threshold``).
    The CP peak alone is fooled by the short deterministic single-carrier
    preamble's lag-N self-match; the PAPR co-requirement rejects it (SC is
    near-constant-envelope), and the CP requirement rejects noise. Anything not
    clearing both is single-carrier (the conservative default -- the SC
    resolver then gates on its own lock, so a mis-defaulted region still fails
    loudly rather than silently mis-decoding).

    Args:
        iq: Region samples (``complex128`` or castable).
        fft_sizes: Candidate OFDM FFT sizes to probe.
        cp_ratio: Cyclic-prefix length as a fraction of the FFT size.
        ofdm_threshold: Decision threshold on the CP metric.
        papr_threshold: Decision threshold on the linear PAPR.

    Returns:
        ``(family, cp_score)`` with ``cp_score`` (the CP-autocorr component) in
        ``[0, 1]``.
    """
    y = np.asarray(iq, dtype=np.complex128)
    cp_best = 0.0
    for n_fft in fft_sizes:
        cp = max(1, int(round(n_fft * cp_ratio)))
        cp_best = max(cp_best, _cp_autocorr_peak(y, n_fft, cp))
    is_ofdm = cp_best >= ofdm_threshold and _papr(y) >= papr_threshold
    family = Family.OFDM if is_ofdm else Family.SINGLE_CARRIER
    return family, cp_best


# Aligned-payload quadrature-rail energy ratio above which a PSK burst is
# judged QPSK (both rails filled) rather than BPSK (Q ~ 0). The Barker
# preamble is always BPSK, so acquisition cannot separate BPSK from QPSK at
# the same sps -- this post-alignment test does.
#
# The discriminator is strongly asymmetric: QPSK's ratio stays >= 0.81 even
# at 8 dB SNR, while a noisy BPSK burst's ratio climbs from ~0 toward 0.5+
# as SNR drops. A 0.5 threshold therefore lets a BPSK burst blindly resolve
# to qpsk_link and return silent wrong bits in the normal-SNR band (measured
# BPSK->QPSK confusion: nonzero from ~11 dB down, e.g. 6/40 @10 dB, 7/40
# @8 dB at 0.5). Raising the threshold to 0.7 -- the top of the sanctioned
# [0.3, 0.7] latitude -- keeps QPSK safe (its ratio never approaches 0.7)
# while pushing BPSK->QPSK confusion out of the normal-SNR band (measured
# clean, 0/40, at 10 dB and above at 0.7).
QPSK_QRAIL_THRESHOLD = 0.7


def _sc_ref(spec: SCProfileSpec) -> Complex:
    """Preamble reference waveform for a catalog profile."""
    if spec.is_fsk:
        return preamble_wave_fsk(spec.profile, gfsk=spec.gfsk)
    return preamble_wave_psk(spec.profile)


def _psk_qrail_ratio(iq: Complex, spec: SCProfileSpec) -> float:
    """mean|Q| / mean|I| of the coherently-aligned payload (BPSK~0, QPSK~1)."""
    centers = sc_aligned_payload_centers(iq, spec.profile)
    if centers.size == 0:
        return 0.0
    num = float(np.mean(np.abs(centers.imag)))
    den = float(np.mean(np.abs(centers.real))) + 1e-9
    return num / den


def _psk_profile_of_order(mod: SCMod, sps: int) -> Optional[SCProfileSpec]:
    """The catalog PSK profile of a given order (BPSK/QPSK) and sps, if any."""
    for spec in SC_CATALOG.values():
        if spec.mod == mod and spec.profile.sps == sps:
            return spec
    return None


def resolve_sc_profile(iq: Complex) -> Tuple[Optional[SCProfileSpec], float]:
    """Blindly resolve a single-carrier region to a catalog profile.

    For each catalog profile, correlates the region against that profile's
    preamble via the CFO-tolerant :func:`core.single_carrier.sc_acquire` and
    keeps the highest lock peak. Below :data:`core.single_carrier.
    SC_SYNC_THRESHOLD` -> no lock (``(None, best)``). PSK profiles that share a
    preamble (same sps: BPSK and QPSK) tie on acquisition; the winner is then
    disambiguated by the payload-order discriminator (:func:`_psk_qrail_ratio`).

    Args:
        iq: Region samples (``complex128`` or castable).

    Returns:
        ``(spec, confidence)`` -- the resolved profile and its ``abs`` lock
        peak in ``[0, 1]``; ``(None, best)`` on no lock.
    """
    signal = np.asarray(iq, dtype=np.complex128)
    best_spec: Optional[SCProfileSpec] = None
    best_peak = 0.0
    for spec in SC_CATALOG.values():
        _, _, peak = sc_acquire(signal, _sc_ref(spec), spec.profile.sps)
        if abs(peak) > best_peak:
            best_peak = abs(peak)
            best_spec = spec
    if best_spec is None or best_peak < SC_SYNC_THRESHOLD:
        return None, best_peak
    if best_spec.mod in (SCMod.BPSK, SCMod.QPSK):
        ratio = _psk_qrail_ratio(signal, best_spec)
        want = SCMod.QPSK if ratio > QPSK_QRAIL_THRESHOLD else SCMod.BPSK
        matched = _psk_profile_of_order(want, best_spec.profile.sps)
        if matched is not None:
            best_spec = matched
    return best_spec, best_peak


def resolve_ofdm_profile(iq: Complex) -> Tuple[Optional[str], float]:
    """Blindly resolve an OFDM region to a catalog profile name.

    Stage 1 (FFT-size-selective sync gate): score each OFDM_CATALOG profile
    with :func:`core.ofdm.ofdm_sync_confidence`; discard those below
    :data:`core.ofdm.OFDM_SYNC_THRESHOLD`. Rejects noise and wrong-FFT-size
    profiles. Stage 2 (CP/layout-selective EVM tiebreak): among locked
    candidates, pick the lowest :func:`_ofdm_data_evm`. Returns ``(None, best)``
    if nothing locks OR the winner's EVM exceeds :data:`OFDM_EVM_MAX` (loud
    no-lock -- the region is not trustworthy OFDM).

    Returns:
        ``(name, sync_confidence)`` on a lock, else ``(None, best_conf)``.
    """
    signal = np.asarray(iq, dtype=np.complex128)
    best_conf = 0.0
    locked: List[Tuple[str, float, float]] = []
    for name, profile in OFDM_CATALOG.items():
        conf = ofdm_sync_confidence(signal, profile)
        best_conf = max(best_conf, conf)
        if conf >= OFDM_SYNC_THRESHOLD:
            locked.append((name, conf, _ofdm_data_evm(signal, profile)))
    if not locked:
        return None, best_conf
    name, conf, evm = min(locked, key=lambda t: t[2])
    if evm > OFDM_EVM_MAX:
        return None, best_conf
    return name, conf
