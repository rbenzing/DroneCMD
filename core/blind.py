"""Blind waveform-family and single-carrier-profile resolution.

Given a detected region with no side information, decide OFDM vs
single-carrier (:func:`classify_family`) and, for single-carrier, infer the
profile from lock confidence (:func:`resolve_sc_profile`, Task 3). Used by the
validation pipeline so decode does not depend on provenance -- provenance is
kept only as ground truth for the profile-ID metric.
"""
from __future__ import annotations

from typing import Tuple, TypeVar

import numpy as np
import numpy.typing as npt

from core.profiles import Family

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
