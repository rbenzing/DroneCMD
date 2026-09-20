"""Raptor-style fountain code PHY for DroneCMD: Robust-Soliton LT inner code
over a systematic sparse precode, per-symbol-CRC erasure detection, and GF(2)
Gaussian-elimination decoding. Reproducible/self-consistent (RFC-inexact),
small-K demonstration scope (see design 0013). No framing import — the outer
CRC-16 lives in core.coding; fountain uses its own per-symbol CRC-8 and an
internal length header.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

Bits = npt.NDArray[np.uint8]
Real = npt.NDArray[np.float64]

_CRC8_POLY = 0x07  # CRC-8 (poly 0x07, init 0x00), MSB-first over a bit array


def crc8(bits: Bits) -> Bits:
    """Compute CRC-8 over a bit array.

    CRC-8 using polynomial 0x07 with zero initialization, MSB-first, over the
    input bit array. Returns the 8-bit CRC as a uint8 array.

    Args:
        bits: Input bit array (uint8 with values 0 or 1).

    Returns:
        8-element uint8 array containing the CRC bits.
    """
    reg = 0
    for bit in np.asarray(bits, dtype=np.uint8):
        reg ^= (int(bit) & 1) << 7
        reg = ((reg << 1) ^ _CRC8_POLY) & 0xFF if (reg & 0x80) else (reg << 1) & 0xFF
    return np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=np.uint8)


def robust_soliton(n_deg: int, c: float, delta: float) -> Real:
    """Compute Robust Soliton degree distribution.

    Robust Soliton probability mass function over degrees 1..n_deg, with index 0
    reserved (set to 0.0, unused). Parameters control ripple size and coverage
    guarantees per the Robust Soliton construction.

    Args:
        n_deg: Maximum degree (K in the source).
        c: Ripple-size scaling constant.
        delta: Target failure probability.

    Returns:
        Array of shape (n_deg + 1,) of float64, with pmf[0] = 0.0 and
        pmf[1:].sum() = 1.0 (a valid probability mass function).
    """
    mu = np.zeros(n_deg + 1, dtype=np.float64)
    rho = np.zeros(n_deg + 1, dtype=np.float64)
    rho[1] = 1.0 / n_deg
    for d in range(2, n_deg + 1):
        rho[d] = 1.0 / (d * (d - 1))
    s = c * np.log(n_deg / delta) * np.sqrt(n_deg)  # expected ripple size
    kf = max(int(np.floor(n_deg / s)), 1) if s > 0 else n_deg
    tau = np.zeros(n_deg + 1, dtype=np.float64)
    for d in range(1, min(kf, n_deg + 1)):
        tau[d] = s / (n_deg * d)
    if 1 <= kf <= n_deg:
        tau[kf] = s * np.log(s / delta) / n_deg if s > 0 else 0.0
    mu = rho + tau
    total = float(mu.sum())
    if total <= 0:
        mu[1] = 1.0
        total = 1.0
    return mu / total


def sample_degree(pmf: Real, rng: np.random.Generator) -> int:
    """Draw a degree from the Robust Soliton pmf.

    Sample a degree from the given probability mass function using the provided
    Generator (deterministic per seeded RNG state). Samples via CDF inverse
    transform.

    Args:
        pmf: Probability mass function (e.g., from robust_soliton()).
        rng: numpy.random.Generator for deterministic, seeded sampling.

    Returns:
        Sampled degree as an integer.
    """
    cdf = np.cumsum(pmf)
    u = float(rng.random()) * float(cdf[-1])
    return int(np.searchsorted(cdf, u, side="left"))


def symbol_neighbors(
    rng: np.random.Generator, degree: int, span: int
) -> npt.NDArray[np.intp]:
    """Draw distinct neighbor indices for LT encoding.

    Return `degree` distinct sorted indices in [0, span) drawn deterministically
    from the given Generator. Clamps degree to [1, span] to ensure valid output.

    Args:
        rng: numpy.random.Generator for deterministic, seeded sampling.
        degree: Requested number of distinct neighbors.
        span: Maximum index (exclusive); number of symbols to choose from.

    Returns:
        Sorted array of intp indices; size = min(degree, span), all in [0, span).
    """
    d = max(1, min(degree, span))
    return np.sort(rng.choice(span, size=d, replace=False)).astype(np.intp)
