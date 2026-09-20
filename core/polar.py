"""Arıkan polar code PHY for DroneCMD: transform, Gaussian-approximation
frozen set, and a CRC-aided successive-cancellation list (CA-SCL) decoder.

Mirrors core.ldpc / core.turbo: pure PHY math, no framing dependency
(scl_decode takes an injected CRC-check callback). Non-bit-reversed
convention: G = F^{⊗m}, F = [[1,0],[1,1]], which is lower-triangular, so
freezing the highest-index input positions forces the last codeword bits to
zero (used for shorten-from-the-end rate matching; see design 0012).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Bits = npt.NDArray[np.uint8]
Real = npt.NDArray[np.float64]


@dataclass(frozen=True)
class PolarCode:
    n: int
    k: int
    design_snr_db: float
    frozen_mask: npt.NDArray[np.bool_]
    info_positions: npt.NDArray[np.intp]


def polar_transform(u: Bits) -> Bits:
    """x = u · F^{⊗m} over GF(2), computed by the in-place butterfly (O(n log n))."""
    x = np.asarray(u, dtype=np.uint8).copy()
    n = x.size
    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            block = x[i : i + step]
            x[i : i + step] = block ^ x[i + step : i + 2 * step]
        step *= 2
    return x
