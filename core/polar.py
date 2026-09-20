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
from typing import Callable, List, Tuple

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


def _phi(x: float) -> float:
    # Chung et al. approximation of the GA phi function, phi(0)=1, decreasing.
    if x <= 0.0:
        return 1.0
    if x < 10.0:
        return float(np.exp(-0.4527 * x**0.86 + 0.0218))
    return float(np.sqrt(np.pi / x) * np.exp(-x / 4.0) * (1.0 - 10.0 / (7.0 * x)))


def _phi_inv(y: float) -> float:
    # Numeric inverse of _phi by bisection on [0, 1e4]; _phi is monotone decreasing.
    if y >= 1.0:
        return 0.0
    if y <= _phi(1e4):
        return 1e4
    lo, hi = 0.0, 1e4
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _phi(mid) > y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gaussian_approx_reliabilities(n: int, design_snr_db: float) -> Real:
    """Per-position mean LLR under the Gaussian approximation (higher = more reliable).

    Non-bit-reversed convention matching ``polar_transform``: at each of the m
    stages, the 'check' (upper) branch degrades via
    ``phi_inv(1-(1-phi(a))^2)`` and the 'variable' (lower) branch improves to
    ``2*a``.
    """
    m = int(round(np.log2(n)))
    mllr = np.zeros(n, dtype=np.float64)
    mllr[0] = 4.0 * (10.0 ** (design_snr_db / 10.0))  # initial mean LLR ~ 2/sigma^2
    for i in range(1, m + 1):
        u = 1 << i
        half = u >> 1
        for j in range(0, n, u):
            for t in range(half):
                a = mllr[j + t]
                mllr[j + t] = _phi_inv(1.0 - (1.0 - _phi(a)) ** 2)
                mllr[j + half + t] = 2.0 * a
    return mllr


def build_code(n: int, k: int, design_snr_db: float) -> PolarCode:
    rel = gaussian_approx_reliabilities(n, design_snr_db)
    # info = the k most reliable positions; ties broken by index for determinism.
    order = np.lexsort((np.arange(n), rel))  # ascending reliability, then index
    info = np.sort(order[n - k :]).astype(np.intp)
    frozen = np.ones(n, dtype=np.bool_)
    frozen[info] = False
    return PolarCode(
        n=n, k=k, design_snr_db=design_snr_db, frozen_mask=frozen, info_positions=info
    )


def build_shortened_mask(code: PolarCode, info_len: int) -> npt.NDArray[np.bool_]:
    """Shorten-from-the-end frozen mask for a frame of ``info_len`` (<= k) bits.

    Force-freeze the top ``s = k - info_len`` input positions (their codeword
    bits are known-0 and dropped), then take the ``info_len`` most-reliable of
    the REMAINING positions as info.
    """
    n, k = code.n, code.k
    if not 0 < info_len <= k:
        raise ValueError("info_len must be in 1..k")
    s = k - info_len
    rel = gaussian_approx_reliabilities(n, code.design_snr_db)
    frozen = np.ones(n, dtype=np.bool_)
    forbidden = set(range(n - s, n))  # force-frozen shortening tail
    order = np.lexsort((np.arange(n), rel))  # ascending reliability
    chosen = 0
    for idx in reversed(order.tolist()):  # most reliable first
        if idx in forbidden:
            continue
        frozen[idx] = False
        chosen += 1
        if chosen == info_len:
            break
    return frozen


def polar_encode(info_bits: Bits, frozen_mask: npt.NDArray[np.bool_]) -> Bits:
    """Scatter info into the unfrozen positions of a length-n vector, transform."""
    n = frozen_mask.size
    info = np.asarray(info_bits, dtype=np.uint8)
    positions = np.where(~frozen_mask)[0]
    if info.size != positions.size:
        raise ValueError("info_bits size must equal the number of unfrozen positions")
    u = np.zeros(n, dtype=np.uint8)
    u[positions] = info
    return polar_transform(u)


def _f(a: Real, b: Real) -> Real:
    """Min-sum check-node combine: sign(a)*sign(b)*min(|a|,|b|).

    Scale-invariant (linearly homogeneous in a common scale of ``a``/``b``),
    which is what keeps the whole SCL decoder scale-invariant.
    """
    combined = np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
    return np.asarray(combined, dtype=np.float64)


def _g(a: Real, b: Real, u: Real) -> Real:
    """Bit-node combine given a decided (0/1-valued) partial-sum vector ``u``."""
    return np.asarray(b + (1.0 - 2.0 * u) * a, dtype=np.float64)


def scl_decode(
    llr: Real,
    frozen_mask: npt.NDArray[np.bool_],
    list_size: int,
    crc_check: Callable[[Bits], bool],
) -> Tuple[Bits, bool]:
    """CRC-aided successive-cancellation list decoder (min-sum f-node).

    Operates in the LLR domain over the same non-bit-reversed butterfly as
    ``polar_transform`` (``L>0 => bit 0``). At each bit position, the bit LLR
    is computed via the recursive SC f/g formula over contiguous halves (the
    encoder's ``x_left = T(u_left) XOR T(u_right)``, ``x_right = T(u_right)``
    structure): the f-branch combines the two channel-LLR halves directly,
    while the g-branch re-transforms the already-decided first-half bits
    (``polar_transform``) to fold them back in as a bit-corrected observation
    of the second half. Every path tracks an approximate path metric
    (``PM += |LLR|`` on a sign mismatch / a forced-zero frozen bit), pruned to
    the best ``list_size`` after each bit. All metrics are linearly
    homogeneous in a common LLR scale, so the decoder is scale-invariant.

    Args:
        llr: Length-n channel LLR vector (``L>0`` favors bit 0).
        frozen_mask: Length-n boolean mask, True at frozen positions.
        list_size: Maximum number of candidate paths to retain.
        crc_check: Callback receiving the decoded info-bit vector (bits at
            the unfrozen positions, in position order) and returning whether
            it passes an externally-defined CRC/frame check.

    Returns:
        Tuple of (info_bits, crc_passed): the info bits of the lowest-PM path
        whose info bits pass ``crc_check``, or the lowest-PM path overall
        (with ``crc_passed=False``) if no path passes.
    """
    n = frozen_mask.size
    channel = np.asarray(llr, dtype=np.float64)

    def bit_llr(u_hat: Bits, i: int) -> float:
        # LLR of bit i given the channel LLRs and the decisions u_hat[:i],
        # via the recursive SC formula (non-bit-reversed, contiguous halves).
        def rec(level_llr: Real, decided: Bits, idx: int, size: int) -> float:
            if size == 1:
                return float(level_llr[0])
            half = size // 2
            if idx < half:
                left = _f(level_llr[:half], level_llr[half:])
                return rec(left, decided[:half], idx, half)
            u_left = decided[:half]  # fully-decided first-half bits
            right = _g(
                level_llr[:half],
                level_llr[half:],
                polar_transform(u_left).astype(np.float64),
            )
            return rec(right, decided[half:], idx - half, half)

        return rec(channel, u_hat, i, n)

    paths: List[Tuple[Bits, float]] = [(np.zeros(n, dtype=np.uint8), 0.0)]
    for i in range(n):
        candidates: List[Tuple[Bits, float]] = []
        for u_hat, pm in paths:
            bit_l = bit_llr(u_hat, i)
            if frozen_mask[i]:
                forced = u_hat.copy()
                forced[i] = 0
                penalty = abs(bit_l) if bit_l < 0 else 0.0
                candidates.append((forced, pm + penalty))
            else:
                for bit in (0, 1):
                    forked = u_hat.copy()
                    forked[i] = bit
                    decided_bit = 0 if bit_l >= 0 else 1
                    penalty = 0.0 if bit == decided_bit else abs(bit_l)
                    candidates.append((forked, pm + penalty))
        candidates.sort(key=lambda path: path[1])
        paths = candidates[:list_size]

    positions = np.where(~frozen_mask)[0]
    # CRC-aided pick: lowest-PM path whose info bits pass crc_check.
    for u_hat, _pm in paths:
        info = u_hat[positions].astype(np.uint8)
        if crc_check(info):
            return info, True
    return paths[0][0][positions].astype(np.uint8), False
