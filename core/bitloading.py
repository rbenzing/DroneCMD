"""Bit-loading for OFDM: square-QAM (orders 2/4/6), per-subcarrier SNR, and
Chow's rate-adaptive loading. Square QAM = two independent Gray-coded sqrt(M)-PAM
rails at unit average energy; order 2 reduces exactly to core.ofdm.qpsk_map.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import erfcinv

Bits = npt.NDArray[np.uint8]
Complex = npt.NDArray[np.complex128]
Real = npt.NDArray[np.float64]

__all__ = ["qam_map", "qam_demap", "qam_soft_demap", "subcarrier_snr", "chow_load"]

LLRs = npt.NDArray[np.float64]


def _gray_inverse(bits_per_rail: int) -> npt.NDArray[np.intp]:
    """Build the inverse-Gray lookup table for one PAM rail.

    Args:
        bits_per_rail: Number of bits carried by a single PAM rail
            (``order // 2`` for a square-QAM constellation).

    Returns:
        Array ``ginv`` of length ``2 ** bits_per_rail`` such that
        ``ginv[g] == n`` for the natural index ``n`` whose Gray code
        (``n ^ (n >> 1)``) equals ``g``.
    """
    L = 1 << bits_per_rail
    ginv = np.zeros(L, dtype=np.intp)
    for n in range(L):
        ginv[n ^ (n >> 1)] = n
    return ginv


def _bits_to_int(bit_groups: Bits, width: int) -> npt.NDArray[np.intp]:
    """Pack MSB-first bit groups into integers.

    Args:
        bit_groups: Array of shape ``(K, width)`` of 0/1 bits, MSB first
            along the last axis.
        width: Number of bits per group.

    Returns:
        Array of shape ``(K,)`` with each row's bits packed into an integer.
    """
    weights = (1 << np.arange(width - 1, -1, -1)).astype(np.intp)
    out: npt.NDArray[np.intp] = (bit_groups.astype(np.intp) * weights).sum(axis=1)
    return out


def _int_to_bits(vals: npt.NDArray[np.intp], width: int) -> Bits:
    """Unpack integers into MSB-first bit groups.

    Args:
        vals: Array of shape ``(K,)`` of non-negative integers, each less
            than ``2 ** width``.
        width: Number of bits to unpack per value.

    Returns:
        Array of shape ``(K, width)`` of 0/1 bits (``uint8``), MSB first.
    """
    shifts = np.arange(width - 1, -1, -1)
    return ((vals[:, None] >> shifts) & 1).astype(np.uint8)


def qam_map(bits: Bits, order: int) -> Complex:
    """Map bits to unit-average-energy square-QAM symbols (order in {2,4,6}).

    Order 2 == core.ofdm.qpsk_map. Per rail: ``order // 2`` bits map to a
    Gray-coded sqrt(M)-PAM level; the amplitude for natural index ``n`` is
    ``(L - 1) - 2n`` (so the all-zero bit group maps to the most-positive
    level, matching QPSK's bit0->+1), scaled to unit average symbol energy
    by ``sqrt((2/3)(M-1))``.

    Args:
        bits: 1-D array of 0/1 values (``uint8``-like) whose length is a
            multiple of ``order``.
        order: Bits per symbol; must be one of ``{2, 4, 6}``.

    Returns:
        Complex128 array of ``bits.size // order`` unit-average-energy
        square-QAM symbols.

    Raises:
        ValueError: If ``order`` is not one of ``{2, 4, 6}``.
    """
    if order not in (2, 4, 6):
        raise ValueError(f"unsupported QAM order: {order}")
    b = np.asarray(bits, dtype=np.uint8).reshape(-1, order)
    bpr = order // 2
    L = 1 << bpr
    ginv = _gray_inverse(bpr)
    i_n = ginv[_bits_to_int(b[:, :bpr], bpr)]
    q_n = ginv[_bits_to_int(b[:, bpr:], bpr)]
    i_amp = (L - 1) - 2.0 * i_n
    q_amp = (L - 1) - 2.0 * q_n
    norm = np.sqrt((2.0 / 3.0) * ((1 << order) - 1))
    out: Complex = ((i_amp + 1j * q_amp) / norm).astype(np.complex128)
    return out


def qam_demap(symbols: Complex, order: int) -> Bits:
    """Hard-decision inverse of :func:`qam_map`.

    Each rail (I and Q) is independently sliced to its nearest sqrt(M)-PAM
    level, then inverse-Gray-decoded back to its bit group.

    Args:
        symbols: 1-D complex array of square-QAM symbols (need not be
            exactly on-constellation; nearest-level slicing is applied).
        order: Bits per symbol; must be one of ``{2, 4, 6}``.

    Returns:
        1-D ``uint8`` array of ``symbols.size * order`` hard-decision bits.

    Raises:
        ValueError: If ``order`` is not one of ``{2, 4, 6}``.
    """
    if order not in (2, 4, 6):
        raise ValueError(f"unsupported QAM order: {order}")
    bpr = order // 2
    L = 1 << bpr
    norm = np.sqrt((2.0 / 3.0) * ((1 << order) - 1))
    s = np.asarray(symbols, dtype=np.complex128) * norm

    def rail(vals: Real) -> npt.NDArray[np.intp]:
        """Slice a PAM rail's real-valued amplitudes to Gray-coded values.

        Args:
            vals: Real-valued rail amplitudes (already de-normalized).

        Returns:
            Array of Gray-coded integer values, one per input amplitude.
        """
        n = np.clip(np.round(((L - 1) - vals) / 2.0), 0, L - 1).astype(np.intp)
        gray: npt.NDArray[np.intp] = (n ^ (n >> 1)).astype(np.intp)
        return gray

    i_g = rail(s.real)
    q_g = rail(s.imag)
    out = np.empty((s.size, order), dtype=np.uint8)
    out[:, :bpr] = _int_to_bits(i_g, bpr)
    out[:, bpr:] = _int_to_bits(q_g, bpr)
    return out.reshape(-1)


def qam_soft_demap(symbols: Complex, order: int, weight: "Real | float" = 1.0) -> LLRs:
    """Soft (max-log LLR) inverse of :func:`qam_map` (order in {2,4,6}).

    Emits ``order`` per-bit LLRs per symbol, in the same MSB-first I-then-Q
    order as :func:`qam_map`, with the repo-wide convention **``L > 0`` => bit
    0** (``L = log P(bit=0)/P(bit=1)``). The level/label tables are derived from
    the same construction as :func:`qam_map`, so soft and hard demap cannot
    drift: hard-slicing (``llr < 0``) reproduces :func:`qam_demap`.

    For each Gray-coded sqrt(M)-PAM rail (I then Q, ``order // 2`` bits each) and
    each bit position, the max-log LLR of received rail value ``r`` is
    ``weight * (min_{levels: bit=1} (r - a)^2 - min_{levels: bit=0} (r - a)^2)``.
    When bit 0 is nearer, ``min_{bit=0}`` is smaller, so ``L > 0`` => bit 0.

    Args:
        symbols: 1-D complex square-QAM symbols (unit-average-energy scale).
        order: Bits per symbol; one of ``{2, 4, 6}``.
        weight: Per-symbol reliability ``|h_k|^2 / N0`` -- a scalar, or a 1-D
            array broadcast over ``symbols``. A global scale does not change
            hard decisions but the *relative* per-symbol weighting is what lets
            a soft FEC decoder exploit strong subcarriers (see ADR-0020).

    Returns:
        1-D ``float64`` array of ``symbols.size * order`` LLRs.

    Raises:
        ValueError: If ``order`` is not one of ``{2, 4, 6}``.
    """
    if order not in (2, 4, 6):
        raise ValueError(f"unsupported QAM order: {order}")
    bpr = order // 2
    L = 1 << bpr
    norm = np.sqrt((2.0 / 3.0) * ((1 << order) - 1))
    s = np.asarray(symbols, dtype=np.complex128).reshape(-1) * norm
    w = np.asarray(weight, dtype=np.float64)

    # Per-rail level amplitudes (natural index n) and their MSB-first Gray bit
    # labels -- exactly qam_map's construction (amp = (L-1) - 2n, label = gray(n)).
    n = np.arange(L, dtype=np.intp)
    amps = (L - 1) - 2.0 * n  # (L,)
    labels = _int_to_bits((n ^ (n >> 1)).astype(np.intp), bpr)  # (L, bpr)

    def rail_llrs(r: Real) -> LLRs:
        dist2 = (r[:, None] - amps[None, :]) ** 2  # (K, L)
        out = np.empty((r.size, bpr), dtype=np.float64)
        big = np.inf
        for j in range(bpr):
            bit1 = labels[:, j] == 1
            min1 = np.min(np.where(bit1[None, :], dist2, big), axis=1)
            min0 = np.min(np.where(~bit1[None, :], dist2, big), axis=1)
            out[:, j] = min1 - min0
        return out

    llr = np.empty((s.size, order), dtype=np.float64)
    llr[:, :bpr] = rail_llrs(s.real)
    llr[:, bpr:] = rail_llrs(s.imag)
    # rail_llrs measures squared distances in the de-normalized integer-PAM
    # domain, which are norm**2 times the unit-average-energy distances. Divide
    # them back so the reliability weight |h_k|**2/N0 multiplies *unit-energy*
    # distances -- otherwise the effective weight would carry an order-dependent
    # norm**2 factor (2/10/42 for orders 2/4/6), mis-weighting high-order
    # carriers relative to QPSK and breaking the BICM premise (see ADR-0020).
    llr /= norm * norm
    if w.ndim == 0:
        llr *= float(w)
    else:
        llr *= w[:, None]
    return llr.reshape(-1)


def subcarrier_snr(h_freq: Complex, noise_var: float) -> Real:
    """Compute per-subcarrier linear SNR from frequency-domain channel gains.

    Args:
        h_freq: 1-D array of complex per-subcarrier channel gains ``H_k``.
        noise_var: Scalar noise variance (must be positive; clamped to a
            small floor to avoid division by zero).

    Returns:
        Float64 array of linear SNR values ``|H_k|^2 / noise_var``, one per
        input subcarrier.
    """
    h = np.asarray(h_freq, dtype=np.complex128)
    nv = max(float(noise_var), 1e-12)
    return (np.abs(h) ** 2 / nv).astype(np.float64)


def _snr_gap(target_ber: float) -> float:
    """Compute the SNR gap Gamma for uncoded square QAM at a target BER.

    Achievable bits per symbol scale as ``log2(1 + SNR/Gamma)``; the
    feasibility SNR for bit-loading order ``o`` is ``Gamma * (2**o - 1)``.
    Uses the standard approximation ``Gamma = (1/3) * [Q^{-1}(target_ber/4)]^2``,
    with ``Q^{-1}(x) = sqrt(2) * erfcinv(2x)``.

    Args:
        target_ber: Target per-bit error rate (in ``(0, 1)``).

    Returns:
        The linear SNR gap Gamma.
    """
    qinv = float(np.sqrt(2.0)) * float(erfcinv(2.0 * (target_ber / 4.0)))
    return (qinv * qinv) / 3.0


def chow_load(
    snr: Real,
    target_ber: float,
    allowed_orders: "tuple[int, ...]" = (0, 2, 4, 6),
) -> npt.NDArray[np.intp]:
    """Assign per-subcarrier bit-loading orders via Chow's rate-adaptive rule.

    Each carrier is assigned the largest allowed order whose feasibility SNR
    ``Gamma * (2**o - 1)`` is less than or equal to the carrier's SNR, so
    every used carrier meets the target BER while total bits are maximized.
    Carriers that cannot support even the smallest positive order are nulled
    (assigned order 0).

    Args:
        snr: 1-D array of per-subcarrier linear SNR values (e.g. from
            :func:`subcarrier_snr`).
        target_ber: Target per-bit error rate used to derive the SNR gap.
        allowed_orders: Candidate bit-loading orders to consider, including
            0 for a nulled carrier. Defaults to ``(0, 2, 4, 6)``.

    Returns:
        Integer (``intp``) array of the same length as ``snr``, giving the
        assigned bit-loading order per subcarrier.
    """
    s = np.asarray(snr, dtype=np.float64)
    gap = _snr_gap(target_ber)
    orders = sorted(o for o in allowed_orders if o > 0)
    alloc = np.zeros(s.size, dtype=np.intp)
    for o in orders:
        required = gap * ((1 << o) - 1)
        alloc[s >= required] = o
    return alloc
