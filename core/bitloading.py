"""Bit-loading for OFDM: square-QAM (orders 2/4/6), per-subcarrier SNR, and
Chow's rate-adaptive loading. Square QAM = two independent Gray-coded sqrt(M)-PAM
rails at unit average energy; order 2 reduces exactly to core.ofdm.qpsk_map.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

Bits = npt.NDArray[np.uint8]
Complex = npt.NDArray[np.complex128]
Real = npt.NDArray[np.float64]

__all__ = ["qam_map", "qam_demap"]


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
