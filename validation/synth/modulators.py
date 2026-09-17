"""Deterministic FSK/GFSK/QPSK/OFDM modulators for the SP1 validation spine.

These generate reference IQ waveforms used to test detection, classification,
and channel-impairment pipelines. All modulators are deterministic (no RNG)
and normalize their output to unit average power as ``complex64``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d

from validation.types import IQSamples, ModScheme


def _bits_from_bytes(data: bytes) -> npt.NDArray[np.float64]:
    """Unpack ``data`` into an MSB-first array of {0.0, 1.0} bits."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.float64)


def _fsk(
    bits: npt.NDArray[np.float64],
    sps: int,
    mod_index: float,
    gaussian_bt: Optional[float],
) -> npt.NDArray[np.complex128]:
    """Binary (G)FSK via phase integration of a (Gaussian-shaped) NRZ signal."""
    symbols = 2.0 * bits - 1.0  # {0,1} -> {-1,+1}
    shape = np.repeat(symbols, sps)
    if gaussian_bt is not None:
        # Gaussian pulse shaping: sigma from BT product over one symbol period.
        sigma = sps * np.sqrt(np.log(2)) / (2 * np.pi * gaussian_bt)
        shape = gaussian_filter1d(shape, sigma=max(sigma, 1e-3), mode="nearest")
    # Frequency deviation: peak phase step so a symbol advances mod_index cycles.
    freq = (mod_index / sps) * shape  # cycles per sample
    phase = 2 * np.pi * np.cumsum(freq)
    result: npt.NDArray[np.complex128] = np.exp(1j * phase)
    return result


def _qpsk(bits: npt.NDArray[np.float64], sps: int) -> npt.NDArray[np.complex128]:
    """QPSK with rectangular pulse shaping (sufficient for SP1)."""
    if len(bits) % 2 == 1:
        bits = np.append(bits, 0.0)
    i_bits = bits[0::2]
    q_bits = bits[1::2]
    # Direct mapping: 0 -> +1/sqrt2, 1 -> -1/sqrt2 (per rail).
    i = (1 - 2 * i_bits) / np.sqrt(2)
    q = (1 - 2 * q_bits) / np.sqrt(2)
    symbols = i + 1j * q
    result: npt.NDArray[np.complex128] = np.repeat(symbols, sps)
    return result


def _ofdm(bits: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
    """OFDM via the shared PHY in :mod:`core.ofdm` (deterministic, no RNG)."""
    from core.ofdm import modulate_ofdm

    return modulate_ofdm(bits.astype(np.uint8))


def modulate(
    data: bytes,
    scheme: ModScheme,
    sps: int = 8,
    *,
    mod_index: float = 0.7,
    bt: float = 0.5,
    rolloff: float = 0.35,
) -> IQSamples:
    """Modulate ``data`` bytes to complex64 IQ, unit average power.

    FSK/GFSK: 1 bit/symbol; QPSK: 2 bits/symbol. Deterministic (no RNG).
    OFDM ignores ``sps`` entirely -- its symbol length is fixed by the shared
    :mod:`core.ofdm` profile (``N + CP`` samples/symbol), not by samples per
    bit/symbol.

    Args:
        data: Payload bytes to modulate (MSB-first bit order).
        scheme: Modulation scheme (``ModScheme.FSK``, ``GFSK``, ``QPSK``, or
            ``OFDM``).
        sps: Samples per symbol. Ignored for ``ModScheme.OFDM``.
        mod_index: FSK/GFSK modulation index (cycles/symbol of deviation).
        bt: GFSK Gaussian filter bandwidth-time product.
        rolloff: Reserved for future pulse-shaped schemes (unused in SP1).

    Returns:
        Unit-average-power IQ samples as ``complex64``. For FSK/GFSK/QPSK,
        length is ``n_symbols * sps`` where ``n_symbols = 8 * len(data)`` for
        FSK/GFSK and ``4 * len(data)`` for QPSK. For OFDM, length is the
        2-symbol STF+LTF preamble plus ``n_ofdm_symbols * 80`` data-symbol
        samples (per the ``core.ofdm`` default profile).

    Raises:
        ValueError: If ``scheme`` is otherwise unsupported.
    """
    if len(data) == 0:
        return np.zeros(0, dtype=np.complex64)
    bits = _bits_from_bytes(data)
    iq: npt.NDArray[np.complex128]
    if scheme == ModScheme.FSK:
        iq = _fsk(bits, sps, mod_index, gaussian_bt=None)
    elif scheme == ModScheme.GFSK:
        iq = _fsk(bits, sps, mod_index, gaussian_bt=bt)
    elif scheme == ModScheme.QPSK:
        iq = _qpsk(bits, sps)
    elif scheme == ModScheme.OFDM:
        iq = _ofdm(bits)
    else:
        raise ValueError(f"modulate() does not support {scheme}")
    p = np.mean(np.abs(iq) ** 2)
    if p > 0:
        iq = iq / np.sqrt(p)
    return iq.astype(np.complex64)
