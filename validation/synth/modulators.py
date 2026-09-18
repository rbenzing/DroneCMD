"""Deterministic FSK/GFSK/BPSK/QPSK/OFDM modulators for the SP1 validation spine.

These generate reference IQ waveforms used to test detection, classification,
and channel-impairment pipelines. All modulators are deterministic (no RNG)
and normalize their output to unit average power as ``complex64``.

Single-carrier schemes (FSK/GFSK/BPSK/QPSK) prepend the shared
:mod:`core.single_carrier` Barker-13x2 preamble waveform to the payload, so
the production single-carrier receivers in that module
(``sc_demodulate_fsk``/``sc_demodulate_psk``) can acquire and decode these
synthetic bursts. OFDM keeps its own STF/LTF preamble from :mod:`core.ofdm`
and is unaffected by this.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d

from validation.types import IQSamples, ModScheme

if TYPE_CHECKING:
    from core.single_carrier import SCProfile

# Schemes carried by the shared core.single_carrier preamble + payload PHY,
# as opposed to OFDM's own STF/LTF-based framing.
_SINGLE_CARRIER_SCHEMES = (
    ModScheme.FSK,
    ModScheme.GFSK,
    ModScheme.BPSK,
    ModScheme.QPSK,
)


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


def _psk_payload(
    bits: npt.NDArray[np.float64],
    sps: int,
    bits_per_symbol: int,
    differential: bool,
) -> npt.NDArray[np.complex128]:
    """BPSK/QPSK payload via the shared ``core.single_carrier`` symbol mapping.

    Uses :func:`core.single_carrier.sc_map_psk` for bit-to-symbol mapping --
    the single source of truth also used by the production PSK receiver
    (``sc_demodulate_psk``'s companion ``sc_demap_psk``) -- rather than an
    ad-hoc local mapping, so transmit and receive agree exactly. Applies
    rectangular pulse shaping via ``np.repeat``, matching the preamble's
    pulse shaping (``core.single_carrier.preamble_wave_psk``).
    """
    from core.single_carrier import sc_diff_encode, sc_map_psk

    symbols = sc_map_psk(bits.astype(np.uint8), bits_per_symbol)
    if differential:
        symbols = sc_diff_encode(symbols)
    result: npt.NDArray[np.complex128] = np.repeat(symbols, sps)
    return result


def _sc_preamble(scheme: ModScheme, profile: SCProfile) -> npt.NDArray[np.complex128]:
    """Shared ``core.single_carrier`` preamble waveform for `scheme`."""
    from core.single_carrier import preamble_wave_fsk, preamble_wave_psk

    if scheme in (ModScheme.FSK, ModScheme.GFSK):
        return preamble_wave_fsk(profile, gfsk=(scheme == ModScheme.GFSK))
    return preamble_wave_psk(profile)


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
    differential: bool = False,
) -> IQSamples:
    """Modulate ``data`` bytes to complex64 IQ, unit average power.

    FSK/GFSK: 1 bit/symbol; BPSK: 1 bit/symbol; QPSK: 2 bits/symbol.
    Deterministic (no RNG).

    Single-carrier schemes (FSK/GFSK/BPSK/QPSK) prepend the shared
    :mod:`core.single_carrier` Barker-13x2 preamble waveform to the payload
    (see module docstring), so the output matches what the production
    single-carrier receivers expect. The preamble is built from a
    ``core.single_carrier.SCProfile`` constructed from *this call's own*
    ``sps``/``mod_index``/``bt`` -- not the module's fixed
    ``DEFAULT_SC_PROFILE`` -- so the preamble and payload always share the
    same PHY parameters, even if ``modulate()`` is called with non-default
    values. (This is the simpler of the two options considered: building a
    per-call ``SCProfile`` is trivial since ``SCProfile``'s fields are
    exactly ``modulate()``'s own ``sps``/``mod_index``/``bt`` parameters, and
    it is a strict improvement over hardcoding ``DEFAULT_SC_PROFILE`` -- the
    two agree whenever ``modulate()`` is called with its default PHY
    parameters, as all current SP1 call sites do.)

    OFDM ignores ``sps`` and ``differential`` entirely -- its symbol length
    is fixed by the shared :mod:`core.ofdm` profile (``N + CP``
    samples/symbol), not by samples per bit/symbol, and it carries its own
    STF/LTF preamble instead of the single-carrier one.

    Args:
        data: Payload bytes to modulate (MSB-first bit order).
        scheme: Modulation scheme (``ModScheme.FSK``, ``GFSK``, ``BPSK``,
            ``QPSK``, or ``OFDM``).
        sps: Samples per symbol. Ignored for ``ModScheme.OFDM``.
        mod_index: FSK/GFSK modulation index (cycles/symbol of deviation).
        bt: GFSK Gaussian filter bandwidth-time product.
        rolloff: Reserved for future pulse-shaped schemes (unused in SP1).
        differential: If True, differentially encode the BPSK/QPSK payload
            symbols (``core.single_carrier.sc_diff_encode``) before pulse
            shaping. Ignored for FSK/GFSK/OFDM.

    Returns:
        Unit-average-power IQ samples as ``complex64``. For FSK/GFSK/BPSK/
        QPSK, length is ``len(preamble) + n_symbols * sps`` where
        ``len(preamble) == 26 * sps`` (the Barker-13x2 preamble) and
        ``n_symbols = 8 * len(data)`` for FSK/GFSK/BPSK, ``4 * len(data)``
        for QPSK. For OFDM, length is the 2-symbol STF+LTF preamble plus
        ``n_ofdm_symbols * 80`` data-symbol samples (per the ``core.ofdm``
        default profile), where ``n_ofdm_symbols = ceil(8 * len(data) /
        96)`` (96 = the profile's QPSK data bits per OFDM symbol).

    Raises:
        ValueError: If ``scheme`` is otherwise unsupported.
    """
    if len(data) == 0:
        return np.zeros(0, dtype=np.complex64)
    bits = _bits_from_bytes(data)
    iq: npt.NDArray[np.complex128]
    if scheme == ModScheme.OFDM:
        iq = _ofdm(bits)
    elif scheme in _SINGLE_CARRIER_SCHEMES:
        from core.single_carrier import SCProfile

        profile = SCProfile(sps=sps, mod_index=mod_index, bt=bt)
        preamble = _sc_preamble(scheme, profile)
        if scheme in (ModScheme.FSK, ModScheme.GFSK):
            gaussian_bt = bt if scheme == ModScheme.GFSK else None
            payload = _fsk(bits, sps, mod_index, gaussian_bt)
        else:
            bits_per_symbol = 1 if scheme == ModScheme.BPSK else 2
            payload = _psk_payload(bits, sps, bits_per_symbol, differential)
        iq = np.concatenate([preamble, payload])
    else:
        raise ValueError(f"modulate() does not support {scheme}")
    p = np.mean(np.abs(iq) ** 2)
    if p > 0:
        iq = iq / np.sqrt(p)
    return iq.astype(np.complex64)
