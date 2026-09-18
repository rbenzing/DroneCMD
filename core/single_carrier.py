"""Shared single-carrier PHY primitives for DroneCMD (single source of truth).

This module will be imported by both the synthetic modulator
(``validation.synth.modulators``) and the production demodulator
(``core.demodulation``) so the transmit and receive sides of FSK/GFSK/BPSK/
QPSK waveforms cannot drift apart -- mirroring how :mod:`core.ofdm` is the
shared PHY for OFDM.

Frame acquisition uses a fixed Barker-13 preamble (repeated twice, giving
26 symbols) correlated against the received signal with a normalized
matched filter. A matched filter maximizes the output signal-to-noise ratio
for a known waveform in additive white noise (North, 1943), and the Barker-13
sequence was chosen for its low-sidelobe autocorrelation (a single sharp peak
at zero lag, sidelobes bounded by ``1/13``), which makes the correlation peak
an unambiguous, high-confidence timing marker even before any channel
equalization has run (Barker, 1953).

References:
    R. H. Barker, "Group synchronizing of binary digital systems," in
    Communication Theory, pp. 273-287, Butterworth, 1953.
    D. O. North, "An analysis of the factors which determine signal/noise
    discrimination in pulsed-carrier systems," Proc. IRE, vol. 51, 1963
    (reprint of 1943 RCA report) -- the matched-filter/correlation-receiver
    result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d

Complex = npt.NDArray[np.complex128]
Real = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SCProfile:
    """Immutable single-carrier PHY parameters shared by TX and RX.

    Attributes:
        sps: Samples per symbol. Governs rectangular pulse-shaping for PSK
            and the phase-integration step size for FSK/GFSK.
        mod_index: FSK/GFSK modulation index (cycles/symbol of frequency
            deviation).
        bt: GFSK Gaussian pulse-shaping filter bandwidth-time product.
    """

    sps: int = 8
    mod_index: float = 0.7
    bt: float = 0.5


# Samples per symbol used by `DEFAULT_SC_PROFILE`. Exposed as a module-level
# constant (in addition to `SCProfile.sps`) since callers that only need the
# oversampling factor -- e.g. to size buffers before a profile exists --
# should not have to construct a profile first.
SC_SPS = 8

DEFAULT_SC_PROFILE = SCProfile(sps=SC_SPS)

# The normalized matched-filter correlation (see `sc_frame_sync`) lives in
# [0, 1] and is ~1 at a true preamble lock. A peak below this threshold means
# the search found no reliable Barker preamble -- either noise, or a spurious
# correlation elsewhere in the payload -- and the recovered frame should not
# be trusted.
SC_SYNC_THRESHOLD = 0.5

# Barker-13: the longest known binary Barker code, chosen for the shared
# preamble because its aperiodic autocorrelation has a single peak of 13 at
# zero lag and sidelobes of magnitude <= 1 everywhere else -- the best
# sidelobe suppression (13:1) of any Barker code, which keeps the matched
# filter's false-lock probability low (Barker, 1953).
BARKER13: Real = np.array(
    [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
    dtype=np.float64,
)

# The preamble is Barker-13 repeated twice (26 symbols): the repeat gives the
# receiver two back-to-back correlation opportunities (robust to a single
# corrupted half) while keeping the preamble short relative to typical drone
# command payloads.
PREAMBLE_SYMBOLS: Real = np.concatenate([BARKER13, BARKER13])


def preamble_wave_psk(profile: SCProfile) -> Complex:
    """Build the BPSK preamble waveform (rectangular pulse shaping).

    The preamble symbols are placed on the real axis (BPSK) and held for
    ``profile.sps`` samples each, matching the rectangular pulse shaping the
    synthetic QPSK modulator (``validation.synth.modulators._qpsk``) uses.

    Args:
        profile: Single-carrier PHY parameters (only ``sps`` is used).

    Returns:
        The preamble waveform as ``complex128``, length
        ``len(PREAMBLE_SYMBOLS) * profile.sps``.
    """
    wave: Complex = np.repeat(PREAMBLE_SYMBOLS.astype(np.complex128), profile.sps)
    return wave


def preamble_wave_fsk(profile: SCProfile, *, gfsk: bool) -> Complex:
    """Build the (G)FSK preamble waveform via phase-integration.

    This duplicates the modulation math of
    ``validation.synth.modulators._fsk`` exactly (rather than importing it,
    which would create a ``validation`` -> ``core`` dependency cycle) so the
    transmit preamble generated in ``validation.synth`` and the receive-side
    reference used by :func:`sc_frame_sync` agree bit-for-bit.

    Args:
        profile: Single-carrier PHY parameters (``sps``, ``mod_index``,
            ``bt``).
        gfsk: If True, apply Gaussian pulse shaping (``profile.bt``) to the
            NRZ symbol stream before phase integration (GFSK). If False,
            integrate the unshaped NRZ stream directly (plain FSK).

    Returns:
        The preamble waveform as ``complex128``, length
        ``len(PREAMBLE_SYMBOLS) * profile.sps``.
    """
    bits = np.where(PREAMBLE_SYMBOLS > 0, 1.0, 0.0)
    symbols = 2.0 * bits - 1.0  # {0,1} -> {-1,+1}
    shape = np.repeat(symbols, profile.sps)
    if gfsk:
        # Gaussian pulse shaping: sigma from BT product over one symbol
        # period (identical to `validation.synth.modulators._fsk`).
        sigma = profile.sps * np.sqrt(np.log(2)) / (2 * np.pi * profile.bt)
        shape = gaussian_filter1d(shape, sigma=max(sigma, 1e-3), mode="nearest")
    # Frequency deviation: peak phase step so a symbol advances mod_index
    # cycles.
    freq = (profile.mod_index / profile.sps) * shape  # cycles per sample
    phase = 2 * np.pi * np.cumsum(freq)
    wave: Complex = np.exp(1j * phase).astype(np.complex128)
    return wave


def sc_frame_sync(
    rx: Complex, ref_wave: Complex, search_span: int
) -> Tuple[int, complex]:
    """Locate the preamble in `rx` via normalized matched-filter correlation.

    For each candidate start offset ``d`` in the searched span, computes the
    normalized cross-correlation between the received window and the
    reference (preamble) waveform::

        c[d] = sum(rx[d:d+L] * conj(ref_wave)) / (||rx[d:d+L]|| * ||ref_wave||)

    where ``L = len(ref_wave)``. Normalizing by both window and reference
    energy makes ``c[d]`` a matched-filter correlation coefficient bounded in
    magnitude by 1 (Cauchy-Schwarz), reaching 1 only at a perfect, noise-free
    alignment -- so ``abs(c[d])`` doubles as a lock-confidence score that is
    independent of received signal amplitude, and ``angle(c[d])`` at the
    best offset is the coarse residual carrier phase (see module docstring
    for the matched-filter/Barker-code rationale).

    Args:
        rx: Received IQ samples to search, as ``complex128``.
        ref_wave: Known reference (preamble) waveform, as ``complex128``.
        search_span: Maximum number of candidate start offsets to test,
            starting from ``d = 0``.

    Returns:
        A tuple ``(preamble_start_index, complex_peak)`` where
        ``abs(complex_peak)`` in ``[0, 1]`` is the lock confidence and
        ``angle(complex_peak)`` is the coarse absolute-phase estimate at that
        offset. Returns ``(0, 0+0j)`` if no full-length window fits (``rx``
        shorter than ``ref_wave``) or ``search_span <= 0``.
    """
    ref = np.asarray(ref_wave, dtype=np.complex128)
    signal = np.asarray(rx, dtype=np.complex128)
    length = ref.size
    ref_norm = float(np.linalg.norm(ref))
    n_positions = min(signal.size - length + 1, search_span)
    if n_positions <= 0:
        return 0, complex(0.0, 0.0)
    correlations: Complex = np.empty(n_positions, dtype=np.complex128)
    for d in range(n_positions):
        window = signal[d : d + length]
        window_norm = float(np.linalg.norm(window))
        correlations[d] = np.sum(window * np.conj(ref)) / (
            window_norm * ref_norm + 1e-12
        )
    best = int(np.argmax(np.abs(correlations)))
    peak = complex(correlations[best])
    return best, peak


def sc_lock_confidence(rx: Complex, ref_wave: Complex, search_span: int) -> float:
    """Lock confidence (``abs`` of the matched-filter peak) for `rx`.

    Args:
        rx: Received IQ samples to search, as ``complex128``.
        ref_wave: Known reference (preamble) waveform, as ``complex128``.
        search_span: Maximum number of candidate start offsets to test.

    Returns:
        ``abs(peak)`` from :func:`sc_frame_sync`, in ``[0, 1]``; ``0.0`` if
        ``rx`` is shorter than ``ref_wave``.
    """
    if len(rx) < len(ref_wave):
        return 0.0
    _, peak = sc_frame_sync(rx, ref_wave, search_span)
    return abs(peak)
