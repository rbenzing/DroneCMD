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
Bits = npt.NDArray[np.uint8]


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

# CFO-hypothesis-search acquisition (see `sc_acquire`). The single
# full-preamble matched filter (`sc_frame_sync`) only holds lock while the
# preamble stays coherent, which caps the acquirable CFO at ~0.003
# cycles/sample at sps=8 -- well short of the OFDM chain's ~0.0156. Searching
# a grid of CFO hypotheses and derotating before each matched-filter pass
# lifts the acquisition range to `SC_CFO_RANGE`, past OFDM parity.
SC_CFO_RANGE = 0.02
# Grid resolution. Chosen as 1/(4 * preamble-length-in-samples) so the
# residual after coarse correction (< SC_CFO_STEP / 2) is comfortably inside
# both the two-halves fine estimator's +/-1/(2*13*sps) range
# (`sc_estimate_cfo_psk`) and the decision-directed loop's pull-in.
SC_CFO_STEP = 1.0 / (4 * len(PREAMBLE_SYMBOLS) * SC_SPS)


def preamble_wave_psk(profile: SCProfile) -> Complex:
    """Build the BPSK preamble waveform (rectangular pulse shaping).

    The preamble symbols are placed on the real axis (BPSK) and held for
    ``profile.sps`` samples each, matching the rectangular pulse shaping the
    synthetic BPSK/QPSK modulator (``validation.synth.modulators``, via
    :func:`sc_map_psk`) uses.

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


def sc_acquire(
    rx: Complex,
    ref_wave: Complex,
    sps: int,
    *,
    cfo_range: float = SC_CFO_RANGE,
    cfo_step: float = SC_CFO_STEP,
) -> Tuple[int, float, complex]:
    """Joint coarse timing + coarse CFO acquisition via a CFO-hypothesis grid.

    The single full-preamble matched filter (:func:`sc_frame_sync`) loses
    coherence -- and therefore lock -- once a carrier frequency offset rotates
    the reference appreciably across its own length, capping the acquirable
    CFO well below the OFDM chain's. This searches a grid of CFO hypotheses in
    ``[-cfo_range, +cfo_range]`` spaced by ``cfo_step``; for each hypothesis it
    derotates ``rx`` and runs the normalized matched filter, then returns the
    ``(start, cfo, peak)`` of the hypothesis maximizing ``abs(peak)``. This
    recovers coarse timing, coarse CFO, and the absolute phase together, with
    acquisition range ``cfo_range``.

    Compute is ``n_cfo * O(n * L)`` matched-filter work (``n_cfo =
    2*cfo_range/cfo_step + 1``), bounded and offline (T&E) -- acceptable; if it
    becomes a hotspot for large captures, vectorize (FFT-domain) later.

    Args:
        rx: Received IQ samples to search, as ``complex128``.
        ref_wave: Known reference (preamble) waveform, as ``complex128``.
        sps: Samples per symbol (search span scales with it, matching the
            single-path receivers).
        cfo_range: Half-width of the CFO grid in cycles/sample.
        cfo_step: Grid spacing in cycles/sample.

    Returns:
        ``(preamble_start_index, coarse_cfo, complex_peak)``. ``abs(peak)`` in
        ``[0, 1]`` is the lock confidence (compare to ``SC_SYNC_THRESHOLD``);
        ``coarse_cfo`` is the best grid CFO in cycles/sample. Returns
        ``(0, 0.0, 0+0j)`` if no full-length window fits.
    """
    signal = np.asarray(rx, dtype=np.complex128)
    ref = np.asarray(ref_wave, dtype=np.complex128)
    if signal.size < ref.size:
        return 0, 0.0, complex(0.0, 0.0)
    n = np.arange(signal.size)
    search_span = sps * 40
    n_steps = int(np.floor(cfo_range / cfo_step))
    hypotheses = np.arange(-n_steps, n_steps + 1) * cfo_step
    best_start = 0
    best_cfo = 0.0
    best_peak = complex(0.0, 0.0)
    for cfo in hypotheses:
        derotated = signal * np.exp(-1j * 2 * np.pi * cfo * n)
        start, peak = sc_frame_sync(derotated, ref, search_span)
        if abs(peak) > abs(best_peak):
            best_start = start
            best_cfo = float(cfo)
            best_peak = peak
    return best_start, best_cfo, best_peak


def sc_map_psk(bits: Bits, bits_per_symbol: int) -> Complex:
    """Map bits to unit-energy PSK symbols (synth Gray convention).

    BPSK (``bits_per_symbol=1``) places each bit on the real axis: bit ``0``
    maps to ``+1``, bit ``1`` maps to ``-1``. QPSK (``bits_per_symbol=2``)
    splits the bitstream into interleaved in-phase/quadrature rails --
    ``i_bits = bits[0::2]``, ``q_bits = bits[1::2]`` -- maps each rail with
    the same 0->+1 / 1->-1 convention, and scales by ``1/sqrt(2)`` so every
    symbol has unit energy.

    Args:
        bits: Input bits as ``uint8`` (0/1 valued). For QPSK, must have even
            length.
        bits_per_symbol: ``1`` for BPSK or ``2`` for QPSK.

    Returns:
        Complex symbols (``complex128``), length ``len(bits) //
        bits_per_symbol``.

    Raises:
        ValueError: If ``bits_per_symbol`` is not 1 or 2, or if QPSK is
            requested with an odd number of bits.
    """
    data = np.asarray(bits, dtype=np.uint8)
    if bits_per_symbol == 1:
        rail = 1.0 - 2.0 * data.astype(np.float64)
        symbols: Complex = rail.astype(np.complex128)
        return symbols
    if bits_per_symbol == 2:
        if data.size % 2 != 0:
            raise ValueError("QPSK mapping requires an even number of bits")
        i_rail = 1.0 - 2.0 * data[0::2].astype(np.float64)
        q_rail = 1.0 - 2.0 * data[1::2].astype(np.float64)
        symbols = ((i_rail + 1j * q_rail) / np.sqrt(2.0)).astype(np.complex128)
        return symbols
    raise ValueError(f"unsupported bits_per_symbol: {bits_per_symbol}")


def sc_demap_psk(symbols: Complex, bits_per_symbol: int) -> Bits:
    """Hard-decision inverse of :func:`sc_map_psk`.

    Args:
        symbols: Received/equalized PSK symbols (``complex128``).
        bits_per_symbol: ``1`` for BPSK or ``2`` for QPSK.

    Returns:
        Recovered bits as ``uint8``, length ``len(symbols) *
        bits_per_symbol``. For QPSK, bits are interleaved
        ``[i0, q0, i1, q1, ...]`` matching :func:`sc_map_psk`'s convention.

    Raises:
        ValueError: If ``bits_per_symbol`` is not 1 or 2.
    """
    syms = np.asarray(symbols, dtype=np.complex128)
    if bits_per_symbol == 1:
        bits: Bits = np.where(syms.real > 0, 0, 1).astype(np.uint8)
        return bits
    if bits_per_symbol == 2:
        i_bits = np.where(syms.real > 0, 0, 1).astype(np.uint8)
        q_bits = np.where(syms.imag > 0, 0, 1).astype(np.uint8)
        bits = np.empty(syms.size * 2, dtype=np.uint8)
        bits[0::2] = i_bits
        bits[1::2] = q_bits
        return bits
    raise ValueError(f"unsupported bits_per_symbol: {bits_per_symbol}")


def sc_diff_encode(symbols: Complex) -> Complex:
    """Differentially encode a PSK symbol stream.

    Computes ``out[k] = out[k-1] * symbols[k]`` with the reference symbol
    ``out[-1] = 1`` (i.e. the first output symbol carries ``symbols[0]``
    directly). Since this recurrence is a running product, it reduces to a
    cumulative product over ``symbols``.

    Args:
        symbols: Data symbols to encode (``complex128``).

    Returns:
        Differentially encoded symbols (``complex128``), same length as
        ``symbols``.
    """
    data = np.asarray(symbols, dtype=np.complex128)
    encoded: Complex = np.cumprod(data)
    return encoded


def sc_diff_decode(
    symbols: Complex, initial_ref: complex = complex(1.0, 0.0)
) -> Complex:
    """Differentially decode a PSK symbol stream (inverse of `sc_diff_encode`).

    Computes ``d[k] = symbols[k] * conj(symbols[k-1])`` with the reference
    ``symbols[-1] = initial_ref``, so ``d[0] = symbols[0] * conj(initial_ref)``.
    For unit-modulus PSK symbols this recovers the original data symbols
    independent of any constant (uncompensated) absolute carrier phase, since
    that phase cancels in the conjugate product -- but only from ``d[1]``
    onward when ``initial_ref`` is the default noiseless unit reference:
    ``symbols[0]`` itself still carries the raw channel phase in that case,
    since there is no earlier received symbol for it to cancel against (see
    `sc_demodulate_psk`, which passes the received last preamble symbol as
    ``initial_ref`` instead, extending the same phase-ambiguity cancellation
    to the first payload symbol).

    Args:
        symbols: Differentially encoded (received) symbols (``complex128``).
        initial_ref: Reference symbol for ``d[0]`` (the ``symbols[-1]``
            slot). Defaults to the unit reference matching
            `sc_diff_encode`'s own convention; a caller that has a received,
            channel-affected symbol known to have been transmitted as ``+1``
            (e.g. the last preamble symbol) can pass it here so ``d[0]``
            benefits from the same phase cancellation as every later symbol.

    Returns:
        Decoded data symbols (``complex128``), same length as ``symbols``.
    """
    data = np.asarray(symbols, dtype=np.complex128)
    prev = np.concatenate([[complex(initial_ref)], data[:-1]])
    decoded: Complex = data * np.conj(prev)
    return decoded


def sc_estimate_cfo_psk(rx_preamble: Complex, sps: int) -> float:
    """Estimate normalized carrier frequency offset from the split preamble.

    The shared preamble (see module docstring) is Barker-13 repeated twice,
    giving two identical 13-symbol halves ``h1``, ``h2``. In the absence of
    CFO, ``h2`` is a noiseless copy of ``h1``; a residual carrier offset
    rotates ``h2`` relative to ``h1`` by ``2*pi*cfo*(13*sps)`` radians (the
    sample gap between the two halves). Correlating the halves and reading
    off the phase of the result gives an unbiased CFO estimate, unambiguous
    for ``abs(cfo) < 1 / (2 * 13 * sps)`` cycles/sample (the phase must stay
    within +/-pi over the 13-symbol gap).

    Args:
        rx_preamble: Received samples spanning (at least) the preamble,
            aligned so that ``rx_preamble[: 26*sps]`` is the two Barker-13
            halves.
        sps: Samples per symbol.

    Returns:
        Estimated CFO in cycles/sample (normalized frequency offset).
    """
    data = np.asarray(rx_preamble, dtype=np.complex128)
    half_len = 13 * sps
    h1 = data[:half_len]
    h2 = data[half_len : 2 * half_len]
    correlation = complex(np.sum(np.conj(h1) * h2))
    cfo = float(np.angle(correlation) / (2 * np.pi * 13 * sps))
    return cfo


def sc_demodulate_psk(
    rx: Complex,
    profile: SCProfile,
    *,
    bits_per_symbol: int,
    differential: bool,
) -> Bits:
    """Full coherent/differential PSK receiver: sync, CFO, demap.

    Pipeline:

    1. Locate the preamble via :func:`sc_frame_sync`; bail out (empty
       result) if the lock confidence is below `SC_SYNC_THRESHOLD`.
    2. Estimate CFO from the preamble (:func:`sc_estimate_cfo_psk`) and
       derotate the entire received signal.
    3. Re-run the matched filter on the derotated signal (small window
       around the already-known start) to get a clean post-CFO phase
       reference.
    4. Coherent mode: derotate the payload by that reference phase (so the
       preamble's known symbols land at ``+1``), sample symbol centers, and
       hard-demap. This is what lets a coherent receiver resolve an
       arbitrary channel phase rotation instead of suffering a fixed
       (e.g. 90-degree) bit-flip ambiguity.
    5. Differential mode: sample symbol centers directly (no absolute-phase
       correction needed) and decode via :func:`sc_diff_decode`, seeded with
       the received last preamble symbol (known to have been transmitted as
       ``+1``, see `PREAMBLE_SYMBOLS`) as the initial reference instead of
       :func:`sc_diff_decode`'s default unit reference -- this extends the
       differential decode's phase-ambiguity cancellation to the first
       payload symbol too (which otherwise carries the raw, uncompensated
       channel phase; see :func:`sc_diff_decode`), before demapping.

    Args:
        rx: Received IQ samples (``complex128`` or castable).
        profile: Single-carrier PHY parameters (uses ``sps``).
        bits_per_symbol: ``1`` for BPSK or ``2`` for QPSK.
        differential: If True, decode differentially encoded symbols
            (phase-ambiguity-tolerant); if False, decode coherently using
            the preamble's absolute-phase reference.

    Returns:
        Recovered payload bits as ``uint8``. Empty array if the preamble
        lock confidence is below `SC_SYNC_THRESHOLD`.
    """
    signal = np.asarray(rx, dtype=np.complex128)
    ref = preamble_wave_psk(profile)

    start, peak = sc_frame_sync(signal, ref, search_span=profile.sps * 40)
    if abs(peak) < SC_SYNC_THRESHOLD:
        return np.array([], dtype=np.uint8)

    cfo = sc_estimate_cfo_psk(signal[start : start + len(ref)], profile.sps)
    n = np.arange(signal.size)
    derotated: Complex = signal * np.exp(-1j * 2 * np.pi * cfo * n)

    resync_span = max(2 * profile.sps, 1)
    _, peak2 = sc_frame_sync(derotated[start:], ref, search_span=resync_span)
    payload = derotated[start + len(ref) :]

    if differential:
        centers = payload[profile.sps // 2 :: profile.sps]
        # The last preamble symbol is known to have been transmitted as
        # +1 (see PREAMBLE_SYMBOLS), so the received (channel-affected)
        # sample at its center is exactly the reference `sc_diff_decode`
        # needs to phase-protect the FIRST payload symbol too, instead of
        # leaving it to carry the raw, uncompensated channel phase.
        preamble_centers = derotated[start : start + len(ref)][
            profile.sps // 2 :: profile.sps
        ]
        last_preamble_symbol = (
            complex(preamble_centers[-1])
            if preamble_centers.size > 0
            else complex(1.0, 0.0)
        )
        deltas = sc_diff_decode(centers, initial_ref=last_preamble_symbol)
        bits: Bits = sc_demap_psk(deltas, bits_per_symbol)
        return bits

    aligned = payload * np.exp(-1j * np.angle(peak2))
    centers = aligned[profile.sps // 2 :: profile.sps]
    bits = sc_demap_psk(centers, bits_per_symbol)
    return bits


def sc_demodulate_fsk(rx: Complex, profile: SCProfile, *, gfsk: bool) -> Bits:
    """Non-coherent (G)FSK receiver: preamble sync, discriminator, bit slice.

    Pipeline:

    1. Locate the preamble via :func:`sc_frame_sync`; bail out (empty
       result) if the lock confidence is below `SC_SYNC_THRESHOLD`.
    2. Recover instantaneous frequency from the payload by differentiating
       the unwrapped instantaneous phase (an FM discriminator) -- the same
       technique the synthetic ``_fsk`` modulator uses, so transmit and
       receive agree.
    3. Because FSK/GFSK detection is non-coherent, a fixed zero threshold
       would be biased by any uncompensated carrier frequency offset (CFO)
       or DC-coupled discriminator bias. Instead, threshold adaptively at
       the mean instantaneous frequency over the whole payload: since the
       synth's NRZ symbol stream is (approximately) zero-mean, the payload
       mean frequency estimates exactly that bias/CFO term, and subtracting
       it recenters the two FSK tones symmetrically around zero. Per
       symbol, average frequency over the inner half of the ``sps``-sample
       window (the pulse-shaped GFSK edges are least reliable) and compare
       to the adaptive threshold.
    4. Direction convention matches the synth ``_fsk`` mapping
       (``symbols = 2*bits - 1``): bit ``1`` -> symbol ``+1`` -> higher
       instantaneous frequency, so ``bit = mean_inner_f > threshold``.

    Args:
        rx: Received IQ samples (``complex128`` or castable).
        profile: Single-carrier PHY parameters (uses ``sps``; ``mod_index``
            and ``bt`` only affect the preamble reference waveform used for
            sync, not the discriminator itself).
        gfsk: If True, use the GFSK preamble reference for sync (must match
            the transmitter's pulse shaping); if False, use the plain FSK
            preamble reference.

    Returns:
        Recovered payload bits as ``uint8``. Empty array if the preamble
        lock confidence is below `SC_SYNC_THRESHOLD` or the payload is
        shorter than one symbol.
    """
    signal = np.asarray(rx, dtype=np.complex128)
    ref = preamble_wave_fsk(profile, gfsk=gfsk)

    start, peak = sc_frame_sync(signal, ref, search_span=profile.sps * 40)
    if abs(peak) < SC_SYNC_THRESHOLD:
        return np.array([], dtype=np.uint8)

    payload = signal[start + len(ref) :]
    sps = profile.sps
    n_sym = payload.size // sps
    if n_sym <= 0:
        return np.array([], dtype=np.uint8)

    phase = np.unwrap(np.angle(payload))
    inst_freq = np.diff(phase, prepend=phase[0])
    threshold = float(np.mean(inst_freq))

    bits: Bits = np.zeros(n_sym, dtype=np.uint8)
    for k in range(n_sym):
        seg = inst_freq[k * sps + sps // 4 : k * sps + 3 * sps // 4]
        bits[k] = 1 if float(np.mean(seg)) > threshold else 0
    return bits
