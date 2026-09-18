"""Shared OFDM PHY primitives for DroneCMD (single source of truth).

The synthetic modulator (``validation.synth.modulators``) and the production
demodulator (``core.demodulation.OFDMDemodulator``) both import this module so
the transmit and receive sides of the OFDM waveform cannot drift apart.

The profile is a fixed 802.11a-style OFDM: a 64-point FFT with a 16-sample
cyclic prefix, 52 occupied subcarriers (48 QPSK data + 4 known pilots), a
Schmidl & Cox timing/CFO preamble (STF) whose two identical time-domain halves
give coarse timing and fractional CFO, and a fully-known long-training symbol
(LTF) used for least-squares per-subcarrier channel estimation.

References:
    T. M. Schmidl and D. C. Cox, "Robust frequency and timing synchronization
    for OFDM," IEEE Trans. Commun., vol. 45, no. 12, pp. 1613-1621, 1997.
    IEEE Std 802.11a-1999, OFDM PHY (subcarrier / pilot layout).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt

Complex = npt.NDArray[np.complex128]
Bits = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class OFDMProfile:
    """Immutable OFDM PHY parameters shared by the modulator and demodulator.

    Attributes:
        fft_size: IFFT/FFT length ``N``.
        cp_len: Cyclic-prefix length in samples.
        data_carriers: Subcarrier indices (may be negative) carrying QPSK data.
        pilot_carriers: Subcarrier indices carrying known pilots.
        pilot_values: Known pilot symbols, aligned to ``pilot_carriers``.
    """

    fft_size: int
    cp_len: int
    data_carriers: Tuple[int, ...]
    pilot_carriers: Tuple[int, ...]
    pilot_values: Tuple[complex, ...]

    @property
    def symbol_len(self) -> int:
        """Samples per OFDM symbol including the cyclic prefix (``N + CP``)."""
        return self.fft_size + self.cp_len

    @property
    def occupied_carriers(self) -> Tuple[int, ...]:
        """All non-null subcarrier indices (data + pilots), ascending."""
        return tuple(sorted(self.data_carriers + self.pilot_carriers))

    @property
    def n_data_bits_per_symbol(self) -> int:
        """QPSK data bits carried by one OFDM symbol (2 per data subcarrier)."""
        return 2 * len(self.data_carriers)


def _build_default_profile() -> OFDMProfile:
    pilots: Tuple[int, ...] = (-21, -7, 7, 21)
    occupied = [k for k in range(-26, 27) if k != 0]
    data = tuple(k for k in occupied if k not in pilots)
    pilot_values: Tuple[complex, ...] = (1 + 0j, 1 + 0j, 1 + 0j, 1 + 0j)
    return OFDMProfile(
        fft_size=64,
        cp_len=16,
        data_carriers=data,
        pilot_carriers=pilots,
        pilot_values=pilot_values,
    )


DEFAULT_OFDM_PROFILE = _build_default_profile()

# The normalized Schmidl & Cox metric (see `_sc_metric`) lives in [0, 1] and
# is ~1 at a true STF lock. A peak below this threshold means the coarse-
# timing search found no reliable STF -- either noise, or a spurious
# correlation elsewhere in the payload -- and the recovered bits should not
# be trusted.
#
# 0.5 alone is unsafe under blind routing: a low-SNR single-carrier region
# (FSK/GFSK) blindly misrouted to the OFDM decoder by
# `core.blind.classify_family` can score above 0.5 on this S&C metric, so 0.5
# would let it decode garbage. 0.8 (a prior attempt) needlessly rejects
# genuine OFDM below ~10 dB. 0.6 is a data-driven midpoint: it fails closed on
# the common misroute cases while preserving genuine OFDM sync to ~6-8 dB
# (OFDM S&C ~0.64 at 6 dB, ~0.75 at 8 dB). NOTE: at very low SNR (<=~5 dB) the
# misrouted-single-carrier and genuine-OFDM S&C distributions OVERLAP (~0.55-
# 0.65), so no single scalar threshold is airtight -- a small residual tail
# (~0.1% of misrouted low-SNR regions) can still leak garbage. That is inherent
# to blind family discrimination at the noise floor (where the whole chain is
# unreliable) and is measured by the profile-ID/BER metrics, not a hard
# guarantee. The loud-failure guarantee holds at normal operating SNR.
OFDM_SYNC_THRESHOLD = 0.6


def qpsk_map(bits: Bits) -> Complex:
    """Map an even-length bit array to unit-energy QPSK symbols.

    Per rail: bit 0 -> +1, bit 1 -> -1, scaled by ``1/sqrt(2)`` (matches the
    single-carrier QPSK convention in ``validation.synth.modulators``). Even
    bits are the in-phase rail, odd bits the quadrature rail.
    """
    pairs = np.asarray(bits, dtype=np.uint8).reshape(-1, 2)
    i = 1.0 - 2.0 * pairs[:, 0].astype(np.float64)
    q = 1.0 - 2.0 * pairs[:, 1].astype(np.float64)
    out: Complex = ((i + 1j * q) / np.sqrt(2.0)).astype(np.complex128)
    return out


def qpsk_demap(symbols: Complex) -> Bits:
    """Hard-decision inverse of :func:`qpsk_map` (sign per rail)."""
    sym = np.asarray(symbols, dtype=np.complex128)
    out = np.empty(2 * sym.size, dtype=np.uint8)
    out[0::2] = (sym.real < 0).astype(np.uint8)
    out[1::2] = (sym.imag < 0).astype(np.uint8)
    return out


def _bins(profile: OFDMProfile, carriers: Tuple[int, ...]) -> npt.NDArray[np.intp]:
    """Map (possibly negative) subcarrier indices to FFT bins ``k mod N``."""
    n = profile.fft_size
    return np.array([k % n for k in carriers], dtype=np.intp)


def _symbol_time(profile: OFDMProfile, spec: Complex) -> Complex:
    """IFFT a length-N spectrum and prepend the cyclic prefix."""
    body = np.fft.ifft(spec, n=profile.fft_size)
    cp = body[profile.fft_size - profile.cp_len :]
    out: Complex = np.concatenate([cp, body]).astype(np.complex128)
    return out


def _stf_freq(profile: OFDMProfile) -> Complex:
    """Schmidl & Cox training spectrum: only even occupied bins are excited.

    Populating only even-index bins makes the IFFT output periodic with period
    ``N/2``, i.e. two identical time-domain halves — the S&C timing property.
    """
    n = profile.fft_size
    spec = np.zeros(n, dtype=np.complex128)
    even_occ = tuple(k for k in profile.occupied_carriers if k % 2 == 0)
    # Deterministic +-1 PN (index-derived); scaled so total power ~ full symbol.
    pn = np.array(
        [1.0 if ((k // 2) % 2 == 0) else -1.0 for k in even_occ], dtype=np.complex128
    )
    spec[_bins(profile, even_occ)] = pn * np.sqrt(2.0)
    return spec


def _ltf_freq(profile: OFDMProfile) -> Tuple[Complex, Complex]:
    """Long-training spectrum on all occupied bins + the known sequence.

    Returns ``(spec, known_seq)`` where ``known_seq`` is aligned to
    ``occupied_carriers`` order for least-squares channel estimation.
    """
    n = profile.fft_size
    occ = profile.occupied_carriers
    seq = np.array(
        [
            (1.0 if (i % 2 == 0) else -1.0) + 1j * (1.0 if (i % 3 == 0) else -1.0)
            for i in range(len(occ))
        ],
        dtype=np.complex128,
    ) / np.sqrt(2.0)
    spec = np.zeros(n, dtype=np.complex128)
    spec[_bins(profile, occ)] = seq
    return spec, seq


def _data_symbol_time(profile: OFDMProfile, data_symbols: Complex) -> Complex:
    """Assemble one OFDM data symbol (data + pilots) into the time domain."""
    n = profile.fft_size
    spec = np.zeros(n, dtype=np.complex128)
    spec[_bins(profile, profile.data_carriers)] = data_symbols
    spec[_bins(profile, profile.pilot_carriers)] = np.asarray(
        profile.pilot_values, dtype=np.complex128
    )
    return _symbol_time(profile, spec)


def modulate_ofdm(bits: Bits, profile: OFDMProfile = DEFAULT_OFDM_PROFILE) -> Complex:
    """Build an OFDM burst: STF + LTF + one or more QPSK data symbols.

    ``bits`` is zero-padded to a multiple of ``profile.n_data_bits_per_symbol``.
    The result is not power-normalized (the synth modulator normalizes).
    """
    nbits = profile.n_data_bits_per_symbol
    b = np.asarray(bits, dtype=np.uint8)
    pad = (-len(b)) % nbits
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    parts = [
        _symbol_time(profile, _stf_freq(profile)),
        _symbol_time(profile, _ltf_freq(profile)[0]),
    ]
    for start in range(0, len(b), nbits):
        parts.append(_data_symbol_time(profile, qpsk_map(b[start : start + nbits])))
    out: Complex = np.concatenate(parts).astype(np.complex128)
    return out


def _sc_metric(rx: Complex, half: int) -> Tuple[npt.NDArray[np.float64], Complex]:
    """Schmidl & Cox timing metric M(d) and the correlation P(d).

    ``P(d) = sum_m conj(r[d+m]) r[d+m+L]``, ``Ra(d) = sum_m |r[d+m]|^2``,
    ``Rb(d) = sum_m |r[d+m+L]|^2``, ``M(d) = |P(d)|^2 / (Ra(d) * Rb(d))`` for
    ``L = half``.

    Normalizing by the product of *both* half-window energies (rather than
    the trailing window's energy alone, as in the plain textbook form) keeps
    ``M(d)`` Cauchy-Schwarz bounded to ``[0, 1]`` -- ``|P(d)| <=
    sqrt(Ra(d) * Rb(d))`` always -- with equality only where the two halves
    are truly proportional (the genuine STF repetition). Without this, a
    trailing window that happens to have anomalously low energy (e.g. at a
    preamble/payload boundary) can drive the plain metric arbitrarily far
    above its nominal peak of 1, creating spurious maxima that defeat a
    global ``argmax`` search for coarse timing.
    """
    dmax = len(rx) - 2 * half
    if dmax <= 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.complex128)
    p = np.zeros(dmax, dtype=np.complex128)
    ra = np.zeros(dmax, dtype=np.float64)
    rb = np.zeros(dmax, dtype=np.float64)
    for d in range(dmax):
        a = rx[d : d + half]
        b = rx[d + half : d + 2 * half]
        p[d] = np.sum(np.conj(a) * b)
        ra[d] = float(np.sum(np.abs(a) ** 2))
        rb[d] = float(np.sum(np.abs(b) ** 2))
    metric = (np.abs(p) ** 2) / (ra * rb + 1e-12)
    return metric, p


def ofdm_sync_confidence(
    rx: Complex, profile: OFDMProfile = DEFAULT_OFDM_PROFILE
) -> float:
    """Report Schmidl & Cox lock confidence for a candidate OFDM region.

    Computes the same normalized S&C metric, over the same bounded search
    window, as the coarse-timing step of :func:`demodulate_ofdm`. Callers
    (notably :class:`core.demodulation.OFDMDemodulator`) use this to gate on
    sync quality *before* trusting bits from :func:`demodulate_ofdm`, since
    that function always returns its best-effort decode even when the STF
    was never actually found.

    Args:
        rx: Complex baseband samples, expected to begin at or near the STF.
        profile: OFDM PHY profile (subcarrier/CP layout) to search against.

    Returns:
        The peak normalized S&C metric in ``[0, 1]`` over the first
        ``profile.symbol_len`` samples (~1.0 at a true STF lock, near 0 for
        noise or a mis-aligned region). ``0.0`` if ``rx`` is too short for
        the metric to be computed at all.
    """
    half = profile.fft_size // 2
    x = np.asarray(rx, dtype=np.complex128)
    metric, _ = _sc_metric(x, half)
    if metric.size == 0:
        return 0.0
    search_span = min(len(metric), profile.symbol_len)
    return float(np.max(metric[:search_span]))


def ofdm_equalized_symbols(
    rx: Complex, profile: OFDMProfile = DEFAULT_OFDM_PROFILE
) -> Complex:
    """Equalized data subcarriers for every data symbol, concatenated.

    Runs the same Schmidl & Cox coarse timing + fractional-CFO correction,
    LTF least-squares channel estimation, and per-symbol one-tap equalization
    with pilot common-phase-error correction as :func:`demodulate_ofdm`, but
    returns the equalized data-subcarrier symbols (the pre-demap signal, in
    subcarrier order, all data symbols concatenated) rather than demapped bits.
    Empty when ``rx`` is shorter than the STF+LTF preamble or yields no data
    symbol. Used by the blind OFDM resolver (:mod:`core.blind`) to score
    trial-demodulation quality; :func:`demodulate_ofdm` demaps its output.
    """
    n = profile.fft_size
    slen = profile.symbol_len
    half = n // 2
    x = np.asarray(rx, dtype=np.complex128)
    if len(x) < 2 * slen:
        return np.zeros(0, dtype=np.complex128)
    metric, p = _sc_metric(x, half)
    if metric.size == 0:
        return np.zeros(0, dtype=np.complex128)
    search_span = min(len(metric), slen)
    d_body = int(np.argmax(metric[:search_span]))
    df = float(np.angle(p[d_body])) / (2.0 * np.pi * half)
    x = x * np.exp(-1j * 2.0 * np.pi * df * np.arange(len(x)))
    occ_bins = _bins(profile, profile.occupied_carriers)
    _, ltf_known = _ltf_freq(profile)
    ltf_start = d_body + slen
    if ltf_start + n > len(x):
        return np.zeros(0, dtype=np.complex128)
    y_ltf = np.fft.fft(x[ltf_start : ltf_start + n], n)
    h = np.ones(n, dtype=np.complex128)
    h[occ_bins] = y_ltf[occ_bins] / ltf_known
    data_bins = _bins(profile, profile.data_carriers)
    pilot_bins = _bins(profile, profile.pilot_carriers)
    pilot_vals = np.asarray(profile.pilot_values, dtype=np.complex128)
    syms = []
    i = 0
    while True:
        b0 = d_body + 2 * slen + i * slen
        if b0 + n > len(x):
            break
        y = np.fft.fft(x[b0 : b0 + n], n)
        pilots_eq = y[pilot_bins] / h[pilot_bins]
        cpe = float(np.angle(np.sum(pilots_eq * np.conj(pilot_vals))))
        data_eq = (y[data_bins] / h[data_bins]) * np.exp(-1j * cpe)
        syms.append(data_eq)
        i += 1
    if not syms:
        return np.zeros(0, dtype=np.complex128)
    out: Complex = np.concatenate(syms).astype(np.complex128)
    return out


def demodulate_ofdm(rx: Complex, profile: OFDMProfile = DEFAULT_OFDM_PROFILE) -> Bits:
    """Recover data bits from an OFDM burst with a full S&C receiver.

    Pipeline: Schmidl & Cox coarse timing + fractional-CFO correction, LS
    channel estimation from the LTF, then per-symbol one-tap equalization with
    pilot-based common-phase-error correction and QPSK demapping.

    Returns an unpacked ``uint8`` bit array (empty if the burst is shorter than
    the STF + LTF preamble).

    Note:
        This function always returns its best-effort decode, even when the
        STF was never actually found within the bounded coarse-timing search
        (see the loop below) -- it does not itself signal a sync failure.
        Callers that cannot guarantee ``rx`` starts at or near the STF (e.g.
        a detector-supplied region) should gate on
        :func:`ofdm_sync_confidence` first; see
        :class:`core.demodulation.OFDMDemodulator`.
    """
    syms = ofdm_equalized_symbols(rx, profile)
    if syms.size == 0:
        return np.zeros(0, dtype=np.uint8)
    return qpsk_demap(syms).astype(np.uint8)
