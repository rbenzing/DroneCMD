"""Calibrated AWGN and channel-impairment models for the SP1 validation spine.

``add_awgn_at_snr`` is the correctness landmine of the whole T&E spine
(spec Sec8): signal power MUST be measured as ``mean(|x|**2)`` (power, not
magnitude/amplitude), and the noise variance must be derived from that power
so that the re-measured ("achieved") output SNR matches the requested
``snr_db`` within a small tolerance. Getting either of those wrong (using
magnitude instead of power, or not splitting the noise variance evenly
across the I and Q rails) silently produces a channel that "runs" but is
mis-calibrated, which corrupts every downstream detection/classification
metric that depends on SNR.

All randomness is drawn from the caller-supplied ``numpy.random.Generator``
so runs are reproducible via :func:`validation.repro.rng`; free functions
such as ``np.random.*`` must never be used here.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from validation.types import ChannelParams, IQSamples


def add_awgn_at_snr(
    signal: IQSamples,
    snr_db: float,
    generator: np.random.Generator,
) -> Tuple[IQSamples, float, float]:
    """Add circularly-symmetric complex AWGN calibrated to ``snr_db``.

    Signal power is measured over the whole array as ``mean(|x|**2)``
    (power, not magnitude) -- callers should pass an all-active packet so
    that idle/silence samples don't bias the calibration low. The noise
    power implied by the target SNR is split evenly across the I and Q
    rails (each rail gets variance ``noise_power / 2``, so their sum has
    the correct total noise power), matching the definition of circularly
    symmetric complex Gaussian noise.

    Args:
        signal: Input IQ samples, ``complex64``.
        snr_db: Target signal-to-noise ratio in dB, ``10*log10(S/N)``.
        generator: Seeded ``numpy.random.Generator`` used for all
            randomness (deterministic given the same seed).

    Returns:
        A tuple ``(noisy, noise_std, achieved_snr_db)`` where ``noisy`` is
        ``signal + noise`` as ``complex64``, ``noise_std`` is the per-
        component (I or Q) noise standard deviation used, and
        ``achieved_snr_db`` is the SNR re-measured from the actual
        (finite-sample) noise realization that was added.
    """
    sig_power = float(np.mean(np.abs(signal) ** 2))
    if sig_power <= 0:
        return signal.copy(), 0.0, float("inf")
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    std = float(np.sqrt(noise_power / 2.0))  # per-component (I, Q) std
    noise = (
        generator.standard_normal(len(signal))
        + 1j * generator.standard_normal(len(signal))
    ) * std
    noisy: IQSamples = (signal + noise).astype(np.complex64)
    achieved_noise_power = float(np.mean(np.abs(noise) ** 2))
    achieved = 10 * np.log10(sig_power / achieved_noise_power)
    return noisy, std, float(achieved)


def apply_channel(
    signal: IQSamples,
    params: ChannelParams,
    generator: np.random.Generator,
) -> Tuple[IQSamples, float]:
    """Apply a full channel model: timing offset, multipath, CFO, then AWGN.

    Impairments are applied in a fixed order, each acting on the output of
    the previous stage:

    1. Timing offset -- prepend ``params.timing_offset`` zero samples.
    2. Multipath -- convolve with ``params.multipath_taps`` (an FIR channel
       impulse response), truncated back to the pre-convolution length so
       the signal length is unaffected by this stage.
    3. CFO / Doppler -- multiply by a complex sinusoid combining
       ``params.cfo_hz`` and ``params.doppler_hz`` (both expressed in
       normalized cycles/sample).
    4. Calibrated AWGN -- :func:`add_awgn_at_snr` at ``params.snr_db``.

    Args:
        signal: Input IQ samples, ``complex64``.
        params: Channel parameters (SNR, CFO, Doppler, multipath, timing).
        generator: Seeded ``numpy.random.Generator`` used for all
            randomness (deterministic given the same seed).

    Returns:
        A tuple ``(iq, achieved_snr_db)``: the fully impaired signal and
        the SNR re-measured from the final AWGN stage.
    """
    x: IQSamples = signal.astype(np.complex64)
    if params.timing_offset > 0:
        zeros = np.zeros(params.timing_offset, dtype=np.complex64)
        x = np.concatenate([zeros, x]).astype(np.complex64)
    if params.multipath_taps:
        taps = np.asarray(params.multipath_taps, dtype=np.complex64)
        x = np.convolve(x, taps, mode="full")[: len(x)].astype(np.complex64)
    if params.cfo_hz or params.doppler_hz:
        n = np.arange(len(x))
        rot = np.exp(1j * 2 * np.pi * (params.cfo_hz + params.doppler_hz) * n)
        x = (x * rot).astype(np.complex64)
    noisy, _std, achieved = add_awgn_at_snr(x, params.snr_db, generator)
    return noisy, achieved
