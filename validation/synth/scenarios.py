"""Protocol x SNR grid -> labeled synthetic captures for the T&E spine.

``build_scenario`` expands a :class:`DatasetSpec` (protocols x SNR grid x
repeats per cell) into a flat list of :class:`~validation.types.LabeledCapture`
objects. Each capture is built as guard-band noise, one modulated-and-AWGN
packet, then more guard-band noise, with ``truth_regions`` marking the exact
sample span of the packet and ``provenance`` carrying enough detail (protocol,
scheme, achieved/requested SNR, payload bytes, seed) to reproduce or audit the
capture later.

All randomness -- payload bytes, the packet's AWGN, and the guard noise --
is drawn from a single :func:`validation.repro.rng` generator seeded from
``spec.seed``, so two calls with the same spec produce bit-identical output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from validation.repro import rng
from validation.synth.channel import add_awgn_at_snr
from validation.synth.modulators import modulate
from validation.types import IQSamples, LabeledCapture, ModScheme


@dataclass
class DatasetSpec:
    """Specification for a synthetic protocol x SNR grid of captures.

    Attributes:
        protocols: Protocol names to generate captures for (outer loop).
        snr_grid_db: Target SNR values in dB (middle loop).
        n_per_cell: Number of captures to generate per (protocol, SNR) cell.
        sample_rate: Sample rate in Hz recorded on each capture.
        seed: Seed for the single generator driving all randomness.
        scheme_by_protocol: Maps each protocol name to its
            :class:`~validation.types.ModScheme`.
        payload_len: Number of random payload bytes to modulate per packet.
        guard: Number of noise-only samples placed before and after the
            packet.
        sps: Samples per symbol passed through to :func:`modulate`.
        differential: If True, differentially encode BPSK/QPSK payloads
            (passed through to :func:`modulate` and recorded in each
            capture's ``provenance["differential"]``). Ignored by FSK/GFSK/
            OFDM schemes.
        pilot_spacing: If > 0, interleave known pilots into coherent BPSK/QPSK
            payloads (passed through to :func:`modulate` and recorded in each
            capture's ``provenance["pilot_spacing"]``) for pilot-aided phase
            tracking. Ignored by differential PSK and FSK/GFSK/OFDM.
    """

    protocols: List[str]
    snr_grid_db: List[float]
    n_per_cell: int
    sample_rate: float
    seed: int
    scheme_by_protocol: Dict[str, ModScheme]
    payload_len: int = 32
    guard: int = 512
    sps: int = 8
    differential: bool = False
    pilot_spacing: int = 0


def _noise(n: int, std: float, generator: np.random.Generator) -> IQSamples:
    """Generate ``n`` complex64 AWGN samples with per-rail std ``std``."""
    if std <= 0 or n <= 0:
        return np.zeros(max(n, 0), dtype=np.complex64)
    z = generator.standard_normal(n) + 1j * generator.standard_normal(n)
    return (z * std).astype(np.complex64)


def build_scenario(spec: DatasetSpec) -> List[LabeledCapture]:
    """Build labeled captures over the protocol x SNR grid in ``spec``.

    For every (protocol, SNR) cell, ``spec.n_per_cell`` captures are
    generated: a random payload is modulated with the protocol's scheme,
    calibrated AWGN is added at the target SNR, and the result is padded
    with ``spec.guard`` noise samples (matching the packet's noise std) on
    each side. ``truth_regions`` records the exact ``[start, end)`` sample
    span of the packet within the capture.

    Args:
        spec: The dataset specification.

    Returns:
        A list of length
        ``len(protocols) * len(snr_grid_db) * n_per_cell`` of
        :class:`~validation.types.LabeledCapture`.
    """
    g = rng(spec.seed)
    captures: List[LabeledCapture] = []
    for proto in spec.protocols:
        scheme = spec.scheme_by_protocol[proto]
        for snr in spec.snr_grid_db:
            for _ in range(spec.n_per_cell):
                payload = g.integers(
                    0, 256, size=spec.payload_len, dtype=np.uint8
                ).tobytes()
                clean = modulate(
                    payload,
                    scheme,
                    sps=spec.sps,
                    differential=spec.differential,
                    pilot_spacing=spec.pilot_spacing,
                )
                noisy_pkt, noise_std, achieved = add_awgn_at_snr(clean, snr, g)
                pre = _noise(spec.guard, noise_std, g)
                post = _noise(spec.guard, noise_std, g)
                iq = np.concatenate([pre, noisy_pkt, post]).astype(np.complex64)
                start = spec.guard
                end = spec.guard + len(noisy_pkt)
                captures.append(
                    LabeledCapture(
                        iq=iq,
                        sample_rate=spec.sample_rate,
                        truth_regions=[(start, end, proto)],
                        provenance={
                            "source": "synth",
                            "protocol": proto,
                            "scheme": scheme.value,
                            "snr_db": float(achieved),
                            "requested_snr_db": float(snr),
                            "payload_hex": payload.hex(),
                            "seed": spec.seed,
                            "differential": spec.differential,
                            "pilot_spacing": spec.pilot_spacing,
                        },
                    )
                )
    return captures
