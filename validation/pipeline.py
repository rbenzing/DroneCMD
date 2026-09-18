"""Injectable detect-then-classify pipeline for validation & T&E.

``DetectClassifyPipeline`` glues an energy-based packet detector to a
protocol classifier, with both stages injected as dependencies. This makes
the pipeline testable with stub detectors/classifiers and requires no
trained models or SDR hardware.

The detector defaults to :func:`core.signal_processing.detect_packets`
wrapped by :func:`default_detector`. The classifier is duck-typed: it must
expose a ``.classify(packet_bytes, signal_metrics=None)`` method returning
either a bare protocol name (``str``, treated as confidence 1.0) or a
``ClassificationResult``-like object exposing ``.predicted_protocol`` and
``.confidence`` (accessed via ``getattr`` so this module never imports
``core.classification.ClassificationResult`` directly).

Detection is scheme-aware, driven by a capture's ``provenance["scheme"]``
(case-insensitive): ``"ofdm"`` selects :func:`default_ofdm_detector` (a
moving-average power envelope tuned to OFDM's high PAPR) instead of the
injected ``detector``. Decode is blind: each detected region is classified
independently by :func:`core.blind.classify_family` into OFDM or
single-carrier (provenance is not consulted), and routed to
:func:`ofdm_region_to_bytes` (blindly resolves the OFDM profile via
:func:`core.blind.resolve_ofdm_profile` and demodulates with the matching
:func:`core.ofdm.demodulate_ofdm` receiver) or
:func:`single_carrier_region_to_bytes` (blindly resolves the single-carrier
profile via :func:`core.blind.resolve_sc_profile` and demodulates with the
matching :mod:`core.single_carrier` receiver) respectively. The old
self-contained inline FSK slicer (formerly ``region_to_bytes``) has been
retired in favor of the core engine.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

from core.signal_processing import detect_packets
from validation.types import Detection, IQSamples, LabeledCapture

DetectorFn = Callable[[IQSamples, float, int], List[Tuple[int, int]]]


class ClassifierLike(Protocol):
    """Structural type for the duck-typed classifier this pipeline accepts.

    Any object with a matching ``classify`` method satisfies this protocol
    (e.g. a trained ``EnhancedProtocolClassifier`` or a test stub) without
    this module importing ``core.classification`` at all.
    """

    def classify(
        self, packet_bytes: bytes, signal_metrics: Optional[Dict[str, float]] = None
    ) -> Union[str, object]:
        ...


def default_detector(
    iq: IQSamples, threshold: float, min_gap: int
) -> List[Tuple[int, int]]:
    """Wrap :func:`core.signal_processing.detect_packets` as a ``DetectorFn``."""
    return detect_packets(iq, threshold=threshold, min_gap=min_gap)


def default_ofdm_detector(
    iq: IQSamples, threshold: float, min_gap: int
) -> List[Tuple[int, int]]:
    """Detect OFDM bursts via a smoothed (moving-average) power envelope.

    OFDM is intrinsically high-PAPR, so the per-sample amplitude threshold in
    :func:`core.signal_processing.detect_packets` fragments a burst into many
    short sub-threshold pieces. Averaging instantaneous power over one OFDM
    symbol fills those nulls, so the burst thresholds as a single region.

    Args:
        iq: Complex baseband samples.
        threshold: Detection threshold; values < 1.0 are a fraction of the
            peak smoothed power.
        min_gap: Minimum region length in samples (shorter regions dropped).

    Returns:
        ``(start, end)`` regions of OFDM activity.
    """
    from core.ofdm import DEFAULT_OFDM_PROFILE

    if len(iq) == 0:
        return []
    power = np.abs(iq).astype(np.float64) ** 2
    window = DEFAULT_OFDM_PROFILE.symbol_len
    if window > 1 and len(power) >= window:
        kernel = np.ones(window, dtype=np.float64) / window
        power = np.convolve(power, kernel, mode="same")
    thr = threshold * float(np.max(power)) if threshold < 1.0 else threshold
    active = power > thr
    edges = np.diff(active.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if len(ends) > 0 and (len(starts) == 0 or starts[0] > ends[0]):
        starts = np.insert(starts, 0, 0)
    if len(starts) > 0 and (len(ends) == 0 or ends[-1] < starts[-1]):
        ends = np.append(ends, len(active))
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s > min_gap]


def single_carrier_region_to_bytes(
    iq_region: IQSamples,
    sample_rate: float,
    *,
    differential: bool = False,
    pilot_spacing: int = 0,
) -> Tuple[bytes, Optional[str]]:
    """Blindly resolve and demodulate a single-carrier region to bytes.

    Runs :func:`core.blind.resolve_sc_profile` to infer the profile (sps +
    modulation, BPSK/QPSK disambiguated) from lock confidence -- no scheme/sps
    hint is taken from the caller. On a lock, demodulates with the resolved
    profile via the shared PH receivers (``sc_demodulate_psk``/
    ``sc_demodulate_fsk``). Only the *profile* is blind; ``differential`` and
    ``pilot_spacing`` are caller-provided payload knobs (not blindly
    detectable; the P2-SC catalog is coherent + pilotless so both default off).

    Args:
        iq_region: Complex baseband samples spanning one detected packet,
            including its Barker preamble near the start.
        sample_rate: Capture sample rate in Hz. Reserved (the blind path
            decodes against the resolved profile directly); kept for signature
            stability and future per-sample-rate profile scaling.
        differential: Passed to the coherent-PSK receiver when the resolved
            profile is PSK; ignored for FSK/GFSK.
        pilot_spacing: Passed to the coherent-PSK receiver; > 0 selects
            pilot-aided tracking (the capture must have used the same spacing).

    Returns:
        ``(packed_bytes, resolved_profile_name)`` on a lock, or ``(b"", None)``
        if the region is empty or resolution/demod did not lock.
    """
    if len(iq_region) == 0:
        return b"", None
    from core.blind import resolve_sc_profile
    from core.single_carrier import sc_demodulate_fsk, sc_demodulate_psk

    iq_c128 = iq_region.astype(np.complex128)
    spec, _conf = resolve_sc_profile(iq_c128)
    if spec is None:
        return b"", None
    if spec.is_fsk:
        bits = sc_demodulate_fsk(iq_c128, spec.profile, gfsk=spec.gfsk)
    else:
        bits = sc_demodulate_psk(
            iq_c128,
            spec.profile,
            bits_per_symbol=spec.bits_per_symbol,
            differential=differential,
            pilot_spacing=pilot_spacing,
        )
    if len(bits) == 0:
        return b"", None
    return np.packbits(bits.astype(np.uint8)).tobytes(), spec.name


def ofdm_region_to_bytes(iq_region: IQSamples) -> Tuple[bytes, Optional[str]]:
    """Blindly resolve the OFDM profile of a region, then demodulate with it.

    Mirrors :func:`single_carrier_region_to_bytes`: resolution and decode use
    the ``core`` PHY directly (not the fixed-profile production
    :class:`core.demodulation.OFDMDemodulator`). Returns ``(packed_bytes,
    resolved_name)`` on a lock, or ``(b"", None)`` on an empty region or when
    the blind resolver reports no trustworthy OFDM lock.

    Args:
        iq_region: Complex baseband samples spanning one detected OFDM
            packet, including its STF+LTF preamble near the start.

    Returns:
        ``(packed_bytes, resolved_profile_name)`` on a lock, or ``(b"",
        None)`` if the region is empty, the blind resolver did not lock, or
        the demod yielded no bits.
    """
    if len(iq_region) == 0:
        return b"", None
    from core.blind import resolve_ofdm_profile
    from core.ofdm import demodulate_ofdm
    from core.profiles import OFDM_CATALOG

    iq_c128 = iq_region.astype(np.complex128)
    name, _ = resolve_ofdm_profile(iq_c128)
    if name is None:
        return b"", None
    bits = demodulate_ofdm(iq_c128, OFDM_CATALOG[name])
    if len(bits) == 0:
        return b"", None
    return np.packbits(bits.astype(np.uint8)).tobytes(), name


def _protocol_and_confidence(result: Union[str, object]) -> Tuple[str, float]:
    """Normalize a classifier result (``str`` or ``ClassificationResult``-like)."""
    proto = getattr(result, "predicted_protocol", result)
    conf = float(getattr(result, "confidence", 1.0))
    return str(proto), conf


class DetectClassifyPipeline:
    """Detect packet regions in a capture, then classify each region.

    Both the detector and classifier are injected dependencies, so the
    pipeline can be exercised in unit tests with stubs — no trained models
    or SDR hardware required.

    Args:
        classifier: Duck-typed classifier exposing
            ``.classify(packet_bytes, signal_metrics=None)``, returning
            either a bare protocol name (``str``) or a
            ``ClassificationResult``-like object with
            ``.predicted_protocol``/``.confidence``.
        detector: Callable ``(iq, threshold, min_gap) -> [(start, end), ...]``
            used to find candidate packet regions for non-OFDM captures
            (``provenance["scheme"]`` unset or not ``"ofdm"``). Defaults to
            :func:`default_detector`.
        threshold: Detection threshold passed through to whichever detector
            is selected.
        min_gap: Minimum detected-region length in samples, passed through
            to whichever detector is selected; regions shorter than this are
            dropped (this filters out short spurious regions -- it does not
            bridge gaps between packets).
        sps: Unused by decode -- single-carrier profile resolution is now
            blind (:func:`single_carrier_region_to_bytes` infers sps itself
            via :func:`core.blind.resolve_sc_profile`). Retained for
            constructor compatibility.
        use_truth_bytes: When ``True``, recover packet bytes from the
            overlapping truth region's ``provenance["payload_hex"]`` instead
            of demodulating the IQ region. Useful for isolating classifier
            accuracy from demodulator quality.
        ofdm_detector: Detector used instead of ``detector`` when a
            capture's ``provenance["scheme"]`` is ``"ofdm"``. Defaults to
            :func:`default_ofdm_detector`.
    """

    def __init__(
        self,
        classifier: ClassifierLike,
        detector: DetectorFn = default_detector,
        threshold: float = 0.05,
        min_gap: int = 256,
        sps: int = 8,
        use_truth_bytes: bool = False,
        ofdm_detector: DetectorFn = default_ofdm_detector,
    ) -> None:
        self.classifier = classifier
        self.detector = detector
        self.threshold = threshold
        self.min_gap = min_gap
        self.sps = sps
        self.use_truth_bytes = use_truth_bytes
        self.ofdm_detector = ofdm_detector

    def _truth_bytes(
        self, capture: LabeledCapture, start: int, end: int
    ) -> Optional[bytes]:
        """Return payload bytes from an overlapping truth region, if any."""
        for ts, te, _proto in capture.truth_regions or []:
            if not (end <= ts or start >= te):  # overlap
                hexstr = capture.provenance.get("payload_hex")
                return bytes.fromhex(hexstr) if hexstr else None
        return None

    def run(self, capture: LabeledCapture) -> List[Detection]:
        """Detect packet regions, then blindly resolve + classify each.

        Detector selection still keys on ``provenance["scheme"]`` (OFDM
        envelope vs energy detector -- detection is scored separately against
        truth). Decode routing is blind: each region is classified by
        :func:`core.blind.classify_family` into OFDM or single-carrier and the
        resolved profile is recorded on the :class:`Detection`. ``differential``
        /``pilot_spacing`` are read from provenance as caller payload knobs.
        """
        from core.blind import classify_family
        from core.profiles import Family

        scheme_hint = str(capture.provenance.get("scheme", "")).lower()
        detector = self.ofdm_detector if scheme_hint == "ofdm" else self.detector
        regions = detector(capture.iq, self.threshold, self.min_gap)
        differential = bool(capture.provenance.get("differential", False))
        pilot_spacing = int(capture.provenance.get("pilot_spacing", 0))
        detections: List[Detection] = []
        for start, end in regions:
            region = capture.iq[start:end]
            resolved: Optional[str] = None
            if self.use_truth_bytes:
                pkt = self._truth_bytes(capture, start, end) or b""
            else:
                family, _ = classify_family(region.astype(np.complex128))
                if family == Family.OFDM:
                    pkt, resolved = ofdm_region_to_bytes(region)
                else:
                    pkt, resolved = single_carrier_region_to_bytes(
                        region,
                        capture.sample_rate,
                        differential=differential,
                        pilot_spacing=pilot_spacing,
                    )
            result = self.classifier.classify(pkt, None)
            proto, conf = _protocol_and_confidence(result)
            detections.append(
                Detection(
                    start=int(start),
                    end=int(end),
                    protocol=proto,
                    confidence=conf,
                    resolved_profile=resolved,
                )
            )
        return detections
