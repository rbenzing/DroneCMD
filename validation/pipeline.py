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


def region_to_bytes(iq_region: IQSamples, sps: int = 8) -> bytes:
    """Demodulate an IQ region to bytes via a simple FSK bit decision.

    This is a self-contained, minimal FSK demodulator: it estimates
    instantaneous frequency from the unwrapped phase and makes a per-symbol
    bit decision from the sign of the mean frequency over the inner half of
    each symbol period. It is not intended to be a high-fidelity
    demodulator; its round-trip behavior is exercised by Task 4's
    modulator/demodulator tests.

    Args:
        iq_region: Complex baseband samples spanning one detected packet.
        sps: Samples per symbol used for bit-boundary alignment.

    Returns:
        Packed bytes (``numpy.packbits``) of the recovered bit stream. Empty
        bytes if the region is shorter than one symbol.
    """
    if len(iq_region) < sps:
        return b""
    phase = np.unwrap(np.angle(iq_region))
    inst_freq = np.diff(phase, prepend=phase[0])
    n_sym = len(iq_region) // sps
    bits = np.zeros(n_sym, dtype=np.uint8)
    for k in range(n_sym):
        seg = inst_freq[k * sps + sps // 4 : k * sps + 3 * sps // 4]
        bits[k] = 1 if float(np.mean(seg)) > 0 else 0
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


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
            used to find candidate packet regions. Defaults to
            :func:`default_detector`.
        threshold: Detection threshold passed through to ``detector``.
        min_gap: Minimum gap (samples) between packets, passed through to
            ``detector``.
        sps: Samples per symbol used by :func:`region_to_bytes` when
            demodulating a region to bytes.
        use_truth_bytes: When ``True``, recover packet bytes from the
            overlapping truth region's ``provenance["payload_hex"]`` instead
            of demodulating the IQ region. Useful for isolating classifier
            accuracy from demodulator quality.
    """

    def __init__(
        self,
        classifier: ClassifierLike,
        detector: DetectorFn = default_detector,
        threshold: float = 0.05,
        min_gap: int = 256,
        sps: int = 8,
        use_truth_bytes: bool = False,
    ) -> None:
        self.classifier = classifier
        self.detector = detector
        self.threshold = threshold
        self.min_gap = min_gap
        self.sps = sps
        self.use_truth_bytes = use_truth_bytes

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
        """Detect and classify all packet regions in ``capture``.

        Args:
            capture: Labeled capture to run detection + classification over.

        Returns:
            One :class:`~validation.types.Detection` per detected region, in
            detection order.
        """
        regions = self.detector(capture.iq, self.threshold, self.min_gap)
        detections: List[Detection] = []
        for start, end in regions:
            if self.use_truth_bytes:
                pkt = self._truth_bytes(capture, start, end) or b""
            else:
                pkt = region_to_bytes(capture.iq[start:end], sps=self.sps)
            result = self.classifier.classify(pkt, None)
            proto, conf = _protocol_and_confidence(result)
            detections.append(
                Detection(
                    start=int(start), end=int(end), protocol=proto, confidence=conf
                )
            )
        return detections
