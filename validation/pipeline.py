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

Both detection and region-to-bytes demodulation are scheme-aware, driven by
a capture's ``provenance["scheme"]`` (case-insensitive). ``"ofdm"`` uses
:func:`default_ofdm_detector` (a moving-average power envelope tuned to
OFDM's high PAPR) instead of the injected ``detector``, and each detected
region is routed through :func:`ofdm_region_to_bytes`, which lazily uses the
core OFDM receiver (:class:`core.demodulation.OFDMDemodulator`). Every other
scheme -- FSK/GFSK/BPSK/QPSK, and unset/unrecognized (defaulted to FSK) --
uses the injected ``detector`` and routes each region through
:func:`single_carrier_region_to_bytes`, which lazily uses the core
preamble-driven single-carrier receivers
(:class:`core.demodulation.FSKDemodulator` /
:class:`core.demodulation.PSKDemodulator`) via
:class:`core.demodulation.DemodulationEngine`. The old self-contained inline
FSK slicer (formerly ``region_to_bytes``) has been retired in favor of the
core engine.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

from core.signal_processing import detect_packets
from validation.types import Detection, IQSamples, LabeledCapture, ModScheme

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
    scheme: ModScheme,
    sample_rate: float,
    sps: int = 8,
    differential: bool = False,
) -> bytes:
    """Demodulate a single-carrier IQ region to bytes via the core engine.

    Lazily imports :mod:`core.demodulation` so the pipeline import path
    stays light, mirroring :func:`ofdm_region_to_bytes`. Maps
    :class:`~validation.types.ModScheme` (``FSK``/``GFSK``/``BPSK``/
    ``QPSK``) to :class:`core.demodulation.ModulationScheme` (defaulting to
    ``FSK`` for an unrecognized scheme) and runs the matching core receiver
    (:class:`~core.demodulation.FSKDemodulator` for FSK/GFSK,
    :class:`~core.demodulation.PSKDemodulator` for BPSK/QPSK) via
    :class:`~core.demodulation.DemodulationEngine`. Both receivers are
    preamble-driven and internally gate on lock confidence -- see
    :mod:`core.single_carrier`.

    Args:
        iq_region: Complex baseband samples spanning one detected packet,
            including its Barker preamble near the start.
        scheme: Single-carrier modulation scheme. OFDM is not handled here
            -- see :func:`ofdm_region_to_bytes`.
        sample_rate: Capture sample rate in Hz. Used only to build a
            ``DemodConfig`` that satisfies its own ``__post_init__``
            validity check -- the underlying single-carrier receivers
            always demodulate against
            ``core.single_carrier.DEFAULT_SC_PROFILE`` (``sps=8``)
            regardless of this value or of ``sps``/``differential`` below.
        sps: Samples per symbol, used only to derive a placeholder
            ``bitrate_bps`` for ``DemodConfig`` (see ``sample_rate`` above).
        differential: Forwarded to ``DemodConfig.differential``; consulted
            only by the BPSK/QPSK receiver, ignored for FSK/GFSK.

    Returns:
        Packed bytes (``numpy.packbits``) of the recovered (unpacked) bit
        stream, or ``b""`` if the region is empty or demodulation failed
        (preamble sync failure or zero bits recovered).
    """
    if len(iq_region) == 0:
        return b""
    from core.demodulation import DemodConfig, DemodulationEngine
    from core.demodulation import ModulationScheme as CoreModulationScheme

    scheme_map: Dict[ModScheme, CoreModulationScheme] = {
        ModScheme.FSK: CoreModulationScheme.FSK,
        ModScheme.GFSK: CoreModulationScheme.GFSK,
        ModScheme.BPSK: CoreModulationScheme.BPSK,
        ModScheme.QPSK: CoreModulationScheme.QPSK,
    }
    core_scheme = scheme_map.get(scheme, CoreModulationScheme.FSK)
    bits_per_symbol = core_scheme.bits_per_symbol
    bitrate_bps = max(1.0, sample_rate / sps * bits_per_symbol)
    if bitrate_bps >= sample_rate / 2:
        bitrate_bps = sample_rate / 4
    cfg = DemodConfig(
        scheme=core_scheme,
        sample_rate_hz=sample_rate,
        bitrate_bps=bitrate_bps,
        differential=differential,
    )
    result = DemodulationEngine(cfg).demodulate(iq_region.astype(np.complex64))
    if not result.is_valid or len(result.bits) == 0:
        return b""
    return np.packbits(result.bits.astype(np.uint8)).tobytes()


def ofdm_region_to_bytes(iq_region: IQSamples) -> bytes:
    """Demodulate an OFDM region to bytes via the core OFDM receiver.

    Lazily imports :mod:`core.demodulation` so the pipeline import path stays
    light. Returns ``b""`` on an empty region or an invalid demod result.

    Args:
        iq_region: Complex baseband samples spanning one detected OFDM
            packet, including its STF+LTF preamble near the start.

    Returns:
        Packed bytes (``numpy.packbits``) of the recovered bit stream, or
        ``b""`` if the region is empty or the OFDM demod could not recover a
        valid burst.
    """
    if len(iq_region) == 0:
        return b""
    from core.demodulation import DemodConfig, ModulationScheme, OFDMDemodulator

    cfg = DemodConfig(scheme=ModulationScheme.OFDM)
    result = OFDMDemodulator(cfg).demodulate(iq_region.astype(np.complex64))
    if not result.is_valid or len(result.bits) == 0:
        return b""
    return np.packbits(result.bits.astype(np.uint8)).tobytes()


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
        sps: Samples per symbol passed through to
            :func:`single_carrier_region_to_bytes` when demodulating a
            non-OFDM region to bytes (see that function's docstring -- the
            core single-carrier receivers use their own fixed profile
            regardless of this value).
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
        """Detect and classify all packet regions in ``capture``.

        Args:
            capture: Labeled capture to run detection + classification over.

        Returns:
            One :class:`~validation.types.Detection` per detected region, in
            detection order.
        """
        scheme_str = str(capture.provenance.get("scheme", "")).lower()
        is_ofdm = scheme_str == "ofdm"
        detector = self.ofdm_detector if is_ofdm else self.detector
        regions = detector(capture.iq, self.threshold, self.min_gap)
        sc_scheme = ModScheme.FSK
        if not is_ofdm:
            try:
                sc_scheme = ModScheme(scheme_str)
            except ValueError:
                sc_scheme = ModScheme.FSK
        differential = bool(capture.provenance.get("differential", False))
        detections: List[Detection] = []
        for start, end in regions:
            if self.use_truth_bytes:
                pkt = self._truth_bytes(capture, start, end) or b""
            elif is_ofdm:
                pkt = ofdm_region_to_bytes(capture.iq[start:end])
            else:
                pkt = single_carrier_region_to_bytes(
                    capture.iq[start:end],
                    sc_scheme,
                    capture.sample_rate,
                    sps=self.sps,
                    differential=differential,
                )
            result = self.classifier.classify(pkt, None)
            proto, conf = _protocol_and_confidence(result)
            detections.append(
                Detection(
                    start=int(start), end=int(end), protocol=proto, confidence=conf
                )
            )
        return detections
