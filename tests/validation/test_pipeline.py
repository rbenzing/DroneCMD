"""Tests for the injectable detect-then-classify pipeline."""
from __future__ import annotations

import numpy as np

from validation.pipeline import DetectClassifyPipeline
from validation.types import Detection, LabeledCapture


class StubClassifier:
    """Returns a fixed protocol with confidence proportional to region energy."""

    def __init__(self, label: str = "mavlink") -> None:
        self.label = label

    def classify(self, packet_bytes: bytes, signal_metrics: object = None) -> str:
        return self.label


def _capture_with_burst() -> LabeledCapture:
    silence = np.zeros(300, dtype=np.complex64)
    burst = (np.ones(400) * 0.9).astype(np.complex64)
    iq = np.concatenate([silence, burst, silence])
    return LabeledCapture(
        iq=iq,
        sample_rate=1e6,
        truth_regions=[(300, 700, "mavlink")],
        provenance={"source": "synth", "protocol": "mavlink"},
    )


def test_pipeline_detects_and_labels() -> None:
    pipe = DetectClassifyPipeline(StubClassifier("mavlink"), threshold=0.2, min_gap=100)
    dets = pipe.run(_capture_with_burst())
    assert len(dets) >= 1
    d = dets[0]
    assert isinstance(d, Detection)
    assert d.protocol == "mavlink"
    assert 250 <= d.start <= 350 and 650 <= d.end <= 750


def test_pipeline_no_detections_on_silence() -> None:
    cap = LabeledCapture(
        iq=np.zeros(2048, dtype=np.complex64),
        sample_rate=1e6,
        truth_regions=None,
        provenance={"source": "synth"},
    )
    pipe = DetectClassifyPipeline(StubClassifier(), threshold=0.05)
    assert pipe.run(cap) == []


def test_injectable_detector_is_used() -> None:
    calls = {"n": 0}

    def fake_detector(iq: np.ndarray, threshold: float, min_gap: int) -> list:
        calls["n"] += 1
        return [(10, 50)]

    pipe = DetectClassifyPipeline(StubClassifier("dji"), detector=fake_detector)
    dets = pipe.run(_capture_with_burst())
    assert calls["n"] == 1
    assert dets[0].protocol == "dji" and (dets[0].start, dets[0].end) == (10, 50)


def test_pipeline_routes_ofdm_scheme_to_ofdm_demod(monkeypatch) -> None:
    import validation.pipeline as P

    calls = {"ofdm": 0, "fsk": 0}
    monkeypatch.setattr(
        P,
        "ofdm_region_to_bytes",
        lambda iq: (calls.__setitem__("ofdm", calls["ofdm"] + 1) or b"\x01"),
    )
    monkeypatch.setattr(
        P,
        "region_to_bytes",
        lambda iq, sps=8: (calls.__setitem__("fsk", calls["fsk"] + 1) or b"\x02"),
    )
    iq = np.concatenate(
        [
            np.zeros(100, np.complex64),
            (np.ones(400) * 0.9).astype(np.complex64),
            np.zeros(100, np.complex64),
        ]
    )
    cap = LabeledCapture(
        iq=iq,
        sample_rate=1e6,
        truth_regions=[(100, 500, "ocusync")],
        provenance={"source": "synth", "scheme": "ofdm"},
    )
    P.DetectClassifyPipeline(StubClassifier("ocusync"), threshold=0.2, min_gap=50).run(
        cap
    )
    assert calls["ofdm"] >= 1 and calls["fsk"] == 0


def test_pipeline_non_ofdm_scheme_uses_fsk_path(monkeypatch) -> None:
    import validation.pipeline as P

    calls = {"ofdm": 0, "fsk": 0}
    monkeypatch.setattr(
        P,
        "ofdm_region_to_bytes",
        lambda iq: (calls.__setitem__("ofdm", calls["ofdm"] + 1) or b"\x01"),
    )
    monkeypatch.setattr(
        P,
        "region_to_bytes",
        lambda iq, sps=8: (calls.__setitem__("fsk", calls["fsk"] + 1) or b"\x02"),
    )
    P.DetectClassifyPipeline(StubClassifier("mavlink"), threshold=0.2, min_gap=100).run(
        _capture_with_burst()  # existing helper; provenance has no "scheme"
    )
    assert calls["fsk"] >= 1 and calls["ofdm"] == 0


def test_pipeline_ofdm_end_to_end_recovers_payload() -> None:
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    payload = bytes(range(24))
    pkt = modulate(payload, ModScheme.OFDM)
    iq = np.concatenate([np.zeros(300, np.complex64), pkt, np.zeros(300, np.complex64)])
    captured = {}

    class Spy:
        def classify(self, packet_bytes, signal_metrics=None):
            captured["b"] = packet_bytes
            return "ocusync"

    cap = LabeledCapture(
        iq=iq,
        sample_rate=1e6,
        truth_regions=[(300, 300 + len(pkt), "ocusync")],
        provenance={"source": "synth", "scheme": "ofdm"},
    )
    dets = DetectClassifyPipeline(Spy(), threshold=0.05, min_gap=64).run(cap)
    assert len(dets) >= 1
    assert captured["b"][: len(payload)] == payload
