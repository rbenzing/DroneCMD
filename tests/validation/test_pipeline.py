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
