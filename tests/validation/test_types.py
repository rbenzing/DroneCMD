from __future__ import annotations

import numpy as np

from validation.types import Detection, LabeledCapture, ModScheme


def test_modscheme_values():
    assert {s.value for s in ModScheme} == {"fsk", "gfsk", "bpsk", "qpsk", "ofdm"}


def test_labeled_capture_construction():
    iq = np.zeros(16, dtype=np.complex64)
    cap = LabeledCapture(
        iq=iq,
        sample_rate=2_048_000.0,
        truth_regions=[(0, 8, "mavlink")],
        provenance={"source": "synth", "snr_db": 10.0},
    )
    assert cap.iq.dtype == np.complex64
    assert cap.truth_regions[0] == (0, 8, "mavlink")
    assert cap.provenance["source"] == "synth"


def test_detection_fields():
    d = Detection(start=0, end=10, protocol="dji", confidence=0.9)
    assert (d.start, d.end, d.protocol, d.confidence) == (0, 10, "dji", 0.9)
