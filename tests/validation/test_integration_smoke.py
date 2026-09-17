"""End-to-end SP1 integration smoke test.

Exercises the full validation & T&E spine in one shot: synthetic dataset
generation (Tasks 4-6), the injectable detect-then-classify pipeline
(Task 9), the evaluation harness + reproducibility manifest (Tasks 2, 11),
and the JSON report writer (Task 7). This is the SP1 definition-of-done
gate test referenced by spec section 15.
"""
from __future__ import annotations

import json

import pytest

from validation import ModScheme, create_pipeline, create_synth_dataset, evaluate
from validation.report import write_report


class OracleStub:
    """Classifier stub that always reports the known protocol."""

    def classify(self, packet_bytes, signal_metrics=None):
        return "mavlink"


@pytest.mark.integration
def test_full_spine_smoke(tmp_path):
    ds = create_synth_dataset(
        protocols=["mavlink"],
        snr_grid_db=[0.0, 20.0],
        n_per_cell=3,
        scheme_by_protocol={"mavlink": ModScheme.FSK},
        seed=11,
        payload_len=24,
    )
    pipe = create_pipeline(OracleStub(), threshold=0.2, min_gap=64)

    r1 = evaluate(ds, pipe, seed=11)
    r2 = evaluate(ds, pipe, seed=11)
    assert r1.manifest.dataset_hash == r2.manifest.dataset_hash

    p = tmp_path / "report.json"
    write_report(r1, p, plots=False)
    doc = json.loads(p.read_text())

    assert doc["detection"]["pd"] >= 0.0
    assert "classification" in doc
    assert "manifest" in doc and doc["manifest"]["dataset_hash"]
