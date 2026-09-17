from __future__ import annotations

import json

import numpy as np

from validation.ingest.labeler import load_labeled


def test_label_from_sidecar_json(tmp_path):
    d = tmp_path / "dji_ocusync"
    d.mkdir()
    iq = (np.ones(64) * 0.5).astype(np.complex64)
    iq.tofile(d / "flight01.iq")
    (d / "flight01.json").write_text(json.dumps({"protocol": "dji_ocusync"}))
    cap = load_labeled(d / "flight01.iq", sample_rate=2_048_000.0)
    assert cap.provenance["protocol"] == "dji_ocusync"
    assert cap.provenance["source"] == "real"
    assert cap.iq.dtype == np.complex64
    assert len(cap.iq) == 64


def test_label_falls_back_to_parent_dir(tmp_path):
    d = tmp_path / "mavlink"
    d.mkdir()
    iq = np.ones(32, dtype=np.complex64)
    iq.tofile(d / "cap.iq")
    cap = load_labeled(d / "cap.iq", sample_rate=1e6)
    assert cap.provenance["protocol"] == "mavlink"
    assert cap.truth_regions is None


def test_regions_from_sigmf_annotations(tmp_path):
    d = tmp_path / "mixed"
    d.mkdir()
    iq = np.ones(200, dtype=np.complex64)
    iq.tofile(d / "flight01.iq")
    # Only a .sigmf-meta sidecar exists for this stem (no .json), so this
    # exercises the .sigmf-meta branch of _read_sidecar.
    meta = {
        "protocol": "mixed",
        "annotations": [
            {
                "core:sample_start": 10,
                "core:sample_count": 40,
                "core:description": "mavlink",
            },
            {
                "core:sample_start": 100,
                "core:sample_count": 50,
                "core:description": "dji",
            },
        ],
    }
    (d / "flight01.sigmf-meta").write_text(json.dumps(meta))
    cap = load_labeled(d / "flight01.iq")
    assert cap.truth_regions == [(10, 50, "mavlink"), (100, 150, "dji")]
    assert cap.provenance["protocol"] == "mixed"


def test_regions_zero_count_returns_none(tmp_path):
    d = tmp_path / "mixed"
    d.mkdir()
    iq = np.ones(100, dtype=np.complex64)
    iq.tofile(d / "flight02.iq")
    meta = {
        "annotations": [
            {
                "core:sample_start": 10,
                "core:sample_count": 0,
                "core:description": "mavlink",
            }
        ],
    }
    (d / "flight02.sigmf-meta").write_text(json.dumps(meta))
    cap = load_labeled(d / "flight02.iq")
    assert cap.truth_regions is None
