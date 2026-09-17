from __future__ import annotations

import json

from validation.report import result_to_dict, write_report
from validation.types import (
    ClassificationMetrics,
    DetectionMetrics,
    RunManifest,
    RunResult,
)


def _result():
    return RunResult(
        detection=DetectionMetrics(
            pd=0.9,
            pfa_per_sec=0.1,
            pfa_per_window=0.01,
            tp=9,
            fp=1,
            fn=1,
            min_detectable_snr_db=0.0,
        ),
        classification=ClassificationMetrics(
            accuracy=0.8,
            confusion={"a": {"a": 4, "b": 1}},
            per_class={"a": {"precision": 0.8}},
            accuracy_by_snr={0.0: 0.7, 20.0: 0.95},
        ),
        manifest=RunManifest(
            seed=42,
            dataset_hash="d",
            config_hash="c",
            git_commit="abc",
            timestamp="t",
            versions={"numpy": "1.0"},
        ),
    )


def test_result_to_dict_has_sections():
    d = result_to_dict(_result())
    assert set(d) >= {"detection", "classification", "manifest"}
    assert d["detection"]["pd"] == 0.9
    assert d["manifest"]["dataset_hash"] == "d"


def test_write_report_json(tmp_path):
    p = tmp_path / "report.json"
    write_report(_result(), p, plots=False)
    loaded = json.loads(p.read_text())
    assert loaded["classification"]["accuracy"] == 0.8
    assert loaded["manifest"]["seed"] == 42
