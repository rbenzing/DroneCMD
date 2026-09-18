"""Tests for the dataset x pipeline -> RunResult evaluation harness."""
from __future__ import annotations

from typing import Optional

from validation.dataset import LabeledDataset
from validation.harness import HarnessConfig, run_evaluation
from validation.pipeline import DetectClassifyPipeline
from validation.synth.scenarios import DatasetSpec, build_scenario
from validation.types import ModScheme


class OracleClassifier:
    """Always returns a fixed protocol label, regardless of the packet.

    An oracle stub: it isolates detector behavior in these tests without
    depending on a trained model or the demodulator's bit-accuracy.
    """

    def __init__(self, label: str) -> None:
        self.label = label

    def classify(
        self, packet_bytes: bytes, signal_metrics: Optional[dict] = None
    ) -> str:
        return self.label


def _dataset() -> LabeledDataset:
    spec = DatasetSpec(
        protocols=["mavlink"],
        snr_grid_db=[20.0],
        n_per_cell=3,
        sample_rate=1e6,
        seed=1,
        scheme_by_protocol={"mavlink": ModScheme.FSK},
        payload_len=16,
    )
    return LabeledDataset(build_scenario(spec))


def test_run_is_deterministic() -> None:
    ds = _dataset()
    pipe = DetectClassifyPipeline(
        OracleClassifier("mavlink"), threshold=0.2, min_gap=64
    )
    r1 = run_evaluation(ds, pipe, HarnessConfig(seed=5))
    r2 = run_evaluation(ds, pipe, HarnessConfig(seed=5))
    assert r1.manifest.dataset_hash == r2.manifest.dataset_hash
    assert r1.detection.pd == r2.detection.pd
    assert r1.classification.accuracy == r2.classification.accuracy
    assert r1.detection.ci["pd"] == r2.detection.ci["pd"]
    assert r1.classification.ci["accuracy"] == r2.classification.ci["accuracy"]


def test_high_snr_detects_and_classifies() -> None:
    ds = _dataset()
    pipe = DetectClassifyPipeline(
        OracleClassifier("mavlink"), threshold=0.2, min_gap=64
    )
    r = run_evaluation(ds, pipe, HarnessConfig())
    assert r.detection.pd > 0.5
    assert r.classification.accuracy > 0.5
    assert r.manifest.dataset_hash
    assert "pd" in r.detection.ci
    assert "accuracy" in r.classification.ci


def test_run_evaluation_reports_profile_id() -> None:
    spec = DatasetSpec(
        protocols=["a", "b"],
        snr_grid_db=[30.0],
        n_per_cell=2,
        sample_rate=2_048_000.0,
        seed=3,
        profile_by_protocol={"a": "ble_1m", "b": "qpsk_link"},
    )
    ds = LabeledDataset(build_scenario(spec))

    class _StubClf:
        def classify(
            self, packet_bytes: bytes, signal_metrics: Optional[dict] = None
        ) -> str:
            return "x"

    pipe = DetectClassifyPipeline(_StubClf())
    result = run_evaluation(ds, pipe, HarnessConfig())
    assert result.profile_id is not None
    assert result.profile_id.accuracy >= 0.9  # blind resolution at 30 dB


def test_profile_id_spans_sc_and_ofdm() -> None:
    """Blind profile-ID accuracy is high on a mixed single-carrier + OFDM
    dataset, and OFDM profile names flow into the confusion matrix."""
    spec = DatasetSpec(
        protocols=["g", "w", "n"],
        snr_grid_db=[30.0],
        n_per_cell=3,
        sample_rate=2_048_000.0,
        seed=8,
        profile_by_protocol={"g": "sik_gfsk", "w": "wifi_20", "n": "ofdm_nb"},
    )
    ds = LabeledDataset(build_scenario(spec))

    class _Clf:
        def classify(
            self, packet_bytes: bytes, signal_metrics: Optional[dict] = None
        ) -> str:
            return "x"

    pipe = DetectClassifyPipeline(_Clf())
    result = run_evaluation(ds, pipe, HarnessConfig())
    assert result.profile_id is not None
    assert result.profile_id.accuracy >= 0.8  # blind resolution at 30 dB
    assert {"sik_gfsk", "wifi_20", "ofdm_nb"} <= set(result.profile_id.confusion.keys())
