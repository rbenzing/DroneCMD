"""Regression test for the write()/from_dir() provenance round-trip.

Guards against the FIX 1 regression where ``LabeledDataset.write()``
persisted a ``.json`` sidecar's full ``provenance`` dict (including
``snr_db``, ``payload_hex``, ``scheme``, ``requested_snr_db``, ``seed``) but
``validation.ingest.labeler.load_labeled`` rebuilt ``provenance`` from
scratch on reload, silently dropping all of it. That made every reloaded
capture report ``snr_db=0.0`` to the harness, collapsing
``accuracy_by_snr``/``min_detectable_snr_db`` into a single bogus 0-dB
bucket regardless of the dataset's actual SNR grid.
"""
from __future__ import annotations

from typing import Optional

from validation.dataset import LabeledDataset
from validation.harness import HarnessConfig, run_evaluation
from validation.pipeline import DetectClassifyPipeline
from validation.synth.scenarios import DatasetSpec, build_scenario
from validation.types import ModScheme


class OracleClassifier:
    """Always returns a fixed protocol label, regardless of the packet.

    An oracle stub: it isolates detector/harness behavior in these tests
    without depending on a trained model or the demodulator's bit-accuracy.
    """

    def __init__(self, label: str) -> None:
        self.label = label

    def classify(
        self, packet_bytes: bytes, signal_metrics: Optional[dict] = None
    ) -> str:
        return self.label


def _synth_dataset() -> LabeledDataset:
    spec = DatasetSpec(
        protocols=["mavlink"],
        snr_grid_db=[15.0, 30.0],
        n_per_cell=2,
        sample_rate=1e6,
        seed=7,
        scheme_by_protocol={"mavlink": ModScheme.FSK},
        payload_len=16,
    )
    return LabeledDataset(build_scenario(spec))


def test_roundtrip_preserves_snr_db_provenance(tmp_path):
    ds = _synth_dataset()
    ds.write(tmp_path)
    reloaded = LabeledDataset.from_dir(tmp_path, sample_rate=1e6)

    assert len(reloaded) == len(ds)
    for capture in reloaded:
        assert capture.provenance.get("source") == "synth"
        assert "snr_db" in capture.provenance
        assert isinstance(capture.provenance["snr_db"], float)
        # Other synth-only fields must also survive the round-trip.
        assert "payload_hex" in capture.provenance
        assert "scheme" in capture.provenance
        assert "requested_snr_db" in capture.provenance


def test_roundtrip_eval_has_multiple_snr_buckets(tmp_path):
    ds = _synth_dataset()
    ds.write(tmp_path)
    reloaded = LabeledDataset.from_dir(tmp_path, sample_rate=1e6)

    pipe = DetectClassifyPipeline(OracleClassifier("mavlink"))
    result = run_evaluation(
        reloaded, pipe, HarnessConfig(detector_threshold=0.2, min_gap=64)
    )

    buckets = result.classification.accuracy_by_snr
    # Before FIX 1, every reloaded capture reported snr_db=0.0, so all
    # pairs fell into a single bogus 0-dB bucket.
    assert len(buckets) > 1
    # The two SNR cells (~15 dB, ~30 dB) must map to genuinely distinct
    # buckets, not just an artifact of rounding within one cell.
    assert min(buckets) < 20.0
    assert max(buckets) > 25.0
