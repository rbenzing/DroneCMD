from __future__ import annotations

from validation import (
    DetectClassifyPipeline,
    LabeledDataset,
    ModScheme,
    RunResult,
    create_pipeline,
    create_synth_dataset,
    evaluate,
)


class Stub:
    def classify(self, packet_bytes, signal_metrics=None):
        return "mavlink"


def test_end_to_end_public_api():
    ds = create_synth_dataset(
        protocols=["mavlink"],
        snr_grid_db=[20.0],
        n_per_cell=2,
        scheme_by_protocol={"mavlink": ModScheme.FSK},
        seed=3,
        payload_len=16,
    )
    assert isinstance(ds, LabeledDataset) and len(ds) == 2
    pipe = create_pipeline(Stub(), threshold=0.2, min_gap=64)
    assert isinstance(pipe, DetectClassifyPipeline)
    result = evaluate(ds, pipe, seed=3)
    assert isinstance(result, RunResult)
    assert result.manifest.dataset_hash
