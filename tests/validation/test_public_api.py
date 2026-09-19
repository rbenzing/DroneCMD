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


def test_default_synth_scheme_gfsk_fallback() -> None:
    from cli import _default_synth_scheme
    from validation.types import ModScheme

    # Known protocols keep their explicit scheme.
    assert _default_synth_scheme("ocusync") == ModScheme.OFDM
    assert _default_synth_scheme("bpsk_link") == ModScheme.BPSK
    assert _default_synth_scheme("dji") == ModScheme.QPSK
    assert _default_synth_scheme("mavlink") == ModScheme.FSK
    # Unknown protocol falls back to GFSK (the default single-carrier scheme),
    # not raw FSK.
    assert _default_synth_scheme("some_new_link") == ModScheme.GFSK


def test_default_synth_profile_mapping() -> None:
    from cli import _default_synth_profile

    assert _default_synth_profile("mavlink") == "sik_gfsk"
    assert _default_synth_profile("ble") == "ble_1m"
    assert _default_synth_profile("dji") == "qpsk_link"
    assert _default_synth_profile("ocusync") == "wifi_20"
    assert _default_synth_profile("bpsk_link") == "psk_c2"
    assert _default_synth_profile("unknown_link") == "sik_gfsk"  # GFSK default
