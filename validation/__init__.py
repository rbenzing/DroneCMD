"""DroneCMD validation & T&E spine (SP1) — public API."""
from __future__ import annotations

from typing import Dict, List, Optional

from validation.dataset import LabeledDataset
from validation.harness import HarnessConfig, run_evaluation
from validation.pipeline import DetectClassifyPipeline
from validation.synth.scenarios import DatasetSpec, build_scenario
from validation.types import (
    ChannelParams,
    ClassificationMetrics,
    Detection,
    DetectionMetrics,
    IQSamples,
    LabeledCapture,
    ModScheme,
    RunManifest,
    RunResult,
)

__all__ = [
    "ModScheme",
    "ChannelParams",
    "LabeledCapture",
    "Detection",
    "DetectionMetrics",
    "ClassificationMetrics",
    "RunManifest",
    "RunResult",
    "IQSamples",
    "LabeledDataset",
    "DetectClassifyPipeline",
    "HarnessConfig",
    "DatasetSpec",
    "create_synth_dataset",
    "create_pipeline",
    "evaluate",
]


def create_synth_dataset(
    protocols: List[str],
    snr_grid_db: List[float],
    n_per_cell: int,
    scheme_by_protocol: Dict[str, ModScheme],
    sample_rate: float = 2_048_000.0,
    seed: int = 42,
    payload_len: int = 32,
    differential: bool = False,
) -> LabeledDataset:
    """Create a synthetic labeled dataset for validation.

    Args:
        protocols: List of protocol names (e.g., ["mavlink", "dji"]).
        snr_grid_db: List of SNR values in dB to generate.
        n_per_cell: Number of samples per SNR value.
        scheme_by_protocol: Mapping of protocol name to ModScheme.
        sample_rate: Sample rate in Hz (default 2.048 MHz).
        seed: Random seed for reproducibility (default 42).
        payload_len: Payload length in bytes (default 32).
        differential: If True, differentially encode BPSK/QPSK payloads
            (default False). Ignored by FSK/GFSK/OFDM schemes.

    Returns:
        A LabeledDataset instance.
    """
    spec = DatasetSpec(
        protocols=protocols,
        snr_grid_db=snr_grid_db,
        n_per_cell=n_per_cell,
        sample_rate=sample_rate,
        seed=seed,
        scheme_by_protocol=scheme_by_protocol,
        payload_len=payload_len,
        differential=differential,
    )
    return LabeledDataset(build_scenario(spec))


def create_pipeline(classifier: object, **kwargs: object) -> DetectClassifyPipeline:
    """Create a detect-then-classify pipeline.

    Args:
        classifier: A classifier instance with a classify() method.
        **kwargs: Additional arguments passed to DetectClassifyPipeline.

    Returns:
        A DetectClassifyPipeline instance.
    """
    return DetectClassifyPipeline(classifier, **kwargs)  # type: ignore[arg-type]


def evaluate(
    dataset: LabeledDataset,
    pipeline: DetectClassifyPipeline,
    seed: int = 42,
    model_hash: Optional[str] = None,
    **kwargs: object,
) -> RunResult:
    """Evaluate a pipeline on a labeled dataset.

    Args:
        dataset: A LabeledDataset instance.
        pipeline: A DetectClassifyPipeline instance.
        seed: Random seed for reproducibility (default 42).
        model_hash: Optional hash of the model for tracking.
        **kwargs: Additional arguments passed to HarnessConfig.

    Returns:
        A RunResult containing detection/classification metrics and manifest.
    """
    config = HarnessConfig(seed=seed, **kwargs)  # type: ignore[arg-type]
    return run_evaluation(dataset, pipeline, config, model_hash=model_hash)
