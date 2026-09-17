"""Evaluation harness: dataset x pipeline -> RunResult for validation & T&E.

``run_evaluation`` orchestrates a :class:`~validation.dataset.LabeledDataset`
through a :class:`~validation.pipeline.DetectClassifyPipeline`, matching
detections to ground truth, aggregating detection and classification
metrics (including bootstrap confidence intervals on the headline Pd and
accuracy figures), and wrapping everything in a reproducibility
:class:`~validation.types.RunManifest` via
:func:`validation.repro.capture_manifest`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from validation.dataset import LabeledDataset
from validation.metrics import (
    bootstrap_ci,
    classification_metrics,
    detection_metrics,
    match_detections,
    overlap_iou,
)
from validation.pipeline import DetectClassifyPipeline
from validation.repro import capture_manifest, hash_config
from validation.types import Detection, RunResult


@dataclass
class HarnessConfig:
    """Configuration governing a single :func:`run_evaluation` call.

    Attributes:
        seed: Seed used for bootstrap confidence intervals (and recorded
            in the run manifest for reproducibility).
        iou_threshold: Minimum IoU for a detection to count as a true
            positive against a truth region, and for a detection to be
            paired with a truth region's label for classification scoring.
        detector_threshold: Energy-detector threshold passed to the
            pipeline.
        min_gap: Minimum gap (samples) between detected packets, passed to
            the pipeline.
        window: Detector window length (samples), used to derive the
            false-alarm rate per window.
        pd_target: Minimum per-SNR-bucket probability of detection required
            for that bucket to count toward the minimum-detectable SNR.
    """

    seed: int = 42
    iou_threshold: float = 0.5
    detector_threshold: float = 0.05
    min_gap: int = 256
    window: int = 1000
    pd_target: float = 0.9


def _match_pair(
    detection: Detection,
    truth_regions: Optional[List[Tuple[int, int, str]]],
    iou_threshold: float,
) -> Optional[Tuple[str, str]]:
    """Pair a detection with the best-overlapping truth region's label.

    Args:
        detection: A single pipeline detection.
        truth_regions: Ground-truth ``(start, end, protocol)`` regions for
            the capture the detection came from.
        iou_threshold: Minimum IoU required to accept the pairing.

    Returns:
        ``(truth_label, pred_label)`` for the best-overlapping truth region
        if its IoU meets ``iou_threshold``, else ``None``.
    """
    best_label: Optional[str] = None
    best_iou = 0.0
    for start, end, protocol in truth_regions or []:
        iou = overlap_iou((detection.start, detection.end), (start, end))
        if iou > best_iou:
            best_iou, best_label = iou, protocol
    if best_label is not None and best_iou >= iou_threshold:
        return best_label, detection.protocol
    return None


def run_evaluation(
    dataset: LabeledDataset,
    pipeline: DetectClassifyPipeline,
    config: HarnessConfig,
    model_hash: Optional[str] = None,
) -> RunResult:
    """Run ``pipeline`` over every capture in ``dataset`` and score the result.

    For each capture: runs the detect-then-classify pipeline, matches
    detections to truth regions (accumulating tp/fp/fn), pairs each
    detection with its best-overlapping truth region's label (for
    classification scoring), and buckets per-capture SNR (from
    ``provenance["snr_db"]``) for the minimum-detectable-SNR calculation.
    Aggregates detection and classification metrics, wires bootstrap
    confidence intervals onto the headline Pd/accuracy figures, and
    captures a reproducibility manifest.

    Args:
        dataset: Labeled captures to evaluate against.
        pipeline: Injectable detect-then-classify pipeline under test. Its
            ``threshold``/``min_gap`` are overwritten from ``config`` before
            running so the harness config is the single source of truth.
        config: Harness configuration (detector params, thresholds, seed).
        model_hash: Optional hash identifying the classifier/model under
            test, recorded in the run manifest.

    Returns:
        A :class:`~validation.types.RunResult` with detection metrics,
        classification metrics (both carrying bootstrap CIs on their
        headline figures), and a reproducibility manifest.
    """
    pipeline.threshold = config.detector_threshold
    pipeline.min_gap = config.min_gap

    matched: List[Tuple[int, int, int]] = []
    pairs: List[Tuple[str, str]] = []
    snr_by_pair: List[float] = []
    total_samples = 0
    pd_by_snr: Dict[float, List[int]] = {}

    for capture in dataset:
        total_samples += len(capture.iq)
        detections = pipeline.run(capture)
        det_spans = [(d.start, d.end) for d in detections]
        tp, fp, fn = match_detections(
            det_spans,
            capture.truth_regions or [],
            iou_threshold=config.iou_threshold,
        )
        matched.append((tp, fp, fn))

        snr = float(capture.provenance.get("snr_db", 0.0))
        pd_by_snr.setdefault(round(snr, 0), []).append(1 if tp > 0 else 0)

        for detection in detections:
            pair = _match_pair(detection, capture.truth_regions, config.iou_threshold)
            if pair is not None:
                pairs.append(pair)
                snr_by_pair.append(snr)

    # Minimum-detectable SNR: lowest bucket whose per-bucket Pd clears the
    # target.
    min_snr: Optional[float] = None
    for snr_bucket in sorted(pd_by_snr):
        if float(np.mean(pd_by_snr[snr_bucket])) >= config.pd_target:
            min_snr = snr_bucket
            break

    sample_rate = next(iter(dataset)).sample_rate if len(dataset) else 1.0
    det_metrics = detection_metrics(
        matched,
        total_samples=total_samples,
        sample_rate=sample_rate,
        window=config.window,
        min_snr=min_snr,
    )
    cls_metrics = classification_metrics(pairs, snr_by_pair=snr_by_pair)

    # Bootstrap confidence intervals on the headline numbers -- every
    # reported metric carries a CI, not just a point estimate.
    det_metrics.ci["pd"] = bootstrap_ci(
        [1.0 if m[0] > 0 else 0.0 for m in matched], seed=config.seed
    )
    cls_metrics.ci["accuracy"] = bootstrap_ci(
        [1.0 if truth == pred else 0.0 for (truth, pred) in pairs], seed=config.seed
    )

    config_hash = hash_config(asdict(config))
    manifest = capture_manifest(
        seed=config.seed,
        dataset_hash=dataset.content_hash(),
        config_hash=config_hash,
        model_hash=model_hash,
    )
    return RunResult(
        detection=det_metrics, classification=cls_metrics, manifest=manifest
    )
