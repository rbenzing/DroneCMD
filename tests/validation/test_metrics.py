from __future__ import annotations

import numpy as np

from validation.metrics import (
    bootstrap_ci,
    classification_metrics,
    detection_metrics,
    match_detections,
    overlap_iou,
)


def test_iou_basic() -> None:
    assert overlap_iou((0, 10), (0, 10)) == 1.0
    assert overlap_iou((0, 10), (10, 20)) == 0.0
    assert abs(overlap_iou((0, 10), (5, 15)) - (5 / 15)) < 1e-9


def test_match_counts() -> None:
    tp, fp, fn = match_detections(
        detected=[(0, 10), (100, 110)],
        truth=[(0, 9, "a"), (200, 210, "b")],
        iou_threshold=0.5,
    )
    assert (tp, fp, fn) == (1, 1, 1)


def test_detection_metrics_pd_pfa() -> None:
    m = detection_metrics(
        matched=[(1, 1, 0), (1, 0, 1)],
        total_samples=2_000_000,
        sample_rate=1_000_000.0,
        window=1000,
    )
    assert abs(m.pd - (2 / 3)) < 1e-9  # tp=2, fn=1
    assert m.fp == 1
    assert abs(m.pfa_per_sec - 0.5) < 1e-9  # 1 FP over 2.0 s


def test_classification_metrics_confusion_and_accuracy() -> None:
    pairs = [("a", "a"), ("a", "b"), ("b", "b"), ("b", "b")]
    cm = classification_metrics(pairs)
    assert abs(cm.accuracy - 0.75) < 1e-9
    assert cm.confusion["a"]["a"] == 1 and cm.confusion["a"]["b"] == 1
    assert cm.per_class["b"]["recall"] == 1.0


def test_bootstrap_ci_brackets_mean() -> None:
    vals = list(np.r_[np.ones(50), np.zeros(50)])  # mean 0.5
    lo, hi = bootstrap_ci(vals, n=500, seed=1)
    assert lo < 0.5 < hi
