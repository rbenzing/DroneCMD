"""Detection and classification metrics for the validation & T&E spine.

All functions here are pure (no I/O, no global state) so they can be
unit-tested with golden values and safely reused across the validation
pipeline. Randomness (bootstrap resampling) is seeded via
:func:`validation.repro.rng` for reproducibility -- never the
``np.random.*`` free functions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from validation.repro import rng
from validation.types import (
    ClassificationMetrics,
    CodedLinkMetrics,
    DetectionMetrics,
    ProfileIdMetrics,
)

Bits = npt.NDArray[np.uint8]


def overlap_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Compute the intersection-over-union of two half-open sample intervals.

    Args:
        a: ``(start, end)`` sample indices of the first interval.
        b: ``(start, end)`` sample indices of the second interval.

    Returns:
        IoU in ``[0.0, 1.0]``. Returns ``0.0`` when the union is empty
        (e.g. both intervals are zero-length).
    """
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match_detections(
    detected: List[Tuple[int, int]],
    truth: List[Tuple[int, int, str]],
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int]:
    """Greedily match detected intervals to truth intervals by IoU.

    Each truth region can be claimed by at most one detection (best IoU
    among unclaimed truth regions wins). Detections that fail to reach
    ``iou_threshold`` against any unclaimed truth region count as false
    positives; unclaimed truth regions count as false negatives.

    Args:
        detected: List of ``(start, end)`` detected sample intervals.
        truth: List of ``(start, end, protocol)`` ground-truth regions.
        iou_threshold: Minimum IoU required to count as a true positive.

    Returns:
        ``(tp, fp, fn)`` counts.
    """
    truth_spans = [(s, e) for (s, e, _p) in truth]
    used: set = set()
    tp = 0
    for d in detected:
        best_j, best_iou = -1, 0.0
        for j, t in enumerate(truth_spans):
            if j in used:
                continue
            iou = overlap_iou(d, t)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_threshold:
            used.add(best_j)
            tp += 1
    fp = len(detected) - tp
    fn = len(truth_spans) - tp
    return tp, fp, fn


def detection_metrics(
    matched: List[Tuple[int, int, int]],
    total_samples: int,
    sample_rate: float,
    window: int,
    roc_points: Optional[List[Tuple[float, float]]] = None,
    min_snr: Optional[float] = None,
) -> DetectionMetrics:
    """Aggregate per-capture ``(tp, fp, fn)`` counts into detection metrics.

    Args:
        matched: Per-capture ``(tp, fp, fn)`` tuples, e.g. from repeated
            calls to :func:`match_detections`.
        total_samples: Total number of IQ samples analyzed across all
            captures (used to derive false-alarm rate per second).
        sample_rate: Sample rate in Hz used to convert samples to seconds.
        window: Detector window length in samples, used to derive the
            false-alarm rate per window.
        roc_points: Optional pre-computed ``(pfa, pd)`` ROC curve points.
        min_snr: Optional minimum SNR (dB) at which detection was reliable.

    Returns:
        A populated :class:`~validation.types.DetectionMetrics`.
    """
    tp = sum(m[0] for m in matched)
    fp = sum(m[1] for m in matched)
    fn = sum(m[2] for m in matched)
    pd = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    duration_s = total_samples / sample_rate if sample_rate > 0 else 0.0
    pfa_per_sec = fp / duration_s if duration_s > 0 else 0.0
    n_windows = max(1, total_samples // max(1, window))
    pfa_per_window = fp / n_windows
    return DetectionMetrics(
        pd=pd,
        pfa_per_sec=pfa_per_sec,
        pfa_per_window=pfa_per_window,
        tp=tp,
        fp=fp,
        fn=fn,
        roc=roc_points or [],
        min_detectable_snr_db=min_snr,
    )


def classification_metrics(
    pairs: List[Tuple[str, str]],
    snr_by_pair: Optional[List[float]] = None,
) -> ClassificationMetrics:
    """Compute confusion matrix, accuracy, and per-class precision/recall/F1.

    Args:
        pairs: List of ``(truth_label, pred_label)`` pairs.
        snr_by_pair: Optional per-pair SNR (dB), aligned with ``pairs``,
            used to bucket accuracy by SNR (rounded to 1 decimal place).

    Returns:
        A populated :class:`~validation.types.ClassificationMetrics`.
    """
    labels = sorted({p for pair in pairs for p in pair})
    confusion: Dict[str, Dict[str, int]] = {t: {p: 0 for p in labels} for t in labels}
    correct = 0
    for truth, pred in pairs:
        confusion[truth][pred] += 1
        if truth == pred:
            correct += 1
    accuracy = correct / len(pairs) if pairs else 0.0
    per_class: Dict[str, Dict[str, float]] = {}
    for c in labels:
        tp = confusion[c][c]
        fp = sum(confusion[t][c] for t in labels if t != c)
        fn = sum(confusion[c][p] for p in labels if p != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1}
    acc_by_snr: Dict[float, float] = {}
    if snr_by_pair is not None:
        buckets: Dict[float, List[int]] = {}
        for (truth, pred), snr in zip(pairs, snr_by_pair):
            buckets.setdefault(round(snr, 1), []).append(int(truth == pred))
        acc_by_snr = {k: float(np.mean(v)) for k, v in sorted(buckets.items())}
    return ClassificationMetrics(
        accuracy=accuracy,
        confusion=confusion,
        per_class=per_class,
        accuracy_by_snr=acc_by_snr,
    )


def profile_id_metrics(
    pairs: List[Tuple[str, str]],
    snr_by_pair: Optional[List[float]] = None,
) -> ProfileIdMetrics:
    """Blind profile-ID accuracy from ``(truth_profile, resolved_profile)`` pairs.

    Args:
        pairs: ``(truth_profile, resolved_profile)`` pairs; a no-lock resolve
            should be passed as the string ``"none"`` by the caller.
        snr_by_pair: Optional per-pair SNR (dB) for bucketed accuracy.

    Returns:
        A populated :class:`~validation.types.ProfileIdMetrics`.
    """
    labels = sorted({p for pair in pairs for p in pair})
    confusion: Dict[str, Dict[str, int]] = {t: {p: 0 for p in labels} for t in labels}
    correct = 0
    for truth, pred in pairs:
        confusion[truth][pred] += 1
        if truth == pred:
            correct += 1
    accuracy = correct / len(pairs) if pairs else 0.0
    acc_by_snr: Dict[float, float] = {}
    if snr_by_pair is not None:
        buckets: Dict[float, List[int]] = {}
        for (truth, pred), snr in zip(pairs, snr_by_pair):
            buckets.setdefault(round(snr, 1), []).append(int(truth == pred))
        acc_by_snr = {k: float(np.mean(v)) for k, v in sorted(buckets.items())}
    return ProfileIdMetrics(
        accuracy=accuracy, confusion=confusion, accuracy_by_snr=acc_by_snr
    )


def coded_link_metrics(
    pairs: List[Tuple[Bits, Optional[Bits]]],
    snr_by_pair: Optional[List[float]] = None,
) -> CodedLinkMetrics:
    """Coded payload BER + frame-error rate (FER).

    Each pair is (truth_payload_bits, decoded_payload_bits or None). A None or
    too-short decode counts all truth bits as errors and the frame as a failure
    (conservative, standard convention). Empty input -> zeros.

    Args:
        pairs: ``(truth_payload_bits, decoded_payload_bits)`` pairs, where the
            decoded array is ``None`` on a loud decode failure (e.g. CRC
            mismatch or no lock).
        snr_by_pair: Optional per-pair SNR (dB) for bucketed BER/FER.

    Returns:
        A populated :class:`~validation.types.CodedLinkMetrics`.
    """
    if not pairs:
        return CodedLinkMetrics(coded_ber=0.0, fer=0.0)
    snrs = snr_by_pair if snr_by_pair is not None else [0.0] * len(pairs)
    tot_bits = tot_errs = tot_fail = 0
    bits_by: Dict[float, int] = {}
    errs_by: Dict[float, int] = {}
    fail_by: Dict[float, int] = {}
    n_by: Dict[float, int] = {}
    for (truth, dec), snr in zip(pairs, snrs):
        n = int(truth.size)
        if dec is None or int(dec.size) < n:
            errs = n
        else:
            errs = int(np.sum(truth != dec[:n]))
        fail = 1 if errs > 0 else 0
        tot_bits += n
        tot_errs += errs
        tot_fail += fail
        key = round(float(snr), 0)
        bits_by[key] = bits_by.get(key, 0) + n
        errs_by[key] = errs_by.get(key, 0) + errs
        fail_by[key] = fail_by.get(key, 0) + fail
        n_by[key] = n_by.get(key, 0) + 1
    ber = tot_errs / tot_bits if tot_bits else 0.0
    fer = tot_fail / len(pairs)
    ber_by_snr = {k: errs_by[k] / bits_by[k] for k in bits_by if bits_by[k]}
    fer_by_snr = {k: fail_by[k] / n_by[k] for k in n_by}
    return CodedLinkMetrics(
        coded_ber=ber, fer=fer, ber_by_snr=ber_by_snr, fer_by_snr=fer_by_snr
    )


def bootstrap_ci(
    values: Sequence[float],
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Compute a percentile bootstrap confidence interval for the mean.

    Args:
        values: Sample of scalar observations (e.g. per-capture accuracy).
        n: Number of bootstrap resamples.
        seed: Seed passed to :func:`validation.repro.rng` for determinism.
        alpha: Significance level; the returned interval covers the
            ``1 - alpha`` central range (e.g. ``alpha=0.05`` -> 95% CI).

    Returns:
        ``(lo, hi)`` interval bounds. Returns ``(0.0, 0.0)`` for empty input.
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return (0.0, 0.0)
    g = rng(seed)
    means = np.empty(n)
    for i in range(n):
        sample = g.choice(arr, size=len(arr), replace=True)
        means[i] = float(np.mean(sample))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi
