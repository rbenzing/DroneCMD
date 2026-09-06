"""
Dataset builder for protocol classifier training.

Loads labeled IQ captures from a directory tree, extracts feature vectors
using the same AdvancedFeatureExtractor used at inference time, and returns
(X, y) arrays ready for sklearn.

Expected directory layout:
    data_dir/
      dji_ocusync/
        flight01.iq
        flight01.json    # sidecar with {"protocol": "dji_ocusync", ...}
        flight02.iq
        ...
      mavlink/
        ...
      unknown/
        ...

The protocol label is read from the sidecar JSON ("protocol" key).  If no
sidecar exists, the parent directory name is used as the label.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from core.classification import (
    AdvancedFeatureExtractor,
    ClassifierConfig,
    FeatureType,
)
from core.signal_processing import SignalProcessor, detect_packets
from utils.fileio import read_iq_file

logger = logging.getLogger(__name__)

IQSamples = npt.NDArray[np.complex64]
FeatureMatrix = npt.NDArray[np.float32]
LabelArray = npt.NDArray[np.str_]


# Accepted label strings — callers can extend this set
KNOWN_LABELS = frozenset(
    {
        "dji_ocusync",
        "dji_lightbridge",
        "dji_wifi",
        "parrot",
        "mavlink",
        "unknown",
    }
)


@dataclass
class DatasetStats:
    """Summary of the built dataset."""

    total_files: int = 0
    total_packets: int = 0
    failed_files: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    feature_dim: int = 0


def _read_label(iq_path: Path) -> str:
    """Return the protocol label for a capture file.

    Prefers the 'protocol' key in the JSON sidecar; falls back to the
    parent directory name.
    """
    sidecar = iq_path.with_suffix(".json")
    if sidecar.exists():
        try:
            with open(sidecar) as f:
                meta = json.load(f)
            label = str(meta.get("protocol", "")).strip().lower()
            if label:
                return label
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read sidecar {sidecar}: {e}")
    return iq_path.parent.name.lower()


def _extract_packets_from_iq(
    iq_data: IQSamples,
    threshold: float = 0.05,
    min_gap: int = 500,
    max_packets: int = 200,
) -> List[bytes]:
    """Detect and extract packet byte strings from an IQ array."""
    processor = SignalProcessor()
    normalized = processor.normalize(iq_data)
    regions = detect_packets(normalized, threshold=threshold, min_gap=min_gap)

    packets: List[bytes] = []
    for start, end in regions[:max_packets]:
        segment = iq_data[start:end]
        # Convert IQ segment to bytes (raw complex64 representation).
        # The feature extractor operates on byte payloads.
        packets.append(segment.tobytes())

    return packets


def build_dataset(
    data_dir: str | Path,
    feature_types: Optional[List[FeatureType]] = None,
    threshold: float = 0.05,
    max_packets_per_file: int = 200,
    min_samples_per_class: int = 10,
) -> Tuple[FeatureMatrix, LabelArray, DatasetStats, List[str]]:
    """Build (X, y) training arrays from a directory of labeled IQ captures.

    Args:
        data_dir: Root directory containing per-label subdirectories.
        feature_types: Feature extraction types (defaults to classifier default set).
        threshold: Packet detection threshold (0–1, relative to signal peak).
        max_packets_per_file: Maximum packets extracted per IQ file.
        min_samples_per_class: Classes with fewer samples are logged as warnings.

    Returns:
        X: float32 feature matrix, shape (n_samples, n_features)
        y: string label array, shape (n_samples,)
        stats: DatasetStats summary
        feature_names: List of feature names (length == n_features)

    Raises:
        ValueError: If no usable samples could be extracted.
        FileNotFoundError: If data_dir does not exist.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if feature_types is None:
        feature_types = ClassifierConfig().feature_types

    extractor = AdvancedFeatureExtractor(feature_types=feature_types)
    stats = DatasetStats()

    all_features: List[npt.NDArray[np.float32]] = []
    all_labels: List[str] = []
    feature_names: List[str] = []

    iq_files = sorted(data_dir.rglob("*.iq"))
    if not iq_files:
        raise ValueError(f"No .iq files found under {data_dir}")

    logger.info(f"Found {len(iq_files)} IQ files in {data_dir}")

    for iq_path in iq_files:
        stats.total_files += 1
        label = _read_label(iq_path)

        try:
            iq_data = read_iq_file(str(iq_path))
        except Exception as e:
            logger.warning(f"Failed to load {iq_path}: {e}")
            stats.failed_files += 1
            continue

        if len(iq_data) < 1024:
            logger.warning(f"Skipping {iq_path}: too short ({len(iq_data)} samples)")
            stats.failed_files += 1
            continue

        packets = _extract_packets_from_iq(
            iq_data,
            threshold=threshold,
            max_packets=max_packets_per_file,
        )

        if not packets:
            logger.warning(
                f"No packets detected in {iq_path} — check threshold or signal quality"
            )
            stats.failed_files += 1
            continue

        for packet_bytes in packets:
            try:
                features, names = extractor.extract_features(packet_bytes)
                if len(features) == 0:
                    continue
                if not np.all(np.isfinite(features)):
                    continue  # skip degenerate feature vectors
                all_features.append(features)
                all_labels.append(label)
                if not feature_names:
                    feature_names = names
                stats.total_packets += 1
                stats.class_counts[label] = stats.class_counts.get(label, 0) + 1
            except Exception as e:
                logger.debug(f"Feature extraction failed for packet in {iq_path}: {e}")

        logger.info(f"  {iq_path.name}: label={label}, packets={len(packets)}")

    if not all_features:
        raise ValueError(
            "No usable feature vectors extracted.  Check that your .iq files contain "
            "real drone traffic and that the detection threshold is appropriate."
        )

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=str)
    stats.feature_dim = X.shape[1]

    # Warn on imbalanced or thin classes
    for cls, count in stats.class_counts.items():
        if count < min_samples_per_class:
            logger.warning(
                f"Class '{cls}' has only {count} samples (minimum recommended: "
                f"{min_samples_per_class}).  Capture more data for this protocol."
            )

    logger.info(
        f"Dataset built: {X.shape[0]} samples, {X.shape[1]} features, "
        f"{len(stats.class_counts)} classes: {dict(stats.class_counts)}"
    )

    return X, y, stats, feature_names
