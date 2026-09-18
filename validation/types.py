from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

IQSamples = npt.NDArray[np.complex64]


class ModScheme(Enum):
    FSK = "fsk"
    GFSK = "gfsk"
    BPSK = "bpsk"
    QPSK = "qpsk"
    OFDM = "ofdm"


@dataclass
class ChannelParams:
    snr_db: float
    cfo_hz: float = 0.0
    doppler_hz: float = 0.0
    multipath_taps: Tuple[complex, ...] = ()
    timing_offset: int = 0


@dataclass
class LabeledCapture:
    iq: IQSamples
    sample_rate: float
    truth_regions: Optional[
        List[Tuple[int, int, str]]
    ]  # (start, end, protocol) or None
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Detection:
    start: int
    end: int
    protocol: str
    confidence: float
    resolved_profile: Optional[str] = None
    payload: Optional[bytes] = None


@dataclass
class DetectionMetrics:
    pd: float
    pfa_per_sec: float
    pfa_per_window: float
    tp: int
    fp: int
    fn: int
    roc: List[Tuple[float, float]] = field(default_factory=list)  # ROC points
    min_detectable_snr_db: Optional[float] = None
    ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class ClassificationMetrics:
    accuracy: float
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    per_class: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )  # precision/recall/f1
    accuracy_by_snr: Dict[float, float] = field(default_factory=dict)
    ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class ProfileIdMetrics:
    accuracy: float
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    accuracy_by_snr: Dict[float, float] = field(default_factory=dict)
    ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class CodedLinkMetrics:
    coded_ber: float
    fer: float
    ber_by_snr: Dict[float, float] = field(default_factory=dict)
    fer_by_snr: Dict[float, float] = field(default_factory=dict)
    ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class RunManifest:
    seed: int
    dataset_hash: str
    config_hash: str
    git_commit: str
    timestamp: str
    versions: Dict[str, str] = field(default_factory=dict)
    model_hash: Optional[str] = None


@dataclass
class RunResult:
    detection: DetectionMetrics
    classification: ClassificationMetrics
    manifest: RunManifest
    profile_id: Optional[ProfileIdMetrics] = None
    coded_link: Optional[CodedLinkMetrics] = None
