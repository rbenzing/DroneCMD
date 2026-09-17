"""Real-capture ingest: load an on-disk IQ capture into a LabeledCapture.

Mirrors the labeling convention used by ``training/dataset.py``:

    data_dir/
      dji_ocusync/
        flight01.iq
        flight01.json    # sidecar with {"protocol": "dji_ocusync", ...}

The protocol label is read from the JSON sidecar (or a SigMF ``.sigmf-meta``
sidecar) when present; otherwise it falls back to the parent directory name.
Ground-truth regions, when available, come from SigMF ``annotations``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.fileio import read_iq_file
from validation.types import IQSamples, LabeledCapture


def _read_sidecar(iq_path: Path) -> Dict[str, Any]:
    """Return sidecar metadata for ``iq_path``, or ``{}`` if none exists."""
    for ext in (".json", ".sigmf-meta"):
        candidate = iq_path.with_suffix(ext)
        if candidate.exists():
            try:
                loaded: Any = json.loads(candidate.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
            return loaded if isinstance(loaded, dict) else {}
    return {}


def _regions_from_sigmf(meta: Dict[str, Any]) -> Optional[List[Tuple[int, int, str]]]:
    """Convert SigMF ``annotations`` into (start, end, protocol) tuples."""
    annotations = meta.get("annotations")
    if not annotations:
        return None
    regions: List[Tuple[int, int, str]] = []
    for annotation in annotations:
        start = int(annotation.get("core:sample_start", 0))
        count = int(annotation.get("core:sample_count", 0))
        protocol = str(annotation.get("core:description", "unknown"))
        if count > 0:
            regions.append((start, start + count, protocol))
    return regions or None


def load_labeled(iq_path: Path, sample_rate: Optional[float] = None) -> LabeledCapture:
    """Load a real IQ capture from disk into a :class:`LabeledCapture`.

    Args:
        iq_path: Path to the raw IQ capture file (e.g. ``flight01.iq``).
        sample_rate: Sample rate to record on the capture. When omitted,
            falls back to a ``sample_rate`` key in the sidecar metadata,
            else ``0.0``.

    Returns:
        A LabeledCapture with the protocol label, optional SigMF-derived
        truth regions, and provenance metadata. When the sidecar carries a
        persisted ``"provenance"`` dict (as written by
        ``LabeledDataset.write``), it is preserved as the base -- so a
        reloaded synthetic capture keeps ``source="synth"``, ``snr_db``,
        ``payload_hex``, ``scheme``, ``requested_snr_db``, and ``seed`` --
        with ``path``/``protocol``/``source`` filled in only where absent.
        A genuine real capture with no persisted provenance still gets
        ``source="real"``.
    """
    iq_path = Path(iq_path)
    iq: IQSamples = read_iq_file(iq_path).astype(np.complex64)
    meta = _read_sidecar(iq_path)
    protocol = meta.get("protocol") or iq_path.parent.name
    sample_rate = sample_rate or float(meta.get("sample_rate", 0.0)) or 0.0

    persisted_provenance = meta.get("provenance")
    prov: Dict[str, Any] = (
        dict(persisted_provenance) if isinstance(persisted_provenance, dict) else {}
    )
    prov.setdefault("source", "real")  # a genuine real capture stays "real"
    prov["path"] = str(iq_path)
    prov.setdefault("protocol", protocol)  # already resolved above

    return LabeledCapture(
        iq=iq,
        sample_rate=sample_rate,
        truth_regions=_regions_from_sigmf(meta),
        provenance=prov,
    )
