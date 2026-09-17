"""Unified, SigMF-backed labeled dataset for validation & T&E.

``LabeledDataset`` wraps a list of :class:`~validation.types.LabeledCapture`
objects and provides a content-addressed hash plus on-disk persistence.

Persistence uses a dual-metadata layout: :func:`write` writes the SigMF
``.sigmf-data``/``.sigmf-meta`` pair (via ``utils.fileio.write_iq_file``) so
the samples round-trip through the standard SigMF reader, and additionally
writes a ``.json`` sidecar containing ``{sample_rate, protocol, annotations,
provenance}``. The ``.json`` sidecar is the reliable label carrier: the
SigMF writer in ``utils.fileio`` does not place ``annotations`` under the
SigMF ``annotations`` key (it nests arbitrary metadata under
``global.user:*`` instead), so ``validation.ingest.labeler.load_labeled``
reads the ``.json`` sidecar first when both exist.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from utils.fileio import FileFormat, write_iq_file
from validation.ingest.labeler import load_labeled
from validation.repro import hash_array
from validation.types import LabeledCapture


class LabeledDataset:
    """An ordered collection of labeled IQ captures.

    Args:
        captures: The labeled captures making up the dataset.
    """

    def __init__(self, captures: List[LabeledCapture]) -> None:
        self._captures = list(captures)

    def __iter__(self) -> Iterator[LabeledCapture]:
        return iter(self._captures)

    def __len__(self) -> int:
        return len(self._captures)

    def content_hash(self) -> str:
        """Return a sha256 hash over each capture's IQ data, truth regions,
        and sample rate.

        The hash is stable across calls for identical data and sensitive to
        any change in sample values, ground-truth labeling, or sample rate
        (so datasets differing only by sample rate hash differently).
        """
        digest = hashlib.sha256()
        for capture in self._captures:
            digest.update(hash_array(capture.iq).encode())
            digest.update(json.dumps(capture.truth_regions, sort_keys=True).encode())
            digest.update(str(capture.sample_rate).encode())
        return digest.hexdigest()

    def write(self, out_dir: Path) -> None:
        """Persist every capture to ``out_dir`` as SigMF pairs + JSON sidecars.

        For each capture this writes ``capture_NNNNN.sigmf-data`` and
        ``capture_NNNNN.sigmf-meta`` (via ``write_iq_file``), plus a
        ``capture_NNNNN.json`` sidecar carrying ``sample_rate``, ``protocol``,
        ``annotations`` (SigMF-style, derived from ``truth_regions``), and
        ``provenance``. See the module docstring for why the JSON sidecar is
        the authoritative label source.

        Args:
            out_dir: Directory to write the dataset into; created if needed.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for index, capture in enumerate(self._captures):
            base = out_dir / f"capture_{index:05d}"
            annotations: List[Dict[str, Any]] = [
                {
                    "core:sample_start": int(start),
                    "core:sample_count": int(end - start),
                    "core:description": protocol,
                }
                for (start, end, protocol) in (capture.truth_regions or [])
            ]
            meta: Dict[str, Any] = {
                "sample_rate": capture.sample_rate,
                "protocol": capture.provenance.get("protocol", "unknown"),
                "annotations": annotations,
                "provenance": capture.provenance,
            }
            write_iq_file(
                base.with_suffix(".sigmf-data"),
                capture.iq,
                file_format=FileFormat.SIGMF,
                metadata=meta,
            )
            base.with_suffix(".json").write_text(json.dumps(meta, default=str))

    @classmethod
    def from_dir(
        cls, path: Path, sample_rate: Optional[float] = None
    ) -> "LabeledDataset":
        """Load a dataset previously persisted by :meth:`write`.

        Prefers SigMF data files (``*.sigmf-data``); falls back to raw
        ``*.iq`` files (recursively) when none are found, mirroring the
        real-capture layout consumed by ``validation.ingest.labeler``.

        Args:
            path: Directory to load captures from.
            sample_rate: Sample rate override passed through to
                ``load_labeled`` for each capture.
        """
        path = Path(path)
        candidates: List[Path] = sorted(path.glob("*.sigmf-data")) or sorted(
            path.rglob("*.iq")
        )
        captures = [
            load_labeled(candidate, sample_rate=sample_rate) for candidate in candidates
        ]
        return cls(captures)
