from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from validation.types import RunResult

logger = logging.getLogger(__name__)


def result_to_dict(result: RunResult) -> Dict[str, Any]:
    return {
        "detection": asdict(result.detection),
        "classification": asdict(result.classification),
        "manifest": asdict(result.manifest),
        "profile_id": asdict(result.profile_id) if result.profile_id else None,
        "coded_link": asdict(result.coded_link) if result.coded_link else None,
    }


def _write_plots(result: RunResult, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # matplotlib optional ([viz] extra)
        logger.warning("Plots requested but matplotlib unavailable: %s", exc)
        return

    stem = path.with_suffix("")
    acc = result.classification.accuracy_by_snr
    if acc:
        xs = sorted(acc)
        fig, ax = plt.subplots()
        ax.plot(xs, [acc[x] for x in xs], marker="o")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_title("Classification accuracy vs SNR")
        fig.savefig(f"{stem}_accuracy_by_snr.png", dpi=120)
        plt.close(fig)

    conf = result.classification.confusion
    if conf:
        labels = sorted(conf)
        mat = [[conf[t].get(p, 0) for p in labels] for t in labels]
        fig, ax = plt.subplots()
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Truth")
        ax.set_title("Confusion")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(f"{stem}_confusion.png", dpi=120)
        plt.close(fig)


def write_report(result: RunResult, path: Path, plots: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result_to_dict(result), indent=2, default=str))
    if plots:
        _write_plots(result, path)
