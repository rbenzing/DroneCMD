from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Optional

import numpy as np

from validation.types import RunManifest


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def hash_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def hash_config(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _versions() -> Dict[str, str]:
    names = ["numpy", "scipy", "scikit-learn"]
    out: Dict[str, str] = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = "unknown"
    return out


def capture_manifest(
    seed: int,
    dataset_hash: str,
    config_hash: str,
    model_hash: Optional[str] = None,
) -> RunManifest:
    return RunManifest(
        seed=seed,
        dataset_hash=dataset_hash,
        config_hash=config_hash,
        git_commit=git_commit(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        versions=_versions(),
        model_hash=model_hash,
    )
