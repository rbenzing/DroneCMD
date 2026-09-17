# Validation & T&E Spine (SP1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained `validation/` package that turns DroneCMD's detect→demod→classify pipeline into measured, reproducible, auditable performance numbers over synthetic + real-ingested ground truth.

**Architecture:** New top-level package (flat layout, mirrors `core/`). Synthetic modulators + a calibrated channel model generate labeled IQ; a real-capture labeler ingests `.iq`/SigMF; both feed a unified `LabeledDataset`. A dependency-injected `DetectClassifyPipeline` wraps the existing detector, demodulator, and classifier; pure-function metrics score detections and classifications with bootstrap CIs; a harness emits a `RunResult` with a reproducibility manifest; a `report` module serializes JSON. A `dronecmd validate` CLI drives it.

**Tech Stack:** Python 3.9+, numpy, scipy, scikit-learn, joblib (all already declared), matplotlib (optional, behind `[viz]`). SigMF via existing `utils/fileio`. pytest for tests.

**Spec:** `docs/superpowers/specs/2026-09-17-validation-te-spine-design.md` (v1.0.0). Executors read both; the plan argues from the spec.

## Global Constraints

- **Python floor:** 3.9 (repo targets 3.9; no `match`, no PEP 604 `X | Y` in annotations evaluated at runtime — use `Optional[...]`/`Union[...]` from `typing`).
- **IQ dtype:** all IQ arrays are `numpy.complex64`. Type alias `IQSamples = npt.NDArray[np.complex64]`.
- **Imports:** flat layout — import as `from core.signal_processing import ...`, `from utils.fileio import ...`, `from validation.types import ...`. No `dronecmd.` prefix.
- **Determinism:** never call `np.random.*` free functions; always thread a `numpy.random.Generator` from `validation.repro.rng(seed)`.
- **No new heavy deps.** matplotlib import must be lazy and guarded (only inside `report.write_report` when `plots=True`).
- **Type safety:** strict mypy is configured; annotate everything, avoid `Any` except in serialization dicts. Run `mypy validation` per task.
- **Every commit message ends with the attribution line:**
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- **Quality gate per task:** `black validation tests && isort validation tests && flake8 validation tests && mypy validation` and the task's tests must pass before commit.
- **Test markers:** reuse existing `slow`/`integration`/`hardware`. SP1 needs no hardware; the one end-to-end CLI test is marked `integration`.

---

### Task 1: Package skeleton + `types.py`

**Files:**
- Create: `validation/__init__.py` (empty placeholder for now — real API in Task 13)
- Create: `validation/types.py`
- Test: `tests/validation/__init__.py` (empty), `tests/validation/test_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `IQSamples`; enum `ModScheme{FSK,GFSK,QPSK,OFDM}`; dataclasses `ChannelParams`, `LabeledCapture`, `Detection`, `DetectionMetrics`, `ClassificationMetrics`, `RunManifest`, `RunResult`. Exact fields below — every later task depends on these names.

- [ ] **Step 1: Write the failing test**

```python
# tests/validation/test_types.py
from __future__ import annotations
import numpy as np
from validation.types import (
    ModScheme, ChannelParams, LabeledCapture, Detection,
    DetectionMetrics, ClassificationMetrics, RunManifest, RunResult,
)

def test_modscheme_values():
    assert {s.value for s in ModScheme} == {"fsk", "gfsk", "qpsk", "ofdm"}

def test_labeled_capture_construction():
    iq = np.zeros(16, dtype=np.complex64)
    cap = LabeledCapture(iq=iq, sample_rate=2_048_000.0,
                         truth_regions=[(0, 8, "mavlink")],
                         provenance={"source": "synth", "snr_db": 10.0})
    assert cap.iq.dtype == np.complex64
    assert cap.truth_regions[0] == (0, 8, "mavlink")
    assert cap.provenance["source"] == "synth"

def test_detection_fields():
    d = Detection(start=0, end=10, protocol="dji", confidence=0.9)
    assert (d.start, d.end, d.protocol, d.confidence) == (0, 10, "dji", 0.9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'validation'`.

- [ ] **Step 3: Write minimal implementation**

```python
# validation/__init__.py
"""DroneCMD validation & T&E spine (SP1)."""
```

```python
# validation/types.py
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
    truth_regions: Optional[List[Tuple[int, int, str]]]  # (start, end, protocol) or None
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Detection:
    start: int
    end: int
    protocol: str
    confidence: float


@dataclass
class DetectionMetrics:
    pd: float
    pfa_per_sec: float
    pfa_per_window: float
    tp: int
    fp: int
    fn: int
    roc: List[Tuple[float, float]] = field(default_factory=list)  # (pfa, pd) points
    min_detectable_snr_db: Optional[float] = None
    ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class ClassificationMetrics:
    accuracy: float
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)  # precision/recall/f1
    accuracy_by_snr: Dict[float, float] = field(default_factory=dict)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_types.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add validation/__init__.py validation/types.py tests/validation/
git commit -m "feat(validation): add SP1 package skeleton and core types

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `repro.py` — seeding, hashing, manifest

**Files:**
- Create: `validation/repro.py`
- Test: `tests/validation/test_repro.py`

**Interfaces:**
- Consumes: `validation.types.RunManifest`.
- Produces: `rng(seed: int) -> np.random.Generator`; `hash_array(a: np.ndarray) -> str`; `hash_config(obj: Any) -> str`; `git_commit() -> str`; `capture_manifest(seed: int, dataset_hash: str, config_hash: str, model_hash: Optional[str] = None) -> RunManifest`.

- [ ] **Step 1: Write the failing test**

```python
# tests/validation/test_repro.py
from __future__ import annotations
import numpy as np
from validation.repro import rng, hash_array, hash_config, capture_manifest

def test_rng_is_deterministic():
    a = rng(42).standard_normal(100)
    b = rng(42).standard_normal(100)
    assert np.array_equal(a, b)

def test_rng_differs_by_seed():
    assert not np.array_equal(rng(1).standard_normal(50), rng(2).standard_normal(50))

def test_hash_array_stable_and_sensitive():
    x = np.arange(10, dtype=np.complex64)
    assert hash_array(x) == hash_array(x.copy())
    y = x.copy(); y[0] += 1
    assert hash_array(x) != hash_array(y)

def test_hash_config_order_independent():
    assert hash_config({"a": 1, "b": 2}) == hash_config({"b": 2, "a": 1})

def test_capture_manifest_has_versions():
    m = capture_manifest(seed=42, dataset_hash="d", config_hash="c")
    assert m.seed == 42 and m.dataset_hash == "d"
    assert "numpy" in m.versions
    assert isinstance(m.timestamp, str) and m.timestamp
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_repro.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'validation.repro'`.

- [ ] **Step 3: Write minimal implementation**

```python
# validation/repro.py
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
            capture_output=True, text=True, timeout=5, check=False,
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_repro.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add validation/repro.py tests/validation/test_repro.py
git commit -m "feat(validation): add reproducibility primitives (rng, hashing, manifest)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Bug-audit F1 — fix `core/signal_processing.detect_packets`

Systematic-debugging cycle for finding **F1** (spec §11/§16): the detector computes `power = np.abs(iq)` (magnitude, not power) and has identical `if np.iscomplexobj / else` branches (dead code). Fix aligns it with the codebase's power convention (`calculate_power`, `estimate_snr`, `normalize` all use `|x|**2`).

**Files:**
- Modify: `core/signal_processing.py:475-482` (the power computation + dead branch inside `detect_packets`)
- Test: `tests/test_signal_processing.py` (add a distinguishing test to `TestDetectPackets`)
- Modify (log): `docs/superpowers/specs/2026-09-17-validation-te-spine-design.md` (§16 appendix, mark F1 fixed)

**Interfaces:**
- Consumes: existing `core.signal_processing.detect_packets(iq, threshold, min_gap)`.
- Produces: same signature; behavior now uses instantaneous power `|x|**2`.

- [ ] **Step 1: Write the failing test (RED)**

Magnitude vs power differ when a sample's amplitude `a` satisfies `a > f*A` but `a**2 < f*A**2` (relative threshold `f`, peak `A`). With peak `1.0`, `f=0.5`, a plateau at amplitude `0.6`: magnitude includes it (`0.6>0.5`), power excludes it (`0.36<0.5`).

```python
# add to tests/test_signal_processing.py, class TestDetectPackets
    def test_uses_power_not_magnitude_semantics(self):
        """A 0.6-amplitude plateau next to a 1.0 peak must NOT be detected
        under power semantics (0.36 < 0.5*1.0), even though magnitude (0.6)
        would exceed a 0.5 relative threshold."""
        plateau = (np.ones(400) * 0.6).astype(np.complex64)
        peak = (np.ones(50) * 1.0).astype(np.complex64)
        silence = np.zeros(200, dtype=np.complex64)
        signal = np.concatenate([silence, plateau, silence, peak, silence])
        regions = detect_packets(signal, threshold=0.5, min_gap=100)
        # Only the 1.0 peak region survives; the 0.6 plateau is below power threshold.
        # (peak length 50 < min_gap 100, so with min_gap filtering we expect ZERO
        #  long regions — the plateau must not appear as a >100-sample region.)
        for start, end in regions:
            # No detected region may fall inside the plateau span [200, 600)
            assert not (start >= 200 and end <= 600), (
                f"plateau region {(start, end)} detected under magnitude semantics"
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_signal_processing.py::TestDetectPackets::test_uses_power_not_magnitude_semantics -v`
Expected: FAIL — current magnitude code detects the 0.6 plateau as a >100-sample region.

- [ ] **Step 3: Fix implementation**

In `core/signal_processing.py`, inside `detect_packets`, replace the identical-branch magnitude block:

```python
    # BEFORE:
    # if np.iscomplexobj(iq_samples):
    #     power = np.abs(iq_samples)
    # else:
    #     power = np.abs(iq_samples)

    # AFTER — instantaneous power, consistent with calculate_power()/estimate_snr():
    power = np.abs(iq_samples).astype(np.float64) ** 2
```

Leave the rest (`threshold = threshold * np.max(power)` relative-threshold logic, edge handling, `min_gap` filter) unchanged.

- [ ] **Step 4: Run tests to verify pass + no regressions**

Run: `pytest tests/test_signal_processing.py -v`
Expected: PASS — new test passes and all existing `TestDetectPackets` tests (`test_detects_burst_above_noise`, `test_no_false_positives_on_silence`, `test_returns_list_of_tuples`) still pass.

- [ ] **Step 5: Log finding + commit**

In the spec §16, update F1 to: `**F1 (fixed):** detect_packets now uses |x|**2 and the dead branch is removed. RED test: test_uses_power_not_magnitude_semantics.`

```bash
git add core/signal_processing.py tests/test_signal_processing.py docs/superpowers/specs/2026-09-17-validation-te-spine-design.md
git commit -m "fix(dsp): detect_packets uses power |x|^2, not magnitude; drop dead branch

Aligns energy detection with the codebase power convention (calculate_power,
estimate_snr, normalize). Bug-audit finding F1. RED test added.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `synth/modulators.py` — FSK, GFSK, QPSK

**Files:**
- Create: `validation/synth/__init__.py` (empty)
- Create: `validation/synth/modulators.py`
- Test: `tests/validation/test_modulators.py`

**Interfaces:**
- Consumes: `validation.types.ModScheme`, `IQSamples`.
- Produces: `modulate(data: bytes, scheme: ModScheme, sps: int = 8, *, mod_index: float = 0.7, bt: float = 0.5, rolloff: float = 0.35) -> IQSamples`. Output is unit-average-power `complex64`, length `= n_symbols * sps` where `n_symbols = 8*len(data)` for FSK/GFSK (1 bit/symbol) and `= 4*len(data)` for QPSK (2 bits/symbol).
- Also produces test helpers `_ref_demod_fsk(iq, sps) -> bytes` and `_ref_demod_qpsk(iq, sps) -> bytes` **in the test file** (reference demodulators — not shipped).

- [ ] **Step 1: Write the failing test (round-trip correctness anchor)**

```python
# tests/validation/test_modulators.py
from __future__ import annotations
import numpy as np
from validation.types import ModScheme
from validation.synth.modulators import modulate

DATA = bytes([0b10110010, 0b01011101, 0xA5, 0x3C])

def _bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

def _ref_demod_fsk(iq: np.ndarray, sps: int) -> np.ndarray:
    # instantaneous frequency = d(phase)/dt; sign at symbol centre -> bit
    phase = np.unwrap(np.angle(iq))
    inst_freq = np.diff(phase, prepend=phase[0])
    n_sym = len(iq) // sps
    bits = np.empty(n_sym, dtype=np.uint8)
    for k in range(n_sym):
        seg = inst_freq[k * sps + sps // 4: k * sps + 3 * sps // 4]
        bits[k] = 1 if np.mean(seg) > 0 else 0
    return bits

def _ref_demod_qpsk(iq: np.ndarray, sps: int) -> np.ndarray:
    n_sym = len(iq) // sps
    out = []
    for k in range(n_sym):
        c = iq[k * sps + sps // 2]
        i_bit = 0 if c.real >= 0 else 1
        q_bit = 0 if c.imag >= 0 else 1
        out.extend([i_bit, q_bit])
    return np.array(out, dtype=np.uint8)

def test_fsk_roundtrip_recovers_bits():
    iq = modulate(DATA, ModScheme.FSK, sps=8)
    assert iq.dtype == np.complex64
    assert len(iq) == 8 * len(DATA) * 8  # sps * n_bits
    rec = _ref_demod_fsk(iq, sps=8)
    assert np.array_equal(rec, _bits(DATA))

def test_qpsk_roundtrip_recovers_bits():
    iq = modulate(DATA, ModScheme.QPSK, sps=8)
    assert len(iq) == 8 * (len(DATA) * 8 // 2)
    rec = _ref_demod_qpsk(iq, sps=8)
    assert np.array_equal(rec, _bits(DATA))

def test_gfsk_roundtrip_recovers_bits():
    iq = modulate(DATA, ModScheme.GFSK, sps=8, bt=0.5)
    rec = _ref_demod_fsk(iq, sps=8)
    assert np.array_equal(rec, _bits(DATA))

def test_output_average_power_normalized():
    iq = modulate(DATA, ModScheme.FSK, sps=8)
    p = float(np.mean(np.abs(iq) ** 2))
    assert abs(p - 1.0) < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_modulators.py -v`
Expected: FAIL — `ModuleNotFoundError: validation.synth.modulators`.

- [ ] **Step 3: Write minimal implementation**

```python
# validation/synth/modulators.py
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d

from validation.types import IQSamples, ModScheme


def _bits_from_bytes(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.float64)


def _fsk(bits: np.ndarray, sps: int, mod_index: float, gaussian_bt: float | None) -> np.ndarray:
    symbols = 2.0 * bits - 1.0  # {0,1} -> {-1,+1}
    shape = np.repeat(symbols, sps)
    if gaussian_bt is not None:
        # Gaussian pulse shaping: sigma from BT product over one symbol period
        sigma = sps * np.sqrt(np.log(2)) / (2 * np.pi * gaussian_bt)
        shape = gaussian_filter1d(shape, sigma=max(sigma, 1e-3), mode="nearest")
    # frequency deviation: peak phase step so a symbol advances mod_index cycles
    freq = (mod_index / sps) * shape  # cycles per sample
    phase = 2 * np.pi * np.cumsum(freq)
    return np.exp(1j * phase)


def _qpsk(bits: np.ndarray, sps: int) -> np.ndarray:
    if len(bits) % 2 == 1:
        bits = np.append(bits, 0.0)
    i_bits = bits[0::2]
    q_bits = bits[1::2]
    # Gray-ish direct mapping: 0 -> +1/sqrt2, 1 -> -1/sqrt2
    i = (1 - 2 * i_bits) / np.sqrt(2)
    q = (1 - 2 * q_bits) / np.sqrt(2)
    symbols = i + 1j * q
    return np.repeat(symbols, sps)  # rectangular pulse shaping (sufficient for SP1)


def modulate(
    data: bytes,
    scheme: ModScheme,
    sps: int = 8,
    *,
    mod_index: float = 0.7,
    bt: float = 0.5,
    rolloff: float = 0.35,
) -> IQSamples:
    """Modulate ``data`` bytes to complex64 IQ, unit average power.

    FSK/GFSK: 1 bit/symbol; QPSK: 2 bits/symbol. Deterministic (no RNG).
    OFDM is intentionally unsupported in SP1 (see Task 15).
    """
    if len(data) == 0:
        return np.zeros(0, dtype=np.complex64)
    bits = _bits_from_bytes(data)
    if scheme == ModScheme.FSK:
        iq = _fsk(bits, sps, mod_index, gaussian_bt=None)
    elif scheme == ModScheme.GFSK:
        iq = _fsk(bits, sps, mod_index, gaussian_bt=bt)
    elif scheme == ModScheme.QPSK:
        iq = _qpsk(bits, sps)
    else:
        raise ValueError(f"modulate() does not support {scheme} in SP1")
    p = np.mean(np.abs(iq) ** 2)
    if p > 0:
        iq = iq / np.sqrt(p)
    return iq.astype(np.complex64)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_modulators.py -v`
Expected: PASS (4 passed). If GFSK round-trip is marginal, the fix is to keep `bt=0.5` (already the default) — sigma stays small enough that symbol centres are unambiguous.

- [ ] **Step 5: Commit**

```bash
git add validation/synth/__init__.py validation/synth/modulators.py tests/validation/test_modulators.py
git commit -m "feat(validation): add FSK/GFSK/QPSK modulators with round-trip tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `synth/channel.py` — calibrated AWGN + impairments

**Files:**
- Create: `validation/synth/channel.py`
- Test: `tests/validation/test_channel.py`

**Interfaces:**
- Consumes: `validation.types.{IQSamples, ChannelParams}`, `validation.repro.rng`.
- Produces:
  - `add_awgn_at_snr(signal: IQSamples, snr_db: float, generator: np.random.Generator) -> Tuple[IQSamples, float, float]` returning `(noisy, noise_std, achieved_snr_db)`. Signal power measured over the whole array (caller passes an all-active packet).
  - `apply_channel(signal: IQSamples, params: ChannelParams, generator: np.random.Generator) -> Tuple[IQSamples, float]` returning `(iq, achieved_snr_db)`; applies, in order: timing offset (prepend zeros), multipath FIR, CFO+Doppler phase rotation, then calibrated AWGN.

- [ ] **Step 1: Write the failing test (SNR calibration is the landmine)**

```python
# tests/validation/test_channel.py
from __future__ import annotations
import numpy as np
from validation.types import ChannelParams
from validation.repro import rng
from validation.synth.channel import add_awgn_at_snr, apply_channel

def _measure_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    noise = noisy - clean
    s = np.mean(np.abs(clean) ** 2)
    n = np.mean(np.abs(noise) ** 2)
    return 10 * np.log10(s / n)

def test_awgn_hits_target_snr():
    g = rng(0)
    clean = (np.exp(1j * 2 * np.pi * 0.03 * np.arange(20000))).astype(np.complex64)
    clean /= np.sqrt(np.mean(np.abs(clean) ** 2))
    for target in (-5.0, 0.0, 10.0, 20.0):
        noisy, _std, achieved = add_awgn_at_snr(clean, target, g)
        assert abs(achieved - target) < 0.5, (target, achieved)
        assert abs(_measure_snr_db(clean, noisy) - target) < 0.7

def test_awgn_is_deterministic_by_seed():
    clean = np.ones(1000, dtype=np.complex64)
    a, _, _ = add_awgn_at_snr(clean, 10.0, rng(7))
    b, _, _ = add_awgn_at_snr(clean, 10.0, rng(7))
    assert np.allclose(a, b)

def test_apply_channel_timing_offset_shifts():
    clean = np.ones(100, dtype=np.complex64)
    p = ChannelParams(snr_db=40.0, timing_offset=10)
    out, _ = apply_channel(clean, p, rng(1))
    assert len(out) == 110
    assert np.mean(np.abs(out[:10])) < np.mean(np.abs(out[10:]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_channel.py -v`
Expected: FAIL — `ModuleNotFoundError: validation.synth.channel`.

- [ ] **Step 3: Write minimal implementation**

```python
# validation/synth/channel.py
from __future__ import annotations
from typing import Tuple
import numpy as np

from validation.types import ChannelParams, IQSamples


def add_awgn_at_snr(
    signal: IQSamples,
    snr_db: float,
    generator: np.random.Generator,
) -> Tuple[IQSamples, float, float]:
    """Add circularly-symmetric complex AWGN calibrated to ``snr_db``.

    Signal power is measured over the array (pass an all-active packet).
    Returns (noisy, noise_std_per_component, achieved_snr_db).
    """
    sig_power = float(np.mean(np.abs(signal) ** 2))
    if sig_power <= 0:
        return signal.copy(), 0.0, float("inf")
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    std = np.sqrt(noise_power / 2.0)  # split across I and Q
    noise = (generator.standard_normal(len(signal))
             + 1j * generator.standard_normal(len(signal))) * std
    noisy = (signal + noise).astype(np.complex64)
    achieved = 10 * np.log10(sig_power / float(np.mean(np.abs(noise) ** 2)))
    return noisy, float(std), float(achieved)


def apply_channel(
    signal: IQSamples,
    params: ChannelParams,
    generator: np.random.Generator,
) -> Tuple[IQSamples, float]:
    x = signal.astype(np.complex64)
    if params.timing_offset > 0:
        x = np.concatenate([np.zeros(params.timing_offset, dtype=np.complex64), x])
    if params.multipath_taps:
        taps = np.asarray(params.multipath_taps, dtype=np.complex64)
        x = np.convolve(x, taps, mode="full")[: len(x)].astype(np.complex64)
    if params.cfo_hz or params.doppler_hz:
        n = np.arange(len(x))
        # cfo_hz/doppler_hz expressed in cycles/sample here (normalized)
        rot = np.exp(1j * 2 * np.pi * (params.cfo_hz + params.doppler_hz) * n)
        x = (x * rot).astype(np.complex64)
    noisy, _std, achieved = add_awgn_at_snr(x, params.snr_db, generator)
    return noisy, achieved
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_channel.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add validation/synth/channel.py tests/validation/test_channel.py
git commit -m "feat(validation): add calibrated AWGN channel + impairments

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: `synth/scenarios.py` — protocol × SNR grid → labeled captures

**Files:**
- Create: `validation/synth/scenarios.py`
- Test: `tests/validation/test_scenarios.py`

**Interfaces:**
- Consumes: `modulate`, `add_awgn_at_snr`, `validation.repro.rng`, `validation.types.{LabeledCapture, ModScheme}`.
- Produces: dataclass `DatasetSpec(protocols: List[str], snr_grid_db: List[float], n_per_cell: int, sample_rate: float, seed: int, scheme_by_protocol: Dict[str, ModScheme], payload_len: int = 32, guard: int = 512)`; `build_scenario(spec: DatasetSpec) -> List[LabeledCapture]`. Each capture: one packet placed after a `guard`-sample noise pad and followed by another; `truth_regions=[(guard, guard+packet_len, protocol)]`; guard noise uses the same `noise_std` as the packet's AWGN so the noise floor is consistent.

- [ ] **Step 1: Write the failing test**

```python
# tests/validation/test_scenarios.py
from __future__ import annotations
import numpy as np
from validation.types import ModScheme
from validation.synth.scenarios import DatasetSpec, build_scenario

def _spec(seed=42):
    return DatasetSpec(
        protocols=["mavlink", "dji"],
        snr_grid_db=[-10.0, 0.0, 20.0],
        n_per_cell=2,
        sample_rate=2_048_000.0,
        seed=seed,
        scheme_by_protocol={"mavlink": ModScheme.FSK, "dji": ModScheme.QPSK},
        payload_len=16,
    )

def test_cardinality_matches_grid():
    caps = build_scenario(_spec())
    assert len(caps) == 2 * 3 * 2  # protocols * snr * n_per_cell

def test_truth_regions_and_provenance():
    caps = build_scenario(_spec())
    for c in caps:
        assert c.truth_regions is not None and len(c.truth_regions) == 1
        start, end, proto = c.truth_regions[0]
        assert 0 <= start < end <= len(c.iq)
        assert proto in ("mavlink", "dji")
        assert c.provenance["source"] == "synth"
        assert "snr_db" in c.provenance
        assert c.iq.dtype == np.complex64

def test_deterministic_by_seed():
    a = build_scenario(_spec(1))[0].iq
    b = build_scenario(_spec(1))[0].iq
    assert np.array_equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_scenarios.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# validation/synth/scenarios.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from validation.repro import rng
from validation.synth.channel import add_awgn_at_snr
from validation.synth.modulators import modulate
from validation.types import IQSamples, LabeledCapture, ModScheme


@dataclass
class DatasetSpec:
    protocols: List[str]
    snr_grid_db: List[float]
    n_per_cell: int
    sample_rate: float
    seed: int
    scheme_by_protocol: Dict[str, ModScheme]
    payload_len: int = 32
    guard: int = 512
    sps: int = 8


def _noise(n: int, std: float, generator: np.random.Generator) -> IQSamples:
    if std <= 0 or n <= 0:
        return np.zeros(max(n, 0), dtype=np.complex64)
    z = generator.standard_normal(n) + 1j * generator.standard_normal(n)
    return (z * std).astype(np.complex64)


def build_scenario(spec: DatasetSpec) -> List[LabeledCapture]:
    g = rng(spec.seed)
    captures: List[LabeledCapture] = []
    for proto in spec.protocols:
        scheme = spec.scheme_by_protocol[proto]
        for snr in spec.snr_grid_db:
            for _ in range(spec.n_per_cell):
                payload = g.integers(0, 256, size=spec.payload_len, dtype=np.uint8).tobytes()
                clean = modulate(payload, scheme, sps=spec.sps)
                noisy_pkt, noise_std, achieved = add_awgn_at_snr(clean, snr, g)
                pre = _noise(spec.guard, noise_std, g)
                post = _noise(spec.guard, noise_std, g)
                iq = np.concatenate([pre, noisy_pkt, post]).astype(np.complex64)
                start = spec.guard
                end = spec.guard + len(noisy_pkt)
                captures.append(LabeledCapture(
                    iq=iq,
                    sample_rate=spec.sample_rate,
                    truth_regions=[(start, end, proto)],
                    provenance={
                        "source": "synth",
                        "protocol": proto,
                        "scheme": scheme.value,
                        "snr_db": float(achieved),
                        "requested_snr_db": float(snr),
                        "payload_hex": payload.hex(),
                        "seed": spec.seed,
                    },
                ))
    return captures
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_scenarios.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add validation/synth/scenarios.py tests/validation/test_scenarios.py
git commit -m "feat(validation): add synthetic scenario builder (protocol x SNR grid)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: `ingest/labeler.py` — real capture → LabeledCapture

**Files:**
- Create: `validation/ingest/__init__.py` (empty)
- Create: `validation/ingest/labeler.py`
- Test: `tests/validation/test_labeler.py`

**Interfaces:**
- Consumes: `utils.fileio.read_iq_file`, `validation.types.LabeledCapture`.
- Produces: `load_labeled(iq_path: Path, sample_rate: Optional[float] = None) -> LabeledCapture`. Protocol label from sidecar JSON (`{"protocol": ...}`) if present, else parent directory name (matches `training/dataset.py` convention). `truth_regions` from SigMF `annotations` (`core:sample_start` + `core:sample_count` + `core:description` as protocol) when present, else `None`. `provenance={"source": "real", "path": ..., "protocol": ...}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/validation/test_labeler.py
from __future__ import annotations
import json
import numpy as np
from validation.ingest.labeler import load_labeled

def test_label_from_sidecar_json(tmp_path):
    d = tmp_path / "dji_ocusync"
    d.mkdir()
    iq = (np.ones(64) * 0.5).astype(np.complex64)
    iq.tofile(d / "flight01.iq")
    (d / "flight01.json").write_text(json.dumps({"protocol": "dji_ocusync"}))
    cap = load_labeled(d / "flight01.iq", sample_rate=2_048_000.0)
    assert cap.provenance["protocol"] == "dji_ocusync"
    assert cap.provenance["source"] == "real"
    assert cap.iq.dtype == np.complex64
    assert len(cap.iq) == 64

def test_label_falls_back_to_parent_dir(tmp_path):
    d = tmp_path / "mavlink"
    d.mkdir()
    iq = np.ones(32, dtype=np.complex64)
    iq.tofile(d / "cap.iq")
    cap = load_labeled(d / "cap.iq", sample_rate=1e6)
    assert cap.provenance["protocol"] == "mavlink"
    assert cap.truth_regions is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_labeler.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# validation/ingest/labeler.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from utils.fileio import read_iq_file
from validation.types import IQSamples, LabeledCapture


def _read_sidecar(iq_path: Path) -> dict:
    for ext in (".json", ".sigmf-meta"):
        p = iq_path.with_suffix(ext)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return {}
    return {}


def _regions_from_sigmf(meta: dict) -> Optional[List[Tuple[int, int, str]]]:
    anns = meta.get("annotations")
    if not anns:
        return None
    out: List[Tuple[int, int, str]] = []
    for a in anns:
        start = int(a.get("core:sample_start", 0))
        count = int(a.get("core:sample_count", 0))
        proto = str(a.get("core:description", "unknown"))
        if count > 0:
            out.append((start, start + count, proto))
    return out or None


def load_labeled(iq_path: Path, sample_rate: Optional[float] = None) -> LabeledCapture:
    iq_path = Path(iq_path)
    iq: IQSamples = read_iq_file(iq_path).astype(np.complex64)
    meta = _read_sidecar(iq_path)
    protocol = meta.get("protocol") or iq_path.parent.name
    sr = sample_rate or float(meta.get("sample_rate", 0.0)) or 0.0
    return LabeledCapture(
        iq=iq,
        sample_rate=sr,
        truth_regions=_regions_from_sigmf(meta),
        provenance={"source": "real", "path": str(iq_path), "protocol": protocol},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_labeler.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add validation/ingest/__init__.py validation/ingest/labeler.py tests/validation/test_labeler.py
git commit -m "feat(validation): add real-capture labeler (sidecar + SigMF annotations)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: `dataset.py` — unified LabeledDataset (SigMF-backed)

**Files:**
- Create: `validation/dataset.py`
- Test: `tests/validation/test_dataset.py`

**Interfaces:**
- Consumes: `utils.fileio.{write_iq_file, FileFormat}`, `validation.ingest.labeler.load_labeled`, `validation.repro.hash_array`, `validation.types.LabeledCapture`.
- Produces: class `LabeledDataset` with `__init__(self, captures: List[LabeledCapture])`, `__iter__`, `__len__`, `content_hash() -> str` (sha256 over concatenated per-capture array hashes + labels), `write(self, out_dir: Path) -> None` (one `.sigmf-data`/`.sigmf-meta` pair + label json per capture), `classmethod from_dir(cls, path: Path, sample_rate: Optional[float] = None) -> "LabeledDataset"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/validation/test_dataset.py
from __future__ import annotations
import numpy as np
from validation.types import LabeledCapture
from validation.dataset import LabeledDataset

def _caps():
    return [
        LabeledCapture(iq=np.ones(32, dtype=np.complex64), sample_rate=1e6,
                       truth_regions=[(4, 20, "mavlink")],
                       provenance={"source": "synth", "protocol": "mavlink"}),
        LabeledCapture(iq=(np.arange(16).astype(np.complex64)), sample_rate=1e6,
                       truth_regions=[(0, 16, "dji")],
                       provenance={"source": "synth", "protocol": "dji"}),
    ]

def test_len_and_iter():
    ds = LabeledDataset(_caps())
    assert len(ds) == 2
    assert [c.provenance["protocol"] for c in ds] == ["mavlink", "dji"]

def test_content_hash_stable_and_sensitive():
    assert LabeledDataset(_caps()).content_hash() == LabeledDataset(_caps()).content_hash()
    mutated = _caps(); mutated[0].iq[0] += 1
    assert LabeledDataset(mutated).content_hash() != LabeledDataset(_caps()).content_hash()

def test_write_then_from_dir_roundtrip(tmp_path):
    LabeledDataset(_caps()).write(tmp_path)
    loaded = LabeledDataset.from_dir(tmp_path, sample_rate=1e6)
    assert len(loaded) == 2
    protos = sorted(c.provenance["protocol"] for c in loaded)
    assert protos == ["dji", "mavlink"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_dataset.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# validation/dataset.py
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Iterator, List, Optional

from utils.fileio import FileFormat, write_iq_file
from validation.ingest.labeler import load_labeled
from validation.repro import hash_array
from validation.types import LabeledCapture


class LabeledDataset:
    def __init__(self, captures: List[LabeledCapture]) -> None:
        self._captures = list(captures)

    def __iter__(self) -> Iterator[LabeledCapture]:
        return iter(self._captures)

    def __len__(self) -> int:
        return len(self._captures)

    def content_hash(self) -> str:
        h = hashlib.sha256()
        for c in self._captures:
            h.update(hash_array(c.iq).encode())
            h.update(json.dumps(c.truth_regions, sort_keys=True).encode())
        return h.hexdigest()

    def write(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(self._captures):
            base = out_dir / f"capture_{i:05d}"
            annotations = []
            for (start, end, proto) in (c.truth_regions or []):
                annotations.append({
                    "core:sample_start": int(start),
                    "core:sample_count": int(end - start),
                    "core:description": proto,
                })
            meta = {
                "sample_rate": c.sample_rate,
                "protocol": c.provenance.get("protocol", "unknown"),
                "annotations": annotations,
                "provenance": c.provenance,
            }
            write_iq_file(base.with_suffix(".sigmf-data"), c.iq,
                          file_format=FileFormat.SIGMF, metadata=meta)
            base.with_suffix(".json").write_text(json.dumps(meta, default=str))

    @classmethod
    def from_dir(cls, path: Path, sample_rate: Optional[float] = None) -> "LabeledDataset":
        path = Path(path)
        caps: List[LabeledCapture] = []
        # Prefer SigMF data files; fall back to raw .iq
        candidates = sorted(path.glob("*.sigmf-data")) or sorted(path.rglob("*.iq"))
        for p in candidates:
            caps.append(load_labeled(p, sample_rate=sample_rate))
        return cls(caps)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_dataset.py -v`
Expected: PASS (3 passed). If `write_iq_file` SigMF round-trip drops annotations, this surfaces bug-audit item §11.5 — log it in the spec §16 and, if needed, persist annotations via the sidecar `.json` (already written) which `load_labeled` reads.

- [ ] **Step 5: Commit**

```bash
git add validation/dataset.py tests/validation/test_dataset.py
git commit -m "feat(validation): add unified SigMF-backed LabeledDataset

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: `pipeline.py` — DetectClassifyPipeline (injectable)

**Files:**
- Create: `validation/pipeline.py`
- Test: `tests/validation/test_pipeline.py`

**Interfaces:**
- Consumes: `core.signal_processing.detect_packets`, `validation.types.{Detection, LabeledCapture}`. Classifier and detector are injected (duck-typed).
- Produces:
  - Type alias `DetectorFn = Callable[[IQSamples, float, int], List[Tuple[int, int]]]`.
  - `default_detector: DetectorFn` (wraps `core.signal_processing.detect_packets`).
  - `class DetectClassifyPipeline(classifier, detector: DetectorFn = default_detector, threshold: float = 0.05, min_gap: int = 256, use_truth_bytes: bool = False)`. `classifier` must expose `.classify(packet_bytes: bytes, signal_metrics: Optional[dict] = None) -> Union[str, ClassificationResult]`.
  - `.run(capture: LabeledCapture) -> List[Detection]`: detect regions → for each, slice IQ → obtain bytes (via injected demod-to-bytes helper `region_to_bytes`, or, if `use_truth_bytes`, from the overlapping truth region's `payload_hex`) → classify → build `Detection`.
  - `region_to_bytes(iq_region: IQSamples, sps: int = 8) -> bytes` using `numpy.packbits` over a simple FSK bit decision (self-contained; the demod-quality bug-audit item §11.3 is bounded by Task 4's round-trip test).

- [ ] **Step 1: Write the failing test (stubs — no models, no hardware)**

```python
# tests/validation/test_pipeline.py
from __future__ import annotations
import numpy as np
from validation.types import LabeledCapture, Detection
from validation.pipeline import DetectClassifyPipeline

class StubClassifier:
    """Returns a fixed protocol with confidence proportional to region energy."""
    def __init__(self, label="mavlink"):
        self.label = label
    def classify(self, packet_bytes, signal_metrics=None):
        return self.label

def _capture_with_burst():
    silence = np.zeros(300, dtype=np.complex64)
    burst = (np.ones(400) * 0.9).astype(np.complex64)
    iq = np.concatenate([silence, burst, silence])
    return LabeledCapture(iq=iq, sample_rate=1e6,
                          truth_regions=[(300, 700, "mavlink")],
                          provenance={"source": "synth", "protocol": "mavlink"})

def test_pipeline_detects_and_labels():
    pipe = DetectClassifyPipeline(StubClassifier("mavlink"), threshold=0.2, min_gap=100)
    dets = pipe.run(_capture_with_burst())
    assert len(dets) >= 1
    d = dets[0]
    assert isinstance(d, Detection)
    assert d.protocol == "mavlink"
    assert 250 <= d.start <= 350 and 650 <= d.end <= 750

def test_pipeline_no_detections_on_silence():
    cap = LabeledCapture(iq=np.zeros(2048, dtype=np.complex64), sample_rate=1e6,
                         truth_regions=None, provenance={"source": "synth"})
    pipe = DetectClassifyPipeline(StubClassifier(), threshold=0.05)
    assert pipe.run(cap) == []

def test_injectable_detector_is_used():
    calls = {"n": 0}
    def fake_detector(iq, threshold, min_gap):
        calls["n"] += 1
        return [(10, 50)]
    pipe = DetectClassifyPipeline(StubClassifier("dji"), detector=fake_detector)
    dets = pipe.run(_capture_with_burst())
    assert calls["n"] == 1
    assert dets[0].protocol == "dji" and (dets[0].start, dets[0].end) == (10, 50)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_pipeline.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# validation/pipeline.py
from __future__ import annotations
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from core.signal_processing import detect_packets
from validation.types import Detection, IQSamples, LabeledCapture

DetectorFn = Callable[[IQSamples, float, int], List[Tuple[int, int]]]


def default_detector(iq: IQSamples, threshold: float, min_gap: int) -> List[Tuple[int, int]]:
    return detect_packets(iq, threshold=threshold, min_gap=min_gap)


def region_to_bytes(iq_region: IQSamples, sps: int = 8) -> bytes:
    if len(iq_region) < sps:
        return b""
    phase = np.unwrap(np.angle(iq_region))
    inst_freq = np.diff(phase, prepend=phase[0])
    n_sym = len(iq_region) // sps
    bits = np.zeros(n_sym, dtype=np.uint8)
    for k in range(n_sym):
        seg = inst_freq[k * sps + sps // 4: k * sps + 3 * sps // 4]
        bits[k] = 1 if float(np.mean(seg)) > 0 else 0
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


def _confidence(result: Union[str, object]) -> Tuple[str, float]:
    proto = getattr(result, "predicted_protocol", result)
    conf = float(getattr(result, "confidence", 1.0))
    return str(proto), conf


class DetectClassifyPipeline:
    def __init__(
        self,
        classifier: object,
        detector: DetectorFn = default_detector,
        threshold: float = 0.05,
        min_gap: int = 256,
        sps: int = 8,
        use_truth_bytes: bool = False,
    ) -> None:
        self.classifier = classifier
        self.detector = detector
        self.threshold = threshold
        self.min_gap = min_gap
        self.sps = sps
        self.use_truth_bytes = use_truth_bytes

    def _truth_bytes(self, capture: LabeledCapture, start: int, end: int) -> Optional[bytes]:
        for (ts, te, _proto) in (capture.truth_regions or []):
            if not (end <= ts or start >= te):  # overlap
                hexstr = capture.provenance.get("payload_hex")
                return bytes.fromhex(hexstr) if hexstr else None
        return None

    def run(self, capture: LabeledCapture) -> List[Detection]:
        regions = self.detector(capture.iq, self.threshold, self.min_gap)
        detections: List[Detection] = []
        for (start, end) in regions:
            if self.use_truth_bytes:
                pkt = self._truth_bytes(capture, start, end) or b""
            else:
                pkt = region_to_bytes(capture.iq[start:end], sps=self.sps)
            result = self.classifier.classify(pkt, None)
            proto, conf = _confidence(result)
            detections.append(Detection(start=int(start), end=int(end),
                                        protocol=proto, confidence=conf))
        return detections
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_pipeline.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add validation/pipeline.py tests/validation/test_pipeline.py
git commit -m "feat(validation): add injectable detect-then-classify pipeline

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: `metrics.py` — detection + classification + bootstrap CIs

**Files:**
- Create: `validation/metrics.py`
- Test: `tests/validation/test_metrics.py`

**Interfaces:**
- Consumes: `validation.types.{Detection, DetectionMetrics, ClassificationMetrics}`, `validation.repro.rng`.
- Produces (all pure functions):
  - `overlap_iou(a: Tuple[int,int], b: Tuple[int,int]) -> float`.
  - `match_detections(detected: List[Tuple[int,int]], truth: List[Tuple[int,int,str]], iou_threshold: float = 0.5) -> Tuple[int,int,int]` → `(tp, fp, fn)`.
  - `detection_metrics(matched: List[Tuple[int,int,int]], total_samples: int, sample_rate: float, window: int, roc_points: Optional[List[Tuple[float,float]]] = None, min_snr: Optional[float] = None) -> DetectionMetrics` (matched = list of per-capture `(tp,fp,fn)`).
  - `classification_metrics(pairs: List[Tuple[str,str]], snr_by_pair: Optional[List[float]] = None) -> ClassificationMetrics` (pairs = `(truth_label, pred_label)`).
  - `bootstrap_ci(values: Sequence[float], n: int = 1000, seed: int = 0, alpha: float = 0.05) -> Tuple[float,float]`.

- [ ] **Step 1: Write the failing test (golden values)**

```python
# tests/validation/test_metrics.py
from __future__ import annotations
import numpy as np
from validation.metrics import (
    overlap_iou, match_detections, detection_metrics,
    classification_metrics, bootstrap_ci,
)

def test_iou_basic():
    assert overlap_iou((0, 10), (0, 10)) == 1.0
    assert overlap_iou((0, 10), (10, 20)) == 0.0
    assert abs(overlap_iou((0, 10), (5, 15)) - (5 / 15)) < 1e-9

def test_match_counts():
    tp, fp, fn = match_detections(
        detected=[(0, 10), (100, 110)],
        truth=[(0, 9, "a"), (200, 210, "b")],
        iou_threshold=0.5,
    )
    assert (tp, fp, fn) == (1, 1, 1)

def test_detection_metrics_pd_pfa():
    m = detection_metrics(matched=[(1, 1, 0), (1, 0, 1)],
                          total_samples=2_000_000, sample_rate=1_000_000.0, window=1000)
    assert abs(m.pd - (2 / 3)) < 1e-9    # tp=2, fn=1
    assert m.fp == 1
    assert abs(m.pfa_per_sec - 0.5) < 1e-9  # 1 FP over 2.0 s

def test_classification_metrics_confusion_and_accuracy():
    pairs = [("a", "a"), ("a", "b"), ("b", "b"), ("b", "b")]
    cm = classification_metrics(pairs)
    assert abs(cm.accuracy - 0.75) < 1e-9
    assert cm.confusion["a"]["a"] == 1 and cm.confusion["a"]["b"] == 1
    assert cm.per_class["b"]["recall"] == 1.0

def test_bootstrap_ci_brackets_mean():
    vals = list(np.r_[np.ones(50), np.zeros(50)])  # mean 0.5
    lo, hi = bootstrap_ci(vals, n=500, seed=1)
    assert lo < 0.5 < hi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_metrics.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# validation/metrics.py
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from validation.repro import rng
from validation.types import ClassificationMetrics, DetectionMetrics


def overlap_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match_detections(
    detected: List[Tuple[int, int]],
    truth: List[Tuple[int, int, str]],
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int]:
    truth_spans = [(s, e) for (s, e, _p) in truth]
    used = set()
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
    tp = sum(m[0] for m in matched)
    fp = sum(m[1] for m in matched)
    fn = sum(m[2] for m in matched)
    pd = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    duration_s = total_samples / sample_rate if sample_rate > 0 else 0.0
    pfa_per_sec = fp / duration_s if duration_s > 0 else 0.0
    n_windows = max(1, total_samples // max(1, window))
    pfa_per_window = fp / n_windows
    return DetectionMetrics(
        pd=pd, pfa_per_sec=pfa_per_sec, pfa_per_window=pfa_per_window,
        tp=tp, fp=fp, fn=fn, roc=roc_points or [], min_detectable_snr_db=min_snr,
    )


def classification_metrics(
    pairs: List[Tuple[str, str]],
    snr_by_pair: Optional[List[float]] = None,
) -> ClassificationMetrics:
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
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1}
    acc_by_snr: Dict[float, float] = {}
    if snr_by_pair is not None:
        buckets: Dict[float, List[int]] = {}
        for (truth, pred), snr in zip(pairs, snr_by_pair):
            buckets.setdefault(round(snr, 1), []).append(int(truth == pred))
        acc_by_snr = {k: float(np.mean(v)) for k, v in sorted(buckets.items())}
    return ClassificationMetrics(
        accuracy=accuracy, confusion=confusion, per_class=per_class,
        accuracy_by_snr=acc_by_snr,
    )


def bootstrap_ci(
    values: Sequence[float],
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float]:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_metrics.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add validation/metrics.py tests/validation/test_metrics.py
git commit -m "feat(validation): add detection/classification metrics + bootstrap CIs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 11: `harness.py` — orchestrate dataset × pipeline → RunResult

**Files:**
- Create: `validation/harness.py`
- Test: `tests/validation/test_harness.py`

**Interfaces:**
- Consumes: `validation.dataset.LabeledDataset`, `validation.pipeline.DetectClassifyPipeline`, `validation.metrics.*`, `validation.repro.{capture_manifest, hash_config}`, `validation.types.{RunResult}`.
- Produces: dataclass `HarnessConfig(seed: int = 42, iou_threshold: float = 0.5, detector_threshold: float = 0.05, min_gap: int = 256, window: int = 1000, pd_target: float = 0.9)`; `run_evaluation(dataset: LabeledDataset, pipeline: DetectClassifyPipeline, config: HarnessConfig, model_hash: Optional[str] = None) -> RunResult`. Builds `(truth_label, pred_label)` pairs from each detection matched to a truth region; computes accuracy-by-SNR from `provenance["snr_db"]`; computes min-detectable-SNR as lowest SNR bucket with per-bucket Pd ≥ `pd_target`.

- [ ] **Step 1: Write the failing test (determinism + wiring)**

```python
# tests/validation/test_harness.py
from __future__ import annotations
import numpy as np
from dataclasses import asdict
from validation.types import ModScheme
from validation.synth.scenarios import DatasetSpec, build_scenario
from validation.dataset import LabeledDataset
from validation.pipeline import DetectClassifyPipeline
from validation.harness import HarnessConfig, run_evaluation

class OracleClassifier:
    """Classifies by peeking at the payload_hex via truth bytes -> always right label.
       For the test we just return based on region energy sign."""
    def __init__(self, label): self.label = label
    def classify(self, packet_bytes, signal_metrics=None): return self.label

def _dataset():
    spec = DatasetSpec(protocols=["mavlink"], snr_grid_db=[20.0], n_per_cell=3,
                       sample_rate=1e6, seed=1,
                       scheme_by_protocol={"mavlink": ModScheme.FSK}, payload_len=16)
    return LabeledDataset(build_scenario(spec))

def test_run_is_deterministic():
    ds = _dataset()
    pipe = DetectClassifyPipeline(OracleClassifier("mavlink"), threshold=0.2, min_gap=64)
    r1 = run_evaluation(ds, pipe, HarnessConfig(seed=5))
    r2 = run_evaluation(ds, pipe, HarnessConfig(seed=5))
    assert r1.manifest.dataset_hash == r2.manifest.dataset_hash
    assert r1.detection.pd == r2.detection.pd
    assert r1.classification.accuracy == r2.classification.accuracy

def test_high_snr_detects_and_classifies():
    ds = _dataset()
    pipe = DetectClassifyPipeline(OracleClassifier("mavlink"), threshold=0.2, min_gap=64)
    r = run_evaluation(ds, pipe, HarnessConfig())
    assert r.detection.pd > 0.5
    assert r.classification.accuracy > 0.5
    assert r.manifest.dataset_hash
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_harness.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# validation/harness.py
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from validation.dataset import LabeledDataset
from validation.metrics import (
    classification_metrics, detection_metrics, match_detections,
)
from validation.pipeline import DetectClassifyPipeline
from validation.repro import capture_manifest, hash_config
from validation.types import RunResult


@dataclass
class HarnessConfig:
    seed: int = 42
    iou_threshold: float = 0.5
    detector_threshold: float = 0.05
    min_gap: int = 256
    window: int = 1000
    pd_target: float = 0.9


def _match_pair(det, truth_regions) -> Optional[Tuple[str, str]]:
    from validation.metrics import overlap_iou
    best = None
    best_iou = 0.0
    for (ts, te, proto) in (truth_regions or []):
        iou = overlap_iou((det.start, det.end), (ts, te))
        if iou > best_iou:
            best_iou, best = iou, proto
    if best is not None and best_iou >= 0.5:
        return (best, det.protocol)
    return None


def run_evaluation(
    dataset: LabeledDataset,
    pipeline: DetectClassifyPipeline,
    config: HarnessConfig,
    model_hash: Optional[str] = None,
) -> RunResult:
    pipeline.threshold = config.detector_threshold
    pipeline.min_gap = config.min_gap

    matched: List[Tuple[int, int, int]] = []
    pairs: List[Tuple[str, str]] = []
    snr_by_pair: List[float] = []
    total_samples = 0
    pd_by_snr: Dict[float, List[int]] = {}

    for cap in dataset:
        total_samples += len(cap.iq)
        dets = pipeline.run(cap)
        det_spans = [(d.start, d.end) for d in dets]
        tp, fp, fn = match_detections(det_spans, cap.truth_regions or [],
                                      iou_threshold=config.iou_threshold)
        matched.append((tp, fp, fn))
        snr = float(cap.provenance.get("snr_db", 0.0))
        pd_by_snr.setdefault(round(snr, 0), []).append(1 if tp > 0 else 0)
        for d in dets:
            pair = _match_pair(d, cap.truth_regions)
            if pair is not None:
                pairs.append(pair)
                snr_by_pair.append(snr)

    # min detectable SNR: lowest bucket with Pd >= target
    min_snr = None
    for snr in sorted(pd_by_snr):
        if float(np.mean(pd_by_snr[snr])) >= config.pd_target:
            min_snr = snr
            break

    det_metrics = detection_metrics(
        matched, total_samples=total_samples,
        sample_rate=next(iter(dataset)).sample_rate if len(dataset) else 1.0,
        window=config.window, min_snr=min_snr,
    )
    cls_metrics = classification_metrics(pairs, snr_by_pair=snr_by_pair)

    # Bootstrap confidence intervals on the headline numbers (spec §2 goal 4).
    from validation.metrics import bootstrap_ci
    det_metrics.ci["pd"] = bootstrap_ci(
        [1.0 if m[0] > 0 else 0.0 for m in matched], seed=config.seed
    )
    cls_metrics.ci["accuracy"] = bootstrap_ci(
        [1.0 if t == p else 0.0 for (t, p) in pairs], seed=config.seed
    )

    cfg_hash = hash_config(asdict(config))
    manifest = capture_manifest(
        seed=config.seed, dataset_hash=dataset.content_hash(),
        config_hash=cfg_hash, model_hash=model_hash,
    )
    return RunResult(detection=det_metrics, classification=cls_metrics, manifest=manifest)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_harness.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add validation/harness.py tests/validation/test_harness.py
git commit -m "feat(validation): add evaluation harness producing RunResult

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 12: `report.py` — JSON report (+ optional plots)

**Files:**
- Create: `validation/report.py`
- Test: `tests/validation/test_report.py`

**Interfaces:**
- Consumes: `validation.types.RunResult`.
- Produces: `result_to_dict(result: RunResult) -> Dict[str, Any]`; `write_report(result: RunResult, path: Path, plots: bool = False) -> None`. JSON includes `detection`, `classification`, and full `manifest`. When `plots=True`, matplotlib is imported lazily and writes `<path stem>_confusion.png` + `<path stem>_accuracy_by_snr.png` next to the JSON; if matplotlib is missing, a warning is logged and JSON still writes.

- [ ] **Step 1: Write the failing test**

```python
# tests/validation/test_report.py
from __future__ import annotations
import json
from validation.types import (
    RunResult, DetectionMetrics, ClassificationMetrics, RunManifest,
)
from validation.report import result_to_dict, write_report

def _result():
    return RunResult(
        detection=DetectionMetrics(pd=0.9, pfa_per_sec=0.1, pfa_per_window=0.01,
                                   tp=9, fp=1, fn=1, min_detectable_snr_db=0.0),
        classification=ClassificationMetrics(accuracy=0.8,
                                             confusion={"a": {"a": 4, "b": 1}},
                                             per_class={"a": {"precision": 0.8}},
                                             accuracy_by_snr={0.0: 0.7, 20.0: 0.95}),
        manifest=RunManifest(seed=42, dataset_hash="d", config_hash="c",
                             git_commit="abc", timestamp="t",
                             versions={"numpy": "1.0"}),
    )

def test_result_to_dict_has_sections():
    d = result_to_dict(_result())
    assert set(d) >= {"detection", "classification", "manifest"}
    assert d["detection"]["pd"] == 0.9
    assert d["manifest"]["dataset_hash"] == "d"

def test_write_report_json(tmp_path):
    p = tmp_path / "report.json"
    write_report(_result(), p, plots=False)
    loaded = json.loads(p.read_text())
    assert loaded["classification"]["accuracy"] == 0.8
    assert loaded["manifest"]["seed"] == 42
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_report.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# validation/report.py
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
        ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
        ax.set_title("Classification accuracy vs SNR")
        fig.savefig(f"{stem}_accuracy_by_snr.png", dpi=120); plt.close(fig)
    conf = result.classification.confusion
    if conf:
        labels = sorted(conf)
        mat = [[conf[t].get(p, 0) for p in labels] for t in labels]
        fig, ax = plt.subplots()
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Truth"); ax.set_title("Confusion")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(f"{stem}_confusion.png", dpi=120); plt.close(fig)


def write_report(result: RunResult, path: Path, plots: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result_to_dict(result), indent=2, default=str))
    if plots:
        _write_plots(result, path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_report.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add validation/report.py tests/validation/test_report.py
git commit -m "feat(validation): add JSON T&E report writer with optional plots

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 13: `validation/__init__.py` — public API + factory functions

**Files:**
- Modify: `validation/__init__.py`
- Test: `tests/validation/test_public_api.py`

**Interfaces:**
- Consumes: everything above.
- Produces (the simple layer, matching the repo's two-layer idiom):
  - `create_synth_dataset(protocols, snr_grid_db, n_per_cell, scheme_by_protocol, sample_rate=2_048_000.0, seed=42, payload_len=32) -> LabeledDataset`.
  - `create_pipeline(classifier, **kwargs) -> DetectClassifyPipeline`.
  - `evaluate(dataset, pipeline, seed=42, **kwargs) -> RunResult`.
  - Re-export the key types/classes at package top level.

- [ ] **Step 1: Write the failing test**

```python
# tests/validation/test_public_api.py
from __future__ import annotations
import validation
from validation import (
    create_synth_dataset, create_pipeline, evaluate,
    LabeledDataset, DetectClassifyPipeline, RunResult, ModScheme,
)

class Stub:
    def classify(self, packet_bytes, signal_metrics=None): return "mavlink"

def test_end_to_end_public_api():
    ds = create_synth_dataset(
        protocols=["mavlink"], snr_grid_db=[20.0], n_per_cell=2,
        scheme_by_protocol={"mavlink": ModScheme.FSK}, seed=3, payload_len=16,
    )
    assert isinstance(ds, LabeledDataset) and len(ds) == 2
    pipe = create_pipeline(Stub(), threshold=0.2, min_gap=64)
    assert isinstance(pipe, DetectClassifyPipeline)
    result = evaluate(ds, pipe, seed=3)
    assert isinstance(result, RunResult)
    assert result.manifest.dataset_hash
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_public_api.py -v`
Expected: FAIL — names not exported.

- [ ] **Step 3: Write minimal implementation**

```python
# validation/__init__.py
"""DroneCMD validation & T&E spine (SP1) — public API."""
from __future__ import annotations
from typing import Dict, List, Optional

from validation.types import (
    ModScheme, ChannelParams, LabeledCapture, Detection,
    DetectionMetrics, ClassificationMetrics, RunManifest, RunResult, IQSamples,
)
from validation.dataset import LabeledDataset
from validation.pipeline import DetectClassifyPipeline
from validation.harness import HarnessConfig, run_evaluation
from validation.synth.scenarios import DatasetSpec, build_scenario

__all__ = [
    "ModScheme", "ChannelParams", "LabeledCapture", "Detection",
    "DetectionMetrics", "ClassificationMetrics", "RunManifest", "RunResult",
    "IQSamples", "LabeledDataset", "DetectClassifyPipeline", "HarnessConfig",
    "DatasetSpec", "create_synth_dataset", "create_pipeline", "evaluate",
]


def create_synth_dataset(
    protocols: List[str],
    snr_grid_db: List[float],
    n_per_cell: int,
    scheme_by_protocol: Dict[str, ModScheme],
    sample_rate: float = 2_048_000.0,
    seed: int = 42,
    payload_len: int = 32,
) -> LabeledDataset:
    spec = DatasetSpec(
        protocols=protocols, snr_grid_db=snr_grid_db, n_per_cell=n_per_cell,
        sample_rate=sample_rate, seed=seed, scheme_by_protocol=scheme_by_protocol,
        payload_len=payload_len,
    )
    return LabeledDataset(build_scenario(spec))


def create_pipeline(classifier: object, **kwargs: object) -> DetectClassifyPipeline:
    return DetectClassifyPipeline(classifier, **kwargs)  # type: ignore[arg-type]


def evaluate(
    dataset: LabeledDataset,
    pipeline: DetectClassifyPipeline,
    seed: int = 42,
    model_hash: Optional[str] = None,
    **kwargs: object,
) -> RunResult:
    config = HarnessConfig(seed=seed, **kwargs)  # type: ignore[arg-type]
    return run_evaluation(dataset, pipeline, config, model_hash=model_hash)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_public_api.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add validation/__init__.py tests/validation/test_public_api.py
git commit -m "feat(validation): add public API factories (simple layer)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 14: CLI — `dronecmd validate synth|ingest|run`

**Files:**
- Modify: `cli.py` — add subparser in `create_parser()` (near the `train` subparser ~line 380), add `cmd_validate(args, config, output)` handler (near `cmd_train` ~line 887), wire dispatch in `main()` (~line 922).
- Test: `tests/test_cli_validate.py`

**Interfaces:**
- Consumes: `validation.create_synth_dataset`, `validation.LabeledDataset`, `validation.create_pipeline`, `validation.evaluate`, `validation.report.write_report`, `core.classification.{EnhancedProtocolClassifier, ClassifierConfig}`.
- Produces: three sub-actions. `run` loads the trained classifier from `--models` (raising the existing `ModelNotTrainedError` if absent); `synth`/`ingest` build/persist a `LabeledDataset` to `--out`.

- [ ] **Step 1: Write the failing test (parser + smoke, no models)**

```python
# tests/test_cli_validate.py
from __future__ import annotations
import json
import numpy as np
import pytest
from cli import create_parser

def test_parser_has_validate_synth():
    parser = create_parser()
    args = parser.parse_args(
        ["validate", "synth", "--protocols", "mavlink", "--snr", "0:20:10",
         "--n", "2", "--out", "ds"]
    )
    assert args.command == "validate"
    assert args.validate_action == "synth"
    assert args.protocols == "mavlink"

@pytest.mark.integration
def test_validate_synth_writes_dataset(tmp_path):
    from cli import cmd_validate
    from utils.config import ConfigManager  # existing config type used by other cmds
    from cli import CLIOutput
    parser = create_parser()
    out = tmp_path / "ds"
    args = parser.parse_args(
        ["validate", "synth", "--protocols", "mavlink,dji", "--snr", "0:20:20",
         "--n", "1", "--out", str(out)]
    )
    cmd_validate(args, ConfigManager(), CLIOutput(json_mode=False))
    assert any(out.glob("*.sigmf-data")) or any(out.glob("*.json"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_validate.py -v`
Expected: FAIL — `validate` subcommand not defined / `cmd_validate` missing.

- [ ] **Step 3: Write minimal implementation**

In `create_parser()` (after the `train` subparser block), add:

```python
    # Validate command (T&E spine, SP1)
    validate_parser = subparsers.add_parser(
        'validate', help='Empirical validation / T&E of detect->classify pipeline'
    )
    validate_sub = validate_parser.add_subparsers(dest='validate_action')

    v_synth = validate_sub.add_parser('synth', help='Generate a synthetic labeled dataset')
    v_synth.add_argument('--protocols', required=True, help='Comma list, e.g. mavlink,dji')
    v_synth.add_argument('--snr', required=True, help='LOW:HIGH:STEP in dB, e.g. -20:20:2')
    v_synth.add_argument('--n', type=int, default=20, help='Captures per (protocol, SNR) cell')
    v_synth.add_argument('--seed', type=int, default=42)
    v_synth.add_argument('--out', required=True, help='Output dataset directory')

    v_ingest = validate_sub.add_parser('ingest', help='Label real captures into a dataset')
    v_ingest.add_argument('--input', required=True, help='Directory of real .iq/.sigmf captures')
    v_ingest.add_argument('--out', required=True, help='Output dataset directory')
    v_ingest.add_argument('--sample-rate', type=float, default=2_048_000.0)

    v_run = validate_sub.add_parser('run', help='Run the harness and write a report')
    v_run.add_argument('--dataset', required=True, help='Dataset directory')
    v_run.add_argument('--models', required=True, help='Trained model directory')
    v_run.add_argument('--report', required=True, help='Output report.json path')
    v_run.add_argument('--plots', action='store_true')
    v_run.add_argument('--seed', type=int, default=42)
```

Add the handler (sync — mirrors `cmd_train`, `cmd_config`):

```python
def cmd_validate(args: argparse.Namespace, config: ConfigManager, output: CLIOutput) -> None:
    from pathlib import Path
    from validation import (
        create_synth_dataset, create_pipeline, evaluate, LabeledDataset, ModScheme,
    )
    from validation.report import write_report

    action = getattr(args, 'validate_action', None)
    if action == 'synth':
        lo, hi, step = (float(x) for x in args.snr.split(':'))
        grid = list(np.arange(lo, hi + step / 2, step))
        protocols = [p.strip() for p in args.protocols.split(',')]
        default_scheme = {"mavlink": ModScheme.FSK, "dji": ModScheme.QPSK}
        scheme_by_protocol = {p: default_scheme.get(p, ModScheme.FSK) for p in protocols}
        ds = create_synth_dataset(
            protocols=protocols, snr_grid_db=grid, n_per_cell=args.n,
            scheme_by_protocol=scheme_by_protocol, seed=args.seed,
        )
        ds.write(Path(args.out))
        output.info(f"Wrote {len(ds)} synthetic captures to {args.out}")
    elif action == 'ingest':
        ds = LabeledDataset.from_dir(Path(args.input), sample_rate=args.sample_rate)
        ds.write(Path(args.out))
        output.info(f"Ingested {len(ds)} captures to {args.out}")
    elif action == 'run':
        from core.classification import EnhancedProtocolClassifier, ClassifierConfig
        clf = EnhancedProtocolClassifier(ClassifierConfig(model_path=Path(args.models)))
        ds = LabeledDataset.from_dir(Path(args.dataset))
        pipe = create_pipeline(clf)
        result = evaluate(ds, pipe, seed=args.seed)
        write_report(result, Path(args.report), plots=args.plots)
        output.info(f"Pd={result.detection.pd:.3f} "
                    f"accuracy={result.classification.accuracy:.3f} -> {args.report}")
    else:
        output.error("Usage: dronecmd validate {synth|ingest|run} ...")
```

In `main()`, add dispatch alongside the other commands:

```python
        elif args.command == 'validate':
            cmd_validate(args, config, output)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_validate.py -v`
Expected: PASS (parser test always; integration test writes a dataset). If `CLIOutput`/`ConfigManager` constructor signatures differ, match the exact usage already in `cli.py`'s `main()` (read lines ~922-998 and mirror them).

- [ ] **Step 5: Commit**

```bash
git add cli.py tests/test_cli_validate.py
git commit -m "feat(cli): add 'dronecmd validate' (synth/ingest/run) T&E command

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 15 (optional/deferred): OFDM modulator

**Deferred from spec D5.** The existing `DemodulationEngine` has no OFDM demodulator, so an OFDM packet cannot traverse the full detect→demod→classify chain. Implement OFDM only for **detection-stage** validation and the **`use_truth_bytes=True`** classification path. Gate behind its own review; do not block SP1 completion on it.

**Files:**
- Modify: `validation/synth/modulators.py` (add `_ofdm` + `ModScheme.OFDM` branch)
- Test: `tests/validation/test_modulators.py` (add OFDM presence/power/length test — no round-trip through core demod)

- [ ] **Step 1:** Write a test asserting `modulate(DATA, ModScheme.OFDM, sps=...)` returns unit-power `complex64` of the expected length (n_subcarriers/CP math), and that an inline IFFT/FFT reference round-trip recovers the QAM symbols.
- [ ] **Step 2:** Run — FAIL (branch raises `ValueError`).
- [ ] **Step 3:** Implement `_ofdm(bits, n_sub=64, cp=16, ...)`: bits → QPSK subcarrier symbols → IFFT → prepend cyclic prefix → serialize → normalize power.
- [ ] **Step 4:** Run — PASS.
- [ ] **Step 5:** Commit `feat(validation): add OFDM modulator (detection-stage only)` with the attribution line.

---

### Task 16: SP1 integration + quality gate + findings log

**Files:**
- Create: `tests/validation/test_integration_smoke.py`
- Modify: `docs/superpowers/specs/2026-09-17-validation-te-spine-design.md` (§16 findings log — final state), bump spec to v1.1.0 if any interface changed during implementation.
- Modify: `pyproject.toml` if `matplotlib` needs adding to the `[viz]` extra (verify it is present; add if missing).

**Interfaces:**
- Consumes: the whole package.
- Produces: an `@pytest.mark.integration` end-to-end test that builds a synthetic dataset, runs the harness with an oracle stub classifier, writes a report, and asserts the report JSON has detection + classification + manifest sections with a non-empty `dataset_hash` and reproducible hashes across two runs.

- [ ] **Step 1: Write the failing test**

```python
# tests/validation/test_integration_smoke.py
from __future__ import annotations
import json
import pytest
from validation import create_synth_dataset, create_pipeline, evaluate, ModScheme
from validation.report import write_report

class Oracle:
    def classify(self, packet_bytes, signal_metrics=None): return "mavlink"

@pytest.mark.integration
def test_full_spine_smoke(tmp_path):
    ds = create_synth_dataset(
        protocols=["mavlink"], snr_grid_db=[0.0, 20.0], n_per_cell=3,
        scheme_by_protocol={"mavlink": ModScheme.FSK}, seed=11, payload_len=24,
    )
    pipe = create_pipeline(Oracle(), threshold=0.2, min_gap=64)
    r1 = evaluate(ds, pipe, seed=11)
    r2 = evaluate(ds, pipe, seed=11)
    assert r1.manifest.dataset_hash == r2.manifest.dataset_hash
    p = tmp_path / "report.json"
    write_report(r1, p, plots=False)
    doc = json.loads(p.read_text())
    assert doc["detection"]["pd"] >= 0.0
    assert "manifest" in doc and doc["manifest"]["dataset_hash"]
```

- [ ] **Step 2:** Run: `pytest tests/validation/test_integration_smoke.py -v -m integration` — should PASS once the whole package is in place (it exercises Tasks 1–13).

- [ ] **Step 3: Full quality gate**

Run:
```bash
black validation tests && isort validation tests && flake8 validation tests && mypy validation
pytest -m "not slow and not hardware" -q
```
Expected: all green. Fix any lint/type issues inline (do not suppress).

- [ ] **Step 4: Finalize findings log**

Update spec §16 with the final status of F1 (fixed, Task 3) and any items discovered during Tasks 8/9 (SigMF annotation round-trip §11.5; demod byte-recovery §11.3). Each entry: module, symptom, RED test name, fix, status.

- [ ] **Step 5: Commit**

```bash
git add tests/validation/test_integration_smoke.py docs/superpowers/specs/2026-09-17-validation-te-spine-design.md pyproject.toml
git commit -m "test(validation): add end-to-end smoke + finalize SP1 bug-audit log

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- §2 goal 1 (synthetic labeled IQ w/ known SNR) → Tasks 4,5,6. ✓
- §2 goal 2 (ingest/label real) → Task 7. ✓
- §2 goal 3 (detect→demod→classify + Pd/Pfa/ROC + confusion/per-class/acc-vs-SNR) → Tasks 9,10,11. Detection ROC sweep is represented by the `roc` field + min-detectable-SNR; a full threshold sweep to populate `roc` points is a harness enhancement noted below.
- §2 goal 4 (bootstrap CI) → Task 10 `bootstrap_ci` (wired into metrics dataclasses' `ci` field; harness currently reports point estimates — see gap G1).
- §2 goal 5 (repro manifest) → Task 2 + Task 11. ✓
- §2 goal 6 (CLI) → Task 14. ✓
- §8 SNR calibration landmine → Task 5 test. ✓
- §11 bug audit F1 → Task 3; §11.5 SigMF round-trip → Task 8 step 4 note; §11.3 demod recovery → bounded by Task 4. §11.2 (two detect_packets) → pipeline injectable detector (Task 9) enables comparison; explicit comparison run is not yet a task (gap G2).
- §5 D5 OFDM → Task 15 (deferred, documented). ✓ (deviation recorded)

**Gaps found and resolved inline:**
- **G1 — CIs wired into harness output (resolved).** Task 11 Step 3 now computes `det_metrics.ci["pd"]` and `cls_metrics.ci["accuracy"]` via `bootstrap_ci` before constructing `RunResult`. Add a harness assertion for `ci` presence to Task 11's tests if desired.
- **G2 — detector divergence comparison.** Add a short follow-on step in Task 14 or a manual analysis: run `evaluate` twice with `create_pipeline(clf, detector=default_detector)` vs a `capture.detector`-backed `DetectorFn`, and record both Pd/Pfa in the findings log. Optional for SP1 sign-off; flagged, not blocking.
- **G3 — full detection ROC sweep.** The `roc` list is populated only if the harness sweeps `detector_threshold`. Mark as an SP1 stretch: loop thresholds in `run_evaluation` and collect `(pfa_per_window, pd)` points. Non-blocking; min-detectable-SNR (the headline) is computed without it.

**2. Placeholder scan:** No "TBD"/"implement later"/"add error handling" left. Deferred items (Task 15, G2, G3) are explicitly scoped with concrete instructions, not vague placeholders.

**3. Type consistency:** `LabeledCapture`, `Detection`, `DetectionMetrics`, `ClassificationMetrics`, `RunManifest`, `RunResult` fields are defined once in Task 1 and referenced identically downstream. `modulate(...)`, `add_awgn_at_snr(...)`, `build_scenario(...)`, `DetectClassifyPipeline.run(...)`, `run_evaluation(...)`, `write_report(...)` signatures match between their producing task and every consuming task. `detect_packets(iq, threshold, min_gap)` matches the real API confirmed in `core/signal_processing.py`.

---

## Execution Handoff

Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. Matches your "written for subagents to complete" requirement.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.
