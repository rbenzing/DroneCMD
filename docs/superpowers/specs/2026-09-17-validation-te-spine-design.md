# Validation & T&E Spine — Sub-project 1 Design

- **Spec version:** 1.0.0
- **Date:** 2026-09-17
- **Status:** Approved for planning (pending user spec review)
- **Author:** rbenzing + Claude
- **Program context:** Turn DroneCMD's *asserted* capabilities into *measured, reproducible, auditable* ones — the "empirical validation" a government program office needs before a capability is accreditable.

### Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2026-09-17 | Initial approved design (SP1). |

### For implementers / subagents

This spec is written so each unit in §5.1 can be handed to an **independent subagent with
only this spec as context** — no prior conversation required. Each row of §5.1 is a
self-contained task with an explicit public interface, dependency list, and the tests it
must pass (§12). Tasks depend only on the interfaces of already-built units, never on their
internals, so they can be dispatched in the dependency order the plan specifies. The
implementation plan (produced next via the writing-plans skill) enumerates these as
discrete, individually-completable, verifiable work items. Spec and plan are versioned in
git; material changes bump the **Spec version** and add a changelog row.

---

## 1. Problem & context

DroneCMD is a ~31k-line SDR framework: capture → demodulation → ML protocol
classification (ensemble + training pipeline) → FHSS → replay → injection, plus a
plugin system. Today "validation" is a thin layer of unit tests (`tests/`) plus
cross-validation metrics baked into `training/train.py`.

There is no way to answer the questions a Test & Evaluation (T&E) reviewer asks:
*what is the probability of detection at a given false-alarm rate, under what SNR,
against what ground truth, reproducibly, with the exact code and data that produced
the number?* This sub-project builds the spine that answers those questions and, as a
side effect, becomes a bug-detection instrument for the DSP/ML code beneath it.

This is the first of a phased program:

- **SP1 (this doc):** Validation spine core — synthetic + real ground truth → T&E
  harness for the **detect → demod → classify** pipeline, with reproducibility primitives.
- **SP2 (future):** Accreditation wrapper — tamper-evident audit log, chain-of-custody,
  signed/PDF T&E report artifacts, model cards + SHAP explainability.
- **SP3 (future):** First new Track-A capability (Remote ID / ASTM F3411 decoder),
  built *against* this harness so it ships with a performance envelope.

## 2. Goals

1. Generate **synthetic** labeled IQ with known protocol, SNR, and channel conditions,
   fully deterministic given a seed.
2. **Ingest and label real** `.iq`/SigMF captures into the same labeled container.
3. Run the **detect → demod → classify** pipeline over that unified dataset and emit:
   - Detection: Pd, Pfa, ROC/DET, min-detectable-SNR (SNR at which Pd ≥ 0.9 at a fixed Pfa).
   - Classification: confusion matrix, per-class ROC, accuracy-vs-SNR envelope.
4. Attach a **bootstrap confidence interval** to every headline metric.
5. Emit a **reproducibility manifest**: seed, dataset hash, model hash, config hash,
   git commit, library versions.
6. Provide a `dronecmd validate` CLI over all of the above.

## 3. Non-goals (deferred)

- Tamper-evident / cryptographically signed reports and chain-of-custody (**SP2**).
- SHAP explainability and model cards (**SP2**).
- Human-readable PDF T&E report (**SP2**); SP1 emits JSON (+ optional plots).
- New capabilities: Remote ID decoder, geolocation/DF, fingerprinting (**SP3**).
- Hardware-in-the-loop / live-SDR evaluation. SP1 is offline, file-based.

## 4. Decisions log (from brainstorming)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Ground truth = **synthetic + real ingestion** | Synthetic gives clean SNR sweeps + aligned truth; real gives reviewer credibility. |
| D2 | Primary T&E target = **detect-then-classify pipeline** | Most compelling single story; exercises detector, demod, and classifier together. |
| D3 | Architecture = **self-contained `validation/` package**, numpy/scipy/sklearn only, SigMF-native | Auditable (no black box), deterministic, Windows-friendly, fits flat-layout idiom. |
| D4 | Reporting for SP1 = **JSON + optional matplotlib plots**; PDF/signed → SP2 | Keeps SP1 focused; JSON is the machine-checkable accreditation artifact. |
| D5 | Modulation schemes = **FSK, GFSK, QPSK, OFDM** | Covers the drone-relevant families (FSK/GFSK for control links, QPSK/OFDM for video/OcuSync-like). |

## 5. Architecture

New top-level package (flat layout, mirrors `core/`, `capture/`, …):

```
validation/
  __init__.py       # library-first public API + factory functions (two-layer idiom)
  types.py          # LabeledCapture, Detection, ChannelParams, RunResult, RunManifest, ModScheme
  synth/
    __init__.py
    modulators.py    # bytes -> IQ (complex64): FSK, GFSK, QPSK, OFDM            [NEW core DSP]
    channel.py       # AWGN@calibrated-SNR, CFO, Doppler, multipath, timing offset
    scenarios.py     # protocol × SNR grid -> DatasetSpec (packets at known offsets in noise)
  ingest/
    __init__.py
    labeler.py       # real .iq/.sigmf -> LabeledCapture (reuses utils/fileio + training sidecar convention)
  dataset.py         # LabeledDataset: one iterator over synth + real, SigMF-backed on disk
  pipeline.py        # DetectClassifyPipeline: wraps detect_packets + demod + classifier (swappable "thing under test")
  metrics.py         # pure functions: detection + classification metrics + bootstrap CIs
  harness.py         # orchestrates dataset × pipeline -> RunResult
  report.py          # RunResult -> report.json (+ optional matplotlib plots behind [viz] extra)
  repro.py           # np.random.default_rng(seed), array/config hashing, RunManifest
```

CLI is wired into the existing `cli.py` as a `dronecmd validate` command group; the
package itself stays import-first so it is usable as a library (matches the codebase's
two-layer API design).

### 5.1 Design-for-isolation summary

| Unit | Does what | Used how | Depends on |
|------|-----------|----------|------------|
| `synth/modulators.py` | bytes → complex64 IQ per scheme | `modulate(bits, scheme, sps, …)` | numpy |
| `synth/channel.py` | apply impairments + calibrated noise | `apply_channel(iq, params, rng) -> (iq, achieved_snr)` | numpy/scipy |
| `synth/scenarios.py` | compose packets into labeled buffers over a grid | `build_scenario(spec) -> Iterable[LabeledCapture]` | modulators, channel |
| `ingest/labeler.py` | real capture → LabeledCapture | `load_labeled(path) -> LabeledCapture` | utils/fileio |
| `dataset.py` | unify + persist (SigMF) + iterate | `LabeledDataset.write()/read()/__iter__` | utils/fileio |
| `pipeline.py` | run detect→demod→classify | `DetectClassifyPipeline.run(iq, sr) -> [Detection]` | core.signal_processing / core.demodulation / core.classification |
| `metrics.py` | score detections/classes vs truth | pure functions → dataclasses | numpy/scipy/sklearn |
| `harness.py` | dataset × pipeline → RunResult | `run(dataset, pipeline, cfg) -> RunResult` | dataset, pipeline, metrics, repro |
| `report.py` | serialize + plot | `write_report(run_result, path, plots=False)` | json, matplotlib (opt) |
| `repro.py` | seeds, hashes, manifest | `rng(seed)`, `hash_array`, `RunManifest.capture()` | numpy, hashlib, importlib.metadata |

Each unit is understandable and testable in isolation; the interfaces above are the
contract the plan will implement against.

## 6. Data model (`types.py`)

- `ModScheme(Enum)`: `FSK`, `GFSK`, `QPSK`, `OFDM`.
- `ChannelParams`: `snr_db: float`, `cfo_hz: float = 0`, `doppler_hz: float = 0`,
  `multipath_taps: tuple[complex,...] = ()`, `timing_offset: int = 0`.
- `LabeledCapture` — **the shared currency** (both synth and real emit it):
  - `iq: IQSamples` (`np.complex64`), `sample_rate: float`
  - `truth_regions: list[tuple[int, int, str]] | None` — `(start, end, protocol)`;
    `None` when a real capture has no region annotations (→ scored classification-only).
  - `provenance: dict` — `{source: "synth"|"real", snr_db, channel, seed, protocol}`.
- `Detection`: `start: int`, `end: int`, `protocol: str`, `confidence: float`.
- `RunResult`: detection metrics, classification metrics, per-SNR breakdown, CIs, `RunManifest`.
- `RunManifest`: `seed`, `dataset_hash`, `model_hash`, `config_hash`, `git_commit`,
  `timestamp`, `versions: {numpy, scipy, sklearn, dronecmd}`.

## 7. Data flow

```
                 synth/scenarios ─┐
                                  ├─> LabeledDataset (SigMF on disk, hashed) ─┐
   real .iq/.sigmf ─> ingest ─────┘                                          │
                                                                             v
   DetectClassifyPipeline ( detect_packets -> slice -> demod -> classify ) <─┘
                                      │
                                      v
                               [Detection] per capture
                                      │
                    metrics (match to truth_regions) + bootstrap CI
                                      │
                                      v
                         RunResult ─> report.json (+ plots)
```

Synthetic data provides **aligned ground truth at all three stages** (we know the
region we placed the packet in, the bytes it carries, and its protocol). Real captures
provide partial labels: protocol always; regions only when SigMF annotations exist.

## 8. Correctness landmine: SNR calibration

The single most error-prone step and the reason the harness is trustworthy only after
the bug audit. `channel.apply_channel` must:

1. Measure **signal power over the active region** (the samples that actually contain
   the packet), using power `mean(|x|²)` — **not** magnitude `|x|`, and **not** averaged
   over the whole buffer including guard noise.
2. Compute noise variance `N0` so that `10*log10(S/N) == snr_db`.
3. Add circularly-symmetric complex Gaussian noise with variance `N0` (split across I/Q).
4. Return the **achieved** SNR (re-measured) and store it on the label — the label is
   ground truth, so it records what actually happened, not the request.

A dedicated test asserts `measured_out_snr ≈ target_snr` within tolerance across the
grid. This same class of bug (magnitude mislabeled as power) already exists in the
detector — see §11.

## 9. Metrics (`metrics.py`, pure functions)

**Detection** — match detected regions to `truth_regions` by overlap (IoU ≥ threshold,
default 0.5):
- TP/FP/FN → **Pd** = TP/(TP+FN). **Pfa** is reported as a **false-alarm rate in false
  alarms per second** (computed from `sample_rate`) as the headline number; a
  per-analysis-window Pfa is also recorded for detectors that operate window-wise. Both
  are defined in `metrics.py` and labeled with their denominator in the report.
- **ROC/DET** by sweeping the detector threshold parameter.
- **min-detectable-SNR**: lowest SNR bin where Pd ≥ 0.9 at a fixed operating Pfa.

**Classification** — on packets matched to truth (or pre-segmented):
- Confusion matrix, per-class precision/recall/F1.
- Per-class ROC from classifier confidence (one-vs-rest).
- **Accuracy-vs-SNR** curve (the headline envelope).

**Uncertainty** — bootstrap resampling (default 1000 resamples, seeded) → 95% CI on Pd,
accuracy, and per-class recall.

All functions take arrays/dataclasses and return dataclasses; no I/O, no global state →
directly unit-testable against hand-built golden inputs.

## 10. CLI (`dronecmd validate`)

```
dronecmd validate synth  --protocols mavlink,dji --snr=-20:20:2 --n 50 --seed 42 --out ds/
dronecmd validate ingest --input captures/ --out ds/
dronecmd validate run    --dataset ds/ --models data/models/ --report report.json [--plots] [--seed 42] [--use-truth-bytes]
```

- `--snr LOW:HIGH:STEP` parses to an inclusive dB grid. A negative `LOW` bound
  requires the `=` form (`--snr=-20:20:2`), since a space-separated value
  starting with `-` (`--snr -20:20:2`) is otherwise parsed by argparse as an
  unrecognized flag; positive-low ranges accept either form
  (`--snr 0:20:2` or `--snr=0:20:2`).
- `run` loads the trained ensemble via existing `ClassifierConfig.model_path` /
  `ModelManager`, raising the existing `ModelNotTrainedError` if models are absent
  (no silent random predictions — consistent with current classifier contract).
- `run --use-truth-bytes` scores the classifier on ground-truth payload bytes
  instead of demodulated bytes; without it, CLI classification of synthetic
  non-FSK schemes (e.g. QPSK) is demod-limited because `region_to_bytes` is a
  fixed FSK demodulator (see §11.3).
- Honors `--json` global output convention already in `cli.py`.

## 11. Bug-audit workstream (cross-cutting, **gating**)

The harness's numbers are meaningful only if the code under test is correct. Before
trusting any metric, run the `systematic-debugging` discipline over the critical path.
Each finding follows RED (a failing test reproducing the bug) → fix → GREEN, and is
logged in the appendix of this spec.

Critical-path modules and known/suspected issues found during design:

1. **`core/signal_processing.py:455` `detect_packets`** — computes `power = np.abs(iq)`
   (**magnitude**, not power `|x|²`) despite the name, and the `if np.iscomplexobj / else`
   branches are **identical** (dead code). Affects threshold semantics → Pd/Pfa. **Confirmed.**
2. **Two divergent `detect_packets`** (`core/signal_processing.py:455` and
   `capture/detector.py:827`) plus a `SignalDetector` class — reconcile or characterize.
   The pipeline adapter runs both through the harness to **quantify** the divergence.
3. `core/demodulation` — verify byte recovery is correct enough that classification is
   testing the classifier, not demod noise (anchored by the modulator round-trip test).
4. `core/classification` feature extraction — must be **deterministic** for a given input
   (repro requires it).
5. `utils/fileio` — SigMF read/write **round-trip** fidelity (dtype `complex64`,
   sample_rate, annotations preserved).

## 12. Testing strategy (TDD)

Written test-first, per the test-driven-development skill:

- **Modulator round-trip** (correctness anchor): `modulate(bits) -> demod -> bits`, 0 BER
  at high SNR, for each scheme.
- **Channel SNR calibration**: measured output SNR ≈ target across the grid (catches the
  power/magnitude and whole-buffer averaging bugs).
- **Metrics golden values**: hand-built detection/truth and confusion inputs → known
  Pd/Pfa/accuracy.
- **Determinism**: same seed → byte-identical `RunResult` (hashes match) across two runs.
- **Dataset/ingest**: SigMF round-trip and label fidelity (synth and real).
- **Detector regression**: characterize both `detect_packets` implementations through the
  harness.

Test markers reuse the existing `slow` / `integration` scheme. Hardware is not required
(SP1 is fully offline). Hardware interfaces stay mocked per project convention.

## 13. Dependencies

No new heavy dependencies. Uses already-declared `numpy`, `scipy`, `scikit-learn`,
`joblib`. `matplotlib` (plots) sits behind the existing `[viz]` extra and is optional at
runtime. SigMF is handled via the existing `utils/fileio` support (no new package
required); if a helper is wanted later it is a light, pure-python dep — decide in the plan.

## 14. Risks & open questions

- **Demod ↔ classifier coupling:** if demod is too lossy, classification accuracy reflects
  demod, not the classifier. Mitigation: the modulator round-trip test bounds demod error;
  optionally offer a "bytes-truth bypass" path that scores the classifier on known bytes to
  separate the two. *(Plan decides whether to build the bypass in SP1 or SP2.)*
- **Real-capture label scarcity:** few annotated real captures may make real-data detection
  metrics thin. Acceptable for SP1 (synthetic carries the SNR sweeps); real data primarily
  provides face-validity.
- **OFDM modulator scope:** a faithful OFDM tx (CP, pilots) is the heaviest modulator. If it
  threatens the plan's size, ship FSK/GFSK/QPSK first and add OFDM as the last plan step.

## 15. Definition of done (SP1)

- `dronecmd validate synth|ingest|run` all work end-to-end on a small dataset.
- `report.json` contains detection + classification metrics with CIs and a full
  reproducibility manifest; re-running with the same seed reproduces identical hashes.
- All new units have unit tests; the five bug-audit items in §11 are each resolved or
  explicitly characterized with a logged finding.
- `black . && isort . && flake8 && mypy validation` and `pytest -m "not slow and not hardware"`
  pass.

## 16. Appendix — bug-audit findings log

_(Populated during implementation; one entry per finding: module, symptom, RED test,
fix, status. Finalized at Task 16, the SP1 definition-of-done gate.)_

- **F1 — `core/signal_processing.detect_packets` power/magnitude confusion (FIXED, Task 3).**
  Symptom: `detect_packets` computed `power = np.abs(iq)` (magnitude, not power `|x|²`)
  despite the name, and its `if np.iscomplexobj(...) / else` branches were byte-identical
  dead code — threshold semantics were therefore wrong for every caller. RED test:
  `tests/test_signal_processing.py::test_uses_power_not_magnitude_semantics`. Fix: power is
  now computed as `np.abs(iq_samples).astype(np.float64) ** 2`, consistent with
  `calculate_power()`/`estimate_snr()` elsewhere in the module, and the dead branch was
  removed. The full test suite (134 → 166 tests as validation-package tests were added
  through Tasks 1–15) stayed green across all five caller modules
  (`capture/detector.py`, `capture/manager.py`, `capture/sniffer.py`, `core/parsing.py`,
  `plugins/protocols/generic.py`, plus `training/dataset.py` and `validation/pipeline.py`).
  **Status: FIXED.**

- **F2 — SigMF annotation round-trip loss (CONFIRMED, §11.5; MITIGATED, DEFERRED).**
  Symptom: `utils.fileio._write_sigmf_data` hardcodes a top-level `"annotations": []` and
  writes any caller-supplied metadata under `global.user:*` instead of the SigMF
  `annotations` array, so annotations passed in through the standard `write_iq_file`
  path do not round-trip through `utils.fileio`. Confirmed by inspection of
  `utils/fileio.py` (`_write_sigmf_data`, ~line 910–945): the `sigmf_meta["annotations"]`
  key is never populated from `metadata`, and extra keys are namespaced under
  `global.user:*` rather than emitted as SigMF annotation objects. Impact: any consumer
  that round-trips a `LabeledCapture`'s `truth_regions` purely through
  `.sigmf-meta`/`utils.fileio` would silently lose region labels.
  Mitigation (shipped, Task 7/8): `validation.dataset.LabeledDataset.write()` uses a
  **dual-metadata** layout — it writes the standard `.sigmf-data`/`.sigmf-meta` pair via
  `utils.fileio.write_iq_file` for sample round-trip, *and* writes a `.json` sidecar
  (`capture_NNNNN.json`) carrying `{sample_rate, protocol, annotations, provenance}`.
  `validation.ingest.labeler.load_labeled` reads the `.json` sidecar first when both exist
  (see `validation/dataset.py` module docstring and `tests/validation/test_labeler.py::
  test_label_from_sidecar_json`), so truth-region fidelity for the validation spine does
  not depend on the SigMF writer's annotation bug. A direct fix in
  `utils.fileio._write_sigmf_data` was scoped out of SP1 (touches a shared, widely-consumed
  I/O module outside the `validation/` file-touch boundary and risks regressing other
  SigMF consumers of `utils.fileio`) — logged as a candidate for SP2.
  **Status: CONFIRMED, mitigated for the validation spine, fix deferred to SP2.**

- **§11.2 — two divergent `detect_packets` implementations (CHARACTERIZED, non-blocking).**
  `core/signal_processing.py:455` and `capture/detector.py` (plus a `SignalDetector` class)
  implement detection independently. Rather than reconciling them for SP1,
  `validation.pipeline.DetectClassifyPipeline` accepts an injectable `detector: DetectorFn`
  (defaulting to `default_detector`, which wraps `core.signal_processing.detect_packets`),
  so either implementation — or a `capture.detector`-backed one — can be run through the
  harness and compared on Pd/Pfa via `evaluate(...)`. The mechanism to quantify the
  divergence exists and is exercised by `tests/validation/test_pipeline.py`'s injectable-
  detector tests; an actual side-by-side divergence run was not required for SP1 sign-off
  (tracked as gap G2 in the plan self-review — optional follow-up, not blocking).
  **Status: CHARACTERIZED / non-blocking.**

- **§11.3 — `core/demodulation` byte-recovery fidelity (BOUNDED, Task 4).**
  Concern: if demodulation is too lossy, classification accuracy in the harness reflects
  demod noise rather than the classifier under test. Bounded by the Task 4 modulator
  round-trip tests — `tests/validation/test_modulators.py::test_fsk_roundtrip_recovers_bits`,
  `test_qpsk_roundtrip_recovers_bits`, `test_gfsk_roundtrip_recovers_bits` — which assert 0
  bit-error-rate at high SNR through independent reference demodulators for each scheme,
  giving a known-good floor against which pipeline-level demod (`validation.pipeline.
  region_to_bytes`) can be judged. `DetectClassifyPipeline` additionally exposes
  `use_truth_bytes=True` to bypass demodulation entirely and score the classifier on
  known-truth bytes, isolating the two failure modes when needed.
  **Status: BOUNDED / non-blocking.**

- **§11.4 — `core/classification` feature-extraction determinism (EXERCISED, Task 11).**
  Requirement: feature extraction must be deterministic for a given input, since
  reproducibility (§2 goal 5) depends on it end-to-end. Exercised by the harness
  determinism test — `tests/validation/test_harness.py::test_run_is_deterministic` — which
  runs `run_evaluation`/`evaluate` twice with the same seed over the same dataset and
  asserts the resulting `RunResult`/manifest hashes are byte-identical; this is further
  confirmed at the whole-package level by `tests/validation/test_integration_smoke.py::
  test_full_spine_smoke` (Task 16), which asserts `r1.manifest.dataset_hash ==
  r2.manifest.dataset_hash` across two independent `evaluate()` calls.
  **Status: EXERCISED / non-blocking.**
