# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DroneCMD is a Python-based SDR (Software-Defined Radio) framework for drone communication analysis. It provides signal capture, protocol classification, packet parsing, and command injection capabilities for commercial drones using SDR hardware (HackRF, RTL-SDR, etc.).

**IMPORTANT**: This is security research and educational software. All development must be for lawful security research and authorized testing only.

## Development Commands

### Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"

# Optional feature sets
pip install -e ".[sdr]"    # HackRF support (pyhackrf)
pip install -e ".[crypto]" # Cryptography features
pip install -e ".[viz]"    # Visualization tools
pip install -e ".[all]"    # All optional features
```

### Testing

**Note**: Tests live in `tests/`. Mock `rtlsdr.RtlSdr` and other hardware interfaces for unit tests. The project uses a flat top-level layout (`core/`, `capture/`, ...) rather than a single `dronecmd` package, so tests import modules directly (e.g. `from core.classification import ...`); `conftest.py` adds the repo root to `sys.path`.

```bash
pytest
pytest --cov=. --cov-report=html
pytest -m "not slow and not hardware"   # Skip slow/hardware-dependent tests
pytest -m integration
pytest -n auto                           # Parallel execution
pytest tests/test_specific.py::test_function_name
```

Test markers: `slow`, `integration`, `hardware`, `async`

### Code Quality
```bash
# Format + lint everything (flat layout — no `dronecmd` package)
black . && isort . && flake8

# Type-check. The CI-enforced strict-clean gate is `validation/` plus the
# hardened signal-core modules below; the remaining legacy modules carry
# pre-existing type debt and are not gated.
mypy validation core/blind.py core/single_carrier.py core/ofdm.py core/selftest.py
mypy core capture plugins utils injector cli.py   # legacy — has known findings

bandit -r core capture plugins utils injector cli.py  # Security scanning
```

CI (`.github/workflows/ci.yml`) runs `black`/`isort`/`flake8` scoped to
`validation/` and `mypy` over `validation/` plus the hardened signal-core
modules (`core/blind.py`, `core/single_carrier.py`, `core/ofdm.py`,
`core/selftest.py`), plus the `pytest` suite (`-m "not slow and not hardware"`)
on Python 3.9–3.12 for every push and PR to `main`.

### CLI Usage
```bash
dronecmd capture --platform hackrf --frequency 2.44e9 --duration 30 --output capture.iq
dronecmd analyze --input capture.iq --protocols mavlink,dji
dronecmd replay --input capture.iq --strategy intelligent --count 5
dronecmd generate fhss --frequency 2.44e9 --data "test payload"
dronecmd config show
dronecmd config set capture.default_sample_rate 2048000

# Empirical validation / T&E of the detect→classify pipeline
dronecmd validate synth  --protocols mavlink,dji --snr=-20:20:2 --n 50 --out ds/
dronecmd validate ingest --input captures/ --out ds/
dronecmd validate run    --dataset ds/ --models models/ --report report.json --plots
```
Note: `--snr` uses the equals form (`--snr=-20:20:2`) for ranges with a
negative lower bound (argparse otherwise treats the value as a flag).

### Versioning & Releases

Versions are **derived from git tags** by `setuptools_scm` — there is no
hardcoded version string to edit. `__init__.__version__` and
`constants.FRAMEWORK_VERSION` both resolve from the installed package metadata
(falling back to the scm-written `_version.py`, then `"0.0.0+unknown"`).

**To cut a release ("version up"):**
```bash
git tag v1.2.3            # semver; the tag IS the version
git push origin v1.2.3
```
Pushing a `v*` tag triggers `.github/workflows/release.yml`, which builds the
sdist + wheel (setuptools_scm pins them to `1.2.3`) and publishes a GitHub
Release with auto-generated notes. The generated `_version.py` is git-ignored.
Do **not** add a static `version =`/`__version__ =` string or reintroduce
bump2version — tags are the single source of truth.

## Architecture

### Two-Layer API Design (Progressive Enhancement)

The codebase deliberately exposes two API levels. **Both must be maintained when adding features.**

**Simple layer** (`capture/` + top-level `__init__.py`):
- Factory functions: `create_capture_manager()`, `create_protocol_classifier()`
- Classes: `CaptureManager`, `PacketSniffer`
- Synchronous, good for prototyping

**Enhanced layer** (`core/`):
- Classes: `EnhancedLiveCapture`, `DemodulationEngine`, `EnhancedFHSSEngine`, `EnhancedProtocolClassifier`, `EnhancedReplayEngine`
- Async/await streaming: `async for samples in capture.stream_samples()`
- Context managers: `async with EnhancedLiveCapture(config) as capture`
- Dataclass config objects: `SDRConfig`, `DemodConfig`, `FHSSConfig`, `ReplayConfig`
- Structured result objects with `detected`, `confidence`, `error_message` fields

Use `asyncio.run()` for top-level async calls. Use `asyncio.create_task()` for background processing. Always clean up SDR resources via async context managers.

### Module Map

- **`core/`** — Enhanced signal processing: `capture.py` (SDR capture), `demodulation.py` (FM/AM/FSK/GFSK/QPSK), `classification.py` (ML-based protocol ID), `fhss.py` (FHSS + FCC compliance), `signal_processing.py` (DSP/FFT utilities), `replay.py` (signal replay), `parsing.py` (packet parsing), `ofdm.py` (OFDM demod chain; the adaptive bit-loaded OFDM path — `modulate_ofdm_loaded`/`demodulate_ofdm_loaded`, plus the bit-map header, per-subcarrier CSI exposure `ofdm_equalized_symbols_csi`, the soft-LLR loaded demod `demodulate_ofdm_loaded_soft`, and the **coded bit-loaded OFDM (BICM)** composition `modulate_coded_ofdm_loaded`/`decode_coded_ofdm_loaded`), `single_carrier.py` (single-carrier receivers + soft-LLR demod), `bitloading.py` (adaptive square-QAM {QPSK/16-QAM/64-QAM} + per-subcarrier SNR + Chow's rate-adaptive loading + the soft square-QAM demapper `qam_soft_demap`), `profiles.py` (parametric link-profile catalog, profile-carried coding), `blind.py` (blind profile resolution: sync gate + trial-demod EVM tiebreak), `coding.py` (channel-coding framework: `Codec` registry, CRC framing, interleaver, convolutional/Viterbi + Reed-Solomon + BCH + LDPC + turbo + polar + fountain codecs — **the seven-family FEC sheet is now complete**), `galois.py` (field-parametric GF(2^m) algebra + Berlekamp-Massey + Chien, shared by RS/BCH), `polar.py` (Arıkan polar transform + Gaussian-approximation frozen set + CRC-aided SCL decoder), `fountain.py` (Raptor-style fountain code: systematic sparse precode + Robust-Soliton LT encoder, per-symbol-CRC erasure detection, GF(2) Gaussian-elimination decoder)
- **`capture/`** — Simple capture layer: `manager.py`, `detector.py`, `sniffer.py`
- **`plugins/`** — Protocol plugin system: `base.py` (abstract base classes), `registry.py` (discovery), `protocols/` (DJI, Parrot, generic, `_template.py`)
- **`utils/`** — Cross-cutting: `config.py` (YAML config + profiles), `logging.py`, `fileio.py` (IQ file formats), `crypto.py`, `compat.py`
- **`injector/`** — Packet injection: `suringe.py` (injection engine), `obfuscation.py`
- **`training/`** — Classifier training pipeline: `dataset.py` (feature extraction from labeled captures), `train.py` (sklearn ensemble training + cross-validation)
- **`validation/`** — Validation & Test-and-Evaluation (T&E) spine: synthetic signal generator + calibrated channel model (`synth/`, including coded modulation via `modulate(coding=...)`), real-capture ingestion (`ingest/`), unified SigMF-backed `LabeledDataset` (`dataset.py`), injectable detect→demod→classify `pipeline.py` (with soft/hard coded-decode branches and blind link resolution), detection/classification metrics **plus the coded-link (coded BER/FER) metric** with bootstrap CIs (`metrics.py`), evaluation `harness.py`, JSON `report.py`, reproducibility manifest (`repro.py`). Library-first public API in `validation/__init__.py`; driven by `dronecmd validate`
- **`cli.py`** — argparse-based CLI (subcommands: `capture`, `analyze`, `replay`, `generate`, `convert`, `config`, `info`, `selftest`, `train`, `validate`) with JSON output support
- **`core/selftest.py`** — hardware-in-the-loop RX self-test: drives a real HackRF (receive-only) across capture parameters, both capture APIs, the analysis pipeline and the CLI, returning a pass/fail result set. Exposed as `dronecmd selftest` and `scripts/hackrf_selftest.py`
- **`constants.py`** — RF frequency ranges, sample rates, protocol constants
- **`exceptions.py`** — Custom exception hierarchy

### Design Docs & Decisions

- **`docs/adr/`** — Architecture Decision Records (Nygard-style): the committed source of truth for *why* the architecture is the way it is. Start at `docs/adr/README.md`.
- **`docs/design/`** — Numbered, committed per-phase design specs (the detailed *how* of each capability).
- Detailed implementation plans and per-task execution ledgers stay **local/git-ignored** under `docs/superpowers/` and `.superpowers/` (see ADR-0011). When you add a capability, keep the module map above, the relevant ADR, and README.md in sync.

### Plugin System

Plugins declare `PluginType` (PROTOCOL, INJECTION, ANALYSIS, DECODER, ENCODER) and `PluginCapability` (DETECT, DECODE, ENCODE, INJECT, ANALYZE, VALIDATE) via a `PluginMetadata` dataclass. Use `plugins/protocols/_template.py` as the starting point.

**To add a new protocol plugin:**
1. Create `plugins/protocols/<name>.py`, inherit from `BaseProtocolPlugin` in `plugins/base.py` (the convenience base the bundled DJI/Parrot/generic plugins use; it supplies default `metadata`/`initialize()`/`validate()`). The fully abstract `ProtocolPlugin` base is also available.
2. Implement `get_name()`, `get_version()`, `get_supported_protocols()`, `detect()` (returns `ProtocolDetectionResult`), and `parse_packet()` (returns `ProtocolParseResult`); override `metadata`/`initialize()`/`validate()` only if the defaults are insufficient.
3. Register in `pyproject.toml` under `[project.entry-points."dronecmd.plugins"]` using the exact class name, e.g. `name = "plugins.protocols.<name>:MyProtocolPlugin"`.

**To add a new capture platform:**
1. Add enum value to `SDRPlatform` in `core/capture.py`
2. Implement initialization in `_initialize_sdr()`
3. Add validation in `SDRConfig`

**To add a new IQ file format:**
1. Add enum to `FileFormat` in `utils/fileio.py`
2. Implement reader/writer with metadata preservation and compression support

## Key Constraints

### Signal Processing
- Use `numpy.complex64` for IQ samples (not `complex128`)
- Apply windowing before FFT operations
- Validate SNR and power metrics; handle empty signals, DC offset, and frequency drift

### SDR Hardware
- Validate sample rates against hardware limits before applying
- Handle USB buffer overruns gracefully
- Support both local USB and TCP-connected SDR devices (`rtlsdr.RtlSdr` for RTL-SDR)

### Type Safety
- Strict `mypy` is configured — avoid `Any`; use `numpy.typing` for array annotations
- Type aliases in use: `FrequencyHz = float`, `IQSamples = npt.NDArray[np.complex64]`, `PacketBytes = bytes`

### Security
- All injection features must validate FCC compliance where applicable
- Docstrings follow Google style; document complex algorithms with references to specs/papers
