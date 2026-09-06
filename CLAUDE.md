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
# Run all checks before committing (flat layout — no `dronecmd` package)
black . && isort . && flake8 && mypy core capture plugins utils injector cli.py

bandit -r core capture plugins utils injector cli.py  # Security scanning
```

### CLI Usage
```bash
dronecmd capture --platform hackrf --frequency 2.44e9 --duration 30 --output capture.iq
dronecmd analyze --input capture.iq --protocols mavlink,dji
dronecmd replay --input capture.iq --strategy intelligent --count 5
dronecmd generate fhss --frequency 2.44e9 --data "test payload"
dronecmd config show
dronecmd config set capture.default_sample_rate 2048000
```

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

- **`core/`** — Enhanced signal processing: `capture.py` (SDR capture), `demodulation.py` (FM/AM/FSK/GFSK/QPSK), `classification.py` (ML-based protocol ID), `fhss.py` (FHSS + FCC compliance), `signal_processing.py` (DSP/FFT utilities), `replay.py` (signal replay), `parsing.py` (packet parsing)
- **`capture/`** — Simple capture layer: `manager.py`, `detector.py`, `sniffer.py`
- **`plugins/`** — Protocol plugin system: `base.py` (abstract base classes), `registry.py` (discovery), `protocols/` (DJI, Parrot, generic, `_template.py`)
- **`utils/`** — Cross-cutting: `config.py` (YAML config + profiles), `logging.py`, `fileio.py` (IQ file formats), `crypto.py`, `compat.py`
- **`injector/`** — Packet injection: `suringe.py` (injection engine), `obfuscation.py`
- **`cli.py`** — Click-based CLI with JSON output support
- **`constants.py`** — RF frequency ranges, sample rates, protocol constants
- **`exceptions.py`** — Custom exception hierarchy

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
