# DroneCMD

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-126%20passing-brightgreen?style=for-the-badge)]()
[![SDR](https://img.shields.io/badge/SDR-HackRF%20%7C%20RTL--SDR-orange?style=for-the-badge)]()
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black)](https://buymeacoffee.com/russellbenzing)

**A production-grade Python framework for SDR-based drone communication analysis**

📡 **SDR Capture** • 🧠 **ML Classification** • 🛰️ **MAVLink / DJI Parsing** • ⚖️ **FCC-Compliant Injection**

[Capabilities](#-key-capabilities) • [Installation](#-installation) • [Architecture](#-architecture) • [License](#-legal)

</div>

---

**DroneCMD** is a production-grade Python framework for SDR-based drone communication analysis. It provides a full signal pipeline — capture → demodulate → classify → parse — with FCC-compliant signal injection, a from-scratch ML classifier training system, and a plugin architecture for extensibility.

> ⚠️ **This software is for lawful security research and authorized testing only.**

---

## 🛰️ Key Capabilities

| Capability | Description |
|---|---|
| Live IQ Capture | HackRF / RTL-SDR with async streaming and USB overrun recovery |
| Protocol Classification | Sklearn ensemble trained on your own labeled IQ captures |
| Packet Parsing | MAVLink v1/v2, DJI OcuSync/Lightbridge, generic protocol analysis |
| FHSS Engine | Frequency hopping with FCC CFR 47 §15.247 compliance enforcement |
| Signal Injection | Power-limited, dwell-time-enforced injection with typed compliance exceptions |
| Plugin System | Extensible per-manufacturer protocol plugins (DJI, Parrot, custom) |
| CLI | Full-featured argparse CLI with JSON output support |

---

## 📋 Requirements

- Python 3.8+
- HackRF tools (`hackrf_info`) or RTL-SDR drivers for hardware capture
- No hardware required for signal analysis, parsing, classification, or testing

---

## 📦 Installation

```bash
git clone https://github.com/rbenzing/DroneCMD.git
cd DroneCMD
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

Optional feature sets:

```bash
pip install -e ".[sdr]"        # HackRF support (pyhackrf)
pip install -e ".[crypto]"     # Reed-Solomon FEC + cryptography
pip install -e ".[viz]"        # Matplotlib / Plotly visualization
pip install -e ".[all]"        # Everything
```

---

## 🖥️ CLI Usage

```bash
# Capture IQ samples from SDR hardware (2.44 GHz needs HackRF; default is rtl_sdr)
dronecmd capture --platform hackrf --frequency 2.44e9 --duration 30 --output capture.iq

# Analyze a capture file
dronecmd analyze --input capture.iq --protocols mavlink,dji

# Train the protocol classifier on labeled captures
dronecmd train --data-dir captures/ --output-dir models/

# Replay a recorded signal (FCC compliance enforced)
dronecmd replay --input capture.iq --strategy intelligent --count 5

# Generate and transmit an FHSS signal
dronecmd generate fhss --frequency 2.44e9 --data "test payload"

# Show/set configuration
dronecmd config show
dronecmd config set capture.default_sample_rate 2048000
```

---

## 🧠 Classifier Training

The classifier is **not pre-trained** — it learns from your own labeled IQ captures. This is intentional: effective classification requires data collected from the specific hardware and environment you are analyzing.

**Collect captures** → **label them** → **train** → **classify**

See [docs/iq_capture_guide.md](docs/iq_capture_guide.md) for the full data collection procedure, directory layout, labeling format, and verification steps.

```bash
# After collecting and labeling captures:
dronecmd train --data-dir captures/ --output-dir models/ --cv-folds 5

# Models are then auto-loaded from models/ on next run
dronecmd analyze --input new_capture.iq
```

---

## 🏗️ Architecture

DroneCMD exposes two API layers that are both maintained:

**Simple layer** — factory functions and synchronous classes for quick scripting:

```python
from capture.manager import CaptureManager

manager = CaptureManager(platform="hackrf", sample_rate=2_048_000)
iq_data = manager.load_file("capture.iq")
packets = manager.extract_packets(iq_data, threshold=0.1)
```

**Enhanced layer** — async streaming, structured configs, typed results:

```python
from core.capture import EnhancedLiveCapture, SDRConfig, SDRPlatform

# Field names are frequency_hz / sample_rate_hz; SDRConfig validates against
# the platform's limits at construction (2.44 GHz requires HackRF).
config = SDRConfig(
    frequency_hz=2.44e9,
    sample_rate_hz=2_048_000,
    platform=SDRPlatform.HACKRF,
)
async with EnhancedLiveCapture(config) as capture:
    async for samples in capture.stream_samples():
        process(samples)  # each block is a numpy complex64 IQ array
```

The `EnhancedProtocolClassifier.classify()` method operates on **extracted
packet bytes**, not raw IQ — run demodulation/packet extraction first, then
classify the resulting bytes.

**Module map:**

- `core/` — Signal pipeline: `capture`, `demodulation`, `classification`, `fhss`, `signal_processing`, `replay`, `parsing`
- `capture/` — Simple layer: `manager`, `detector`, `sniffer`
- `injector/` — Packet injection engine with FCC compliance enforcement
- `plugins/` — Protocol plugin system: `base`, `registry`, `protocols/` (DJI, Parrot, generic)
- `training/` — Classifier training pipeline: `dataset`, `train`
- `utils/` — IQ file I/O, YAML config, logging, crypto, compat
- `cli.py` — argparse-based CLI entry point

---

## 🔌 Plugin Development

Use `plugins/protocols/_template.py` as the starting point. Subclass
`BaseProtocolPlugin` (from `plugins/base.py`) and implement `get_name()`,
`get_version()`, `get_supported_protocols()`, `detect()` (returns
`ProtocolDetectionResult`), and `parse_packet()` (returns `ProtocolParseResult`).
Register the exact class name in `pyproject.toml`:

```toml
[project.entry-points."dronecmd.plugins"]
myplugin = "plugins.protocols.myplugin:MyProtocolPlugin"
```

---

## 🧪 Testing

```bash
pytest                                          # All tests
pytest -m "not slow and not hardware"           # Skip hardware/slow tests
pytest tests/test_fhss.py -v                    # Single file
pytest -n auto                                  # Parallel execution
```

126 unit tests, no hardware required. Hardware-dependent tests are marked `@pytest.mark.hardware` and excluded by default.

---

## ⚖️ Compliance

Signal injection enforces hard limits — violations raise typed exceptions, not silent log warnings:

| Violation | Exception |
|---|---|
| Power exceeds limit | `PowerLimitError` |
| Dwell time exceeded | `DwellTimeError` |
| Frequency out of band | `FrequencyViolationError` |

These are non-bypassable. `InjectionConfig.validate()` returns a list of errors before transmission begins.

---

## ⚖️ Legal

This software is for educational and authorized security research **only**. Use against hardware, drones, or networks you do not own or do not have explicit permission to test is prohibited.

© 2025 Russell Benzing | MIT License

---

## 👤 About the Author

Built by **Russell Benzing**. DroneCMD is a defensive/research SDR toolkit for lawful drone-communication analysis and authorized testing.

---

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/rbenzing/DroneCMD/issues)
- **Data collection guide**: [docs/iq_capture_guide.md](docs/iq_capture_guide.md)

If DroneCMD is useful to you, you can support the work:

<div align="center">

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black)](https://buymeacoffee.com/russellbenzing)

</div>
