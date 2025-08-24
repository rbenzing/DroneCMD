# DroneCMD: Full-Spectrum Drone Command & Interference Toolkit

**DroneCMD** is a Python-based framework that enables signal analysis, classification, live capture, and command injection for popular commercial drones using SDR (Software-Defined Radio) hardware such as HackRF.

> ⚠️ This toolkit is intended for lawful security research and educational purposes only.

---

## ✈️ Features

- 📡 Live IQ Capture using HackRF and pyrtlsdr
- 🧠 Protocol Classification via Machine Learning
- 🧬 Full Packet Parser for Signal Inspection
- 🎮 Plugin-based Command Injection per Manufacturer
- 🧰 Modular CLI for scripting and automation
- 🧪 Testable, extensible Python project

---

## 🚀 Quickstart

1. Clone the repository:

```bash
git clone https://github.com/your-org/dronecmd.git
cd dronecmd
```

2. Install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Use the CLI:

```bash
python3 -m dronecmd.cli.cli capture --freq 2450 --duration 5 --output cap.iq
python3 -m dronecmd.cli.cli identify --input cap.iq
python3 -m dronecmd.cli.cli inject --plugin dji --interface hackrf0 --command '{"takeoff": true}'
```

---

## 🔌 Supported Plugins

- DJI
- Parrot
- Autel (coming soon)
- Skydio (coming soon)

Use `dronecmd list-plugins` to view available modules.

---

## ⚙️ Developer Notes

- Python 3.8+ required
- HackRF tools must be installed for `pyhackrf`
- All packet encoders are extensible and plugin-friendly

---

## 🛡 Legal Disclaimer

This software is for educational and authorized research purposes **only**. Use of this software against networks, drones, or hardware that you do not own or explicitly have permission to test is strictly prohibited.

---

© 2025 Russell Benzing | MIT License