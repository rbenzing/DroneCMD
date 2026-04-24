# Recording Real Drone IQ Captures for Classifier Training

This guide covers everything needed to build a labeled IQ dataset from actual
drone traffic.  The classifier (`core/classification.py`) requires real training
data — it will raise `ModelNotTrainedError` until you train it.

---

## Hardware Required

| Hardware | Role | Approximate cost |
|---|---|---|
| HackRF One | Primary wideband capture (1 MHz – 6 GHz, TX/RX) | ~$340 |
| RTL-SDR v3 | Secondary / budget capture (500 kHz – 1.75 GHz, RX only) | ~$30 |
| Directional antenna (patch or Yagi, 2.4 GHz / 5.8 GHz) | Improve SNR at range | $20–80 |
| USB 3.0 host with ≥4 GB RAM | Sustained high-sample-rate writes | — |

**Minimum viable**: RTL-SDR v3 + stock antenna captures DJI and Parrot traffic
at ≤ 30 m.  HackRF is needed for 5.8 GHz OcuSync captures and for
transmit-path validation.

---

## Software Setup

```bash
# Install SDR system libraries (Ubuntu/Debian)
sudo apt install hackrf rtl-sdr sox

# Test HackRF
hackrf_info

# Test RTL-SDR
rtl_test -t

# Install DroneCMD with SDR support
pip install -e ".[sdr,dev]"
```

On Windows: install Zadig USB drivers for both devices before running the
above (`zadig.akeo.ie`).

---

## Target Frequencies by Protocol

| Protocol | Frequency | Sample rate | Notes |
|---|---|---|---|
| DJI OcuSync 2/3 | 2.400–2.483 GHz | 20 MHz | Requires HackRF |
| DJI OcuSync 2/3 (5.8 GHz) | 5.725–5.850 GHz | 20 MHz | Requires HackRF |
| DJI Lightbridge | 2.400–2.483 GHz | 10 MHz | |
| DJI WiFi (Spark, Tello) | 2.4 GHz or 5.8 GHz | 20 MHz | 802.11 carrier |
| Parrot ARSDK | 2.4 GHz | 5 MHz | |
| MAVLink over telemetry radio | 433 MHz / 915 MHz | 2 MHz | 3DR radios |
| MAVLink over 900 MHz FHSS | 902–928 MHz | 2 MHz | SiK firmware |

Start at 2.4 GHz — most consumer drones operate there, RTL-SDR can reach it,
and it gives the most training variety.

---

## Capture Procedure

### 1. Position the drone

- Place the drone on the ground, powered on but not in flight
- Stand 5–20 m away with line of sight to the drone
- Controller should be powered on and linked
- Aim directional antenna at the drone if using one

### 2. Capture with DroneCMD

```bash
# DJI OcuSync 2.4 GHz — 30 second capture
dronecmd capture \
  --frequency 2.44e9 \
  --sample-rate 20e6 \
  --duration 30 \
  --gain 40 \
  --output captures/dji_ocusync_2g4_flight01.iq

# Parrot at 2.4 GHz
dronecmd capture \
  --frequency 2.44e9 \
  --sample-rate 5e6 \
  --duration 30 \
  --output captures/parrot_arsdk_01.iq

# MAVLink telemetry at 433 MHz
dronecmd capture \
  --frequency 433.5e6 \
  --sample-rate 2e6 \
  --duration 60 \
  --output captures/mavlink_433_01.iq
```

Or directly with HackRF tools (useful for raw validation):

```bash
hackrf_transfer -r captures/raw_2g4.iq -f 2440000000 -s 20000000 -n 600000000
```

### 3. Capture background noise (no drone active)

This is required to train the "unknown/noise" class and prevent false positives:

```bash
dronecmd capture \
  --frequency 2.44e9 \
  --sample-rate 20e6 \
  --duration 30 \
  --output captures/background_noise_2g4_01.iq
```

### 4. Repeat for statistical confidence

Minimum per class for a deployable classifier:

| Class | Min captures | Min total duration |
|---|---|---|
| dji_ocusync | 20 | 10 minutes |
| parrot | 10 | 5 minutes |
| mavlink | 10 | 5 minutes |
| unknown/noise | 20 | 10 minutes |

More is better.  Vary: distance (5 m, 15 m, 30 m), orientation, indoor vs
outdoor, and flight state (idle, hover, active flight).

---

## Labeling

Each capture file needs a corresponding label file.  Create a JSON sidecar:

```bash
# Automatically written if using dronecmd capture (--include-metadata is default)
# Manual example:
cat > captures/dji_ocusync_2g4_flight01.json << 'EOF'
{
  "protocol": "dji_ocusync",
  "frequency_hz": 2440000000,
  "sample_rate_hz": 20000000,
  "capture_date": "2025-06-01",
  "drone_model": "DJI Mini 3 Pro",
  "distance_m": 10,
  "environment": "outdoor_open",
  "flight_state": "hovering",
  "notes": ""
}
EOF
```

Accepted `protocol` label strings (must match exactly):

- `dji_ocusync` — DJI OcuSync 2.x / 3.x
- `dji_lightbridge` — DJI Lightbridge (legacy)
- `dji_wifi` — DJI WiFi-based protocols (Spark, Tello)
- `parrot` — Parrot ARSDK / Skycontroller
- `mavlink` — MAVLink over any carrier
- `unknown` — background noise or unidentified signals

---

## Verifying a Capture

Before using a file for training, verify it contains signal:

```bash
dronecmd analyze --input captures/dji_ocusync_2g4_flight01.iq
```

Key things to check in the output:
- `signal_power_dbfs` should be above –40 dBFS (strong drone signal)
- `packets_found` should be > 0
- `protocol_distribution` will say `unknown` until the classifier is trained —
  that is expected at this stage

You can also view the spectrum with:

```bash
# Requires pip install -e ".[viz]"
python -c "
import numpy as np
import matplotlib.pyplot as plt
from dronecmd.utils.fileio import read_iq_file

iq = read_iq_file('captures/dji_ocusync_2g4_flight01.iq')
freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), 1/20e6))
psd = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(iq))) + 1e-12)
plt.plot(freqs / 1e6, psd)
plt.xlabel('Frequency offset (MHz)')
plt.ylabel('Power (dBFS)')
plt.title('Spectrum')
plt.show()
"
```

You should see clear spectral peaks above the noise floor for real drone traffic.

---

## Directory Layout

Organize captures before running the training pipeline:

```
captures/
  dji_ocusync/
    flight01.iq       # 30 s, 20 MHz, 2.44 GHz
    flight01.json     # label sidecar
    flight02.iq
    flight02.json
    ...
  parrot/
    session01.iq
    session01.json
    ...
  mavlink/
    telemetry01.iq
    telemetry01.json
    ...
  unknown/
    noise01.iq
    noise01.json
    ...
```

Then run training:

```bash
dronecmd train --data-dir captures/ --output-dir models/
```

See `docs/training_pipeline.md` for full training documentation.

---

## Troubleshooting

**No signal detected:**
- Increase gain (`--gain 50` or higher for RTL-SDR)
- Move drone closer
- Check antenna connection
- Verify drone is powered on and controller is linked

**RTL-SDR overflow errors (`O` characters in output):**
- Reduce sample rate (`--sample-rate 2e6` instead of 20e6)
- Use faster USB port (USB 3.0)
- Close other applications

**HackRF "hackrf_open() failed":**
- Unplug and replug
- Check udev rules: `sudo cp /usr/share/hackrf/udev/53-hackrf.rules /etc/udev/rules.d/ && sudo udevadm control --reload`

**Captures look like pure noise even near drone:**
- DJI OcuSync uses spread-spectrum — the signal will look noise-like at 20 MHz bandwidth.  This is correct.  Verify with `dronecmd analyze` packet detection, not visual inspection.
- Try 5 MHz bandwidth at the OcuSync center channel (2440 MHz) to see individual hop packets.
