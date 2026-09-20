# ADR-0019: HackRF receive backends — SoapySDR with a `hackrf_transfer` CLI fallback

- **Status:** Accepted
- **Date:** 2026-09-20
- **Deciders:** rbenzing

## Context

`SDRPlatform.HACKRF` was declared but had no working capture backend — selecting
it fell through to `NotImplementedError`. HackRF is a primary target device for
this framework, so a real receive path was needed.

Two facts constrain the implementation:

- The natural in-process path is the **SoapySDR** Python bindings against the
  `SoapyHackRF` module. But the SoapySDR Python bindings are a compiled C
  extension whose ABI is tied to a specific CPython version. On Windows the
  common distribution (PothosSDR) ships those bindings for **Python 3.9 only**,
  while DroneCMD here runs on Python 3.13 — so `import SoapySDR` is not possible
  in the application interpreter, and the SoapySDR backend cannot be exercised
  in-process on that host.
- The stock **`hackrf_transfer` / `hackrf_info` CLI** (from `libhackrf`) has no
  such constraint: it is a native binary driven over a subprocess, so it works
  regardless of the Python interpreter's ABI. On this host it captures real RF
  successfully where the SoapySDR bindings cannot load.

All HackRF use in this project is **receive-only** — transmit stays gated behind
explicit legal/safety authorization and is deliberately out of scope for these
backends.

## Decision

We will provide **two receive-only HackRF backends** in `core/capture.py`, both
implementing the existing `SDRHardwareInterface` protocol, and select between
them automatically:

- **`SoapyHackRFHardware`** — in-process SoapySDR (`driver=hackrf`) RX stream.
  Used when the SoapySDR Python bindings are importable (`SOAPY_SDR_AVAILABLE`).
- **`HackRFTransferHardware`** — a portable fallback that shells out to the
  `hackrf_transfer -r` CLI per `read_samples` call, loading the recorded signed
  8-bit interleaved I/Q as `complex64`. Used when SoapySDR is unavailable but the
  CLI is found (`HACKRF_TRANSFER_AVAILABLE`). It locates `hackrf_transfer` /
  `hackrf_info` on `PATH` and, failing that, at the default PothosSDR install
  locations.

`_create_hardware_interface` prefers SoapySDR, then the CLI backend, and raises a
clear, actionable error if neither is available. Neither backend ever invokes
transmit mode.

## Consequences

### Positive
- HackRF capture works out of the box on hosts where the SoapySDR Python bindings
  are ABI-mismatched (notably Windows + PothosSDR on a modern interpreter).
- The in-process SoapySDR path is retained where it is usable (Linux, or a Python
  matching the SoapyHackRF bindings) for lower per-call overhead.
- Selection is automatic and best-available; callers keep using `SDRPlatform.HACKRF`.

### Negative / trade-offs
- The CLI backend is not streaming-native: each `read_samples` call is a separate
  `hackrf_transfer` invocation (process spawn + temp file), so it is unsuitable
  for continuous low-latency streaming and adds per-call startup latency.
- Two backends to maintain, with slightly different gain semantics (SoapySDR
  `setGain` vs. the CLI's LNA/VGA/amp flag mapping).

### Neutral / notes
- The SoapySDR backend is validated by mocked unit tests; the CLI backend is
  validated by mocked unit tests **and** an on-air RX smoke test against real
  hardware (`@pytest.mark.hardware`). Both on-air smoke tests skip cleanly when a
  device or the required runtime is absent.
- See the README "HackRF hardware setup" section for the PothosSDR + Zadig
  install steps and the Python-3.9 SoapySDR-binding note.
