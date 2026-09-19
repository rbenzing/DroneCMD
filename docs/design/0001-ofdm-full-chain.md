# OFDM Full-Chain Integration — Design Spec

**Design #:** 0001
**Phase:** P1

**Date:** 2026-09-17
**Status:** Accepted — implemented, shipped in v0.3.0
**Sub-project:** SP1 follow-on (Task 15 "OFDM", previously deferred)

## Goal

Make OFDM a first-class modulation scheme across DroneCMD's demodulation
engine and the validation/T&E spine, so an OFDM burst travels the full
**modulate → channel → detect → demod → classify** chain and its recovered
bits are provably correct (0 BER noiseless; BER decreasing with SNR through
the calibrated channel).

## Motivation

OFDM was deferred in SP1 because `core/demodulation.py` had no OFDM
demodulator, so an OFDM packet could not be demodulated in the chain —
`ModScheme.OFDM` existed only as an enum placeholder and
`validation.synth.modulators.modulate()` raised `ValueError` for it. OFDM is
the modulation family behind modern drone links (e.g. DJI OcuSync,
WiFi-based control), so a defensible T&E spine must measure detect→classify
performance on OFDM, not just single-carrier FSK/GFSK/QPSK.

## Chosen fidelity

**Full Schmidl–Cox receiver** (user decision): preamble-based coarse timing
and fractional CFO estimation, CP-based framing, LS channel estimation from a
known long-training symbol, per-subcarrier one-tap equalization, and
pilot-based common-phase-error (CPE) tracking on data symbols. This survives
the AWGN, multipath, timing-offset, and (bounded) CFO/Doppler impairments in
`validation/synth/channel.py`.

## Scope: P1 of a phased effort

This spec is **Phase 1 (P1)** of a larger OFDM/demod effort that will be
built incrementally on a single branch and merged as one PR. Later phases,
each with their own spec brainstormed just before build:

- **P2 — Multiple OFDM profiles** (parameterized `OFDMProfile` registry).
- **P3 — Coding + interleaving** (FEC + block interleaver over the payload).
- **P4 — Adaptive bit-loading** (per-subcarrier QAM order from channel state).
- **PE — QPSK / single-carrier through the core `DemodulationEngine`.**

P1 introduces the shared `OFDMProfile` object precisely so P2–P4 extend it
without rework. P1 itself is deliberately the uncoded, single-profile,
QPSK-subcarrier baseline.

## Non-goals (for P1 specifically)

- **Integer-CFO estimation.** Only the fractional CFO within the Schmidl–Cox
  acquisition range (|Δf| < one subcarrier spacing) is corrected. Synthetic
  scenarios keep CFO within this range. Documented as an SP-level bound.
  (Stays out of scope across all phases unless raised.)
- **QPSK/single-carrier through the core `DemodulationEngine`** — deferred to
  **PE**. In P1 their pipeline demod behavior is unchanged (the existing
  inline FSK slicer / `--use-truth-bytes`).
- **Coding/interleaving** — deferred to **P3**. P1 payload bits map straight
  to QPSK subcarriers, uncoded.
- **Adaptive bit/power loading** — deferred to **P4**. P1 uses a fixed
  QPSK-per-subcarrier profile.
- **Multiple profiles** — deferred to **P2**. P1 ships the one fixed
  802.11a-style profile (though via the parameterized `OFDMProfile` object).
- **Real over-the-air OFDM capture ingestion tuning.** Ingestion is
  scheme-agnostic already; no OFDM-specific ingest work.

## PHY profile (fixed, 802.11a-style)

Single source of truth: `core/ofdm.py::OFDMProfile` / `DEFAULT_OFDM_PROFILE`.

| Parameter | Value |
|---|---|
| FFT size `N` | 64 |
| Cyclic prefix `CP` | 16 |
| OFDM symbol length | `N + CP` = 80 samples |
| Occupied subcarriers | 52: indices `-26..-1` and `1..26` |
| DC subcarrier (index 0) | null |
| Edge subcarriers (`±27..±31`, `-32`) | null |
| Pilot subcarriers | 4 at `k ∈ {-21, -7, 7, 21}`, fixed BPSK values `+1` (index-fixed polarity) |
| Data subcarriers | 48 (the occupied set minus pilots) |
| Subcarrier modulation | QPSK, Gray-mapped, 2 bits/subcarrier |
| **Data bits per OFDM data symbol** | **96** (48 × 2) |
| Output dtype / power | `complex64`, unit average power |

FFT convention: standard `numpy.fft.fft`/`ifft` with subcarrier index `k`
mapped to bin `k mod N` (i.e. `fftshift` layout for the ±26 occupied set).
QPSK Gray map (per subcarrier, MSB = I bit):
`00→(+1+1j)/√2, 01→(+1-1j)/√2, 10→(-1+1j)/√2, 11→(-1-1j)/√2`.

## Frame structure

```
[ STF ][ LTF ][ DATA_1 ][ DATA_2 ] ... [ DATA_M ]
  80     80      80         80             80        (samples each)
```

- **STF (Schmidl–Cox timing symbol):** load only *even* occupied subcarriers
  with a fixed PN sequence; the IFFT then produces two identical time-domain
  halves (each `N/2 = 32` samples, plus CP). Enables the S&C timing metric and
  fractional CFO estimate.
- **LTF (long training / channel-estimation symbol):** all 52 occupied
  subcarriers loaded with a fixed known QPSK sequence. Enables LS channel
  estimation `H(k) = Y_LTF(k) / X_LTF_known(k)`.
- **DATA symbols:** 48 QPSK data subcarriers + 4 known pilots each.

Payload framing: `bits = np.unpackbits(payload)`; zero-pad to a multiple of 96;
`M = len(bits)//96` data symbols. (Round-trip tests use payload lengths that
are exact multiples of 96 bits = 12 bytes, so recovery is byte-exact with no
padding ambiguity — matching the existing modulator round-trip test style.)

## Modulator (`validation/synth/modulators.py`)

- New `_ofdm(bits) -> NDArray[complex128]` thin wrapper delegating waveform
  synthesis to `core.ofdm` primitives (deterministic, no RNG — training and
  pilot sequences are fixed constants).
- Wire into `modulate()`: `elif scheme == ModScheme.OFDM: iq = _ofdm(bits)`;
  remove the `ValueError` for OFDM and the "unsupported in SP1" docstrings.
  `sps` is ignored for OFDM (the profile fixes symbol length); document this.
- Preserve the existing unit-average-power normalization tail.

## Shared OFDM module (`core/ofdm.py`, new)

Pure, dependency-light (`numpy` + `scipy.fft` optional; use `numpy.fft`),
fully typed with `numpy.typing`. Contents:

- `@dataclass(frozen=True) class OFDMProfile` — fields: `fft_size`, `cp_len`,
  `data_carriers: tuple[int,...]`, `pilot_carriers: tuple[int,...]`,
  `pilot_values: tuple[complex,...]`. `DEFAULT_OFDM_PROFILE` instance above.
- Subcarrier index → FFT-bin helpers.
- `stf_time(profile) -> NDArray[complex128]` and
  `ltf_time(profile) -> NDArray[complex128]` (with CP), plus the known
  frequency-domain LTF sequence for channel estimation.
- `map_bits_to_symbols(bits, profile) -> NDArray[complex128]` (QPSK Gray) and
  the inverse `demap_symbols_to_bits`.
- `modulate_ofdm(bits, profile) -> NDArray[complex128]` — full frame
  (STF+LTF+DATA), pre-normalization.
- Receiver helpers (pure functions): `schmidl_cox_metric(rx, L)`,
  `estimate_fractional_cfo(rx, d, L)`, `estimate_channel_ls(ltf_rx, profile)`,
  `equalize(Y, H)`, `estimate_cpe(pilots_rx, profile)`.

## Core demodulator (`core/demodulation.py`)

- `ModulationScheme.OFDM = "ofdm"`. Add to `bits_per_symbol` (→ 96) and to
  `requires_coherent_detection` (→ True).
- `DemodConfig`: add `ofdm_fft_size: int = 64`, `ofdm_cp_len: int = 16`.
  Special-case `samples_per_symbol` (→ `ofdm_fft_size + ofdm_cp_len`) and
  `symbol_rate_hz` (→ `sample_rate_hz / (fft+cp)`) for OFDM so those
  properties are meaningful and not derived from the single-carrier bitrate.
  `__post_init__` validation is unchanged (defaults pass; the OFDM demod does
  not use `bitrate_bps`).
- `class OFDMDemodulator(BaseDemodulator)` implementing `demodulate(iq) ->
  DemodulationResult`:
  1. Empty / shorter-than-`2*(N+CP)` input → `is_valid=False` with an
     `error_message` (mirrors existing demods' empty-input handling).
  2. Schmidl–Cox metric → coarse STF start `d̂`.
  3. Fractional CFO from `angle(P(d̂))` → derotate whole burst.
  4. Locate LTF (`d̂ + 80`), CP-strip + FFT → LS channel estimate `H(k)`.
  5. For each data symbol (`d̂ + 160 + i*80`): CP-strip, FFT, equalize with
     `H(k)`, estimate CPE from the 4 pilots and derotate, QPSK-demap the 48
     data subcarriers → 96 bits.
  6. Concatenate all symbols' bits into `result.bits` as an **unpacked**
     `uint8` array (one bit per element), matching `FSKDemodulator`'s
     `result.bits` convention (packing to bytes happens in the caller, not
     here). Populate quality metrics (SNR/EVM best-effort) consistent with
     other demods.
- Register `OFDMDemodulator` in `DemodulationEngine._create_demodulators()`
  and in the `demodulate()` scheme-override branch.

## Pipeline integration (`validation/pipeline.py`)

Make the demod step scheme-aware **without changing the default (FSK) path**:

- `region_to_bytes()` stays exactly as-is (the FSK reference slicer).
- Add `ofdm_region_to_bytes(iq_region) -> bytes` that lazily imports
  `core.demodulation`, runs `OFDMDemodulator` on the region, and returns
  `np.packbits(result.bits).tobytes()` (or `b""` on an invalid/empty result)
  — the same bit→byte packing `region_to_bytes` uses.
- `DetectClassifyPipeline.run()` reads the capture's known scheme from
  `capture.provenance.get("scheme")`. Dispatch:
  `scheme == "ofdm"` → `ofdm_region_to_bytes`; otherwise → `region_to_bytes`
  (current behavior). `use_truth_bytes` bypass is unchanged and still wins
  when set.
- No new required constructor args (backward compatible). Lazy import keeps
  `core.demodulation`'s heavier deps out of the pipeline import path.

Regression guarantee: existing captures/tests carry no `provenance["scheme"]`
== "ofdm" (they are FSK/QPSK), so they follow the identical FSK path.

## CLI & scenarios

- `validation/synth/scenarios.py`: no change (already scheme-agnostic via
  `scheme_by_protocol`). Verify OFDM flows through `build_scenario`.
- `cli.py` `validate synth`: extend the `default_scheme` map with
  `"ocusync": ModScheme.OFDM` (DJI OcuSync is OFDM-based), so
  `dronecmd validate synth --protocols ocusync ...` produces OFDM captures.
  Refresh the `--use-truth-bytes` help text: OFDM now demodulates for real;
  QPSK remains demod-limited.

## Testing strategy

New `tests/validation/test_ofdm.py`:

- **Round-trip, noiseless (anchor):** `modulate(payload, OFDM)` →
  `OFDMDemodulator` → **exact** payload bits recovered (0 BER), payload =
  12- or 24-byte multiple of 96 bits.
- **Power normalization:** OFDM output average power ≈ 1.0 (±0.05), dtype
  `complex64`.
- **BER vs SNR through `channel.py`:** BER at high SNR (≥ ~20 dB) ≈ 0; BER is
  monotone-ish decreasing across a small SNR sweep (assert high-SNR BER <
  low-SNR BER and high-SNR BER below a small threshold). Uses the seeded
  `repro.rng` for determinism.
- **CFO + timing robustness within bound:** apply a small `ChannelParams`
  CFO and `timing_offset` via `apply_channel`; assert recovery still low-BER
  at high SNR.
- **Frame-too-short / empty region:** `OFDMDemodulator` returns an invalid
  result (no crash); `ofdm_region_to_bytes` returns `b""`.
- **Pipeline routing:** an OFDM `LabeledCapture` (with
  `provenance["scheme"]="ofdm"`) routes through the OFDM demod and yields
  non-empty bytes; a non-OFDM capture still uses the FSK path (spy/monkeypatch).

Extend `tests/validation/test_modulators.py` with an inline independent OFDM
reference demod for the round-trip (consistent with the file's existing
"independent reference demod" convention), OR keep the OFDM round-trip in
`test_ofdm.py` — implementer's choice, but the independent-reference principle
holds (the round-trip anchor must not simply call the production receiver
against itself for the *modulator* correctness claim; a minimal inline
FFT+demap reference is acceptable).

## File structure

- **Create:** `core/ofdm.py`, `tests/validation/test_ofdm.py`
- **Modify:** `core/demodulation.py`, `validation/synth/modulators.py`,
  `validation/pipeline.py`, `cli.py`, `README.md`, `CLAUDE.md`

## Global constraints (carried from the project)

- IQ samples are `numpy.complex64` (not `complex128`) at all public
  boundaries; intermediate math may use `complex128` but outputs cast to
  `complex64`.
- All randomness via a caller-supplied/`repro.rng` `numpy.random.Generator`;
  never `np.random.*` free functions. (OFDM waveform synthesis is
  deterministic — no RNG at all.)
- `mypy validation` must stay strict-clean. `core/ofdm.py` typed cleanly with
  `numpy.typing` (avoid `Any`). Note `core.*` is `follow_imports=silent` for
  the gate, so core errors won't fail CI, but new code is written clean.
- Google-style docstrings; document the S&C algorithm with a reference
  (Schmidl & Cox, 1997).
- `black` / `isort` / `flake8` clean on touched files (line length 88).
- Backward compatibility: no existing test changes behavior; the pipeline's
  default demod path is byte-for-byte unchanged.

## Risks & mitigations

- **Modulator/demodulator drift** → single `core/ofdm.py` profile shared by
  both; round-trip test is the anchor.
- **S&C plateau ambiguity picking the wrong start** → use the metric peak
  with a plateau-averaged timing estimate; validate with the timing-offset
  test.
- **DemodConfig single-carrier assumptions leaking into OFDM** → OFDM demod
  never reads `bitrate_bps`; the two properties are special-cased.
- **Heavy `core` imports slowing the pipeline** → lazy import inside the OFDM
  branch only.
