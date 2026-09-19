# Single-Carrier CFO/Phase Hardening (PH) — Design Spec

**Design #:** 0003
**Phase:** PH

**Date:** 2026-09-18
**Status:** Accepted — implemented, shipped in v0.3.0
**Phase:** PH (hardening), inserted before P2 on `feature/ofdm-full-chain` (one final PR)

## Goal

Close the CFO-robustness gap between the single-carrier receivers (built in PE) and the OFDM chain: (1) widen the CFO acquisition range via a CFO-hypothesis search, and (2) fix the coherent-PSK mid-SNR phase non-monotonicity via decision-directed phase tracking, plus an optional pilot-aided tracking mode selectable by parameter.

## Motivation

PE's single-carrier receivers achieve BER=0 at high SNR and fail cleanly at low SNR, but fall short of OFDM's CFO robustness: the acquisition range is ~0.0029 cyc/sample (vs OFDM's ~0.0156) because timing uses a CFO-intolerant full-preamble matched filter, and coherent-PSK BER is non-monotonic at mid-SNR (5–15 dB) because a single preamble-point CFO estimate is applied as one global derotation with no continuous phase tracking. The user asked to reach OFDM parity.

## Decisions (user-approved)

- **CFO acquisition:** implement a **CFO-hypothesis search NOW** (no preamble change; PE's Barker preamble + most tests stand). The OFDM-style short-segment-repeated STF preamble redesign is noted as an optional future refinement, not built in this phase.
- **Phase tracking:** **decision-directed (default, pilotless) AND pilot-aided as an opt-in parameter.** Pilot-aided is off by default (`pilot_spacing = 0`); when enabled it inserts periodic known pilots. Both paths exist; pilots are parameterized so the default framing is unchanged.
- **Default modulation → GFSK.** GFSK is the most common controllable drone C2/telemetry modulation (RC links, SiK/MAVLink, BLE) and the real-world norm over raw FSK. Make GFSK the framework's default single-carrier scheme: the CLI `validate synth` fallback scheme becomes `GFSK` (was raw `FSK`), and GFSK is documented as the canonical single-carrier default in the profile/README/CLAUDE notes (docs land in the effort's final consolidation). OFDM remains first-class as the DJI/WiFi video-link scheme. This is a small, additive change folded into this phase (GFSK is non-coherent single-carrier: it gets the wider CFO acquisition; phase tracking is PSK-only, so GFSK is unaffected by it).

## Non-goals

- STF preamble redesign (future refinement; tracked, not built).
- FSK/GFSK phase tracking (non-coherent — not applicable; FSK still benefits from the wider CFO acquisition).
- Any change to the OFDM (P1) chain.
- Multiple profiles / coding / bit-loading (P2/P3/P4).

## Architecture

All new DSP lives in the shared `core/single_carrier.py` (single source of truth); the core demods and synth modulator thread the new parameters.

### 1. CFO-hypothesis-search acquisition (`core/single_carrier.py`)

- `sc_acquire(rx, ref_wave, sps, *, cfo_range, cfo_step) -> tuple[int, float, complex]` — grid-search CFO over `[-cfo_range, +cfo_range]` in `cfo_step`; for each hypothesis, derotate `rx` and run the existing normalized matched filter (`sc_frame_sync`) against `ref_wave`; return the `(start, cfo, peak)` maximizing `abs(peak)`. This yields coarse timing + coarse CFO + absolute phase together, with acquisition range = `cfo_range`.
  - `cfo_range` default ≈ 0.02 cyc/sample (> OFDM's 0.0156); `cfo_step` fine enough that the residual after coarse correction is within the two-halves fine estimator's range and the phase-tracking pull-in (e.g. `1/(4·len(ref_wave))`).
- The PSK/FSK receivers replace their current single matched-filter acquisition with `sc_acquire` → derotate by the coarse CFO → refine with the existing two-halves `sc_estimate_cfo_psk` (PSK) → proceed. Lock gate still on `abs(peak)`.
- Compute: `n_cfo × O(n·L)` matched-filter passes. Bounded and offline (T&E), acceptable; documented.

### 2. Decision-directed phase tracking (coherent PSK, default)

- After acquisition + absolute-phase alignment, run a first-order decision-directed phase loop across the payload symbols: for each symbol, derotate by the current tracked phase, make the hard decision, compute the phase error `angle(sym · conj(decided))`, and update the tracked phase by `α · error` (small loop gain `α`, tuned in tests). Corrects slow residual phase drift → fixes the mid-SNR non-monotonicity.
- Applies to **coherent BPSK/QPSK** only. FSK is non-coherent (skip). Differential is phase-immune (skip; it already works). Default when `pilot_spacing == 0`.

### 3. Pilot-aided tracking (optional, parameterized)

- New parameter `pilot_spacing: int = 0` (0 = off / pilotless). When `> 0`, the **modulator** inserts a known pilot symbol (fixed BPSK `+1`) after every `pilot_spacing` payload symbols; the **demodulator** (told the same `pilot_spacing`) estimates the residual phase at each pilot and linearly interpolates the correction across the intervening payload symbols, then strips the pilots and demaps the payload.
- Framing: `N` payload symbols → interleaved with `floor((N-1)/pilot_spacing)` pilots. The demod reconstructs payload symbol positions from `pilot_spacing`. Payload-length/round-trip math accounts for the interleave.
- Threaded end-to-end: `modulate(..., pilot_spacing=0)`, `DemodConfig.pilot_spacing`, `provenance["pilot_spacing"]`, `DatasetSpec.pilot_spacing` → `build_scenario` → `create_synth_dataset`, `single_carrier_region_to_bytes`, and a CLI `--pilot-spacing`. Default 0 everywhere → PE's behavior/framing unchanged.

## Testing strategy

- **Wide-CFO recovery:** PSK (coherent + differential) and FSK recover bits (BER≈0 at high SNR) at CFO magnitudes near ±0.015 cyc/sample that PE could not acquire — proving the search widened the range toward OFDM parity.
- **Mid-SNR monotonicity:** coherent-PSK BER-vs-SNR across 5/10/15/20/25 dB is now (approximately) monotonically decreasing (the PE non-monotonic wobble is gone) — with decision-directed tracking on, pilotless.
- **Pilot-aided mode:** `pilot_spacing > 0` round-trip recovers exact payload (pilots inserted + stripped correctly); characterize its residual-phase benefit vs pilotless at a chosen SNR. Confirm `pilot_spacing = 0` is byte-for-byte the PE pilotless path.
- **Zero regression:** PE's existing single-carrier round-trip/BER tests (pilotless default) and all OFDM tests stay green. `mypy validation` clean.

## Global constraints (carried from PE)

- `numpy.complex64` at public boundaries; `complex128` internal; deterministic synthesis (pilots/preamble are constants; no RNG).
- `mypy validation` strict-clean; `core/single_carrier.py` typed, no `Any`; Google docstrings; black/isort/flake8 (88).
- Zero regression to OFDM and to the PE pilotless single-carrier default.
- New parameters default to the PE behavior (`pilot_spacing=0`, tracking on for coherent PSK) so existing datasets/tests are unaffected except where they opt in.

## Risks

- **CFO-search compute** — bounded (`n_cfo` grid) and offline; documented. If it becomes a hotspot for large captures, vectorize (FFT-domain) later.
- **Pilot framing math** — the interleave changes payload-length bookkeeping; contained to the `pilot_spacing > 0` path and covered by round-trip tests.
- **DD loop-gain tuning** — an `α` too large is noisy, too small is sluggish; tuned and pinned by the mid-SNR monotonicity test.
- **Interaction with differential** — pilots/DD tracking apply to coherent PSK; differential stays on its phase-immune path (no DD, no pilots) to avoid double-correction.

## Resolved decisions (user-approved)

1. **DD tracking is always-on for coherent PSK** when `pilot_spacing == 0` (it strictly helps and needs no framing change); no separate flag.
2. **Pilot pattern:** fixed BPSK `+1` inserted every `pilot_spacing` payload symbols. Simple/fixed; revisit only if low-SNR pilot detection proves weak.
3. **CFO search defaults:** `cfo_range ≈ 0.02` cyc/sample, `cfo_step ≈ 1/(4·len(preamble))`, exposed as tunable module constants in `core/single_carrier.py`.
4. **Default modulation = GFSK** (see Decisions): CLI synth fallback `FSK → GFSK`; GFSK documented as the canonical single-carrier default.
