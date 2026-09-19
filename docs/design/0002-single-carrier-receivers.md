# Single-Carrier Full Receivers (PE) — Design Spec

**Design #:** 0002
**Phase:** PE

**Date:** 2026-09-17
**Status:** Accepted — implemented, shipped in v0.3.0
**Phase:** PE of the phased OFDM/demod effort (built on `feature/ofdm-full-chain`, one final PR)

## Goal

Route every single-carrier scheme (FSK, GFSK, BPSK, QPSK) through the core
`DemodulationEngine` with proper **preamble-driven receivers**, retiring the
validation pipeline's inline FSK bit-slicer (`region_to_bytes`). Single-carrier
bursts must recover bits through channel CFO / timing offset / AWGN (and mild
multipath) — the same rigor the OFDM chain got in P1 — with a coherent
(preamble-aided, absolute) PSK path as default **and** a selectable
differential (DPSK/DQPSK) path.

## Motivation

Today the validation pipeline demodulates single-carrier regions with a trivial
inline FSK slicer (`region_to_bytes`) that only coincidentally matches the synth
FSK modulator; QPSK does not round-trip through it at all (P1 documented it as
"demod-limited"). The core `FSKDemodulator`/`PSKDemodulator` exist but (a) have
no test coverage, (b) demap QPSK with inverted polarity vs. the synth Gray map,
(c) estimate FSK tones by fragile FFT peak-finding, and (d) have only a
simplified carrier recovery with no phase-ambiguity resolution — so they do not
round-trip the synth waveforms. PE makes the single-carrier chain real and
consistent, so the T&E spine measures genuine detect→demod→classify performance
for single-carrier schemes, not a slicer artifact.

## Chosen approach (user decisions)

- **Full single-carrier receivers** (not an adapter): build proper
  preamble-driven sync in the core demods, mirroring `core/ofdm.py`.
- **Phase resolution: BOTH** — preamble-aided coherent (absolute) PSK/QPSK as
  the default, plus a selectable differential (DBPSK/DQPSK) mode.

## Non-goals

- Retuning real-capture ingestion (scheme-agnostic already).
- OOK/ASK/MSK/AFSK receivers (out of scope; only FSK, GFSK, BPSK, QPSK).
- Turning the legacy demods into production SDR receivers — the target is
  robust recovery of the *synthetic* validation waveforms through the channel
  model within documented bounds (CFO within the preamble acquisition range,
  timing offset within the search window, mild multipath), mirroring OFDM's
  SP-level scope.
- Coding/interleaving/bit-loading (those are P3/P4).

## Architecture

Mirror the OFDM pattern: a shared PHY module is the single source of truth for
TX and RX so they cannot drift; the core demodulators become thin adapters over
it; the pipeline routes through the core engine.

### `core/single_carrier.py` (new) — shared single-carrier PHY

Single source of truth for the preamble contract, TX builders, and RX sync.

- `@dataclass(frozen=True) class SCProfile` — `sps: int` (samples/symbol, default
  8), `mod_index: float` (FSK deviation), `bt: float` (GFSK), and the fixed
  preamble definition. `DEFAULT_SC_PROFILE` uses `sps=8`, matching the synth
  modulators.
- **Preamble:** a fixed, known sequence of `N_PRE` BPSK training symbols
  (`±1`) chosen for good autocorrelation and a repeated structure. Concretely:
  two identical halves of a length-`L` sequence (e.g. a length-13 Barker code,
  so `N_PRE = 26`), enabling (a) matched-filter **timing** from the correlation
  peak, (b) **fractional CFO** from the phase progression between the two
  identical halves (a symbol-domain Schmidl & Cox estimate), and (c) **absolute
  carrier phase** from the correlation-peak phase (resolving the PSK ambiguity).
  The preamble bits are constants in this module (deterministic; no RNG).
- **TX builders** (deterministic, unit-average-power `complex64` after the
  synth normalization tail):
  - `sc_preamble_symbols(profile) -> NDArray[complex128]` — the known BPSK
    training symbols.
  - The preamble is modulated with the *payload scheme* so it sees the same
    channel: for FSK/GFSK the preamble bits are (G)FSK-modulated; for PSK the
    known BPSK symbols are pulse-shaped like the payload. The synth modulators
    (below) prepend this.
- **RX sync primitives** (pure functions, fully typed):
  - `sc_frame_sync(rx, ref_preamble_wave, sps) -> (start_idx, peak_complex)` —
    normalized matched-filter cross-correlation of `rx` against the known
    preamble waveform; returns the payload-start sample and the complex peak
    (its magnitude = lock confidence, its phase = absolute-phase estimate),
    searched within a bounded window (mirroring OFDM's bounded search).
  - `sc_estimate_cfo(rx_preamble, profile, scheme) -> float` — normalized CFO:
    for PSK, the phase slope across preamble symbols (or the two-halves phase
    difference); for FSK/GFSK, the mean FM-discriminator output over the
    (balanced) preamble.
  - `sc_lock_confidence(rx, ...) -> float` — normalized correlation peak in
    `[0,1]`; used to gate loud sync-failure (mirrors OFDM's confidence gate).

### Receivers (in `core/single_carrier.py`, called by the core demods)

- `sc_demodulate_fsk(rx, profile, *, gfsk: bool) -> NDArray[uint8]`
  1. Frame sync (matched filter on the FSK-modulated preamble) → payload start,
     lock confidence.
  2. CFO estimate from the preamble discriminator bias → derotate / subtract.
  3. Per-symbol non-coherent decision: FM discriminator (instantaneous
     frequency) integrated over the inner half of each symbol → sign → bit.
     (A hardened version of P1's `region_to_bytes` logic, now preamble-aligned
     and CFO-corrected.)
- `sc_demodulate_psk(rx, profile, *, bits_per_symbol: int, differential: bool)
  -> NDArray[uint8]`
  1. Frame sync (matched filter on the BPSK preamble) → payload start, complex
     peak (absolute phase), lock confidence.
  2. CFO estimate + correction (phase slope across preamble).
  3. **Coherent (default):** derotate by the preamble peak phase so the known
     preamble aligns to `+1`, resolving the 90°/180° ambiguity; then symbol
     timing (sample at centers) and QPSK/BPSK demap using the **synth Gray
     convention** (bit 0 → +, bit 1 → −, per rail; `i=bits[0::2]`,
     `q=bits[1::2]`) — the correct, non-inverted mapping.
  4. **Differential (`differential=True`):** decode from phase *differences*
     between consecutive symbols (DBPSK/DQPSK), which is immune to absolute
     phase, so step 3's derotation is not required for correctness (the
     preamble is still used for timing/CFO). Payload is differentially
     *encoded* by the modulator in this mode.

### Synth modulators (`validation/synth/modulators.py`)

- Prepend the shared `core.single_carrier` preamble (scheme-modulated) to the
  FSK/GFSK/QPSK/BPSK payload. `modulate()` gains a `differential: bool = False`
  keyword; when set (PSK family), the payload symbols are differentially
  encoded. BPSK support is added (currently only FSK/GFSK/QPSK exist).
- The unit-average-power normalization tail is unchanged.
- `ModScheme` gains `BPSK` (and the differential variants are represented by the
  `differential` flag carried in capture provenance, not new enum members — see
  Open Questions).

### Core demodulators (`core/demodulation.py`)

- `FSKDemodulator.demodulate` and `PSKDemodulator.demodulate` delegate their
  front-end to `core.single_carrier` (replacing the FFT-peak tone estimation and
  the simplified carrier recovery). They keep the `DemodConfig` /
  `DemodulationResult` interface, store **unpacked** bits in `result.bits`, gate
  on `sc_lock_confidence` for loud sync-failure (mirroring OFDM), and read
  `config.samples_per_symbol` (= `sample_rate/symbol_rate`) and a new
  `config.differential` flag.
- `DemodConfig` gains `differential: bool = False`. `bits_per_symbol` /
  timing come from the existing scheme metadata.

### Pipeline (`validation/pipeline.py`)

- **Retire `region_to_bytes`** (the inline FSK slicer). Add a scheme-aware
  single-carrier demod that runs the capture's known scheme through the core
  `DemodulationEngine`, threading a correct `DemodConfig` (scheme + differential
  from `provenance`, `sample_rate` from the capture, `bitrate = sample_rate/sps`
  → `samples_per_symbol = sps`). OFDM keeps its P1 `ofdm_region_to_bytes` path;
  detection stays scheme-aware (OFDM detector for OFDM; the existing amplitude
  detector for single-carrier, which is constant-modulus and detects fine).
- `use_truth_bytes` bypass unchanged.

## Testing strategy

- **Round-trip anchors (noiseless, 0 BER)** for FSK, GFSK, BPSK, QPSK, and
  differential DQPSK/DBPSK — through the core `single_carrier` receivers.
- **BER-vs-SNR** for each scheme through `channel.py` AWGN (BER decreasing;
  high-SNR BER ≈ 0).
- **CFO + timing-offset robustness** within the documented preamble-acquisition
  range for each scheme (coherent PSK must recover absolute bits; differential
  must be phase-immune).
- **Lock gate:** unsynced / pure-noise region → `is_valid=False` (loud), like
  OFDM.
- **Pipeline round-trip** per scheme end-to-end (detect → core demod → bytes),
  exact payload recovery.
- **Regression:** OFDM path unchanged; update P1's single-carrier
  `test_modulators.py` round-trips (waveforms now carry a preamble — the inline
  reference demods must skip it). Full suite green; `mypy validation` clean.

## Global constraints (carried)

- `numpy.complex64` at public boundaries; `complex128` internal DSP.
- Deterministic waveform synthesis (no RNG; preamble/training are constants).
- `mypy validation` strict-clean; `core/single_carrier.py` typed with
  `numpy.typing`, no `Any`. `core/*` is `follow_imports=silent` for the gate but
  written clean.
- Google docstrings; cite references for sync algorithms.
- black/isort/flake8 clean (line 88).
- Zero regression to the OFDM (P1) path.

## Risks

- **Retiring `region_to_bytes` changes the currently-passing FSK pipeline
  path.** Mitigation: the new receiver must pass the same (updated) round-trip +
  BER tests; the change is gated by the task/whole-branch reviews.
- **Legacy core demods are being reworked in place.** They have no existing
  tests, so PE adds their first real coverage; risk is contained to the
  single-carrier path.
- **CFO acquisition range** is bounded by the preamble length/structure —
  documented, and synthetic scenarios stay within it (as OFDM does).
- **Differential mode doubles the PSK demap surface.** Mitigation: differential
  is a thin add-on over the same sync front-end (only the demap differs).

## Resolved decisions (user-approved)

1. **Differential representation:** carried as a `differential: bool` field on
   `DemodConfig` + a `provenance["differential"]` flag on captures (and a
   `differential` keyword on synth `modulate()`). No `DBPSK/DQPSK` enum churn;
   works uniformly for BPSK and QPSK.
2. **Preamble sequence:** length-13 Barker code repeated ×2 (26 BPSK symbols) —
   strong autocorrelation for timing, two identical halves for the CFO estimate.
   Revisit only if acquisition proves weak in testing.
3. **BPSK scope:** included — add `ModScheme.BPSK` to the validation layer (it
   falls out of the PSK receiver almost for free and is needed for a sensible
   DBPSK).
