# P3b — Convolutional Code + Soft-Decision Viterbi (Design)

**Design #:** 0007
**Phase:** P3b

**Status:** Accepted — implemented, shipped in v0.3.0
**Date:** 2026-09-18
**Author:** rbenzing (with Claude)
**Program:** P3 (channel coding), sub-phase **P3b** — the first real codec, on the P3a framework.
**Predecessors:** P1/PE/PH, P2-SC/P2-OFDM, **P3a** (coding framework: `Codec` interface, registry, CRC framing, interleaver, soft-LLR demod path, coded-BER/FER metric).
**Scope note:** Security-research / educational software. Lawful research / authorized testing only.

---

## 1. Overview

### Goal

Implement the classical rate-1/2 K=7 convolutional code with **soft-decision Viterbi** decoding and **puncturing** to the IEEE 802.11a/g rates 2/3 and 3/4, plugged into the P3a `Codec` framework. This is the first codec that **consumes** P3a's soft-LLR demod path, and it demonstrates coding gain well beyond P3a's repetition code.

### What P3a already provides (build on, don't rebuild)

- `core.coding`: `CodeFamily.CONVOLUTIONAL`; `CodingSpec(name, family, k, n, params)` with a `soft_input` property; the `Codec` protocol (`encode(Bits)->Bits`, `decode(SoftOrHard)->DecodeResult`); `make_codec(spec)` (currently raises `NotImplementedError` for CONVOLUTIONAL); `CODING_CATALOG["conv_k7_r12"]` already registered (`constraint_length=7`, `generators_octal=(0o133, 0o171)`, `soft_input=True`); `frame_with_crc`/`check_and_strip_crc`; `interleave`/`deinterleave` (value-type-agnostic, LLR-safe); `CODING_INTERLEAVE_DEPTH`.
- `core.single_carrier.sc_soft_bits(rx, profile, *, bits_per_symbol, noise_var=None) -> LLRs`.
- `core.profiles`: `SCProfileSpec(..., coding=None)`, `coding_of`, `SC_CATALOG`.
- `validation.pipeline.single_carrier_region_to_bytes`: its coded branch currently does **hard** demod → `deinterleave` → `codec.decode` → `check_and_strip_crc` (built for the hard-input `rep3`).
- The `coded_link` (coded-BER/FER) metric.

### Non-goals

- No tail-biting (zero-tail termination only).
- No sliding-window/streaming Viterbi (full-block traceback; our frames are small).
- No new modulation, no LLR recalibration (Viterbi is scale-invariant — see §3).
- No other codec families (P3c+).

---

## 2. The encoder + puncturing (`ConvCodec.encode`)

**Mother code:** rate-1/2, constraint length K=7 (6-stage shift register, 64 states), generator polynomials **133/171 octal** (the standard NASA/CCSDS/802.11 code). For each input bit, two output bits are produced from the two generator taps XORed over the register.

**Zero-tail termination:** append K−1 = 6 zero bits to the info stream before encoding, so the encoder starts and ends in state 0. The decoder exploits the known terminal state; the 6 tail bits are dropped after decode.

**Puncturing** (applied to the rate-1/2 coded stream, standard IEEE 802.11a patterns; `1`=keep, `0`=puncture, read column-major over the two output rails):
- rate **1/2**: no puncturing.
- rate **2/3**: pattern `[1, 1, 1, 0]` — keep 3 of every 4 coded bits.
- rate **3/4**: pattern `[1, 1, 1, 0, 0, 1]` — keep 4 of every 6 coded bits.

`ConvCodec.encode(info_bits)` = conv-encode(info ‖ 6 zero bits) → puncture(pattern) → coded bits. The puncture pattern is read from `spec.params["puncture"]` (absent/`None` ⇒ rate 1/2).

---

## 3. Soft-decision Viterbi decoder (`ConvCodec.decode`)

Input is the (de-interleaved) received stream — **LLRs** for a soft-input profile, or hard bits (mapped to ±1 LLRs) as a total-function fallback. Steps:

1. **De-puncture:** reinsert the punctured positions as **erasures** (LLR = 0.0, contributing nothing to any branch metric), restoring the full rate-1/2 LLR stream of length `2 * n_stages`, where `n_stages = info_len + 6`.
2. **Trellis:** 64 states. Precompute, per state and input bit, the next state and the 2-bit expected output. **Soft branch metric** for an edge with expected output `(c0, c1)` and received LLRs `(L0, L1)`: `bm = (1 - 2*c0)*L0 + (1 - 2*c1)*L1` (correlation; larger = better match). Erasure positions (`L=0`) contribute 0, the correct handling of punctured bits.
3. **Forward pass (add-compare-select):** accumulate path metrics maximizing total correlation; store the surviving predecessor per state per stage.
4. **Traceback:** start from the known terminal **state 0** (zero-tail); trace survivors back to stage 0; emit the input-bit sequence; **drop the last 6 tail bits** → `info_len` decoded bits.
5. Return `DecodeResult(bits, meta={"n_stages": ...})`. `info_len` is derived from the received length (`n_stages = de-punctured_len // 2`, `info_len = n_stages - 6`) — the codec is self-describing; no external length needed. A too-short input yields an empty `bits` array.

**Scale invariance:** the ACS argmax depends only on relative branch-metric ordering, so a global LLR scale factor (P3a's ~2× conservative calibration) does not change the decode. P3b therefore uses P3a's `sc_soft_bits` **as-is**; LLR recalibration stays a P3e/turbo concern.

---

## 4. Framework integration

### 4a. `core/coding.py`
- `make_codec` gains a `CodeFamily.CONVOLUTIONAL` branch → `ConvCodec(spec)` (reads `constraint_length`, `generators_octal`, and optional `puncture` from `spec.params`).
- `CODING_CATALOG` gains `conv_k7_r23` (k=2, n=3, `puncture=(1,1,1,0)`) and `conv_k7_r34` (k=3, n=4, `puncture=(1,1,1,0,0,1)`); both share `constraint_length=7`, `generators_octal=(0o133,0o171)`, `soft_input=True`. `conv_k7_r12` is unchanged.

### 4b. `core/profiles.py`
- One new PHY-distinct coded profile: `conv_bpsk = SCProfileSpec("conv_bpsk", SCMod.BPSK, SCProfile(sps=32), coding="conv_k7_r12")`. **sps=32 is unique** in `SC_CATALOG` (existing: 4/8/16), so blind resolution separates it cleanly. Existing profiles unchanged.

### 4c. `validation/pipeline.py` — consume the soft-LLR path
Extend `single_carrier_region_to_bytes`'s coded branch to pick the demod by the codec's decision type:
- Resolve the profile; look up `CODING_CATALOG[spec.coding]`.
- If that spec's **`soft_input`** is true: `llrs = sc_soft_bits(iq_c128, spec.profile, bits_per_symbol=spec.bits_per_symbol)`; `deint = deinterleave(llrs, CODING_INTERLEAVE_DEPTH)` (LLR-safe); `frame = make_codec(spec).decode(deint).bits`; then `check_and_strip_crc`.
- Else (hard-input, e.g. `rep3`): the existing hard path, unchanged.
- Loud-on-failure unchanged: empty LLRs / no lock / CRC fail → `(b"", None)`.

This is where P3a's `sc_soft_bits` is first consumed. (The OFDM soft-decode path via `ofdm_soft_bits` is analogous and deferred until an OFDM soft-coded profile exists.)

No change to the synth encode chain (P3a's `modulate(coding=...)` already calls `codec.encode` generically — `ConvCodec.encode` slots in) or to the `coded_link` metric.

---

## 5. Testing strategy

TDD; executed subagent-driven with per-task review. Buckets:

1. **Encoder correctness:** rate-1/2 K=7 with generators 133/171 produces the known output for a reference input (e.g. an impulse → the generator tap sequences); zero-tail appends 6 zeros and returns state 0.
2. **Puncturing:** puncture then de-puncture (erasure-insert) round-trips the rate-1/2 stream; punctured lengths match the rate (2/3, 3/4).
3. **Noiseless round-trip, all rates:** `encode → decode` recovers the payload with `crc_ok` for rates 1/2, 2/3, 3/4 (via `make_codec(CODING_CATALOG[name])`).
4. **Soft Viterbi error correction:** with several bit-flips / low-SNR LLRs within the code's correcting power, `decode` recovers the exact payload; beyond it, CRC fails loudly.
5. **Coding gain (headline):** on the SNR sweep at matched PHY, soft-Viterbi rate-1/2 `conv_k7_r12` BER < `rep3` BER < uncoded BER at low SNR (convolutional strongly beats rate-1/3 repetition), measured through the full synth → `sc_soft_bits` → Viterbi → CRC chain; rates 2/3, 3/4 still beat uncoded (weaker than 1/2).
6. **Blind end-to-end:** a `conv_bpsk` dataset routes through the pipeline's **soft** coded-decode path, blind-resolves `conv_bpsk`, decodes correctly, and populates `coded_link` (low coded BER at normal SNR).
7. **Loud-failure + regression:** CRC loud-failure at very low SNR; full existing suite (P1/PE/PH/P2/P3a, 190 tests) green; `mypy validation` clean.

---

## 6. Global constraints (bind every task)

- `mypy validation` is the CI gate — clean (no `Any`; `numpy.typing`). New/modified `core/*` files `mypy`-clean, minimal diffs, no whole-file `black` reformat, no new legacy findings. Lint every touched test file (CI gates `tests/validation`).
- IQ boundary `complex64` / internal `complex128`; LLRs `float64`; bits `uint8`.
- Deterministic encode (no RNG).
- **Loud on failure:** CRC failure or no lock ⇒ `b""` — no silent wrong payload at normal SNR. The new soft-decode branch preserves the hard-path guarantees; existing hard/uncoded paths byte-for-byte unchanged.
- Additive: `conv_k7_r12` and existing profiles/tests unchanged; the pipeline soft branch is gated on `soft_input`.
- Executed subagent-driven with TDD and per-task review; on `feature/ofdm-full-chain`; no PR/tag this sub-phase.
- **Note on subagent stalls:** the full validation suite (~5 min) has twice stalled implementer subagents waiting on a backgrounded run; task briefs should instruct running only targeted test files, with the full-suite closing gate run in the foreground (or by the controller).

---

## 7. Out of scope / future

- Tail-biting convolutional codes; sliding-window Viterbi.
- The remaining families: RS (P3c), BCH (P3d), LDPC (P3e), turbo (P3f), polar (P3g), fountain (P3h).
- LLR recalibration (P3e/turbo, where absolute scale matters).
- An OFDM soft-coded profile + the OFDM soft-decode pipeline branch (analogous; when needed).
- Docs consolidation + P2/P3a parked cleanups (final consolidation).
