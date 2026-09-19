# P3c — Reed-Solomon (errors + erasures) over GF(2^8) (Design)

**Design #:** 0008
**Phase:** P3c

**Status:** Accepted — implemented, shipped in v0.3.0
**Date:** 2026-09-18
**Author:** rbenzing (with Claude)
**Program:** P3 (channel coding), sub-phase **P3c** — the first algebraic block code; introduces the reusable Galois-field algebra that P3d (BCH) builds on.
**Predecessors:** P1/PE/PH, P2-SC/P2-OFDM, **P3a** (coding framework: `Codec` interface, registry, CRC framing, interleaver, soft-LLR demod path, coded-BER/FER metric), **P3b** (convolutional + soft Viterbi; established the pipeline **soft coded-decode branch**).
**Scope note:** Security-research / educational software. Lawful research / authorized testing only.

---

## 1. Overview

### Goal

Implement standard **Reed-Solomon** over **GF(2^8)** with **errors-and-erasures** decoding (syndromes → Berlekamp-Massey → Chien search → Forney), plugged into the P3a `Codec` framework. Erasures are sourced by a **scale-invariant reliability rule** on P3b's soft-LLR path, so the errors-and-erasures capability is exercised end-to-end (not dead unit-tested code). Two recognizable industry/mil-spec codes are registered and used **shortened** for our small (~26-byte) frames. A new reusable `core/galois.py` (field + polynomial algebra + Berlekamp-Massey + Chien) is the foundation P3d (BCH) reuses directly.

### What P3a/P3b already provide (build on, don't rebuild)

- `core.coding`: `CodeFamily.REED_SOLOMON`; `CodingSpec(name, family, k, n, params)` with a `soft_input` property; the `Codec` protocol (`encode(Bits)->Bits`, `decode(SoftOrHard)->DecodeResult`); `make_codec(spec)` (currently raises `NotImplementedError` for `REED_SOLOMON`); `CODING_CATALOG["rs_255_223"]` already registered as a capability descriptor (`gf_m=8, t=16, symbol_bits=8`); `frame_with_crc`/`check_and_strip_crc`; `interleave`/`deinterleave` (value-type-agnostic, LLR-safe); `CODING_INTERLEAVE_DEPTH`.
- `core.single_carrier.sc_soft_bits(rx, profile, *, bits_per_symbol, noise_var=None) -> LLRs`.
- `core.profiles`: `SCProfileSpec(..., coding=None)`, `coding_of`, `SC_CATALOG`.
- `validation.pipeline.single_carrier_region_to_bytes`: the **soft coded-decode branch** built in P3b — for a `soft_input` codec it runs `sc_soft_bits → deinterleave(LLRs) → codec.decode → check_and_strip_crc`. **RS reuses this branch unchanged.**
- The `coded_link` (coded-BER/FER) metric.

### Non-goals

- No interop with real CCSDS/DVB frames (we use the **conventional Berlekamp basis**, not the CCSDS dual-basis representation). Self-consistent encode/decode is what we test.
- No explicit erasure channel plumbed through the `Codec.decode` signature (rejected Approach C); erasures are derived inside the RS decoder from the LLRs it already receives.
- No OFDM RS profile (SC only, as with P3b).
- No other codec families (P3d BCH reuses the `galois` algebra; P3e+ later).

---

## 2. `core/galois.py` — reusable Galois-field algebra (new file)

A standalone, **field-size-parametric** module so P3d (BCH over GF(2^6)) reuses it without change.

- `class GF2m(m: int, prim_poly: int)`:
  - Builds `exp`/`log` tables of size `2**m` using generator **α = 2** and the reduction polynomial `prim_poly`.
  - `add(a, b) -> int` = `a ^ b` (subtraction identical).
  - `mul(a, b) -> int`, `div(a, b) -> int`, `inv(a) -> int` via log/antilog tables (0 handled explicitly).
  - GF-polynomial helpers (coefficient lists, highest-degree-last or -first fixed and documented): `poly_add`, `poly_scale`, `poly_mul`, `poly_eval` (Horner), `poly_div` (returns quotient, remainder).
  - `n = 2**m - 1` (field's natural codeword length; RS uses this, shortened downward).
- Module constant `GF256 = GF2m(8, 0x11D)` (primitive polynomial x⁸+x⁴+x³+x²+1).
- `berlekamp_massey(field: GF2m, syndromes: list[int], erasure_locator: list[int] | None = None) -> list[int]`: returns the (errata) locator polynomial Λ(x). When `erasure_locator` is given, BM is initialized/seeded so the returned locator is the combined errata locator. Field-parametric; shared by RS and BCH.
- `chien_search(field: GF2m, locator: list[int], n: int) -> list[int]`: returns the list of positions `i ∈ [0, n)` where `locator(α^{-i}) == 0` (i.e. error/erasure positions). Field-parametric; shared.

Forney error-magnitude evaluation stays RS-specific (§4) — binary BCH needs only positions.

All public functions are pure and deterministic. Type-clean under `mypy validation` (no `Any`; integer lists, `numpy.typing` only where arrays cross the boundary).

---

## 3. RS encoder (`_ReedSolomon.encode`)

Systematic Reed-Solomon.

- **Parameters** (from `spec.params`): `gf_m` (8), `t` (error-correction power), `symbol_bits` (8), `prim_poly` (0x11D), `fcr` (first consecutive root = **1**). Field `n_full = 2**gf_m - 1 = 255`.
- **Generator polynomial:** `g(x) = ∏_{i=fcr}^{fcr+2t-1} (x − α^i)` over GF(2^8) — degree `2t`.
- **Systematic parity:** for message polynomial `m(x)` (message symbols as coefficients), parity `p(x) = (m(x) · x^{2t}) mod g(x)`; codeword symbols = `message ‖ parity` (message symbols first, `2t` parity symbols last).
- **Shortening:** parity is always `2t` symbols regardless of message length. For an `s`-symbol payload the shortened codeword is `s + 2t` symbols (the `k_full − s` leading zero symbols are implicit, never transmitted). No separate config — `s` is derived from the input length.
- **`encode(Bits)`:** pack info bits → symbols MSB-first, 8 bits/symbol (frames are byte-aligned: payload bytes + 2-byte CRC ⇒ bit length is a multiple of 8; assert this). RS-encode the `s` symbols → `s + 2t` symbols → unpack MSB-first to a `uint8` bit array. Deterministic, no RNG.

---

## 4. RS decoder — errors + erasures (`_ReedSolomon.decode`)

Input is the (de-interleaved) received stream — **LLRs** for the soft-input RS profile, or hard bits (total-function fallback).

1. **Bits & erasure flags.**
   - Map bits by LLR sign (`L < 0 ⇒ bit 1`, matching P3a convention), pack MSB-first into `s + 2t` GF(2^8) symbols.
   - Per symbol, **reliability** = min over its 8 bits of `|LLR|`. Flag symbol as an **erasure** when its reliability `< factor · median(|LLR|)` over the block (default `factor` a small constant, e.g. 0.5 — recorded in params). This rule is **scale-invariant**: multiplying every LLR by a positive constant leaves the flags unchanged (dodges P3a's ~2× calibration dependency). **Cap** the flagged count at `2t` (keep the least-reliable ones) so erasures alone can never exceed the correction budget.
   - Hard-bit input (`dtype.kind != 'f'`): erasure set empty (errors-only degenerate case).
2. **Syndromes.** `S_j = R(α^{fcr+j})` for `j = 0 … 2t−1`, where the received symbol at index `i` has locator `α^i` (this convention holds directly for shortened `N = s + 2t < 255`). All-zero syndromes ⇒ clean word ⇒ emit the `s` message symbols.
3. **Errata decode.** Build the erasure locator from flagged positions; compute Forney-modified syndromes; run `berlekamp_massey` (seeded with the erasure locator) → combined **errata** locator Λ(x); `chien_search` → error/erasure positions; **Forney** algorithm (error evaluator Ω(x) = S(x)Λ(x) mod x^{2t}; magnitude at position `i` = `Ω(X⁻¹)/Λ'(X⁻¹)` with X = α^i) → error magnitudes; XOR-correct the received symbols.
4. **Loud on failure.** Declare uncorrectable when: `2·(#errors) + (#erasures) > 2t`, or the Chien root count disagrees with the locator degree, or a magnitude computation is singular. On failure, emit the **received message symbols unchanged** (no correction) so `check_and_strip_crc` rejects the frame — never a silent wrong "success."
5. **Output.** Unpack the corrected `s` message symbols → `uint8` bits. `DecodeResult(bits, meta={"n_erasures": …, "n_errors": …, "decode_ok": …})`. A too-short input (`< 2t` symbols) yields an empty `bits` array.

**Scale invariance** (headline property to test): the erasure rule is median-relative and the corrected symbols depend only on hard values + flags, so a global LLR scale factor does not change the decode.

---

## 5. Framework integration

### 5a. `core/coding.py`
- `make_codec` gains a `CodeFamily.REED_SOLOMON` branch → `_ReedSolomon(spec)`.
- `CODING_CATALOG`:
  - **Update** `rs_255_223` params to `{gf_m: 8, t: 16, symbol_bits: 8, prim_poly: 0x11D, fcr: 1, soft_input: True}` (CCSDS/deep-space RS(255,223)). No working codec existed for the old descriptor ⇒ no behavior change; additive.
  - **Add** `rs_255_239` = same params with `t: 8` (DVB-S / IEEE 802.16 RS(255,239)) — the lighter code used as the blind PHY profile (faster tests, still a recognized standard).
- Imports the `galois` algebra; Forney lives here (RS-specific).

### 5b. `core/profiles.py`
- One new PHY-distinct coded profile: `rs_bpsk = SCProfileSpec("rs_bpsk", SCMod.BPSK, SCProfile(sps=64), coding="rs_255_239")`. **sps=64 is unique** in `SC_CATALOG` (existing: 4/8/16/32), so blind resolution separates it cleanly. Existing profiles unchanged.

### 5c. `validation/pipeline.py` — **no change**
RS is `soft_input=True`, so `single_carrier_region_to_bytes` already routes it through P3b's soft branch: `sc_soft_bits(iq_c128, spec.profile, bits_per_symbol=…) → deinterleave(LLRs, CODING_INTERLEAVE_DEPTH) → make_codec(spec).decode(deint).bits → check_and_strip_crc`. Erasure-flagging happens **inside** `_ReedSolomon.decode` from the LLRs it receives; `deinterleave` is an exact inverse, so RS repacks symbol-aligned bits correctly. Loud-on-failure unchanged (empty LLRs / no lock / CRC fail → `(b"", None)`).

### 5d. synth / metric — **no change**
`modulate(coding=…)` already calls `codec.encode` generically (`_ReedSolomon.encode` slots in). The `coded_link` metric is unchanged.

---

## 6. Testing strategy

TDD; executed subagent-driven with per-task review. Buckets:

1. **GF(2^8) field laws:** `mul`/`inv`/`div` correctness (`a·a⁻¹ = 1` for all `a≠0`), distributivity on samples, `α^255 = 1`, log/exp round-trip. Plus a **GF(2^6) smoke check** (constructs `GF2m(6, 0x43)`; verifies field laws) to prove the parametric field is P3d-ready.
2. **Berlekamp-Massey + Chien:** a known syndrome set → the known locator polynomial; Chien returns the known root positions.
3. **Encode systematic:** parity length is `2t`; a clean codeword's syndromes are all zero (`c(α^{fcr+j}) = 0`).
4. **Noiseless round-trip, both codes (shortened):** `encode → decode` recovers the payload with `crc_ok` for `rs_255_223` (t=16) and `rs_255_239` (t=8) via `make_codec(CODING_CATALOG[name])`.
5. **Errors-only correction:** inject up to `t` symbol errors → exact recovery; `t+1` symbol errors → uncorrectable, CRC fails loudly.
6. **Errors + erasures (explicit mask):** with `2e + f ≤ 2t` (e errors, f erasures) → exact recovery; `2e + f > 2t` → failure. Unit test passes erasure positions directly.
7. **Headline — reliability-flagged erasures (soft):** an error burst beyond the errors-only power but within `2e + f ≤ 2t` once the burst symbols are flagged as erasures → recovered through the soft path where errors-only would fail; **scale-invariance** — multiplying all LLRs by 10 yields an identical decode.
8. **Coding gain:** on the SNR sweep at matched PHY, RS BER < uncoded BER at low SNR, measured through the full synth → `sc_soft_bits` → RS → CRC chain.
9. **Blind end-to-end:** an `rs_bpsk` dataset routes through the pipeline's **soft** coded-decode path, blind-resolves `rs_bpsk`, decodes correctly, and populates `coded_link` (low coded BER at normal SNR).
10. **Loud-failure + regression:** CRC loud-failure at very low SNR; full existing suite (P1/PE/PH/P2/P3a/P3b) green; `mypy validation` clean.

---

## 7. Global constraints (bind every task)

- `mypy validation` is the CI gate — clean (no `Any`; `numpy.typing`). New/modified `core/*` files `mypy`-clean, minimal diffs, no whole-file `black` reformat, no new legacy findings. Lint every touched test file (CI gates `tests/validation`).
- IQ boundary `complex64` / internal `complex128`; LLRs `float64`; bits `uint8`; GF symbols plain Python `int` inside `galois`.
- Deterministic encode (no RNG).
- **Loud on failure:** CRC failure, no lock, or uncorrectable RS ⇒ `b""` / unchanged received bits that CRC rejects — no silent wrong payload at normal SNR. Existing hard/uncoded/conv paths byte-for-byte unchanged.
- Additive: `rs_255_223` gains a working codec + params but no existing test changes behavior; existing profiles/tests unchanged; the pipeline soft branch is reused, not modified.
- Executed subagent-driven with TDD and per-task review; on `feature/ofdm-full-chain`; **no PR/tag** this sub-phase.
- **Subagent-stall guard:** the full validation suite (~6 min) and the blind-resolve set (~13 min) have repeatedly stalled implementer subagents on backgrounded runs. Task briefs instruct running only targeted test files; the full-suite closing gate runs in the foreground (or by the controller).

---

## 8. Out of scope / future

- BCH (P3d — reuses `galois` GF + BM + Chien; adds binary syndrome/locator specifics).
- LDPC (P3e), turbo (P3f), polar (P3g), fountain (P3h).
- Real CCSDS/DVB dual-basis interop; erasure-channel plumbing through the `Codec.decode` signature (Approach C).
- An OFDM soft-coded RS profile + OFDM soft-decode branch.
- LLR recalibration (P3e/turbo, where absolute scale matters — RS here is scale-invariant by construction).
- Docs consolidation + P2/P3a/P3b parked cleanups (final consolidation).
