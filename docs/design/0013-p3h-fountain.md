# P3h — Raptor-style fountain code (LT + precode) over an erasure channel (Design)

**Design #:** 0013
**Phase:** P3h

**Date:** 2026-09-20
**Status:** Accepted — implemented (unreleased).
**Author:** rbenzing (with Claude)
**Program:** P3 (channel coding), sub-phase **P3h** — the final codec family: a
Raptor-style fountain (rateless erasure) code. Completes the seven-family
capability sheet (uncoded, repetition, convolutional, RS, BCH, LDPC, turbo,
polar, fountain).
**Predecessors:** P3a (framework), P3b–P3g (all prior codecs). ADRs 0005, 0006,
0007, 0008, 0013, 0014, 0015, 0016.
**Scope note:** Security-research / educational software. Lawful research /
authorized testing only.

---

## 1. Overview

### Goal

Implement a **Raptor-style fountain code** — an outer **systematic sparse
precode** plus an inner **LT (Luby Transform) code** with a **Robust Soliton**
degree distribution — decoded over an **erasure channel** by **GF(2) Gaussian
elimination**, on the P3a `Codec` framework. Before this sub-phase, the
framework carried only a placeholder `fountain_lt` descriptor (`k=0, n=0`
rateless; `{kind:"lt", c:0.03, delta:0.5}`) and `CodeFamily.FOUNTAIN`, with
`make_codec` raising for `FOUNTAIN`. As implemented (see §4), `fountain_lt`
is replaced by the `fountain_r05`/`fountain_r10`/`fountain_r15` overhead
profiles and `make_codec` builds `FOUNTAIN` like every other family.

### The conceptual shift from every prior codec

Fountain codes are **erasure** codes, not error-correctors. Their value is
recovering the K source symbols once *enough* encoded symbols arrive **intact**.
Three design commitments make this fit DroneCMD's bit-error AWGN pipeline:

1. **Per-symbol CRC erasure detection.** Our channel produces bit errors, not
   clean packet drops. Each *encoded* symbol carries a small CRC; the receiver
   marks a symbol **erased** if its CRC fails. This turns the bit-error channel
   into a symbol-erasure channel, the model fountain codes are built for. Makes
   fountain a **hard-input** codec (`soft_input=False`), riding the pipeline's
   hard branch (`sc_demodulate_psk`), exactly like BCH.
2. **Rateless realized as fixed overhead.** The `Codec.encode(info)->bits`
   contract needs a deterministic length. Emit **N = ⌈(1+ε)·K⌉** encoded
   symbols for a fixed overhead ε (catalog param). Multiple ε profiles show the
   overhead/recovery trade-off.
3. **Raptor precode kills the small-K error floor.** Drone payloads are tens of
   bytes ⇒ K in the tens — LT's weak regime (a plain-LT error floor). The
   systematic sparse precode adds R redundant intermediate symbols so the
   combined GF(2) system is full-rank with high probability at small K.

### Scale invariance / no LLR issue

Fountain is hard-input: it consumes hard bits from `sc_demodulate_psk`, detects
erasures by per-symbol CRC, and solves a GF(2) system. There is no soft-LLR
metric and therefore **no `sc_soft_bits` recalibration question** (unlike the
soft codecs; consistent with BCH being hard-input). Records as **ADR-0017**.

### Non-goals

- Not RFC 5053 / RFC 6330 (RaptorQ) bit-compatibility (interop out of scope,
  per the RS/LDPC/turbo/polar precedent — reproducible, self-consistent
  construction is the bar).
- Not large-K / streaming / file-transfer use (fountain's ideal regime). This
  is a faithful small-K *demonstration* within the T&E framework.
- Not systematic LT (encoded symbols are non-systematic XOR combinations).
- Not soft/LLR-based erasure marking, not genie erasures (per-symbol CRC only).
- Not inactivation decoding (plain GF(2) Gaussian elimination — fine at small K).

---

## 2. `core/fountain.py` — new module (mirrors `core/ldpc.py`)

### 2a. Symbols, length header & seeded RNG
- The codec prepends a fixed **16-bit length header** (the frame bit-length) to
  the pipeline's frame (`payload | CRC-16`), giving the codec-internal
  `payload_bits = [len16 | payload | CRC-16]`. This is split into **K source
  symbols of S bits** each (`S` a catalog param; zero-pad the last symbol;
  `K = ⌈len(payload_bits) / S⌉`). Symbols are `uint8` bit-vectors of length S;
  XOR is bitwise.
- **Length recovery (rateless has no fixed trailer to key off):** at decode, the
  transmitted length is always `N·(S + crc_sym)`, so **N** is exact and **K** is
  the unique integer with `N = ⌈(1+ε)·K⌉` (strictly increasing in K ⇒ invertible
  on its image; no valid K ⇒ clean fail). K sizes the decode matrix. The exact
  frame bit-length then comes from the recovered **len16 header** (inside the
  solved source symbols), used only for the final trim — so there is no
  chicken-and-egg (K comes from the length; the header is read *after* the
  solve). A corrupted header ⇒ wrong trim ⇒ the outer CRC-16 fails **loud**.
- All degree/neighbor choices come from a **fixed seed** (catalog param) so TX
  and RX agree without side information. `numpy.random.default_rng(seed + i)`
  per encoded symbol i gives its degree + neighbor set deterministically.

### 2b. Precode (outer, systematic, sparse)
- `build_precode(K, seed, precode_rate) -> R parity rows`: generate
  `R = round((1/precode_rate - 1) · K)` (i.e. rate `K/(K+R)`) parity symbols;
  parity row j = a seeded sparse subset (fixed small degree, e.g. 3–5) of the K
  source indices. Intermediate set **I = [source(K) | parity(R)]**, length
  **L = K + R**. Parity symbol values are computed at encode time as the XOR of
  their source neighbors, so the precode constraints `parity_j = ⊕ sources` hold
  by construction and are also emitted as decoder constraint rows.

### 2c. Robust Soliton degree distribution
- `robust_soliton(L, c, delta) -> pmf` over degrees 1..L: the standard Ideal
  Soliton ρ plus the Robust spike τ (with `S_rs = c·ln(L/delta)·√L`),
  normalized. `sample_degree(rng)` draws from its CDF. Uses the descriptor's
  `c`, `delta`.

### 2d. LT encoder (inner) + per-symbol CRC
- `fountain_encode(frame_bits, spec) -> bits`:
  1. Split frame into K source symbols; build the R precode parity symbols ⇒ L
     intermediates.
  2. For i in 0..N-1 (`N = ceil((1+ε)·K)`): `rng_i = default_rng(seed+i)`; draw
     degree d from Robust Soliton over L; pick d distinct intermediate indices;
     encoded symbol = XOR of those intermediates.
  3. Append a per-symbol CRC (`crc_sym` bits, e.g. CRC-8) over the S-bit symbol.
  4. Concatenate all N `(S + crc_sym)`-bit encoded symbols → the coded bit
     stream (uint8). Deterministic given the frame + seed.

### 2e. Decoder — erasure detection + GF(2) Gaussian elimination
- `fountain_decode(received_bits, spec, frame_len) -> (frame_bits, ok)`:
  1. Reshape into N `(S + crc_sym)`-bit encoded symbols; for each, recompute the
     per-symbol CRC and **mark erased** if it fails. Keep the surviving symbols
     (and their known seed-derived intermediate-index sets).
  2. Build the GF(2) constraint matrix `A` (rows × L intermediates): one row per
     surviving LT symbol (its neighbor incidence), plus the R precode rows
     (`parity_j ⊕ neighbors = 0`). RHS `b` = the surviving symbol S-bit values
     (precode rows RHS = 0).
  3. **Solve `A · I = b` over GF(2)** by Gaussian elimination (row-reduce the
     incidence matrix; apply the same row ops as symbol-XORs to the RHS
     S-bit-vectors). If full rank over the K source columns, recover I; else
     decode fails (loud).
  4. Extract the K source symbols from I, concatenate, truncate to `frame_len`
     → frame bits.
- `frame_len` (the true frame bit length) must be recoverable: the codec conveys
  it the way `_LDPC`/`_Turbo` convey `info_len` — see §3.

---

## 3. `_Fountain` codec (in `core/coding.py`)

- `__init__(spec)`: read `symbol_bits` (S), `c`, `delta`, `seed`,
  `precode_rate`, `precode_degree`, `overhead` (ε). The per-symbol CRC is a
  fixed **CRC-8** (`core.fountain.crc8`), not a tunable param — `crc_sym = 8`.
- `encode(info_bits)`: `info_bits` is the pipeline's frame (`payload|CRC-16`);
  prepend the 16-bit length header (§2a) → `payload_bits`; `K = ⌈len /S⌉`;
  `N = ⌈(1+ε)·K⌉`; run `fountain_encode` → N `(S+crc_sym)`-bit encoded symbols.
- `decode(received)`: hard branch — bits in (fountain is hard-input; if ever fed
  floats, map `L>0⇒bit0`); `N = len /(S+crc_sym)`; recover K from N and ε; run
  `fountain_decode` (per-symbol-CRC erasures → GF(2) solve → read len16 header →
  trim); return the frame (`payload|CRC-16`) as
  `DecodeResult(bits=frame_bits, meta={"decode_ok": ok, "erasures": n_erased})`.
  On any failure (rank-deficient solve, no valid K, header out of range) return
  empty/short bits with `decode_ok=False` (outer CRC then fails loud anyway).
- `make_codec`: add `if spec.family == CodeFamily.FOUNTAIN: return _Fountain(spec)`
  — this is the **last** family, so after P3h `make_codec` raises for nothing
  and the `test_make_codec_unimplemented_families_raise` test must be updated to
  expect **all families build** (no `NotImplementedError` branch left; the
  `make_codec` fallback `raise` becomes unreachable for catalog specs but stays
  as a guard for an unknown family).

## 4. Catalog & profile

- **Catalog**: replace/extend `fountain_lt` with **multiple overhead profiles**
  sharing a Raptor param block:
  - `fountain_r05` (ε=0.5), `fountain_r10` (ε=1.0), `fountain_r15` (ε=1.5),
    each `CodingSpec(name, FOUNTAIN, 0, 0, {kind:"raptor", c:0.03, delta:0.5,
    symbol_bits:S, precode_rate:P, precode_degree:PD, seed:SD, overhead:ε})`
    (`k=0, n=0` rateless). Params pinned in implementation:
    `symbol_bits=16`, `precode_rate=0.95`, `precode_degree=4`, `seed=7`
    (shared across all three ε profiles). The per-symbol CRC is a fixed CRC-8
    (hardcoded, not a catalog param).
- **Profile** (`core/profiles.py`): `fountain_bpsk = SCProfileSpec(
  "fountain_bpsk", SCMod.BPSK, SCProfile(sps=<U>), coding="fountain_r10")` where
  `<U>` is a **small unique** sps (not in {4,8,16,32,48,64,128,256,512}; e.g.
  **96**) — the P3g lesson: keep blind-resolve cost low; uniqueness (not
  magnitude) is what blind resolution needs.

## 5. Integration — no pipeline change
`soft_input=False` → hard branch (`sc_demodulate_psk`) reused unchanged; erasure
detection + GF(2) solve internal to the codec; outer CRC-16 loud-fail; all
existing uncoded/rep3/conv/RS/BCH/LDPC/turbo/polar paths byte-for-byte
unchanged.

## 6. Testing strategy

1. **Robust Soliton**: pmf sums to 1, degrees in 1..L; deterministic given
   (L,c,delta); the τ spike is at ⌊L/S_rs⌋.
2. **Precode**: L = K+R; parity rows are XOR of their source neighbors (holds by
   construction); deterministic given (K, seed).
3. **Encode structure**: output length = N·(S+crc_sym) with N=⌈(1+ε)·K⌉;
   per-symbol CRC valid on a clean encode; deterministic given frame+seed.
4. **Noiseless round-trip** (all overhead profiles): `encode → decode` with no
   erasures recovers the payload with `decode_ok`; frame_len recovered exactly.
5. **Erasure recovery (headline)**: puncture/corrupt a fraction of encoded
   symbols (flip bits so their per-symbol CRC fails) up to the code's capability
   → GF(2) solve still recovers the payload; beyond capability → **loud fail**
   (outer CRC false, never a silent wrong payload). Show the Raptor precode
   recovers a case where plain-LT peeling would stall.
6. **GF(2) solver**: rank-deficient system → clean failure (no crash, decode_ok
   False); full-rank → exact recovery.
7. **Erasure recovery vs overhead**: higher ε tolerates a higher erasure rate
   (the multi-profile trade-off) — a monotone check across r05/r10/r15.
8. **End-to-end (erasure/FER metric)**: through synth BPSK → `sc_demodulate_psk`
   (hard) → per-symbol CRC erasures → fountain decode → outer CRC, at an SNR
   giving a moderate symbol-erasure rate, fountain **recovers the payload**
   where the uncoded frame does not (FER, not BER — fountain is an erasure
   code). Payload sized to give an adequate K (larger than the tiny default).
9. **Blind e2e**: `fountain_bpsk` resolves + hard-decodes + recovers payload.
10. **Loud-fail + regression**: full existing suite (P1–P3g) green; `mypy
    validation` clean; families test updated (all build).

## 7. Global constraints (bind every task)

- `mypy validation` clean (no `Any`; `numpy.typing`); minimal diffs; lint
  touched test files (black/isort/flake8).
- complex64 at IQ boundary; bits uint8; GF(2) work in uint8. Deterministic
  encode (fixed seed); the precode + LT structure is seed-reproducible.
- **Loud on failure** CRC-gated (outer CRC-16); fountain never emits a silent
  wrong payload. `decode_ok` reflects the outer CRC.
- **Hard-input** (`soft_input=False`); no `sc_soft_bits` involvement.
- Additive; on branch `feature/p3h-fountain` (from `main`); no PR/tag this
  sub-phase (merge/version deferred to the user, as for prior sub-phases).
- Stall guard: implementers run only targeted test files; controller runs the
  full-suite closing gate; never background a slow run silently; never run the
  whole `test_blind_resolve.py` (use `-k`). Keep the `fountain_bpsk` profile at
  a **small unique sps** (blind-resolve cost) and gain/e2e payloads sized for
  < 30 s targeted runs (GF(2) GE is cheap at small L, but keep K modest).

## 8. Out of scope / future

- RaptorQ (RFC 6330) / inactivation decoding / systematic fountain; large-K /
  streaming / true rateless (unbounded symbol generation); soft-decision
  fountain (BP over LLRs); interop.
- P4 (bit-loading) — the next program phase after the FEC family sheet is
  complete.
- Docs on completion: mark this spec Accepted (implemented); add **ADR-0017**
  (Raptor-style fountain: per-symbol-CRC erasure model + GF(2)-GE decode,
  hard-input, small-K demonstration scope); update ADR + design indices,
  README/CLAUDE module maps (add `core/fountain.py`), and note fountain
  completes the seven-family sheet.
