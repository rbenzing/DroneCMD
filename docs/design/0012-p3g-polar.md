# P3g — Arıkan polar code + CRC-aided SCL (Design)

**Design #:** 0012
**Phase:** P3g

**Date:** 2026-09-19
**Status:** Accepted — implemented (unreleased).
**Author:** rbenzing (with Claude)
**Program:** P3 (channel coding), sub-phase **P3g** — the third iterative/soft
codec after LDPC (P3e) and turbo (P3f): an Arıkan polar code with a CRC-aided
successive-cancellation list (CA-SCL) decoder.
**Predecessors:** P3a (framework), P3b (soft Viterbi + puncture helpers),
P3c (galois), P3d (BCH), P3e (LDPC; ADR-0013 scale-invariant soft decode),
P3f (turbo; ADR-0014). Builds on the demod phase-ramp fix (ADR-0015), which
makes end-to-end coding-gain measurement through `sc_soft_bits` valid again.
ADRs 0005, 0006, 0013, 0014, 0015.
**Scope note:** Security-research / educational software. Lawful research /
authorized testing only.

---

## 1. Overview

### Goal

Implement an **Arıkan polar code** (block length **n = 256 = 2⁸**, rates
**1/2, 1/3, 2/3**) on the P3a `Codec` framework, decoded by a
**CRC-aided successive-cancellation list (CA-SCL)** decoder with
**`list_size = 8`**, reusing the soft pipeline branch unchanged. The frozen
set is chosen by a **Gaussian-approximation** density-evolution construction at
a fixed design SNR. The framework already carries the capability descriptor
`polar_256_128` (`{soft_input: True, list_size: 8}`) and `CodeFamily.POLAR`;
`make_codec` raises for `POLAR` today.

### CRC reuse — one frame = one polar block

The framework's soft codecs are **single-block per frame**: `_Turbo.encode`
raises when the CRC-framed frame exceeds one block (`info_len > K`), and shortens
for smaller frames. Polar follows the same contract — **one CRC-framed frame
maps to exactly one polar block**. The whole frame is `frame_with_crc(payload)`
= `[payload | CRC-16]`, so the frame CRC *is* the block CRC. CA-SCL reuses that
existing CRC-16 (from `core.coding`) for path selection at zero extra overhead —
no per-block inner CRC, no multi-block ambiguity. Per-block payload capacity is
`(K − 16) / 8` bytes: **14 B** (r½, K=128), **8 B** (r⅓, K=85), **19 B**
(r⅔, K=170). Larger payloads / multi-block framing are out of scope (Sec 7).

### Scale invariance ⇒ no LLR recalibration

The decoder works entirely in the LLR domain with:
- **min-sum f-node** `f(a, b) = sign(a)·sign(b)·min(|a|, |b|)`,
- **g-node** `g(a, b, û) = b + (1 − 2û)·a`,
- **approximate path metric** `PM += |LLR|` when the path's bit decision
  disagrees with the LLR sign (0 otherwise).

Every one of these is **linearly homogeneous** in a global input-LLR scale, so
scaling all LLRs by `k > 0` scales every path metric by `k` — the *ordering* of
paths, the surviving list, and the final hard decision are all **unchanged**.
Polar CA-SCL is therefore **scale-invariant** (empirically `decode(k·llr) ==
decode(llr)` for `k ∈ 1e-3 … 1e3`), exactly as for LDPC min-sum (ADR-0013),
turbo max-log-MAP (ADR-0014), and soft Viterbi. **P3g does not recalibrate
`sc_soft_bits`** (records as **ADR-0016**). Exact (`tanh`/`log-MAP`) f-nodes
would be scale-sensitive and are out of scope.

### Non-goals

- Not multi-block frames, not n = 512+, not systematic polar encoding.
- Not the exact 5G-NR reliability sequence / interop (per RS/LDPC/turbo
  precedent — reproducible construction, interop out of scope).
- No plain SC or metric-only SCL (CA-SCL only).
- No `sc_soft_bits` recalibration.

---

## 2. `core/polar.py` — new module (mirrors `core/ldpc.py`)

### 2a. Polar transform (encoder core)
- `m = 8`, `n = 2**m = 256`. Kernel `F = [[1, 0], [1, 1]]`;
  generator `G = F^{⊗m}` (Kronecker power). Non-bit-reversed convention used
  consistently for encode and decode (equivalent for our purposes; simpler
  indexing).
- `polar_transform(u) -> x`: the length-n GF(2) transform, computed by the
  in-place butterfly (`O(n log n)`), **not** a dense `G` matmul.
- `PolarCode` dataclass: `n`, `k`, `frozen_mask` (length-n bool; True = frozen),
  `info_positions` (sorted indices where `frozen_mask` is False).

### 2b. Frozen set — Gaussian approximation
- `gaussian_approx_reliabilities(n, design_snr_db) -> np.ndarray`: standard GA
  density evolution. Seed the root LLR mean `mllr = 4·10^(design_snr_db/10)`;
  recurse the check/variable updates through the m stages using the `phi`
  function and its inverse; the per-position mean LLRs give the reliability
  ordering.
- `build_frozen_mask(n, k, design_snr_db)`: freeze the `n − k` least-reliable
  positions; the `k` most-reliable are info. Deterministic given
  `(n, k, design_snr_db)`. `design_snr_db` is a module/catalog constant.

### 2c. Encoder
- `polar_encode(info_bits, code) -> x`: scatter `k` info bits into
  `code.info_positions` of a length-n vector `u` (frozen positions = 0), apply
  `polar_transform`, return the length-n codeword `x` (uint8).

### 2d. CA-SCL decoder
- `scl_decode(llr, code, list_size, crc_check) -> (info_bits, passed)`:
  LLR-domain SCL over the butterfly.
  - Maintain up to `list_size` paths, each with a partial decision vector and a
    path metric. At each info bit, fork every path into û ∈ {0, 1}; at each
    frozen bit, force û = 0 (its known value). Update PM by the approximate
    `|LLR|`-on-disagreement rule; prune to the best `list_size` by PM.
  - f/g node LLRs recursed with the **min-sum f-node** and **g-node** above.
  - **CRC-aided selection:** extract each surviving path's `k` info bits, and
    of those return the **lowest-PM path whose bits pass `crc_check`**; if none
    pass, return the lowest-PM path and `passed = False`.
- `crc_check` is injected by the codec so `core.polar` does not import framing;
  the codec passes the `core.coding` CRC-16 verifier.

### 2e. Shortening (shorten-from-the-end; conveys L like LDPC/turbo)
The pipeline's `check_and_strip_crc` treats the **last 16 bits as the CRC**, so
`decode` must return exactly the `L`-bit frame — the codec must convey `L`. It
does so, like `_LDPC`/`_Turbo`, by transmitting a **variable-length** codeword,
using the standard *shorten-from-the-end* scheme (Wang & Liu). `G = F^{⊗m}` is
**lower-triangular**, so freezing the highest-index `s = K − L` input positions
forces the **last `s` codeword bits to 0** (each `x_j` depends only on
`u_j … u_{n-1}`). Those `s` known-0 codeword bits are **not transmitted**, so:

- **Transmitted length** `M = n − s = (n − K) + L` (varies with `L`; `= n` at
  `L = K`), exactly mirroring `_LDPC`'s `info_len + (n − k)`.
- **Frozen set:** the `s` shortened tail positions are force-frozen; the
  remaining `n − s` positions are frozen/info by **GA reliability** to leave
  exactly `L` info positions. (Reliability is respected among the transmittable
  positions; the forced tail is the shortening overhead.)
- **Decode** reconstructs the length-`n` LLR vector by appending `s` `+1e6`
  LLRs (the known-0 tail) — the same `+1e6` known-zero convention as LDPC/turbo.

`L = K` is the nominal catalog rate (`M = n`).

---

## 3. `_Polar` codec (in `core/coding.py`)

- `__init__(spec)`: read `k`, `n`, `list_size`, `design_snr_db`; build the
  `PolarCode` once (frozen mask cached on the instance).
- `encode(info_bits)`: raise if `info_bits.size > k` ("payload exceeds polar
  block k"); set `L = info_bits.size`, `s = K − L`; build the shortened frozen
  mask (force-freeze the top `s` positions + GA-fill to `L` info); scatter info,
  `polar_transform`, and transmit the first `M = n − s` codeword bits (drop the
  known-0 tail). (Matches `_LDPC`/`_Turbo`'s variable-length single-block
  contract.)
- `decode(received)`: soft branch — cast float LLRs directly (`L>0 ⇒ bit 0`),
  hard branch — map bits→±LLR (as the other soft codecs do). Recover
  `L = received_len − (n − K)`, `s = K − L`; append `s` `+1e6` LLRs for the
  known-0 tail; rebuild the same shortened frozen mask; run `scl_decode` with
  the injected CRC-16 verifier; return
  `DecodeResult(bits=info[:L], meta={"decode_ok": passed})`.
- `make_codec`: add `if spec.family == CodeFamily.POLAR: return _Polar(spec)`.
- **Regression:** enabling `POLAR` removes it from the "unimplemented families
  raise" set — extend `test_make_codec_unimplemented_families_raise`'s builds
  tuple additively (drop `POLAR`, keep `FOUNTAIN`), pre-empted in the pre-flight
  conflict scan (the recurring per-sub-phase regression).

## 4. Catalog & profile

- **Catalog** (`core/coding.py`): keep `polar_256_128` (r½, K=128); add params
  `{soft_input: True, list_size: 8, design_snr_db: <D>, rate: "1/2"}`. Add
  `polar_256_85` (K=85, r⅓) and `polar_256_170` (K=170, r⅔) with the same
  `design_snr_db` and `list_size`. `design_snr_db` value fixed during
  implementation (measured to give a clean waterfall; ~1–3 dB typical).
- **Profile** (`core/profiles.py`): `polar_bpsk = SCProfileSpec("polar_bpsk",
  SCMod.BPSK, SCProfile(sps=1024), coding="polar_256_128")` — **sps = 1024
  unique** (existing 4/8/16/32/64/128/256/512).

## 5. Integration — no pipeline change
`soft_input=True` → soft branch reused; shortening internal to the codec;
loud-on-failure CRC-gated; `sc_soft_bits` untouched; all existing
uncoded/rep3/conv/RS/BCH/LDPC/turbo paths byte-for-byte unchanged.

## 6. Testing strategy

1. **GA frozen set**: `build_frozen_mask` deterministic; exactly `k` info
   positions; monotone with reliability (a higher-reliability position is never
   frozen while a lower one is info).
2. **Transform**: `polar_transform` (butterfly) equals a dense `F^{⊗m}` matmul
   reference on random inputs (small n and n=256).
3. **Noiseless round-trip, all 3 rates** incl. heavy shortening:
   `encode → (bits→±LLR) → decode` recovers the payload with `decode_ok`.
4. **Error correction**: corrects low-SNR LLRs within capability; beyond → CRC
   fails **loud** (`decode_ok=False`, never a silent wrong payload).
5. **CRC-aided list selection**: a case where the max-likelihood (lowest-PM)
   path is wrong but a CRC-passing path exists in the list → CA-SCL returns the
   CRC-passing one (demonstrates the CRC aid over plain SCL).
6. **Scale-invariance (headline)**: `decode(k·llr) == decode(llr)` for
   `k ∈ {1e-3 … 1e3}` (justifies no `sc_soft_bits` recalibration).
7. **Coding gain (end-to-end)**: polar BER < uncoded BER through synth →
   `sc_soft_bits` → CA-SCL → CRC, at a low SNR (now valid post-ADR-0015). Near-
   full-block payload to avoid the shortening waterfall (per the LDPC lesson).
8. **Blind e2e**: `polar_bpsk` resolves + soft-decodes + populates `coded_link`.
9. **Loud-fail + regression**: full existing suite (P1–P3f) green; `mypy
   validation` clean.

## 7. Global constraints (bind every task)

- `mypy validation` clean (no `Any`; `numpy.typing`); minimal diffs; lint
  touched test files (black/isort/flake8).
- complex64 boundary; LLRs float64; bits uint8. Deterministic encode + frozen
  set (fixed `design_snr_db`); the polar transform is exact GF(2).
- **Loud on failure** CRC-gated; CA-SCL never emits a silent wrong payload.
- **No `sc_soft_bits` recalibration** (min-sum f-node + `|LLR|` PM is
  scale-invariant).
- Additive; on branch `feature/p3g-polar` (from `main`); no PR/tag this
  sub-phase (merge/version deferred to the user, as for prior sub-phases).
- Stall guard: implementers run only targeted test files; controller runs the
  full-suite closing gate; never background a slow run silently; never run the
  whole `test_blind_resolve.py` (use `-k`). Pure-Python SCL is the slow part
  (the implemented decoder recomputes from the root per bit, ~`O(list·n²·log n)`;
  a memoized cache would reach `O(list·n·log n)` — future optimization) — keep
  gain/e2e trial counts and payloads sized for < 30 s targeted runs, per the
  LDPC/turbo precedent.

## 8. Out of scope / future

- Multi-block frames; n = 512+; systematic polar; exact 5G reliability
  sequence / interop; plain SC or metric-only SCL; adaptive list size.
- Fountain (P3h); bit-loading (P4).
- Docs on completion: mark this spec Accepted (implemented); add **ADR-0016**
  (polar CA-SCL, min-sum f-node + `|LLR|` path metric, scale-invariant → no
  recalibration); update the ADR and design indices, README/CLAUDE module maps.
