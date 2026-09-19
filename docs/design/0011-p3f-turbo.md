# P3f — LTE-style turbo code + iterative max-log-MAP (Design)

**Design #:** 0011
**Phase:** P3f

**Date:** 2026-09-19
**Status:** Design (awaiting review)
**Author:** rbenzing (with Claude)
**Program:** P3 (channel coding), sub-phase **P3f** — the second iterative soft codec (turbo).
**Predecessors:** P3a (framework), P3b (soft Viterbi + puncture helpers), P3c (galois), P3d (BCH), P3e (LDPC; ADR-0013 scale-invariant soft decode). ADRs 0005, 0006, 0008, 0013.
**Scope note:** Security-research / educational software. Lawful research / authorized testing only.

---

## 1. Overview

### Goal

Implement an **LTE-style turbo code** — two recursive systematic convolutional (RSC) constituent encoders (K=4, generators 13/15 octal), a **QPP interleaver**, rate **1/3** plus punctured **1/2** — decoded by an **iterative max-log-MAP** SISO decoder with **0.7 extrinsic scaling**, on the P3a `Codec` framework, reusing the soft pipeline branch unchanged.

### Scale invariance ⇒ no LLR recalibration

Max-log-MAP uses `max` (not `max*`) in the BCJR recursions; every state/branch metric and extrinsic is linearly homogeneous in a global input-LLR scale, and a **constant** extrinsic-scale factor (0.7) preserves that. So the hard decision is **scale-invariant**, and P3a's ~2× conservative `sc_soft_bits` scale is harmless — exactly as for LDPC min-sum (ADR-0013), soft Viterbi, and the RS erasure rule. **P3f does not recalibrate `sc_soft_bits`** (records as ADR-0014). Full log-MAP (with the max* correction) would be scale-sensitive and is out of scope.

### What predecessors provide (build on)

- `core.coding`: `CodeFamily.TURBO`; `Codec`; `CodingSpec.soft_input`; `make_codec` (raises for TURBO today); `CODING_CATALOG["turbo_r13"]` descriptor (`constraint_length=4, generators_octal=(0o13,0o15), soft_input=True, max_iters=8`); CRC framing; `interleave`/`deinterleave`; **`_puncture`/`_depuncture`** (P3b, reused for rate-1/2 puncturing).
- `core.single_carrier.sc_soft_bits` (LLRs, `L>0⇒bit0`).
- `validation.pipeline` soft coded-decode branch (reused unchanged).
- `coded_link` metric.

### Non-goals

- Not the exact LTE interleaver table — a **QPP formula** with verified-valid parameters (interop out of scope, per the RS/LDPC precedent).
- No full log-MAP (max* correction); no other block sizes; no tail-biting (tail-terminated).
- No `sc_soft_bits` recalibration.

---

## 2. `core/turbo.py` — new module (mirrors `core/ldpc.py`)

### 2a. RSC constituent encoder
Recursive systematic convolutional, K=4 (3 memory bits, 8 states). Feedback `g0 = 13₈ = 1 + D + D³`; feedforward `g1 = 15₈ = 1 + D² + D³`. For each input bit: systematic output = input; parity = RSC output given feedback. **Tail termination**: after the info bits, feed 3 tail bits chosen to return the register to state 0 (the tail depends on the current state via the feedback). Encodes info → (systematic, parity) each of length `K + 3`.

### 2b. QPP interleaver
`π(i) = (f1·i + f2·i²) mod K` over fixed block size **K = 256** (= 2⁸). Valid permutation iff `f1` odd and `f2` even (Sun–Takeshita, for K a power of two). Parameters are module constants (e.g. `f1=31, f2=64`); the builder **verifies bijection** (asserts `sorted(π(range(K))) == range(K)`). `interleave_qpp`/`deinterleave_qpp` over K.

### 2c. Turbo encoder
- systematic `s` = info (K bits, padded).
- parity1 `p1` = RSC(info) parity (+ its 3 tail bits & tail parity).
- parity2 `p2` = RSC(π(info)) parity (+ its tail).
- Rate-1/3 codeword = interleave-of `[s | p1 | p2]` (systematic + both parities + termination bits), a deterministic layout.
- **Rate 1/2**: puncture the two parity streams with the standard alternating pattern (keep p1 on even indices, p2 on odd) via `_puncture`; systematic always kept. Reuses P3b `_puncture`/`_depuncture`.

### 2d. Max-log-MAP SISO + turbo iteration
- **BCJR (max-log-MAP)** for one RSC constituent: forward `α` and backward `β` state-metric recursions using `max` (log domain); branch metric `γ` from channel systematic+parity LLRs and the a-priori LLR; per-bit **extrinsic** LLR = max-over-1-paths − max-over-0-paths − systematic − a-priori.
- **Turbo loop**: SISO-1 (natural order) produces extrinsic → ×0.7 → interleave → a-priori for SISO-2 (interleaved order) → extrinsic → ×0.7 → deinterleave → a-priori for SISO-1; repeat `max_iters`. Final per-bit LLR = systematic + both extrinsics; hard decision by sign. Scale-invariant.
- **Rate 1/2 decode**: de-puncture parities to erasures (LLR 0) before BCJR (reuse `_depuncture`), exactly as P3b's Viterbi handles punctured bits.

### 2e. Shortening
Pad info to K; encode; transmit real-info systematic + parities (+tails); decode pins the known-zero pad positions with large a-priori LLRs (+1e6). Mirrors LDPC.

---

## 3. Catalog & profile

- **Catalog**: update `turbo_r13` params `{constraint_length:4, generators_octal:(0o13,0o15), soft_input:True, max_iters:8, extrinsic_scale:0.7, block_k:256, qpp:(31,64)}`; add `turbo_r12` (same, `puncture` pattern for rate 1/2).
- **`core/profiles.py`**: `turbo_bpsk = SCProfileSpec("turbo_bpsk", SCMod.BPSK, SCProfile(sps=512), coding="turbo_r13")` — **sps=512 unique** (existing 4/8/16/32/64/128/256).

## 4. Integration — no pipeline change
`soft_input=True` → soft branch reused; shortening internal to the codec; loud-on-failure CRC-gated; `sc_soft_bits` untouched.

## 5. Testing strategy

1. **RSC encoder**: systematic output equals input; register returns to state 0 after the 3 tail bits; known reference parity for a small input.
2. **QPP**: `π` is a genuine bijection over K; `f1` odd / `f2` even.
3. **Turbo encode structure**: rate-1/3 length; systematic prefix recoverable; rate-1/2 puncture length.
4. **Noiseless round-trip both rates**: `encode → decode` recovers payload with `crc_ok`.
5. **Error correction**: corrects low-SNR LLRs within capability; beyond → CRC fails **loud**.
6. **Scale-invariance (headline)**: ×k LLR ⇒ identical decode (justifies no recalibration).
7. **Coding gain**: turbo BER < uncoded BER through synth → `sc_soft_bits` → turbo → CRC (near-full-rate payload to avoid the shortening waterfall, per the LDPC lesson).
8. **Blind e2e**: `turbo_bpsk` resolves + soft-decodes + populates `coded_link`.
9. **Loud-fail + regression**: full existing suite (P1–P3e) green; `mypy validation` clean.

## 6. Global constraints (bind every task)

- `mypy validation` clean (no `Any`; `numpy.typing`); minimal diffs; lint touched test files.
- complex64 boundary; LLRs float64; bits uint8. Deterministic encode; QPP params fixed + bijection-verified.
- **Loud on failure** CRC-gated; existing uncoded/rep3/conv/RS/BCH/LDPC paths byte-for-byte unchanged.
- **No `sc_soft_bits` recalibration** (max-log-MAP + constant extrinsic scale is scale-invariant).
- Additive; on branch `feature/p3f-turbo` (stacked on `feature/p3e-ldpc`); no PR/tag this sub-phase.
- Stall guard: implementers run only targeted test files; controller runs the full-suite closing gate; never background a slow run; never run the whole `test_blind_resolve.py` (use `-k`).

## 7. Out of scope / future

- Exact LTE interleaver tables / interop; full log-MAP; tail-biting; other block sizes.
- Polar (P3g), fountain (P3h); bit-loading (P4).
- Docs on completion: mark spec Accepted; add **ADR-0014** (turbo max-log-MAP + 0.7 extrinsic scaling, scale-invariant → no recalibration); update indices.
