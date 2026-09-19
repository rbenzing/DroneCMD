# P3e — 802.11n QC-LDPC + normalized min-sum (Design)

**Design #:** 0010
**Phase:** P3e

**Date:** 2026-09-19
**Status:** Accepted — implemented (unreleased; awaiting merge)
**Author:** rbenzing (with Claude)
**Program:** P3 (channel coding), sub-phase **P3e** — the first iterative soft codec (LDPC).
**Predecessors:** P3a (coding framework), P3b (soft Viterbi; pipeline soft branch), P3c (Reed-Solomon; `core/galois.py`), P3d (BCH). ADRs 0005 (coding framework), 0006 (soft-LLR + loud-on-failure), 0008 (FEC family choices).
**Scope note:** Security-research / educational software. Lawful research / authorized testing only.

---

## 1. Overview

### Goal

Implement an **802.11n-style quasi-cyclic LDPC** at block length **n=648** (lifting size **Z=27**), code rates **1/2, 2/3, 3/4**, with an iterative **normalized min-sum** soft-decision decoder, plugged into the P3a `Codec` framework and reusing the P3b/P3c **soft coded-decode pipeline branch unchanged**. The code uses 802.11n's dimensions and IRA/dual-diagonal structure via a deterministic, reproducible construction (§2a) — **not** the exact IEEE base matrices, which cannot be bit-verified in this environment and are unnecessary since interop is out of scope.

### Key property: scale invariance ⇒ no LLR recalibration

Normalized min-sum is **scale-invariant for the hard decision**: every message scales linearly with a global input-LLR scale factor, and the per-iteration sign pattern (hence the decoded bits and the `H·ĉᵀ=0` stopping test) is unchanged. Therefore P3a's ~2× conservative `sc_soft_bits` calibration is **harmless for LDPC**, exactly as it is for soft Viterbi and the RS median-relative erasure rule. **P3e does not recalibrate `sc_soft_bits`.** This supersedes ADR-0006's expectation that "LDPC must recalibrate the LLR scale" — recorded in a new ADR-0013.

### What P3a–P3d already provide (build on, don't rebuild)

- `core.coding`: `CodeFamily.LDPC`; `Codec` protocol; `CodingSpec` with `soft_input`; `make_codec` (raises `NotImplementedError` for LDPC today); `CODING_CATALOG["ldpc_648_r12"]` descriptor (`soft_input=True, max_iters=50`); `frame_with_crc`/`check_and_strip_crc`; `interleave`/`deinterleave` (LLR-safe); `CODING_INTERLEAVE_DEPTH`.
- `core.single_carrier.sc_soft_bits(...) -> LLRs` (convention `L>0 ⇒ bit 0`).
- `core.profiles`: `SCProfileSpec(..., coding=None)`, `SC_CATALOG`, `coding_of`.
- `validation.pipeline.single_carrier_region_to_bytes`: the **soft** coded-decode branch (`sc_soft_bits → deinterleave(LLRs) → codec.decode → check_and_strip_crc`). **LDPC reuses it unchanged.**
- The `coded_link` (coded BER/FER) metric.

### Non-goals

- No interop with real 802.11n frames, and **not the exact IEEE base matrices** (802.11n dimensions + IRA/dual-diagonal structure only, via the reproducible construction in §2a). Like RS's conventional (non-CCSDS) basis, we verify **self-consistent encode/decode + structural fidelity + coding gain**; decoding over-the-air 802.11n is out of scope.
- No sum-product / layered / offset-min-sum decoding (normalized min-sum only).
- No other block lengths (1296/1944) or rate 5/6.
- No `sc_soft_bits` recalibration (see above).

---

## 2. `core/ldpc.py` — new module (mirrors the `core/galois.py` pattern)

Keeping the LDPC machinery out of `coding.py` (which would balloon) in a dedicated, testable module.

### 2a. Base (prototype) matrices — reproducible 802.11n-style construction
We build **802.11n-dimensioned** QC base matrices (**not** the exact IEEE tables — interop is out of scope and the exact tables cannot be bit-verified here, so we do not claim them). Same dimensions/structure as 802.11n n=648:
- Block grid `n_b = 24` columns, `Z = 27`; `m_b = 12/8/6` rows for rate `1/2 / 2/3 / 3/4` (so `k_b = 12/16/18`, `k = 324/432/486`).
- Each base entry is `-1` (a Z×Z all-zero block) or an integer `0..Z-1` (the Z×Z identity **cyclically right-shifted** by that amount).

**Construction (deterministic, seeded, IRA/staircase — the 802.11n family form):**
- **Parity part** (last `m_b` block-columns): a **dual-diagonal (staircase)** structure — bidiagonal identity blocks (row `i` and `i-1`) plus a single weight-3 first parity column — chosen so `H_parity` is **invertible over GF(2)** and supports recursive (back-substitution) encoding. Parity variable-nodes therefore have degree 2 (degree 3 for the first), exactly the accumulator structure 802.11n uses.
- **Info part** (first `k_b` block-columns): each info block-column gets **weight 3** (3 nonzero blocks) at pseudo-random distinct block-rows with pseudo-random shifts `0..Z-1`, drawn from a **fixed seed**, and constrained to be **4-cycle-free (girth ≥ 6)** by rejecting placements that create a length-4 cycle. Deterministic → reproducible and testable.

This is a legitimate irregular repeat-accumulate QC-LDPC of 802.11n's dimensions. **Functional bar:** because it is not the exact standard code, the guard is (a) **structural verification** (dims, degree profile — info dv=3 / dual-diagonal parity, Z=27 lifting, 4-cycle-free, `H_parity` invertible) and (b) the **coding-gain test** — a code that does not beat uncoded fails the phase, so the construction must actually work.

### 2b. `build_h(rate) -> Tanner graph`
Expand a base matrix by Z=27: each `-1` → zero block; each shift value → a cyclically shifted Z×Z identity. Produce the sparse parity-check `H` of shape `((n_b−k_b)·Z) × (n_b·Z)` = 324/216/162 × 648, represented as per-check variable-index lists (and the transpose, per-variable check-index lists) for message passing.

### 2c. Encoder (systematic)
802.11n QC-LDPC is systematic (first k columns are info). Parity is solved from `H·cᵀ = 0` over GF(2): split `H = [H_info | H_parity]`; `parity = H_parity⁻¹ (H_info · info)` over GF(2). The 802.11 parity submatrix has a dual-diagonal structure enabling recursive back-substitution; equivalently, precompute the parity solve (GF(2) elimination) **once per rate** and cache it. `encode(info) = info ‖ parity`. Deterministic.

### 2d. Decoder — normalized min-sum
Operate on channel LLRs over the Tanner graph:
1. Initialize each variable node's LLR from the channel (`sc_soft_bits`, convention `L>0 ⇒ bit 0`).
2. **Check-node update:** message to variable `v` = `α · (∏ sign of other incoming) · min(|other incoming|)`, with normalization `α ≈ 0.8` (`norm_factor`, from `spec.params`).
3. **Variable-node update:** sum of channel LLR + all incoming check messages (excluding the target edge).
4. **Hard decision:** sign of the total LLR per variable; if `H·ĉᵀ = 0`, stop.
5. Repeat to `max_iters`; return the systematic info bits.

Scale-invariant (§1). Returns `DecodeResult(bits, meta={"iters":…, "converged":…})`.

### 2e. Shortening (frames ≤ k)
Zero-pad info to k, encode to n, and transmit only the **real info bits + the n−k parity bits** (the known-zero pad positions are not transmitted). `encode(info)` returns `len(info) + (n−k)` bits. `decode` reinserts a large **+∞-style LLR** (e.g. `+1e6`) at the known-zero pad positions, runs min-sum over the full n, and extracts the real info. Mirrors RS/BCH shortening. Parity length per rate: r1/2 → 324, r2/3 → 216, r3/4 → 162.

---

## 3. Catalog & profile

### 3a. `core/coding.py`
- `make_codec` gains a `CodeFamily.LDPC` branch → constructs `_LDPC` from `core.ldpc`.
- Catalog: **update** `ldpc_648_r12` params to `{soft_input: True, max_iters: 50, rate: "1/2", norm_factor: 0.8}`; **add** `ldpc_648_r23` (rate 2/3) and `ldpc_648_r34` (rate 3/4), same shape.

### 3b. `core/profiles.py`
One new PHY-distinct profile: `ldpc_bpsk = SCProfileSpec("ldpc_bpsk", SCMod.BPSK, SCProfile(sps=256), coding="ldpc_648_r12")`. **sps=256 is unique** in `SC_CATALOG` (existing: 4/8/16/32/64/128). Rate-1/2 for the clearest coding-gain demonstration.

### 3c. `validation/pipeline.py` — no change
LDPC is `soft_input=True` → already routed through the soft branch. Erasure/shortening handling is internal to `_LDPC.decode`. Loud-on-failure preserved (CRC-gated).

### 3d. synth / metric — no change
`modulate(coding=…)` calls `codec.encode` generically; `coded_link` unchanged.

---

## 4. Testing strategy

TDD; subagent-driven with per-task review. Buckets:

1. **Structural fidelity of the construction:** each rate's base matrix has the 802.11n dimensions (`m_b`×24) and the target degree profile (info block-columns weight 3; dual-diagonal parity); the expanded `H` has shape 324/216/162 × 648; the info part is **4-cycle-free (girth ≥ 6)**; `H_parity` is **invertible over GF(2)**; construction is deterministic (fixed seed → identical H across runs).
2. **Encode systematic:** `H·cᵀ = 0` for encoded codewords at all three rates; info sits in the systematic positions.
3. **Noiseless round-trip, all rates:** `encode → (clean LLRs) → min-sum decode` recovers the payload with `crc_ok`.
4. **Error correction:** min-sum corrects error patterns within the code's capability; beyond it, CRC fails **loudly** (no silent wrong payload).
5. **Scale-invariance (headline):** decoding `llr` and `k·llr` for any `k>0` yields identical bits — the property that justifies skipping recalibration.
6. **Coding gain:** LDPC BER < uncoded BER at low SNR through the full synth → `sc_soft_bits` → min-sum → CRC chain.
7. **Blind end-to-end:** an `ldpc_bpsk` dataset blind-resolves `ldpc_bpsk`, soft-decodes, and populates `coded_link`.
8. **Loud-failure + regression:** CRC loud-failure at very low SNR; full existing suite (P1–P3d) green; `mypy validation` clean.

---

## 5. Global constraints (bind every task)

- `mypy validation` is the CI gate — clean (no `Any`; `numpy.typing`). New/modified `core/*` files `mypy`-clean, minimal diffs, no whole-file `black` reformat. Lint every touched `tests/validation` file.
- IQ boundary `complex64` / internal `complex128`; LLRs `float64`; bits `uint8`.
- Deterministic encode (no RNG).
- **Loud on failure:** CRC failure / no lock / non-convergent-and-inconsistent decode ⇒ `b""` / bits that fail CRC — never a silent wrong payload at normal SNR. Existing uncoded/rep3/conv/RS/BCH paths byte-for-byte unchanged.
- **No `sc_soft_bits` recalibration** (normalized min-sum is scale-invariant); do not alter the shared soft-LLR function.
- Additive: `ldpc_648_r12` gains a working codec + params but no existing test changes behavior; existing profiles/tests unchanged; the pipeline soft branch is reused, not modified.
- Executed subagent-driven with TDD and per-task review; on a **new branch off `main`** (`feature/p3e-ldpc`); no PR/tag this sub-phase (merge decision deferred to the user).
- **Subagent-stall guard:** run only targeted test files in implementer tasks; the full-suite closing gate runs foreground / by the controller. Never background a slow test run in a subagent; never run the whole `test_blind_resolve.py` file — use `-k`.

---

## 6. Out of scope / future

- Interop with real 802.11n frames; sum-product / layered / offset min-sum decoding.
- Block lengths 1296/1944; rate 5/6.
- Turbo (P3f), polar (P3g), fountain (P3h); bit-loading (P4).
- **Docs:** on completion, mark this spec **Accepted — shipped**; add **ADR-0013** (LDPC decoder = normalized min-sum, scale-invariant → LLR recalibration not required) with a cross-note updating ADR-0006 and ADR-0008; update the `docs/design/` index and README/CLAUDE module notes if needed.
