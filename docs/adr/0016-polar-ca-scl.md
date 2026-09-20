# ADR-0016: Polar decoder — CA-SCL, min-sum f-node (scale-invariant)

- **Status:** Accepted
- **Date:** 2026-09-20
- **Deciders:** rbenzing (with Claude)

## Context

P3g adds an Arıkan polar code (`core/polar.py`, n=256, rates 1/2, 1/3, 2/3),
the third iterative/soft codec after LDPC (P3e, ADR-0013) and turbo (P3f,
ADR-0014). Three decisions were open: the SC-list decoder variant, the frozen-set
construction, and — per the standing [ADR-0006](0006-soft-llr-and-loud-on-failure.md)
obligation — whether polar needs `sc_soft_bits` LLR recalibration. As with LDPC's
802.11n tables, the exact 5G-NR reliability sequence is not reproduced; polar's
pieces here are again **formulas** (Gaussian-approximation density evolution),
so there is no unobtainable-table problem, only an interop non-goal.

## Decision

- **Decoder: CRC-aided successive-cancellation list (CA-SCL), list_size=8**,
  with a **min-sum f-node** (`f(a,b) = sign(a)·sign(b)·min(|a|,|b|)`) and an
  **approximate `|LLR|`-on-disagreement path metric** (`PM += |LLR|` when a
  path's bit decision disagrees with the LLR sign), not plain SC or a
  metric-only (non-CRC-aided) list decoder.
- **No `sc_soft_bits` recalibration for polar.** Both the min-sum f-node and the
  `|LLR|` path-metric rule are linearly homogeneous in a global input-LLR scale,
  so scaling every LLR by `k > 0` scales every path metric by `k` — path
  ordering, the surviving list, and the final hard decision are all unchanged.
  CA-SCL is therefore **scale-invariant** (empirically:
  `decode(k·llr) == decode(llr)` for `k ∈ 1e-3 … 1e3`), exactly as for LDPC
  normalized min-sum (ADR-0013), turbo max-log-MAP (ADR-0014), and soft Viterbi.
  P3a's conservative `sc_soft_bits` scale is therefore harmless here too. Exact
  (`tanh`/log-MAP) f-nodes would be scale-sensitive and are out of scope.
- **Frozen set: Gaussian-approximation (GA) density evolution** at a fixed
  `design_snr_db=2.0` (measured; gives a clean waterfall for all three catalog
  rates), not the 3GPP 5G-NR reliability sequence. Deterministic given
  `(n, k, design_snr_db)`.
- **CRC reuse, one frame = one polar block.** Like `_LDPC`/`_Turbo`, `_Polar` is
  single-block per frame: the whole CRC-framed frame (`[payload | CRC-16]`) maps
  to exactly one polar block, so CA-SCL's list-selection CRC check reuses the
  existing framework CRC-16 at zero extra overhead — no per-block inner CRC, no
  multi-block ambiguity. Multi-block frames are out of scope.
- **Rate matching: shorten-from-the-end** (Wang & Liu), mirroring `_LDPC`'s
  `info_len + (n − k)` and `_Turbo`'s variable-length contract. Because
  `G = F^{⊗m}` is lower-triangular, freezing the highest-index input positions
  forces the last codeword bits to a known 0, so those bits are simply not
  transmitted; decode reinserts them as `+1e6` LLRs (the same known-zero
  convention as LDPC/turbo).
- **Profile `polar_bpsk` uses `sps=48`** — a unique small value, not the
  existing doubling pattern (4/8/16/32/64/128/256/512) the other coded BPSK
  profiles follow. Blind-resolve trial-demod cost scales with `sps` (more
  samples per symbol to trial-demod per candidate profile), and polar's
  CA-SCL decode is itself the most expensive step in the blind pipeline
  (pure-Python list decoding; see the cost note under Consequences) — so
  `sps=48` was chosen to keep blind
  resolution of `polar_bpsk` tractable rather than following the doubling
  convention, which would have made it the most expensive profile in the
  catalog on both axes at once.
- **Interop out of scope**, per the RS/LDPC/turbo precedent: reproducible,
  self-consistent construction + a genuine coding-gain measurement is the bar,
  not bit-compatibility with 5G-NR polar.

## Consequences

### Positive
- Sidesteps the LLR-recalibration trap entirely; the shared `sc_soft_bits`
  stays untouched (no risk to Viterbi/RS/BCH/LDPC/turbo). CA-SCL's CRC-aided
  list selection recovers cases where the raw lowest-path-metric candidate is
  wrong but a CRC-passing path exists in the list — a genuine gain over plain
  SC or non-CRC-aided list decoding, at zero extra CRC overhead by reusing the
  framework CRC-16.

### Negative / trade-offs
- Not bit-compatible with 5G-NR polar (acceptable — interop out of scope).
- Pure-Python SCL is slow; fine for T&E, not real-time. The current
  implementation recomputes the SC recursion from the root per bit (a
  `polar_transform` per g-branch), so its true cost is ~`O(list · n² · log n)`
  rather than the `O(list · n · log n)` of a memoized SCL — correctness is
  unaffected and n=256 stays within the test budget, but a memoized node-LLR
  cache is the obvious future optimization. The `polar_bpsk` profile's unique
  `sps=48` is partly a symptom of this cost, needed to keep blind-resolution
  runs inside the project's `<30s` targeted-test budget.
- A heavily-shortened polar frame loses effective rate and moves off the GA
  design point — coding-gain demonstrations use a near-full-block payload
  (14 B at `polar_256_128`, `(128-16)/8`), per the same lesson LDPC/turbo
  coding-gain tests already encode.

### Neutral / notes
- Verified: scale-invariant decode across LLR factors 1e-3…1e3
  (`test_polar_scale_invariance`); GA frozen set deterministic with exactly
  `k` info positions (`test_polar_roundtrip_all_rates_with_shortening`); an
  explicit L=K (no-shortening) round-trip for each catalog rate confirms
  `build_shortened_mask(code, K) == code.frozen_mask`
  (`test_polar_full_rate_l_equals_k_all_rates`); end-to-end coding gain through
  the real (ADR-0015-fixed) `sc_soft_bits` demod, mirroring the turbo
  end-to-end gain test (`test_polar_beats_uncoded_low_snr`: 0 residual errors
  polar vs 3 uncoded at 6.0 dB / 8 trials, 14 B payload, sps=8). Cross-refs:
  ADR-0005 (framework), ADR-0006 (soft-LLR/loud-on-failure), ADR-0013 (LDPC),
  ADR-0014 (turbo), ADR-0015 (soft-demod phase-ramp fix, a precondition for a
  valid end-to-end gain measurement).
