# ADR-0013: LDPC decoder — normalized min-sum (scale-invariant)

- **Status:** Accepted
- **Date:** 2026-09-19
- **Deciders:** rbenzing (with Claude)

## Context

P3e adds LDPC, the first *iterative* soft codec. Two decisions were open: the
decoder algorithm and whether the LDPC path requires the LLR recalibration that
[ADR-0006](0006-soft-llr-and-loud-on-failure.md) flagged as a standing
obligation for soft codecs. Separately, the exact IEEE 802.11n base matrices
could not be obtained or bit-verified in this environment (paywalled standard;
scanned/binary reproductions), and interop with real 802.11n is out of scope.

## Decision

- **Decoder: normalized min-sum** (message passing over the Tanner graph;
  check-node update = sign-product × min-magnitude, scaled by a normalization
  factor α≈0.8), not sum-product/belief-propagation.
- **No LLR recalibration for LDPC.** Normalized min-sum is **scale-invariant for
  the hard decision**: every message is linearly homogeneous in a global
  input-LLR scale, so the per-iteration sign pattern, the `H·ĉᵀ=0` stop test, and
  the decoded bits are unchanged. P3a's ~2× conservative `sc_soft_bits` scale is
  therefore harmless here, exactly as for soft Viterbi (ADR-0006) and the RS
  median-relative erasure rule. **This supersedes ADR-0006's expectation that
  "LDPC must recalibrate the LLR scale."** Turbo (P3f) and polar (P3g), if
  implemented with scale-sensitive decoders, remain subject to that obligation.
- **Code: an 802.11n-*style* QC-LDPC**, not the exact IEEE tables — 802.11n
  dimensions (n=648, Z=27, rates 1/2, 2/3, 3/4) and IRA/dual-diagonal structure,
  built by a deterministic, seeded, 4-cycle-free construction (`core/ldpc.py`).
  Interop is out of scope; the guard is structural fidelity + a genuine
  coding-gain measurement, mirroring the RS "conventional basis" precedent.

## Consequences

### Positive
- Sidesteps the LLR-recalibration trap entirely; the shared `sc_soft_bits` stays untouched (no risk to Viterbi/RS/BCH).
- Near-BP performance with a simpler, hardware-standard decoder; scale invariance is a clean, testable property.

### Negative / trade-offs
- Not bit-compatible with real 802.11n (acceptable — interop out of scope).
- Pure-Python min-sum is slow (iterative); fine for T&E, not real-time.
- A heavily-shortened LDPC frame loses effective rate and can sit below the min-sum waterfall — coding-gain demonstrations use a near-design-rate payload.

### Neutral / notes
- Verified: scale-invariant decode across LLR factors 1e-3…1e3; ~344× BER improvement vs uncoded at a mid-SNR Monte-Carlo. Cross-refs: ADR-0005 (framework), ADR-0006 (soft-LLR/loud-on-failure), ADR-0008 (FEC family choices — LDPC now shipped alongside conv/RS/BCH).
