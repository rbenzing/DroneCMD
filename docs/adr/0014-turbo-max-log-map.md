# ADR-0014: Turbo decoder — max-log-MAP + extrinsic scaling (scale-invariant)

- **Status:** Accepted
- **Date:** 2026-09-19
- **Deciders:** rbenzing (with Claude)

## Context

P3f adds an LTE-style turbo code (RSC K=4, generators 13/15 octal; QPP
interleaver; rate 1/3 + punctured 1/2). Two decisions were open: the SISO
decoder algorithm, and whether turbo needs the LLR recalibration
[ADR-0006](0006-soft-llr-and-loud-on-failure.md) flags for soft codecs. As with
LDPC (ADR-0013), the exact standard (LTE) interleaver table is not reproduced —
but turbo's pieces are **formulas** (RSC generators, QPP `π(i)=(f1·i+f2·i²) mod K`),
so unlike LDPC there was no unobtainable-table problem.

## Decision

- **Decoder: iterative max-log-MAP with a constant 0.7 extrinsic-scaling
  factor** (not full log-MAP). Two SISO BCJR decoders exchange extrinsics
  through the QPP interleaver for `max_iters` (=8).
- **No LLR recalibration for turbo.** Max-log-MAP uses `max` (not `max*`); every
  metric/extrinsic is linearly homogeneous in a global input-LLR scale, and the
  0.7 factor is a *relative* scaling — so the full turbo loop's hard decision is
  **scale-invariant** (empirically: `decode(k·llr)==decode(llr)` for k∈1e-3…1e3).
  P3a's conservative `sc_soft_bits` scale is therefore harmless, as for LDPC
  (ADR-0013) and Viterbi. Full log-MAP would be scale-sensitive and is out of scope.
- **Code: 802.11n-style is LDPC's story; turbo uses a QPP interleaver with
  deterministic, bijection-verified parameters** (K=256, f1=31, f2=64) — not the
  exact LTE table. Interop out of scope; self-consistency + coding gain are the bar.

## Consequences

### Positive
- No recalibration; shared `sc_soft_bits` untouched (no risk to other codecs). Near-log-MAP performance.

### Negative / trade-offs
- Not bit-compatible with LTE turbo. Pure-Python BCJR is slow (K × iters × 2 SISO). Max-log-MAP diverges on badly-corrupted input, but so would any decoder.

### Neutral / notes
- **Coding gain is demonstrated on a controlled AWGN-LLR channel**, not through `sc_soft_bits` like the RS/BCH/LDPC gain tests. This is deliberate and honest (the textbook way coding gain is measured), and **necessary** because a **phase/sync issue in the shared soft-demod (`core/single_carrier.py` `sc_soft_bits`) emits ~40–63% wrong-*sign* LLRs on ~40% of frames at the low SNR where a strong code shows gain** — no FEC can correct 60% sign errors, so an end-to-end turbo gain test would measure that demod defect, not the code. **This demod issue is independent of turbo and was latent for the other soft codecs (dodged by higher-SNR test operating points).** It was subsequently root-caused (a spurious residual-CFO phase ramp the soft path never tracked out) and **fixed** — see [ADR-0015](0015-soft-demod-phase-ramp-fix.md); turbo now also has an end-to-end coding-gain test through the fixed `sc_soft_bits`, alongside this AWGN-LLR characterization. Turbo decoder correctness is proven by the noiseless/error-correction/scale-invariance unit tests + a codec-level noisy round-trip. Cross-refs: ADR-0006, ADR-0013, ADR-0015.
