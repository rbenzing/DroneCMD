# ADR-0015: Soft-demod phase-ramp fix — DD payload tracking + L&R CFO estimator

- **Status:** Accepted
- **Date:** 2026-09-19
- **Deciders:** rbenzing (with Claude)

## Context

While validating turbo coding gain (P3f), the soft-LLR demod
(`core/single_carrier.py` `sc_soft_bits`) was found to emit ~40–63% wrong-**sign**
LLRs on a large fraction of frames at low SNR (4–8 dB). No FEC corrects ~60%
sign errors, so turbo (and, latently, every soft codec: convolutional, LDPC)
was capped **by the demod, not the code** — turbo gain had to be shown on a
controlled AWGN-LLR channel ([ADR-0014](0014-turbo-max-log-map.md)), and the
issue was filed for its own investigation.

Systematic root-cause analysis (instrumented, codec-free reproduction, AWGN
only, **zero injected CFO**) found:

- The bad frames were exactly the frames with the largest-magnitude **residual
  CFO estimate**, and sign-error rate tracked `|resid_cfo|`.
- The coarse CFO grid was always correct (0); the culprit was the two-halves
  residual estimator `sc_estimate_cfo_psk` emitting a small **spurious** CFO
  from preamble noise (its CRLB noise tail) even when the true CFO is 0.
- A CFO is a **phase ramp** across the payload. The `peak2` preamble alignment
  removes only a *constant* phase, so the ramp survives, rotating later symbols
  past ±90° and flipping their signs — worse on longer frames (turbo's long
  rate-1/3 codewords hit it hardest).
- **Asymmetry:** the hard receiver (`sc_demodulate_psk`) already runs a
  decision-directed phase loop (`sc_track_phase_dd`) that tracks the ramp out;
  the soft path (`sc_aligned_payload_centers` → `sc_soft_bits`) deliberately
  stopped **before** tracking. That missing stage was the defect.

Two candidate mitigations were measured and rejected as primary fixes:
- **Coherence gating** of the estimator: refuted by data — the two-halves
  coherence of noise-only frames and genuine-CFO frames is statistically
  identical (coherence measures preamble SNR, not CFO presence), so no
  threshold separates them without suppressing real CFO corrections.
- **A "more robust" estimator alone:** the existing two-halves correlator is
  already near-optimal (lowest variance of the single-lag candidates); the
  spurious residual is the CRLB tail, not a bug.

## Decision

- **Decision-directed payload phase tracking on the soft path (load-bearing).**
  `sc_soft_bits` applies the same `sc_track_phase_dd` loop the hard receiver
  uses, before forming LLRs. It is source-agnostic — it tracks out any residual
  ramp (spurious *or* genuine) — and is a no-op on a clean burst (exact
  round-trip preserved). This alone drives low-SNR bad frames (sign-error > 20%)
  from 15/40 to **0/40** at 6 dB.
- **Lower-variance L&R CFO estimator (defense in depth).**
  `sc_estimate_cfo_psk` is replaced with a data-aided Luise & Reggiannini
  multi-lag estimator over the *known* preamble (strip the ±1 modulation with
  the reference, average the autocorrelation over lags `1..N/2`). Measured
  ~25–30% lower estimate variance at 4–8 dB than the two-halves correlator. Its
  small bias is below the noise it replaces, and the DD loop absorbs it. Shared
  by the hard receiver and the blind resolver, so both benefit.
- **No `sc_soft_bits` recalibration / no scale change.** The soft codecs remain
  scale-invariant (ADR-0013/0014); this fix is about phase, not LLR magnitude.

## Consequences

### Positive
- Genuine **end-to-end** coding gain: turbo through the real `sc_soft_bits`
  now decodes to 0 residual errors vs 35/17/5 uncoded at 4/5/6 dB (previously
  20–600× *worse* than uncoded). The fix lifts the ceiling on every soft codec,
  not just turbo. Hard receiver and blind resolver get the lower-variance CFO
  estimate for free.

### Negative / trade-offs
- The soft path now runs a per-symbol Python loop (`sc_track_phase_dd`), a
  small cost. L&R adds an `O(N/2)`-lag loop over the preamble (bounded, offline).
- Changing the shared `sc_estimate_cfo_psk` shifts hard-path and blind-resolver
  behavior slightly; verified against the existing CFO round-trip and
  blind-resolution suites (all green).

### Neutral / notes
- Coding-gain tests keep **both** framings: the controlled AWGN-LLR test
  (decoder-isolating characterization) and the new end-to-end test through the
  fixed demod. Regression guard:
  `test_sc_soft_bits_no_phase_ramp_sign_flips_low_snr`. Supersedes the
  "separate finding" note in [ADR-0014](0014-turbo-max-log-map.md).
