# ADR-0006: Soft-LLR demod path + loud-on-failure decode contract

- **Status:** Accepted
- **Date:** 2026-09-19
- **Deciders:** rbenzing (with Claude)

## Context

Soft-decision decoders (Viterbi, and later LDPC/turbo/polar) need
log-likelihood ratios, not hard bits. And a decoder that silently emits a wrong
payload at normal SNR is worse than one that fails — downstream parsing would
trust garbage. Both concerns are cross-cutting across every codec.

## Decision

- **Soft-LLR demod path:** `sc_soft_bits` / `ofdm_soft_bits` produce calibrated
  LLRs with the convention **`L > 0 ⇒ bit 0`**, calibrated via an EVM/noise-var
  estimate.
- **Routing:** the pipeline sends `soft_input` codecs through the soft branch
  (`soft_bits → deinterleave(LLRs) → decode`) and hard codecs through the hard
  branch; both then verify CRC.
- **Loud on failure (non-negotiable):** every coded decode is **CRC-gated**. On
  CRC failure, no lock, or an uncorrectable block, the pipeline returns `b""` /
  unchanged received bits that CRC rejects — **never a silent wrong payload** at
  normal SNR.
- **Calibration caveat:** the LLR scale is ~2× conservative. Soft-decision
  Viterbi is **scale-invariant** (argmax over correlations), so this is harmless
  for convolutional decoding; the RS erasure rule is made **scale-invariant by
  construction** (median-relative). Absolute-scale-sensitive decoders **must
  recalibrate** — a standing cross-phase obligation. (Update: **LDPC does not** —
  its normalized min-sum decoder is scale-invariant; see
  [ADR-0013](0013-ldpc-normalized-min-sum.md). The obligation still holds for
  turbo/polar if built with scale-sensitive decoders.)

## Consequences

### Positive
- One trustworthy failure semantic across all codes; no silent corruption.
- Soft codecs share one calibrated LLR source.

### Negative / trade-offs
- The conservative LLR scale is a latent trap for future scale-sensitive decoders — must be remembered.

### Neutral / notes
- Verified empirically: thousands of Monte-Carlo beyond-correction-bound trials produced zero silent-wrong payloads.
