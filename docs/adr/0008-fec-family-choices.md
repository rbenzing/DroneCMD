# ADR-0008: FEC family choices and decision types

- **Status:** Accepted
- **Date:** 2026-09-19
- **Deciders:** rbenzing (with Claude)

## Context

Within the coding framework (ADR-0005) we must choose *which* FEC families to
implement and *how* each decodes (hard vs soft, and — for the algebraic codes —
whether to correct erasures). Each choice trades demonstrability, realism, and
implementation cost.

## Decision

The families shipped through P3d, each carried by a PHY-distinct profile with a
**unique `sps`** so blind resolution separates them:

- **Convolutional K=7 (133/171) + soft-decision Viterbi** — rate 1/2 with
  802.11 puncturing to 2/3 and 3/4. Soft input; scale-invariant decode. Profile
  `conv_bpsk` (sps=32).
- **Reed-Solomon over GF(2⁸)** — industry/mil-spec RS(255,223) t=16 and
  RS(255,239) t=8, used shortened. **Soft input with scale-invariant
  reliability-flagged erasures** (errors-and-erasures, `2e+f ≤ 2t`), so the
  erasure capability is exercised end-to-end. Profile `rs_bpsk` (sps=64).
- **Primitive binary BCH** — BCH(255,239) t=2, BCH(255,223) t=4 over GF(2⁸), and
  BCH(63,51) t=2 over GF(2⁶). **Hard-decision, errors-only** (decode = RS
  machinery minus Forney: locate, then flip). Profile `bch_bpsk` (sps=128).

Uncoded and rate-3 repetition (`rep3`) are the baselines.

## Consequences

### Positive
- Covers convolutional + two algebraic families with contrasting decision types (soft vs hard) and erasure handling.
- Each profile is blind-resolvable and shows genuine coding gain vs uncoded.

### Negative / trade-offs
- Pure-Python RS/BCH decode is slow (~0.7–1.5 s per corrected block) — fine for T&E, not for real-time.
- High-rate codes (e.g. RS(255,239)) only beat uncoded above a modest SNR crossover.

### Neutral / notes
- Remaining families — LDPC (P3e), turbo (P3f), polar (P3g), fountain (P3h) — and bit-loading (P4) are deferred. LDPC/turbo/polar are soft and must recalibrate LLR scale (ADR-0006).
