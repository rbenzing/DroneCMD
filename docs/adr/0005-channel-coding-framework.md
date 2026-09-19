# ADR-0005: Channel-coding framework (Codec protocol + registry)

- **Status:** Accepted
- **Date:** 2026-09-19
- **Deciders:** rbenzing (with Claude)

## Context

The project adds several FEC families (convolutional, Reed-Solomon, BCH, and
later LDPC/turbo/polar/fountain). Without a common contract, each would wire into
the pipeline differently, and blind resolution could not know which code a PHY
carries.

## Decision

We define one coding framework in `core/coding.py`:

- A **`Codec` protocol**: `encode(Bits) -> Bits`, `decode(SoftOrHard) ->
  DecodeResult`.
- **`CodingSpec`** (name, family, k, n, params; with `.rate` and `.soft_input`)
  and a **`CODING_CATALOG`** registry, built by a **`make_codec(spec)`** factory.
- **CRC-16-CCITT framing** (`frame_with_crc`/`check_and_strip_crc`) as the
  integrity check for every coded payload.
- A **value-type-agnostic block interleaver** (`interleave`/`deinterleave`,
  `CODING_INTERLEAVE_DEPTH`) that is safe for both hard bits and soft LLRs.
- **Profile-carried coding**: an `SCProfileSpec` names its code, so blind
  resolution → decode is uniform.
- A **coded-link metric** (`coded_link`: coded BER/FER) in the validation spine.

## Consequences

### Positive
- New codecs slot in behind one interface; the pipeline and metric are code-agnostic.
- CRC gives every code a uniform integrity signal (see ADR-0006).

### Negative / trade-offs
- The single `decode(SoftOrHard)` signature pushes soft/hard handling into each codec.

### Neutral / notes
- The catalog is a complete capability sheet; families without a working codec yet raise `NotImplementedError` from `make_codec`.
