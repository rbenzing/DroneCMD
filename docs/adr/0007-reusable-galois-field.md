# ADR-0007: Reusable GF(2^m) algebra shared by RS and BCH

- **Status:** Accepted
- **Date:** 2026-09-19
- **Deciders:** rbenzing (with Claude)

## Context

Reed-Solomon (P3c) and BCH (P3d) both need finite-field arithmetic and the same
syndrome-decoding steps (error-locator search, root finding). Implementing the
field twice would duplicate subtle, bug-prone code and risk divergent
conventions.

## Decision

We factor the shared algebra into **`core/galois.py`**, field-parametric on
`(m, prim_poly)`:

- `GF2m` — exp/log tables (generator α=2), `mul`/`div`/`inv`/`pow`, and
  polynomial ops (`poly_add`/`poly_mul`/`poly_eval`/`poly_div`), coefficients
  **highest-degree-first**. `GF256 = GF2m(8, 0x11D)` is the shared constant.
- Generic **`berlekamp_massey`** (error-locator search, "Convention A" — the
  caller supplies Forney-reduced syndromes and combines erasures) and
  **`chien_search`** (root → error position).

Reed-Solomon adds Forney magnitude evaluation (symbol errors); BCH needs only
the error *positions* and flips those bits (binary). The field is genuinely
parametric — exercised at **GF(2⁶)** and **GF(2⁸)**.

## Consequences

### Positive
- One tested field/decoder core; BCH decode is "RS minus Forney".
- Parametric on field size, so future codes over other GF(2^m) reuse it.

### Negative / trade-offs
- The syndrome/locator **orientation conventions** (`err_loc[::-1]` into Chien, the Forney-syndrome pad) are shared and must be honored exactly — an easy place to introduce a subtle bug.

### Neutral / notes
- A latent bug once hid here: RS's erasure path mixed two conventions and was caught only when BCH-style testing exercised non-empty erasures. `berlekamp_massey`'s internal budget check is intentionally guarded by Chien root-count + post-correction syndrome recheck + CRC.
