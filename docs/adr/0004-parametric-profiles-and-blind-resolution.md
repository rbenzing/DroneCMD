# ADR-0004: Parametric link profiles + blind resolution

- **Status:** Accepted
- **Date:** 2026-09-19
- **Deciders:** rbenzing (with Claude)

## Context

A capture arrives with no reliable prior on its PHY. We need to recognize and
demodulate many single-carrier and OFDM link types without being told which one,
and to add new link types without rewriting the resolver.

## Decision

We use **parametric link-profile catalogs** plus a **blind resolver**:

- `core/profiles.py` holds `SCProfileSpec` (single-carrier) and OFDM
  `OFDMProfile` catalogs, each keyed by a discriminant that blind resolution can
  measure — **`sps` for single-carrier**, **FFT size / cyclic-prefix for OFDM**.
- `core/blind.py` resolves blind: a **sync/family gate** followed by a
  **trial-demodulation data-symbol-EVM tiebreak** among candidates that pass the
  gate.
- Thresholds are **data-driven** — measured from the observed separation between
  genuine and non-genuine cases (e.g. `OFDM_EVM_MAX`, `OFDM_FAMILY_THRESHOLD`,
  `OFDM_SYNC_THRESHOLD`) — and keep an honest low-SNR tail rather than a
  theoretical cutoff.
- A discriminator that cannot be confirmed blind is **dropped**, not shipped
  (e.g. the `wifi_20_altpilot` layout variant, whose CPE bias did not separate).

## Consequences

### Positive
- New PHYs are added as catalog entries; the resolver generalizes.
- Thresholds reflect measured reality, with known failure modes at low SNR.

### Negative / trade-offs
- Every new profile needs a **unique blind key** (e.g. a distinct `sps`); collisions are not resolvable.
- Thresholds are empirical and must be re-measured if the channel model changes.

### Neutral / notes
- Coding is carried by the profile (`SCProfileSpec.coding`), which is how a coded PHY is recognized and decoded.
