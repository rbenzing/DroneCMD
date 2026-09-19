# ADR-0011: Design records and documentation drift control

- **Status:** Accepted
- **Date:** 2026-09-19
- **Deciders:** rbenzing (with Claude)

## Context

The signal/coding stack was built as a long sequence of designed phases (P1
through P3d), but the detailed design work lived only in **local, git-ignored**
SDD artifacts (`docs/superpowers/`, `.superpowers/`). As a result the committed
docs (README, CLAUDE.md) drifted badly — their module map and capability list
described roughly the pre-P1 project, and none of OFDM, blind resolution, or the
FEC stack was discoverable in-repo. Decisions had no durable, reviewable home.

## Decision

We establish an in-repo decision/design record with an explicit split:

- **`docs/adr/`** — Nygard-style Architecture Decision Records: short, durable,
  numbered records of *why* (this file is one). The distilled, committed source
  of truth for architecture decisions.
- **`docs/design/`** — the numbered, committed **design specs** (one per phase),
  recording the detailed design of each capability.
- **`docs/superpowers/` and `.superpowers/` stay git-ignored** — implementation
  plans and per-task execution ledgers are execution scaffolding, not design
  records, and would only add noise to history.
- **README.md and CLAUDE.md are kept in sync** with the module map and
  capabilities as features land, and link to `docs/adr/` and `docs/design/`.

## Consequences

### Positive
- Decisions and designs are discoverable and reviewable in the repo; drift has a brake.
- Reviewers get the "why" (ADR) and the "how" (design spec) without the local process artifacts.

### Negative / trade-offs
- ADRs and the module map need discipline to keep current — a new obligation on each feature.

### Neutral / notes
- This ADR records the very change that created `docs/adr/` and `docs/design/` and moved the design specs out of the ignored path. New design specs are now authored directly under `docs/design/` (numbered), not the ignored location.
