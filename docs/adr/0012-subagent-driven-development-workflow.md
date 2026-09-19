# ADR-0012: Subagent-driven development workflow

- **Status:** Accepted
- **Date:** 2026-09-19
- **Deciders:** rbenzing (with Claude)

## Context

The signal/coding stack is a large, multi-phase build with subtle correctness
requirements (DSP, finite-field algebra, decoder conventions). Doing it in one
long context risks context pollution and shallow review; doing it ad hoc risks
latent bugs slipping through.

## Decision

Each phase follows **brainstorm → spec → plan → subagent-driven execution**:

- A design spec (now under `docs/design/`, ADR-0011) is approved before any code.
- An implementation plan carries complete, transcribable code per task.
- Execution dispatches a **fresh implementer subagent per task**, followed by a
  **task review** (spec compliance + code quality), a bounded **fix loop**, and a
  **whole-branch final review on the most capable model** per phase — tracked in a
  per-plan ledger.
- The **controller runs the closing gate** (full `pytest` suite + `mypy
  validation`) in the foreground, because slow suites stall backgrounded
  subagents; a **controller-recovery** step commits verified work when a subagent
  stalls.
- Every change is **additive** and preserves the loud-on-failure contract
  (ADR-0006).

## Consequences

### Positive
- High review coverage caught real latent bugs (e.g. the RS erasure-convention bug, found because a later phase exercised the untested path).
- Controller context stays focused on coordination.

### Negative / trade-offs
- Slower and more token-intensive than ad-hoc coding; many subagent round-trips.

### Neutral / notes
- Process artifacts (plans, ledgers) remain local/git-ignored (ADR-0011); models are chosen per role (cheap for transcription, mid for judgment, most-capable for final review).
