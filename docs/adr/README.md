# Architecture Decision Records

This directory holds **Architecture Decision Records (ADRs)** — short, durable,
numbered notes capturing a significant decision, its context, and its
consequences. They are the committed source of truth for *why* DroneCMD is built
the way it is. For the detailed *how* of each capability, see the design specs in
[`../design/`](../design/); see [ADR-0011](0011-design-records-and-drift-control.md)
for how these fit together and how we keep the docs from drifting.

## Format

Nygard-style: **Status · Context · Decision · Consequences**. Copy
[`0000-template.md`](0000-template.md) for a new record; give it the next number,
never renumber existing ones, and mark superseded records rather than deleting
them.

## Index

| # | Decision | Status |
|---|----------|--------|
| [0001](0001-two-layer-api.md) | Two-layer API (simple + enhanced) | Accepted |
| [0002](0002-validation-te-spine.md) | Validation & T&E spine as the correctness backbone | Accepted |
| [0003](0003-iq-dtype-and-typing-conventions.md) | IQ dtype and type-safety conventions | Accepted |
| [0004](0004-parametric-profiles-and-blind-resolution.md) | Parametric link profiles + blind resolution | Accepted |
| [0005](0005-channel-coding-framework.md) | Channel-coding framework (Codec protocol + registry) | Accepted |
| [0006](0006-soft-llr-and-loud-on-failure.md) | Soft-LLR demod path + loud-on-failure decode contract | Accepted |
| [0007](0007-reusable-galois-field.md) | Reusable GF(2^m) algebra shared by RS and BCH | Accepted |
| [0008](0008-fec-family-choices.md) | FEC family choices and decision types | Accepted |
| [0009](0009-versioning-via-git-tags.md) | Versioning derived from git tags | Accepted |
| [0010](0010-plugin-system.md) | Plugin system for protocol extensibility | Accepted |
| [0011](0011-design-records-and-drift-control.md) | Design records and documentation drift control | Accepted |
| [0012](0012-subagent-driven-development-workflow.md) | Subagent-driven development workflow | Accepted |
| [0013](0013-ldpc-normalized-min-sum.md) | LDPC decoder — normalized min-sum (scale-invariant) | Accepted |
| [0014](0014-turbo-max-log-map.md) | Turbo decoder — max-log-MAP + extrinsic scaling (scale-invariant) | Accepted |
| [0015](0015-soft-demod-phase-ramp-fix.md) | Soft-demod phase-ramp fix — DD payload tracking + L&R CFO estimator | Accepted |
| [0016](0016-polar-ca-scl.md) | Polar decoder — CA-SCL, min-sum f-node (scale-invariant) | Accepted |
