# ADR-0003: IQ dtype and type-safety conventions

- **Status:** Accepted
- **Date:** 2026-09-19
- **Deciders:** rbenzing (with Claude)

## Context

SDR DSP mixes many numeric kinds (complex samples, log-likelihood ratios, bits,
finite-field symbols). Inconsistent dtypes cause silent precision loss and
memory bloat; unconstrained `Any` erodes type safety in exactly the numeric code
where mistakes are hardest to see. The legacy modules predate strict typing.

## Decision

We fix these conventions across the signal/coding stack:

- **IQ boundary is `numpy.complex64`**; internal DSP promotes to `complex128`
  where precision matters.
- **LLRs are `float64`; bits are `uint8`; GF(2^m) symbols are plain Python `int`**
  inside `core/galois.py`.
- Apply windowing before FFTs; validate SNR/power; handle empty signals, DC
  offset, and drift.
- **Strict `mypy` is the CI gate on `validation/`** (no `Any`; use
  `numpy.typing`). Legacy modules (`core/`, `capture/`, `plugins/`, …) carry
  known pre-existing type debt and are **not** gated; new `core/` code stays
  clean but may use narrow, commented `# type: ignore`/`cast` to match existing
  patterns without expanding the debt class.

## Consequences

### Positive
- Predictable precision/memory; the gated surface stays `Any`-free.
- New numeric code is held to the strict bar without a repo-wide migration.

### Negative / trade-offs
- Two standards (gated vs legacy) can confuse contributors; `mypy core/...` still emits legacy findings.

### Neutral / notes
- Type aliases in use: `FrequencyHz = float`, `IQSamples = NDArray[complex64]`, `PacketBytes = bytes`.
